import Foundation
import MLX
import LTX25Engine

/// Dedicated shifted BF16 neighborhood kernels. No dense global attention and
/// no gathered kernel-volume×channel buffers. Independently authored from the
/// trained operator contract; experimental math remains separately selectable.
enum MLXDiffusionVideoMetal {
  private static let normalize=MLXFast.metalKernel(name:"weetodd_diffvae_head_norm_adjacent_rope_v1",
    inputNames:["x","weight","time_table","height_table","width_table"],outputNames:["out"],source:"""
    uint lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x;
    uint local_query=group/HEADS;
    uint head=group%HEADS;
    uint global_query=local_query+QUERY_START;
    uint voxel=global_query%(FRAMES*HEIGHT*WIDTH);
    uint t=voxel/(HEIGHT*WIDTH),h=(voxel/WIDTH)%HEIGHT,w=voxel%WIDTH;
    ulong offset=(ulong(local_query)*HEADS+head)*64;
    uint first=2*lane;
    float a=float(x[offset+first]),b=float(x[offset+first+1]);
    float inverse=rsqrt(simd_sum(a*a+b*b)/64.0f+1.0e-6f);
    a=a*inverse*float(weight[first]);b=b*inverse*float(weight[first+1]);
    float cosine,sine;
    if(lane<8){uint i=(t*8+lane)*2;cosine=time_table[i];sine=time_table[i+1];}
    else if(lane<20){uint i=(h*12+lane-8)*2;cosine=height_table[i];sine=height_table[i+1];}
    else{uint i=(w*12+lane-20)*2;cosine=width_table[i];sine=width_table[i+1];}
    out[offset+first]=bfloat(a*cosine-b*sine);
    out[offset+first+1]=bfloat(a*sine+b*cosine);
    """)
  private static let attention=MLXFast.metalKernel(name:"weetodd_diffvae_shifted_neighborhood_v1",
    inputNames:["q","k","v"],outputNames:["out"],source:"""
    uint lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x;
    uint query=group/HEADS,head=group%HEADS;
    uint global_query=query+QUERY_START;
    uint volume=FRAMES*HEIGHT*WIDTH;
    uint batch=global_query/volume,voxel=global_query%volume;
    int t=int(voxel/(HEIGHT*WIDTH)),h=int((voxel/WIDTH)%HEIGHT),w=int(voxel%WIDTH);
    int t0=clamp(t-KT/2,0,FRAMES-KT),h0=clamp(h-KH/2,0,HEIGHT-KH),w0=clamp(w-KW/2,0,WIDTH-KW);
    ulong qo=(ulong(query)*HEADS+head)*64+2*lane;
    float qa=float(q[qo]),qb=float(q[qo+1]);
    float maximum=-INFINITY;
    for(int dt=0;dt<KT;++dt){for(int dh=0;dh<KH;++dh){for(int dw=0;dw<KW;++dw){
      ulong row=ulong(batch)*volume+ulong((t0+dt)*HEIGHT+h0+dh)*WIDTH+w0+dw;
      ulong ko=(row*HEADS+head)*64+2*lane;
      float dot=simd_sum(qa*float(k[ko])+qb*float(k[ko+1]))*0.125f;
      maximum=max(maximum,dot);
    }}}
    float denominator=0.0f;
    for(int dt=0;dt<KT;++dt){for(int dh=0;dh<KH;++dh){for(int dw=0;dw<KW;++dw){
      ulong row=ulong(batch)*volume+ulong((t0+dt)*HEIGHT+h0+dh)*WIDTH+w0+dw;
      ulong ko=(row*HEADS+head)*64+2*lane;
      float dot=simd_sum(qa*float(k[ko])+qb*float(k[ko+1]))*0.125f;
      denominator+=exp(dot-maximum);
    }}}
    float a=0.0f,b=0.0f;
    for(int dt=0;dt<KT;++dt){for(int dh=0;dh<KH;++dh){for(int dw=0;dw<KW;++dw){
      ulong row=ulong(batch)*volume+ulong((t0+dt)*HEIGHT+h0+dh)*WIDTH+w0+dw;
      ulong ko=(row*HEADS+head)*64+2*lane;
      float dot=simd_sum(qa*float(k[ko])+qb*float(k[ko+1]))*0.125f;
      float probability=exp(dot-maximum)/denominator;
      a+=probability*float(v[ko]);b+=probability*float(v[ko+1]);
    }}}
    out[qo]=bfloat(a);out[qo+1]=bfloat(b);
    """)
  static func normRotary(_ x:MLXArray,weight:MLXArray,shape:[Int],queryStart:Int) throws -> MLXArray {
    guard shape.count == 6,shape[5] == 64,x.ndim == 3,x.shape[1...] == [shape[4],64],
      x.dtype == .bfloat16,weight.dtype == .bfloat16,weight.shape == [64],queryStart>=0,
      queryStart+x.shape[0]<=shape[0]*shape[1]*shape[2]*shape[3] else {
      throw LTXError.invalid("Experimental DiffVAE Metal normalization requires validated BF16 head64 queries.")
    }
    try Task.checkCancellation()
    var tables:[MLXArray]=[]
    for (length,dimension) in zip(Array(shape[1...3]),[16,24,24]) {
      let frequency=exp(-Float(log(10000.0))*MLXArray(stride(from:0,to:dimension,by:2)).asType(.float32)/Float(dimension))
      let angles=MLXArray(0..<length).asType(.float32).expandedDimensions(axis:1)*frequency.expandedDimensions(axis:0)
      let table=stacked([cos(angles),sin(angles)],axis:-1);eval(table);tables.append(table)
    }
    let groups=try MLXDiffusionVideoPlan.product([x.shape[0],shape[4],32],limit:Int(Int32.max))
    return normalize([x,weight]+tables,template:[("HEADS",shape[4]),("FRAMES",shape[1]),("HEIGHT",shape[2]),("WIDTH",shape[3]),("QUERY_START",queryStart)],
      grid:(groups,1,1),threadGroup:(32,1,1),outputShapes:[x.shape],outputDTypes:[.bfloat16])[0]
  }
  static func attend(q:MLXArray,k:MLXArray,v:MLXArray,shape:[Int],kernel:[Int],queryStart:Int) throws -> MLXArray {
    guard shape.count == 6,shape[5] == 64,k.shape == shape,v.shape == shape,q.ndim == 3,
      q.shape[1...] == [shape[4],64],q.dtype == .bfloat16,k.dtype == .bfloat16,v.dtype == .bfloat16,
      kernel.count == 3,kernel.allSatisfy({ $0>0 && $0%2 == 1 }),
      zip(Array(shape[1...3]),kernel).allSatisfy({ $0 >= $1 }),queryStart>=0,
      queryStart+q.shape[0]<=shape[0]*shape[1]*shape[2]*shape[3] else {
      throw LTXError.invalid("Experimental DiffVAE Metal attention requires complete BF16 head64 shifted neighborhoods.")
    }
    try Task.checkCancellation()
    let groups=try MLXDiffusionVideoPlan.product([q.shape[0],shape[4],32],limit:Int(Int32.max))
    return attention([q,k,v],template:[("HEADS",shape[4]),("FRAMES",shape[1]),("HEIGHT",shape[2]),("WIDTH",shape[3]),("QUERY_START",queryStart),
      ("KT",kernel[0]),("KH",kernel[1]),("KW",kernel[2])],grid:(groups,1,1),threadGroup:(32,1,1),outputShapes:[q.shape],outputDTypes:[.bfloat16])[0]
  }
}
