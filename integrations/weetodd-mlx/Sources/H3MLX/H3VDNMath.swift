import Foundation
import MLX

/// Independent Swift implementation of the released VDN state algebra.
/// FP32 Cholesky and triangular solve reuse this project's verified Metal
/// algorithm; no matrix round trip through Python or approximate solver.
enum H3VDNMath {
  static func retention(frameMeans:MLXArray,down:MLXArray,up:MLXArray,
    bias:MLXArray,logScale:MLXArray,heads:Int,dim:Int) throws -> MLXArray {
    guard frameMeans.ndim == 2,frameMeans.dtype == .float32,frameMeans.shape[0] > 0,
      (1...56).contains(heads),(1...128).contains(dim),down.ndim == 2,
      down.shape[0] > 0,down.shape[1] == frameMeans.shape[1],
      up.shape == [heads*dim,down.shape[0]],bias.shape == [heads*dim],
      logScale.shape == [heads],
      [down,up,bias,logScale].allSatisfy({ $0.dtype.isFloatingPoint }) else {
      throw H3CheckpointError.invalid("Invalid VDN FP32 retention projections.")
    }
    let hidden=matmul(frameMeans,down.asType(.float32).T)
    let delta=(matmul(hidden,up.asType(.float32).T)+bias.asType(.float32))
      .reshaped([frameMeans.shape[0],heads,dim])
    return exp(-exp(logScale.asType(.float32)).reshaped([1,heads,1])
      * logAddExp(delta,MLXArray(Float(0))))
  }

  private static let cholesky=MLXFast.metalKernel(name:"weetodd_h3_vdn_cholesky_fp32",
    inputNames:["a"],outputNames:["l"],source:"""
    uint row=thread_position_in_threadgroup.x;
    uint batch=threadgroup_position_in_grid.x;
    uint base=batch*D*D;
    float values[D];
    for(uint j=0;j<D;++j) values[j]=a[base+row*D+j];
    threadgroup float column[D];
    threadgroup float diagonal;
    for(uint k=0;k<D;++k) {
      if(row==k) diagonal=metal::sqrt(values[k]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float entry=row>=k ? values[k]/diagonal : 0.0f;
      column[row]=entry;l[base+row*D+k]=entry;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(row>k) for(uint j=k+1;j<=row;++j) values[j]=metal::fma(-entry,column[j],values[j]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """)
  private static let triangular=MLXFast.metalKernel(name:"weetodd_h3_vdn_triangular_inverse_fp32",
    inputNames:["l"],outputNames:["result"],source:"""
    uint column=thread_position_in_threadgroup.x;
    uint batch=threadgroup_position_in_grid.x;
    uint base=batch*D*D;
    float values[D];
    for(uint row=0;row<D;++row) {
      float value=row==column ? 1.0f : 0.0f;
      for(uint j=0;j<row;++j) value=metal::fma(-l[base+row*D+j],values[j],value);
      values[row]=value/l[base+row*D+row];
      result[base+row*D+column]=values[row];
    }
    """)
  private static let temporalKernel=MLXFast.metalKernel(name:"weetodd_h3_vdn_temporal_five_tap",
    inputNames:["spatial","weight"],outputNames:["result"],source:"""
    uint i=thread_position_in_grid.x;
    if(i>=SIZE) return;
    int frame=i/FRAME_STRIDE;uint channel=i%CHANNELS;T sum=T(0);
    for(int tap=0;tap<5;++tap) {
      int source_frame=frame+tap-2;T product=T(0);
      if(source_frame>=0 && source_frame<FRAMES) {
        uint source=source_frame*FRAME_STRIDE+i%FRAME_STRIDE;
        product=T(float(spatial[source])*float(weight[channel*5+tap]));
      }
      sum=T(float(sum)+float(product));
    }
    result[i]=sum;
    """)

  static func inverse(_ matrix:MLXArray) throws -> MLXArray {
    guard matrix.ndim >= 2,matrix.dtype == .float32,
      (1...128).contains(matrix.dim(-1)),matrix.dim(-2) == matrix.dim(-1),
      matrix.size > 0 else { throw H3CheckpointError.invalid("VDN solve requires finite FP32 square matrices of width at most 128.") }
    try Task.checkCancellation()
    let dimension=matrix.dim(-1)
    let lower=cholesky([matrix],template:[("D",dimension)],
      grid:(matrix.size/dimension,1,1),threadGroup:(dimension,1,1),
      outputShapes:[matrix.shape],outputDTypes:[.float32])[0]
    let invLower=triangular([lower],template:[("D",dimension)],
      grid:(matrix.size/dimension,1,1),threadGroup:(dimension,1,1),
      outputShapes:[matrix.shape],outputDTypes:[.float32])[0]
    let result=matmul(invLower.swappedAxes(-1,-2),invLower)
    guard MLX.isFinite(result).all().item(Bool.self) else {
      throw H3CheckpointError.invalid("VDN Cholesky solve failed; no approximate result was accepted.")
    }
    return result
  }

  static func temporal(_ spatial:MLXArray,weights:MLXArray) throws -> MLXArray {
    guard spatial.ndim == 4,spatial.shape.allSatisfy({ $0 > 0 }),
      spatial.dtype.isFloatingPoint,weights.shape == [spatial.dim(-1),1,5] else {
      throw H3CheckpointError.invalid("VDN temporal filter requires FHWC and C-by-1-by-5 weights.")
    }
    return temporalKernel([spatial,weights.asType(spatial.dtype)],
      template:[("T",spatial.dtype),("SIZE",spatial.size),("CHANNELS",spatial.dim(-1)),
        ("FRAMES",spatial.shape[0]),("FRAME_STRIDE",spatial.size/spatial.shape[0])],
      grid:(spatial.size,1,1),threadGroup:(256,1,1),outputShapes:[spatial.shape],outputDTypes:[spatial.dtype])[0]
  }

  static func scan(transitions:MLXArray,injections:MLXArray,start:MLXArray) throws
    -> (prefix:MLXArray,suffix:MLXArray) {
    guard transitions.ndim == 4,transitions.shape == injections.shape,
      transitions.dtype == .float32,injections.dtype == .float32,start.dtype == .float32,
      transitions.shape[0] > 0,Array(transitions.shape.dropFirst()) == start.shape,
      transitions.dim(-1) == transitions.dim(-2) else {
      throw H3CheckpointError.invalid("Invalid VDN recurrence state.")
    }
    var state=start;var prefix:[MLXArray]=[];var suffix:[MLXArray]=[]
    for frame in 0..<transitions.shape[0] {
      try Task.checkCancellation();state=addMM(injections[frame],state,transitions[frame]);prefix.append(state)
    }
    state=start
    for frame in (0..<transitions.shape[0]).reversed() {
      try Task.checkCancellation();state=addMM(injections[frame],state,transitions[frame]);suffix.append(state)
    }
    return (stacked(prefix),stacked(Array(suffix.reversed())))
  }

  static func gather(prefix:MLXArray,suffix:MLXArray,alpha:MLXArray,text:MLXArray,
    bounds:[(Int,Int)]) throws -> MLXArray {
    guard prefix.ndim == 4,prefix.shape == suffix.shape,prefix.dtype == .float32,
      suffix.dtype == .float32,alpha.dtype == .float32,text.dtype == .float32,
      bounds.count == prefix.shape[0],Array(prefix.shape.dropFirst()) == text.shape,
      alpha.shape == [prefix.shape[0],prefix.shape[1],prefix.dim(-1)],
      bounds.allSatisfy({ $0.0 <= $0.1 }) else {
      throw H3CheckpointError.invalid("Invalid VDN complement-window state.")
    }
    let frames=prefix.shape[0]
    let logs=concatenated([MLXArray.zeros([1,alpha.shape[1],alpha.shape[2]],dtype:.float32),
      log(maximum(alpha,MLXArray(Float(1e-12)))).cumsum(axis:0)],axis:0)
    let before=concatenated([text.expandedDimensions(axis:0),prefix],axis:0)
    let after=concatenated([suffix,text.expandedDimensions(axis:0)],axis:0)
    let lower=MLXArray(bounds.map { Int32(min(max($0.0,0),frames)) })
    let upper=MLXArray(bounds.map { Int32($0.1 >= frames-1 ? frames : max($0.1+1,0)) })
    let rows=MLXArray(0..<frames)
    let decayBefore=exp(take(logs,rows+1,axis:0)-take(logs,lower,axis:0)).expandedDimensions(axis:2)
    let decayAfter=exp(take(logs,upper,axis:0)-take(logs,rows,axis:0)).expandedDimensions(axis:2)
    return take(before,lower,axis:0)*decayBefore+take(after,upper,axis:0)*decayAfter
  }
}
