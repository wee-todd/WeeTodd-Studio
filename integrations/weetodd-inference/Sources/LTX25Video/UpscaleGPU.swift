import Foundation
import Metal

extension VideoGPU {
  /// Group statistics span all temporal/spatial sites. Residual addition is after
  /// the affine normalization and before SiLU; it is not a pixel RMS operation.
  func groupNormSilu(_ x: VideoTensor, weight: [Float], bias: [Float], groups: Int = 32,
    residual: VideoTensor? = nil, checkCancelled: () throws -> Void = { try Task.checkCancellation() }) throws -> VideoTensor {
    guard groups > 0, groups <= 1024, x.shape[3] % groups == 0,
      weight.count == x.shape[3], bias.count == weight.count,
      weight.allSatisfy(\.isFinite), bias.allSatisfy(\.isFinite),
      residual == nil || residual!.shape == x.shape else { throw VideoDecodeError.invalid("Invalid upscaler group normalization.") }
    try checkCancelled()
    return try autoreleasepool {
      let w = try tensor(weight,shape: [1,1,1,weight.count])
      let b = try tensor(bias,shape: [1,1,1,bias.count])
      let out = try buffer(count: x.count), command = try makeCommand()
      guard let encoder = command.makeComputeCommandEncoder(), let state = pipelines["group_norm_silu"] else {
        throw VideoDecodeError.invalid("Cannot encode group normalization.")
      }
      encoder.setComputePipelineState(state)
      for (i,buffer) in [x.buffer,w.buffer,b.buffer,residual?.buffer ?? x.buffer,out].enumerated() {
        encoder.setBuffer(buffer,offset: 0,index: i)
      }
      let params = [x.count/x.shape[3],x.shape[3],groups,residual == nil ? 0 : 1].map(UInt32.init)
      params.withUnsafeBytes { encoder.setBytes($0.baseAddress!,length: $0.count,index: 5) }
      encoder.dispatchThreadgroups(MTLSize(width: groups,height: 1,depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256,height: 1,depth: 1))
      encoder.endEncoding(); try finish(command); try checkCancelled()
      return VideoTensor(buffer: out,shape: x.shape)
    }
  }

  static let upscaleShader = """
  kernel void group_norm_silu(device const float* x [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],device const float* residual [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint C=p[1], width=C/p[2], count=p[0]*width;
    float sum=0;
    for(uint i=lane;i<count;i+=256) sum+=x[(i/width)*C+group*width+i%width];
    sums[lane]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step>0;step/=2) {
      if(lane<step) sums[lane]+=sums[lane+step];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean=sums[0]/float(count);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0;
    for(uint i=lane;i<count;i+=256) { float d=x[(i/width)*C+group*width+i%width]-mean; sum+=d*d; }
    sums[lane]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step>0;step/=2) {
      if(lane<step) sums[lane]+=sums[lane+step];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse=rsqrt(sums[0]/float(count)+1e-5f);
    for(uint i=lane;i<count;i+=256) {
      uint c=group*width+i%width,index=(i/width)*C+c;
      float value=(x[index]-mean)*inverse*weight[c]+bias[c];
      if(p[3]) value+=residual[index];
      out[index]=value/(1.0f+exp(-value));
    }
  }
  """
}
