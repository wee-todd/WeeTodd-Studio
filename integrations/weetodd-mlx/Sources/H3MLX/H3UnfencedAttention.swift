import Foundation
import MLX

/// Full-key dense attention with unchanged FP32 arithmetic and BF16 output.
/// Only the four execution fences around register loads/MMA are omitted.
enum H3UnfencedAttention {
  private static let kernel=MLXFast.metalKernel(name:"weetodd_h3_dense_unfenced_v1",inputNames:["q","k","v","scale_value"],
      outputNames:["output"],source:"""
      using namespace mlx::steel;
      using T = bfloat; using MaskType = bfloat; using AccumType = float;
      constexpr int BQ = 32, BK = 16, BD = 128, WM = BQ/8, WN = 1;
      constexpr bool align_Q = (QUERY_ROWS % BQ)==0, align_K = (KEY_ROWS % BK)==0;
      constexpr bool has_mask=false,do_causal=false,has_sinks=false;
      const device T* Q=q; const device T* K=k; const device T* V=v; device T* O=output;
      AttnParams params_value{
        1,HEADS,BD,QUERY_ROWS,KEY_ROWS,1,0,
        (QUERY_ROWS+BQ-1)/BQ,(KEY_ROWS+BK-1)/BK,QUERY_ROWS/BQ,KEY_ROWS/BK,
        QUERY_ROWS%BQ,KEY_ROWS%BK,KEY_ROWS-QUERY_ROWS,
        {q_strides[0],q_strides[1],q_strides[2]},
        {k_strides[0],k_strides[1],k_strides[2]},
        {v_strides[0],v_strides[1],v_strides[2]},
        {QUERY_ROWS*HEADS*BD,BD,HEADS*BD}
      };
      params_value.scale=scale_value;
      thread const AttnParams* params=&params_value;
      thread const AttnMaskParams* mask_params=nullptr;
      const device MaskType* mask=nullptr; const device T* sinks=nullptr;
      uint simd_lane_id=thread_index_in_simdgroup;
      uint simd_group_id=simdgroup_index_in_threadgroup;
      uint3 tid=threadgroup_position_in_grid,lid=thread_position_in_threadgroup;
      """+H3UnfencedSteelAttentionSource.body,header:H3SteelAttentionSource.header,ensureRowContiguous:false)

  static func evaluate(query:MLXArray,key:MLXArray,value:MLXArray,range:Range<Int>,
    maximumStageBytes:UInt64) throws -> MLXArray {
    try Task.checkCancellation()
    guard H3RegenerateAttentionPlan.unfencedRuntimeIsQualified(ProcessInfo.processInfo.operatingSystemVersion) else {
      throw H3CheckpointError.invalid("Unfenced H3 attention requires its qualified Metal compiler/OS.")
    }
    return try MLX.withError {error in
      let sliced=try H3RegenerateAttention.validatedQuery(queryOwner:query,keysOwner:key,valuesOwner:value,range:range)
      try error.check()
      let rows=range.count,bytes=UInt64(rows*56*128*2),active=UInt64(Memory.activeMemory)
      guard bytes<=UInt64(GPU.deviceInfo().maxBufferSize),active<=maximumStageBytes,bytes<=maximumStageBytes-active else {
        throw H3CheckpointError.invalid("Unfenced H3 attention output exceeds stage memory or real buffer bounds.")
      }
      let output:MLXArray
      if range.count<=8 {
        output=try H3RegenerateAttention.forcedAttend(queryOwner:query,keysOwner:key,valuesOwner:value,range:range)
      } else {
        output=kernel([sliced,key,value,MLXArray(1/Float(128).squareRoot())],
        template:[("HEADS",56),("QUERY_ROWS",rows),("KEY_ROWS",key.shape[2])],
        grid:(((rows+31)/32)*32,56*4,1),threadGroup:(32,4,1),
        outputShapes:[[1,rows,56,128]],outputDTypes:[.bfloat16])[0].transposed(0,2,1,3)
      }
      try error.check();eval(output);try error.check();try Task.checkCancellation()
      guard UInt64(Memory.activeMemory)<=maximumStageBytes else {
        throw H3CheckpointError.invalid("Unfenced H3 attention exceeded its stage memory budget.")
      }
      return output
    }
  }
}
