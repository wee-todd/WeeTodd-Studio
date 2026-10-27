import MLX

/// M3-compatible indexed Steel attention plus small mapped-row reduction and
/// restore kernels. No gathered K/V tensors or token-by-token mask exist.
enum H3OmniMetal {
  static func summaries(query: MLXArray,key: MLXArray,value: MLXArray,
    plan:H3OmniVSAPlan) throws -> [MLXArray] {
    try Task.checkCancellation()
    let count=plan.sizes.count,heads=query.shape[1]
    let sums=summaryKernel([query,key,value,MLXArray(plan.indices),MLXArray(plan.sizes.map(Int32.init))],
      template:[("HEADS",heads),("TILES",count)],grid:(128*32,count,heads),threadGroup:(256,1,1),
      outputShapes:Array(repeating:[1,heads,count,128],count:3),outputDTypes:[.float32,.float32,.float32])
    let sizes=MLXArray(plan.sizes.map(Float.init),[1,1,count,1])
    let result=sums.map{$0/sizes};eval(result)
    return result
  }

  static func attend(query:MLXArray,key:MLXArray,value:MLXArray,routes:MLXArray,
    plan:H3OmniVSAPlan) throws -> MLXArray {
    try Task.checkCancellation()
    let heads=query.shape[1],count=plan.sizes.count,sparse=count-plan.denseTiles
    let result=attentionKernel([query,key,value,routes,MLXArray(plan.sizes.map(Int32.init)),MLXArray(plan.indices)],
      template:[("HEADS",heads),("QUERY_ROWS",sparse*128),("KEY_ROWS",count*128),
        ("PREFIX_TILES",plan.denseTiles),("VIDEO_TILES",sparse),("ROUTE_TILES",plan.selectedTilesPerQuery)],
      grid:(sparse*128,heads*4,1),threadGroup:(32,4,1),
      outputShapes:[[1,sparse*128,heads,128]],outputDTypes:[.bfloat16],initValue:0)[0].transposed(0,2,1,3)
    eval(result)
    return result
  }

  static func restore(dense:MLXArray,sparse:MLXArray,compression:MLXArray,
    gate:MLXArray,plan:H3OmniVSAPlan) throws -> MLXArray {
    try Task.checkCancellation()
    let result=restoreKernel([dense,sparse,compression,gate,MLXArray(plan.rowSlots),MLXArray(plan.denseOrdinals)],
      template:[("HEADS",gate.shape[1]),("ROWS",plan.rows),("DENSE_TILES",plan.denseTiles)],
      grid:(gate.size,1,1),threadGroup:(256,1,1),outputShapes:[gate.shape],outputDTypes:[.bfloat16])[0]
    eval(result)
    return result
  }

  private static let summaryKernel=MLXFast.metalKernel(
    name:"weetodd_omni128_mapped_summaries",inputNames:["q","k","v","row_map","sizes"],
    outputNames:["qc","kc","vc"],source:"""
      uint lane=thread_index_in_simdgroup;
      uint feature=thread_position_in_grid.x/32;
      uint tile=threadgroup_position_in_grid.y;
      uint head=threadgroup_position_in_grid.z;
      float qa=0,ka=0,va=0;
      for(uint slot=lane;slot<128;slot+=32) {
        if(slot<uint(sizes[tile])) {
          long row=long(row_map[tile*128+slot]);
          qa+=float(q[long(head)*q_strides[1]+row*q_strides[2]+long(feature)*q_strides[3]]);
          ka+=float(k[long(head)*k_strides[1]+row*k_strides[2]+long(feature)*k_strides[3]]);
          va+=float(v[long(head)*v_strides[1]+row*v_strides[2]+long(feature)*v_strides[3]]);
        }
      }
      float qs=simd_sum(qa),ks=simd_sum(ka),vs=simd_sum(va);
      if(lane==0) {
        uint out=(head*TILES+tile)*128+feature;
        qc[out]=qs;kc[out]=ks;vc[out]=vs;
      }
      """,ensureRowContiguous:false)

  private static let attentionKernel=MLXFast.metalKernel(
    name:"weetodd_omni128_mapped_bf16_d128",
    inputNames:["q","k","v","routes","tile_sizes","row_map"],outputNames:["output"],
    source:"""
      using namespace mlx::steel;
      using T=bfloat;using MaskType=bfloat;using AccumType=float;
      constexpr int BQ=32,BK=16,BD=128,WM=4,WN=1;
      constexpr bool align_Q=true,align_K=true,has_mask=false;
      constexpr bool do_causal=false,has_sinks=false;
      const device T* Q=q;const device T* K=k;const device T* V=v;
      auto ROUTES=routes;auto TILE_SIZES=tile_sizes;auto ROW_MAP=row_map;
      device T* O=output;
      AttnParams params_value{
        1,HEADS,BD,QUERY_ROWS,KEY_ROWS,1,0.08838834764831845f,
        QUERY_ROWS/BQ,KEY_ROWS/BK,QUERY_ROWS/BQ,KEY_ROWS/BK,0,0,0,
        {q_strides[0],q_strides[1],q_strides[2]},
        {k_strides[0],k_strides[1],k_strides[2]},
        {v_strides[0],v_strides[1],v_strides[2]},
        {QUERY_ROWS*HEADS*BD,BD,HEADS*BD}
      };
      thread const AttnParams* params=&params_value;
      thread const AttnMaskParams* mask_params=nullptr;
      const device MaskType* mask=nullptr;const device T* sinks=nullptr;
      uint simd_lane_id=thread_index_in_simdgroup;
      uint simd_group_id=simdgroup_index_in_threadgroup;
      uint3 tid=threadgroup_position_in_grid;uint3 lid=thread_position_in_threadgroup;
      """+H3SteelAttentionSource.mappedBody(tileRows:128),
    header:H3SteelAttentionSource.header+H3SteelAttentionSource.originalRowLoader,ensureRowContiguous:false)

  private static let restoreKernel=MLXFast.metalKernel(
    name:"weetodd_omni128_signed_compression_restore",
    inputNames:["dense","sparse","compression","gate","slots","ordinals"],outputNames:["output"],source:"""
      uint i=thread_position_in_grid.x;
      if(i>=HEADS*ROWS*128) return;
      long d=i%128,row=(i/128)%ROWS,head=i/(ROWS*128);
      long slot=slots[row],tile=slot/128,ordinal=ordinals[row];
      bfloat attention;
      if(ordinal>=0) {
        attention=dense[head*dense_strides[1]+ordinal*dense_strides[2]+d*dense_strides[3]];
      } else {
        long r=slot-DENSE_TILES*128;
        attention=sparse[head*sparse_strides[1]+r*sparse_strides[2]+d*sparse_strides[3]];
      }
      float pooled=compression[head*compression_strides[1]+tile*compression_strides[2]+d*compression_strides[3]];
      float g=float(gate[head*gate_strides[1]+row*gate_strides[2]+d*gate_strides[3]]);
      // Upstream multiplies FP32 compression by the raw signed linear gate,
      // then casts that product before adding to the BF16 attention result.
      bfloat product=bfloat(pooled*g);
      output[i]=bfloat(float(attention)+float(product));
      """,ensureRowContiguous:false)
}
