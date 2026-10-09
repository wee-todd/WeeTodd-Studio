import MLX

/// Pool original VSA rows without padded Q/K/V or full Float32 copies.
/// The 64-row reduction follows the pinned MLX column-reduction schedule:
/// pair r/r+32, then reduce the 32 partials with simd_sum. Keep division in MLX.
enum H3TileSummary {
  static func evaluate(query: MLXArray, key: MLXArray, value: MLXArray,
    tiles: H3FastTiles) throws -> [MLXArray] {
    guard query.ndim == 4, query.shape == key.shape, key.shape == value.shape,
      query.shape[0] == 1, (1...56).contains(query.shape[1]), query.shape[2] == tiles.rows,
      query.shape[3] == 128, query.dtype == key.dtype, key.dtype == value.dtype,
      query.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid H3 tile summary tensors.")
    }
    try Task.checkCancellation()
    let count = tiles.sizes.count, heads = query.shape[1]
    let sizes = MLXArray(tiles.sizes.map(Float.init),[1,1,count,1])
    guard query.dtype == .bfloat16, Device.defaultDevice().deviceType == .gpu else {
      let indices = MLXArray(tiles.indices)
      let mask = MLXArray(tiles.sizes.flatMap { size in
        (0..<64).map { Float($0 < size ? 1 : 0) }
      },[1,1,count*64,1]).asType(query.dtype)
      return [query,key,value].map {
        (take($0,indices,axis:2)*mask).asType(.float32)
          .reshaped([1,heads,count,64,128]).sum(axis:3)/sizes
      }
    }
    let sums = kernel([query,key,value,MLXArray(tiles.indices),
      MLXArray(tiles.sizes.map(Int32.init))],
      template:[("HEADS",heads),("TILES",count)],
      grid:(4*256,count,heads),threadGroup:(256,1,1),
      outputShapes:Array(repeating:[1,heads,count,128],count:3),
      outputDTypes:[.float32,.float32,.float32])
    let result = sums.map { $0/sizes }
    eval(result)
    try Task.checkCancellation()
    return result
  }

  // Independently implemented mapped-row reduction; no third-party kernel
  // source is embedded. Shared planes hold only 32 rows by 32 features each.
  private static let kernel = MLXFast.metalKernel(
    name:"weetodd_h3_exact_mapped_tile_summaries",
    inputNames:["q","k","v","row_map","sizes"],
    outputNames:["qc","kc","vc"],source:"""
      uint lane = thread_index_in_simdgroup;
      uint group = simdgroup_index_in_threadgroup;
      uint tid = group * 32 + lane;
      uint local_row = tid / 8;
      uint local_feature = (tid % 8) * 4;
      uint feature_base = threadgroup_position_in_grid.x * 32;
      uint tile = threadgroup_position_in_grid.y;
      uint head = threadgroup_position_in_grid.z;
      threadgroup float qpart[1024], kpart[1024], vpart[1024];
      float qa[4] = {0.0f,0.0f,0.0f,0.0f};
      float ka[4] = {0.0f,0.0f,0.0f,0.0f};
      float va[4] = {0.0f,0.0f,0.0f,0.0f};
      for (uint part = 0; part < 2; ++part) {
        uint slot = local_row + part * 32;
        long row = long(row_map[tile * 64 + slot]);
        float mask = slot < uint(sizes[tile]) ? 1.0f : 0.0f;
        for (uint i = 0; i < 4; ++i) {
          long feature = long(feature_base + local_feature + i);
          long qo = long(head)*long(q_strides[1])+row*long(q_strides[2])+feature*long(q_strides[3]);
          long ko = long(head)*long(k_strides[1])+row*long(k_strides[2])+feature*long(k_strides[3]);
          long vo = long(head)*long(v_strides[1])+row*long(v_strides[2])+feature*long(v_strides[3]);
          qa[i] = float(bfloat(float(q[qo])*mask)) + qa[i];
          ka[i] = float(bfloat(float(k[ko])*mask)) + ka[i];
          va[i] = float(bfloat(float(v[vo])*mask)) + va[i];
        }
      }
      for (uint i = 0; i < 4; ++i) {
        uint slot = local_row * 32 + local_feature + i;
        qpart[slot] = qa[i]; kpart[slot] = ka[i]; vpart[slot] = va[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < 4; ++i) {
        uint feature = group * 4 + i;
        uint slot = lane * 32 + feature;
        float qs = simd_sum(qpart[slot]);
        float ks = simd_sum(kpart[slot]);
        float vs = simd_sum(vpart[slot]);
        if (lane == 0) {
          uint output = (head*TILES+tile)*128+feature_base+feature;
          qc[output] = qs; kc[output] = ks; vc[output] = vs;
        }
      }
      """,ensureRowContiguous:false)
}
