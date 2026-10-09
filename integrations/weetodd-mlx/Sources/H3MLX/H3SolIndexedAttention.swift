import MLX

/// Experimental Sol consumer only. No VSA compression/gate or sampler routing.
/// Original K/V row/head views remain in place; only pooled means are compact.
enum H3SolIndexedAttention {
  private static let kernel = MLXFast.metalKernel(
    name: "weetodd_h3_sol_original_rows_bf16_d128",
    inputNames: ["q", "k", "v", "kc", "vc", "route_bits", "tile_sizes", "scale_value"],
    outputNames: ["output"], source: """
      using namespace mlx::steel;
      using T = bfloat;
      using MaskType = bfloat;
      using AccumType = float;
      constexpr int BQ = 32, BK = 16, BD = 128, WM = 4, WN = 1;
      constexpr bool align_Q = true, align_K = true, has_mask = false;
      constexpr bool do_causal = false, has_sinks = false;
      const device T* Q = q;
      const device T* K = k;
      const device T* V = v;
      const device T* KC = kc;
      const device T* VC = vc;
      auto ROUTE_BITS = route_bits;
      auto TILE_SIZES = tile_sizes;
      device T* O = output;
      AttnParams params_value{
        1, HEADS, BD, QUERY_ROWS, KEY_ROWS, 1, 0, QUERY_ROWS / BQ,
        KEY_ROWS / BK, QUERY_ROWS / BQ, KEY_ROWS / BK, 0, 0, 0,
        {q_strides[0], q_strides[1], q_strides[2]},
        {k_strides[0], k_strides[1], k_strides[2]},
        {v_strides[0], v_strides[1], v_strides[2]},
        {HEADS * QUERY_ROWS * BD, QUERY_ROWS * BD, BD}
      };
      params_value.scale = scale_value;
      thread const AttnParams* params = &params_value;
      thread const AttnMaskParams* mask_params = nullptr;
      const device MaskType* mask = nullptr;
      const device T* sinks = nullptr;
      uint simd_lane_id = thread_index_in_simdgroup;
      uint simd_group_id = simdgroup_index_in_threadgroup;
      uint3 tid = threadgroup_position_in_grid;
      uint3 lid = thread_position_in_threadgroup;
      """ + H3SolSteelAttentionSource.body,
    header: H3SolSteelAttentionSource.header, ensureRowContiguous: false)

  static func evaluate(query: MLXArray, key: MLXArray, value: MLXArray,
    prepared: H3SolRouting.Prepared) throws -> MLXArray {
    try evaluate(query: query, key: key, value: value,
      keyMeans: prepared.keyMeansBF16, valueMeans: prepared.valueMeansBF16,
      exactRouteBits: prepared.exactRouteBits, geometry: prepared.geometry)
  }

  static func evaluate(query: MLXArray, key: MLXArray, value: MLXArray,
    keyMeans: MLXArray, valueMeans: MLXArray, exactRouteBits: MLXArray,
    geometry: H3SolGeometry) throws -> MLXArray {
    try Task.checkCancellation()
    let rows = geometry.rows, heads = geometry.heads
    let blocks = (rows + 63) / 64, words = (blocks + 31) / 32
    let shape = [1, heads, rows, 128], pooledShape = [1, heads, blocks, 128]
    guard Device.defaultDevice().deviceType == .gpu,
      geometry.dimension == 128, geometry.blockSize == 64, geometry.queryBlockSize == 64,
      [query, key, value].allSatisfy({ $0.shape == shape && $0.dtype == .bfloat16 }),
      [keyMeans, valueMeans].allSatisfy({ $0.shape == pooledShape && $0.dtype == .bfloat16 }),
      exactRouteBits.shape == [1, heads, blocks, words], exactRouteBits.dtype == .uint32 else {
      throw H3CheckpointError.invalid("Invalid Sol BF16 indexed attention geometry or storage.")
    }
    // Runtime strides are reliable after materialization. Keep original row
    // and head strides, reject feature stepping/reversal rather than copying
    // the large Q/K/V payload into gathered storage.
    eval([query, key, value, keyMeans, valueMeans, exactRouteBits])
    guard [query, key, value].allSatisfy({
      $0.strides.last == 1 && $0.strides.allSatisfy { $0 >= 0 }
    }) else {
      throw H3CheckpointError.invalid("Sol indexed attention requires contiguous features and nonnegative strides.")
    }
    let meansK = contiguous(keyMeans), meansV = contiguous(valueMeans)
    let bits = contiguous(exactRouteBits)
    let sizes = MLXArray((0..<blocks).map { Int32(min(64, rows - $0 * 64)) })
    let result = kernel([query, key, value, meansK, meansV, bits, sizes, MLXArray(geometry.scale)],
      template: [("HEADS", heads), ("QUERY_ROWS", rows), ("KEY_ROWS", rows),
        ("QUERY_TILES", blocks), ("ROUTE_WORDS", words)],
      grid: (((rows + 31) / 32) * 32, heads * 4, 1), threadGroup: (32, 4, 1),
      outputShapes: [shape], outputDTypes: [.bfloat16])[0]
    eval(result)
    try Task.checkCancellation()
    return result
  }
}
