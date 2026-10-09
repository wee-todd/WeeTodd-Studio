import MLX

/// Reads routed 64-row tiles in place. The native 32×16 BF16/FP32
/// accumulator order is retained; no gathered K/V or token mask is allocated.
enum H3IndexedAttention {
  private static let kernel = makeKernel(originalRows:false)
  private static let originalKernel = makeKernel(originalRows:true)
  private static func makeKernel(originalRows:Bool) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
    name: originalRows ? "weetodd_h3_vsa_original_rows_bf16_d128" : "weetodd_h3_vsa_indexed_padded_bf16_d128",
    inputNames: ["q", "k", "v", "routes", "tile_sizes"] + (originalRows ? ["row_map"]:[]), outputNames: ["output"],
    source: """
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
      auto ROUTES = routes;
      auto TILE_SIZES = tile_sizes;
      device T* O = output;
      AttnParams params_value{
        1, HEADS, BD, QUERY_ROWS, KEY_ROWS, 1, 0.08838834764831845f,
        QUERY_ROWS / BQ, KEY_ROWS / BK, QUERY_ROWS / BQ, KEY_ROWS / BK, 0, 0, 0,
        {q_strides[0], q_strides[1], q_strides[2]},
        {k_strides[0], k_strides[1], k_strides[2]},
        {v_strides[0], v_strides[1], v_strides[2]},
        {QUERY_ROWS * HEADS * BD, BD, HEADS * BD}
      };
      thread const AttnParams* params = &params_value;
      thread const AttnMaskParams* mask_params = nullptr;
      const device MaskType* mask = nullptr;
      const device T* sinks = nullptr;
      uint simd_lane_id = thread_index_in_simdgroup;
      uint simd_group_id = simdgroup_index_in_threadgroup;
      uint3 tid = threadgroup_position_in_grid;
      uint3 lid = thread_position_in_threadgroup;
      """ + (originalRows ? "auto ROW_MAP = row_map;\n" + H3SteelAttentionSource.originalRowBody : H3SteelAttentionSource.indexedBody),
    header: H3SteelAttentionSource.header + (originalRows ? H3SteelAttentionSource.originalRowLoader : ""), ensureRowContiguous: false)
  }

  static func evaluate(query: MLXArray, key: MLXArray, value: MLXArray,
    routes: MLXArray, tiles: H3FastTiles, originalRows:Bool = false) throws -> MLXArray {
    let count = tiles.sizes.count, video = count - tiles.prefixTiles
    guard Device.defaultDevice().deviceType == .gpu,
      query.ndim == 4, key.ndim == 4, key.shape == value.shape,
      query.dtype == .bfloat16, key.dtype == .bfloat16, value.dtype == .bfloat16,
      query.shape[0] == 1, key.shape[0] == 1,
      (1...56).contains(query.shape[1]), query.shape[1] == key.shape[1],
      query.shape[2] == (originalRows ? tiles.rows : video * 64),
      key.shape[2] == (originalRows ? tiles.rows : count * 64),
      query.shape[3] == 128, key.shape[3] == 128,
      routes.ndim == 4, routes.dtype == .int32,
      Array(routes.shape.prefix(3)) == [1, query.shape[1], video],
      (1...count).contains(routes.shape[3]) else {
      throw H3CheckpointError.invalid("Invalid indexed FastH3 BF16 tile attention geometry.")
    }
    try Task.checkCancellation()
    // The pinned MLX stride accessor is stable only after evaluation. Preserve
    // strided heads/rows without copying the full packed Q/K/V buffers, but
    // reject feature views the Steel block loaders cannot address.
    eval([query, key, value])
    // Broadcast feature axes need materialization; positive row/head strides
    // remain views so the production fused QKV layout is not copied.
    let query = query.strides.last == 0 ? contiguous(query) : query
    let key = key.strides.last == 0 ? contiguous(key) : key
    let value = value.strides.last == 0 ? contiguous(value) : value
    eval([query,key,value])
    guard [query, key, value].allSatisfy({ $0.strides.last == 1 && $0.strides.allSatisfy { $0 >= 0 } }) else {
      throw H3CheckpointError.invalid("Indexed FastH3 requires contiguous features and nonnegative strides.")
    }
    let denseRoutes = contiguous(routes)
    // Generated routes are trusted by the consumer; reject malformed indices
    // before the kernel can address outside its K/V buffers.
    guard denseRoutes.min().item(Int32.self) >= 0, denseRoutes.max().item(Int32.self) < count else {
      throw H3CheckpointError.invalid("Indexed FastH3 routes exceed the available tiles.")
    }
    let heads = query.shape[1]
    let consumer = originalRows ? originalKernel : kernel
    let inputs = [query,key,value,denseRoutes,MLXArray(tiles.sizes.map(Int32.init))]
      + (originalRows ? [MLXArray(tiles.indices)] : [])
    let output = consumer(inputs,
      template: [("HEADS", heads), ("QUERY_ROWS", video * 64), ("KEY_ROWS", count * 64),
        ("PREFIX_TILES", tiles.prefixTiles), ("VIDEO_TILES", video), ("ROUTE_TILES", routes.shape[3])],
      grid: (video * 64, heads * 4, 1), threadGroup: (32, 4, 1),
      outputShapes: [[1, video * 64, heads, 128]], outputDTypes: [.bfloat16], initValue: 0)[0]
      .transposed(0, 2, 1, 3)
    eval(output)
    try Task.checkCancellation()
    return output
  }
}
