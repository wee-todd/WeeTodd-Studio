import MLX

enum RelayInputSafety {
  private static let kernel = MLXFast.metalKernel(name: "weetodd_relay_input_safety_v1",
    inputNames: ["q", "k", "v", "sizes"], outputNames: ["flags"], source: """
      uint worker = thread_position_in_grid.x, start = worker * 32;
      bool valid = true;
      auto safe = [](float x) {
        float a = metal::abs(x);
        return a == 0.0f || (a >= 1e-10f && a <= 1e8f);
      };
      #pragma clang loop unroll(disable)
      for (uint c = 0; c < 32; ++c) {
        uint i = start + c;
        if (i < uint(sizes[0])) valid = valid && safe(float(q[i]));
        if (i < uint(sizes[1])) valid = valid && safe(float(k[i]));
        if (i < uint(sizes[2])) valid = valid && safe(float(v[i]));
      }
      bool group_valid = simd_all(valid);
      if (thread_index_in_simdgroup == 0) flags[worker / 32] = group_valid;
      """)

  /// One streaming scan, then a tiny reduction. Avoid full-size abs/comparison
  /// temporaries and a host branch; the result stays on the device for all tiles.
  static func evaluate(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    guard [q, k, v].allSatisfy({ $0.size > 0 && $0.size <= Int(Int32.max)
      && [DType.float32, .bfloat16].contains($0.dtype) }) else { return MLXArray(false) }
    let n = max(q.size, max(k.size, v.size)), groups = (n + 1023) / 1024
    let sizes = MLXArray([q.size, k.size, v.size].map(Int32.init))
    let flags = kernel([q, k, v, sizes], grid: (groups * 32, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [[groups]], outputDTypes: [.bool])[0]
    return all(flags)
  }
}
