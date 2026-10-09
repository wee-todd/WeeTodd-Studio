import MLX

/// Decoder rotary with the same rounding boundaries as separate MLX products.
/// Signed runtime strides avoid copying interleaved QKV slices before rotation.
enum H3VideoRotary {
  private static let kernel = MLXFast.metalKernel(
    name: "weetodd_h3_video_rotary_rounded",
    inputNames: ["x", "cosine", "sine"], outputNames: ["output"],
    source: """
      uint i = thread_position_in_grid.x;
      if (i >= BATCH * ROWS * HEADS * 64) return;
      uint channel = i % 64;
      uint head = (i / 64) % HEADS;
      uint row = (i / (64 * HEADS)) % ROWS;
      uint batch = i / (64 * HEADS * ROWS);
      long base = long(batch) * long(x_strides[0])
        + long(row) * long(x_strides[1]) + long(head) * long(x_strides[2]);
      long offset = base + long(channel) * long(x_strides[3]);
      if (channel >= 48) { output[i] = x[offset]; return; }
      uint paired = channel < 24 ? channel + 24 : channel - 24;
      float rotated = float(x[base + long(paired) * long(x_strides[3])]);
      if (channel < 24) rotated = -rotated;
      long co = long(batch) * long(cosine_strides[0])
        + long(row) * long(cosine_strides[1]) + long(channel) * long(cosine_strides[3]);
      long so = long(batch) * long(sine_strides[0])
        + long(row) * long(sine_strides[1]) + long(channel) * long(sine_strides[3]);
      if (HALF_MODE) {
        half first = half(float(x[offset]) * float(cosine[co]));
        half second = half(rotated * float(sine[so]));
        output[i] = half(float(first) + float(second));
      } else {
        // Volatile intermediates forbid contracting the two rounded products
        // into an FMA, which would change the original Float32 computation.
        volatile float first = float(x[offset]) * float(cosine[co]);
        volatile float second = rotated * float(sine[so]);
        output[i] = first + second;
      }
      """, ensureRowContiguous: false)

  static func apply(_ value: MLXArray, cosine: MLXArray, sine: MLXArray) throws -> MLXArray {
    try Task.checkCancellation()
    guard value.ndim == 4, (1...4).contains(value.shape[0]),
      (1...16_384).contains(value.shape[1]), (1...32).contains(value.shape[2]),
      value.shape[3] == 64, [.float16, .float32].contains(value.dtype),
      cosine.shape == [value.shape[0], value.shape[1], 1, 48],
      sine.shape == cosine.shape, cosine.dtype == value.dtype, sine.dtype == value.dtype else {
      throw H3CheckpointError.invalid("H3 video rotary requires matching decoder tensors.")
    }
    if Device.defaultDevice().deviceType == .gpu {
      return kernel([value, cosine, sine],
        template: [("BATCH", value.shape[0]), ("ROWS", value.shape[1]),
          ("HEADS", value.shape[2]), ("HALF_MODE", value.dtype == .float16)],
        grid: (value.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [value.shape], outputDTypes: [value.dtype])[0]
    }
    let rotated = concatenated([-value[.ellipsis,24..<48],value[.ellipsis,0..<24]],axis:-1)
    return concatenated([value[.ellipsis,0..<48]*cosine+rotated*sine,
      value[.ellipsis,48..<64]],axis:-1)
  }
}
