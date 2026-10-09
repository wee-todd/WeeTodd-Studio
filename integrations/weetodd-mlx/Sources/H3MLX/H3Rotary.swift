import MLX

/// Fuse rotary elementwise work while preserving BF16 rounding at both
/// multiplies and at the sum. Runtime strides address transposed norm outputs
/// directly, so applying rotary does not materialize another full Q/K input.
enum H3Rotary {
  private static let kernel = MLXFast.metalKernel(
    name: "weetodd_h3_rotary_rounded_bf16",
    inputNames: ["x", "cosine", "sine"], outputNames: ["output"],
    source: """
      uint i = thread_position_in_grid.x;
      if (i >= HEADS * ROWS * WIDTH) return;
      uint channel = i % WIDTH;
      uint row = (i / WIDTH) % ROWS;
      uint head = i / (WIDTH * ROWS);
      long base = long(head) * long(x_strides[1]) + long(row) * long(x_strides[2]);
      long offset = base + long(channel) * long(x_strides[3]);
      if (channel >= ROTARY) { output[i] = x[offset]; return; }
      uint paired = channel < ROTARY / 2 ? channel + ROTARY / 2 : channel - ROTARY / 2;
      float rotated = float(x[base + long(paired) * long(x_strides[3])]);
      if (channel < ROTARY / 2) rotated = -rotated;
      long cosOffset = long(row) * long(cosine_strides[2]) + long(channel) * long(cosine_strides[3]);
      long sinOffset = long(row) * long(sine_strides[2]) + long(channel) * long(sine_strides[3]);
      bfloat first = bfloat(float(x[offset]) * float(cosine[cosOffset]));
      bfloat second = bfloat(rotated * float(sine[sinOffset]));
      output[i] = bfloat(float(first) + float(second));
      """, ensureRowContiguous: false)

  static func apply(_ value: MLXArray, cosine: MLXArray, sine: MLXArray,
    rotaryWidth: Int = 96) -> MLXArray {
    if value.dtype == .bfloat16, cosine.dtype == .bfloat16, sine.dtype == .bfloat16,
      Device.defaultDevice().deviceType == .gpu,
      value.ndim == 4, value.shape[0] == 1,
      rotaryWidth > 0, rotaryWidth.isMultiple(of: 2), rotaryWidth <= value.shape[3],
      cosine.shape == [1, 1, value.shape[2], rotaryWidth], sine.shape == cosine.shape {
      return kernel([value, cosine, sine],
        template: [("HEADS", value.shape[1]), ("ROWS", value.shape[2]),
          ("WIDTH", value.shape[3]), ("ROTARY", rotaryWidth)],
        grid: (value.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [value.shape], outputDTypes: [.bfloat16])[0]
    }
    let first = value[.ellipsis, 0..<(rotaryWidth / 2)]
    let second = value[.ellipsis, (rotaryWidth / 2)..<rotaryWidth]
    let rotated = concatenated([-second, first], axis: -1)
    let leading = value[.ellipsis, 0..<rotaryWidth] * cosine + rotated * sine
    return concatenated([leading, value[.ellipsis, rotaryWidth..<value.shape.last!]], axis: -1)
  }
}
