import Foundation
import MLX
import MLXNN
import MLXRandom
import TensorIO

/// Bounded single-still path through H3's causal video VAE encoder. This is
/// the deterministic posterior mean used by Ref2VA; FL2VA sampling is a
/// separate contract. Weighted blocks are read from the installed file only
/// when called and are released before the next weighted stage.
public enum H3VideoVAEEncoder {
  private static let channels = [128, 256, 256, 512, 512, 1024]
  private static let temporalFactors = [1, 2, 2, 1, 1, 1]
  private static let spatialFactors = [2, 2, 2, 2, 1, 1]

  /// Validate all spatial-encoder tensors from the safetensors directory
  /// before any weighted Qwen, VAE or transformer stage begins.
  public static func preflight(checkpointURL: URL) throws {
    _ = try H3VideoVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    try validateEncoderHeader(file)
    try file.checkUnchanged(at: checkpointURL)
  }

  public static func encodeStill(checkpointURL: URL, rgb8: [UInt8],
    width: Int, height: Int) throws -> MLXArray {
    try encodeStill(checkpointURL: checkpointURL, rgb8: rgb8,
      width: width, height: height, samplePosterior: false)
  }

  /// FL2VA samples the posterior independently of the generation seed. Seed
  /// MLXRandom once before a sequence of keyframes so each draw is distinct.
  public static func encodeKeyframe(checkpointURL: URL, rgb8: [UInt8],
    width: Int, height: Int) throws -> MLXArray {
    try encodeStill(checkpointURL: checkpointURL, rgb8: rgb8,
      width: width, height: height, samplePosterior: true)
  }

  /// Encode a 24 fps reference movie on the model's 17-frame clip grid.
  /// Only one clip enters the encoder at a time; the final short clip repeats
  /// its last pixel frame, matching the released VAE's token-drop contract.
  /// The deterministic posterior means have shape [1, 2 + 5*n, H/16, W/16, 24].
  public static func encodeVideo(checkpointURL: URL, rgb8: [UInt8],
    frameCount: Int, width: Int, height: Int) throws -> MLXArray {
    guard frameCount >= 5, frameCount <= 175,
      (frameCount - 5).isMultiple(of: 17),
      (32...256).contains(width), (32...256).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      rgb8.count == frameCount * width * height * 3 else {
      throw H3CheckpointError.invalid("H3 reference video needs 5 + 17*n frames on the 24 fps grid.")
    }
    return try encodeFrames(checkpointURL: checkpointURL, rgb8: rgb8,
      frameCount: frameCount, width: width, height: height,
      samplePosterior: false)
  }

  /// Control guides use the output canvas and complete 15-second timeline,
  /// while ordinary Ref2VA retains its existing smaller media admission.
  /// Full-canvas temporal convolution terms are released sequentially by
  /// default; the explicit fallback permits installed-checkpoint parity probes.
  public static func encodeControlVideo(checkpointURL: URL, rgb8: [UInt8],
    frameCount: Int, width: Int, height: Int,
    progress: (Int, Int) -> Void = { _, _ in },
    releaseTemporalConvolutionTerms: Bool = true) throws -> MLXArray {
    guard (5...362).contains(frameCount), (frameCount - 5).isMultiple(of: 17),
      (32...2048).contains(width), (32...2048).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      width * height <= H3Geometry.maximumCanvasPixels,
      rgb8.count == frameCount * width * height * 3,
      rgb8.count <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("H3 control video exceeds aligned output canvas or frame limits.")
    }
    return try encodeFrames(checkpointURL: checkpointURL, rgb8: rgb8,
      frameCount: frameCount, width: width, height: height, samplePosterior: false,
      progress: progress, releaseTemporalConvolutionTerms: releaseTemporalConvolutionTerms)
  }

  private static func encodeStill(checkpointURL: URL, rgb8: [UInt8],
    width: Int, height: Int, samplePosterior: Bool) throws -> MLXArray {
    try encodeFrames(checkpointURL: checkpointURL, rgb8: rgb8,
      frameCount: 1, width: width, height: height,
      samplePosterior: samplePosterior)
  }

  private static func encodeFrames(checkpointURL: URL, rgb8: [UInt8],
    frameCount: Int, width: Int, height: Int,
    samplePosterior: Bool,
    progress: (Int, Int) -> Void = { _, _ in },
    releaseTemporalConvolutionTerms: Bool = false) throws -> MLXArray {
    guard checkpointURL.isFileURL,
      (32...2048).contains(width), (32...2048).contains(height),
      width * height <= H3Geometry.maximumCanvasPixels,
      width.isMultiple(of: 16), height.isMultiple(of: 16),
      frameCount > 0, frameCount <= 362,
      rgb8.count == frameCount * width * height * 3 else {
      throw H3CheckpointError.invalid("H3 video encoder requires bounded 16-pixel RGB geometry.")
    }
    _ = try H3VideoVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    try validateEncoderHeader(file)
    try Task.checkCancellation()

    func read(_ name: String, shape: [Int]) throws -> MLXArray {
      guard let record = file.tensors[name], record.dtype == "F16",
        record.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 video encoder tensor: \(name)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: Float16.self)
      }
      eval(value)
      return value
    }
    func conv(_ input: MLXArray, name: String, out: Int, kernel: Int,
      spatialPadding: Int = 0, temporalPadding: Int = 0,
      stride: IntOrTriple = 1) throws -> MLXArray {
      let weight = try read(name + ".weight",
        shape: [out, kernel, kernel, kernel, input.shape[4]])
      let bias = try read(name + ".bias", shape: [out])
      return try H3VideoVAEEncoderOps.causalConv(input,
        weight: weight, bias: bias, spatialPadding: spatialPadding,
        temporalPadding: temporalPadding, stride: stride, releaseTemporalTerms: releaseTemporalConvolutionTerms)
    }
    func norm(_ input: MLXArray, name: String) throws -> MLXArray {
      let count = input.shape[4]
      return try H3VideoVAEEncoderOps.temporallyIsolatedGroupNorm(input,
        weight: read(name + ".weight", shape: [count]),
        bias: read(name + ".bias", shape: [count]))
    }
    func block(_ input: MLXArray, name: String, out: Int) throws -> MLXArray {
      try residualBlock(input, name: name, out: out,
        normalize: { try norm($0, name: $1) },
        convolve: { try conv($0, name: $1, out: $2, kernel: $3,
          spatialPadding: $4, temporalPadding: $5) })
    }

    let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 1, 3])
    let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 1, 3])
    var moments: [MLXArray] = []
    let frameBytes = width * height * 3
    let clipCount = frameCount == 1 ? 1 : (frameCount + 16) / 17
    for clipIndex in 0..<clipCount {
      try Task.checkCancellation()
      let first = clipIndex * 17
      let count = frameCount == 1 ? 1 : 17
      var floats = [Float]()
      floats.reserveCapacity(count * frameBytes)
      for localFrame in 0..<count {
        let sourceFrame = min(first + localFrame, frameCount - 1)
        let lower = sourceFrame * frameBytes
        floats.append(contentsOf: rgb8[lower..<(lower + frameBytes)]
          .map { Float($0) / 255 })
      }
      let pixels = MLXArray(floats, [1, count, height, width, 3])
      var value = ((pixels - mean) / std).asType(.float16)
      value = try conv(value, name: "encoder.conv_in", out: channels[0],
        kernel: 3, spatialPadding: 1, temporalPadding: 2)
      for index in channels.indices {
        try Task.checkCancellation()
        let out = channels[index]
        for layer in 0..<2 {
          value = try block(value,
            name: "encoder.down.\(index).block.\(layer)", out: out)
          eval(value)
          Memory.clearCache()
        }
        let temporal = temporalFactors[index]
        let spatial = spatialFactors[index]
        if temporal * spatial > 1 {
          if spatial == 2 {
            value = try H3VideoVAEEncoderOps.reflectSpatial(value,
              axis: 2, before: 0, after: 1)
            value = try H3VideoVAEEncoderOps.reflectSpatial(value,
              axis: 3, before: 0, after: 1)
          }
          value = try conv(value,
            name: "encoder.down.\(index).downsample.conv", out: out,
            kernel: 3, temporalPadding: 2,
            stride: IntOrTriple((temporal, spatial, spatial)))
          eval(value)
          Memory.clearCache()
        }
      }
      value = try norm(value, name: "encoder.norm_out")
      value = try conv(silu(value), name: "encoder.conv_out", out: 48,
        kernel: 3, spatialPadding: 1, temporalPadding: 2)
      value = try conv(value, name: "quant_conv", out: 48, kernel: 1)
      let expected = count == 1 ? 1 : 5
      guard value.shape == [1, expected, height / 16, width / 16, 48] else {
        throw H3CheckpointError.invalid("H3 video encoder produced unexpected latent geometry.")
      }
      moments.append(value)
      eval(value)
      Memory.clearCache()
      progress(clipIndex + 1, clipCount)
      try Task.checkCancellation()
    }
    let joined = moments.count == 1 ? moments[0] : concatenated(moments, axis: 1)
    let latentFrames = frameCount == 1 ? 1 : (frameCount - 5) / 17 * 5 + 2
    let latentMean = joined[0..<1, 0..<latentFrames, 0..<(height / 16),
      0..<(width / 16), 0..<24].asType(.float32)
    let latent: MLXArray
    if samplePosterior {
      let logVariance = clip(joined[0..<1, 0..<latentFrames, 0..<(height / 16),
        0..<(width / 16), 24..<48].asType(.float32), min: -30, max: 20)
      let deviation = exp(0.5 * logVariance)
      latent = (latentMean + deviation * MLXRandom.normal(latentMean.shape))
        .asType(.float16).asType(.float32)
    } else {
      latent = latentMean
    }
    eval(latent)
    Stream.gpu.synchronize()
    Memory.clearCache()
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return latent
  }

  /// Each evaluated projection leaves its normalization scope before the next
  /// convolution. Keep the full temporal batch and operation order unchanged.
  static func residualBlock(_ input: MLXArray, name: String, out: Int,
    normalize: (MLXArray, String) throws -> MLXArray,
    convolve: (MLXArray, String, Int, Int, Int, Int) throws -> MLXArray) throws -> MLXArray {
    func firstProjection(_ input: MLXArray) throws -> MLXArray {
      let first = try normalize(input, name + ".norm1")
      return try convolve(silu(first), name + ".conv1", out, 3, 1, 2)
    }
    func secondProjection(_ input: MLXArray) throws -> MLXArray {
      let second = try { () throws -> MLXArray in
        let hidden = try firstProjection(input)
        return try normalize(hidden, name + ".norm2")
      }()
      return try convolve(silu(second), name + ".conv2", out, 3, 1, 2)
    }
    let projected = try secondProjection(input)
    let residual = input.shape[4] == out ? input
      : try convolve(input, name + ".nin_shortcut", out, 1, 0, 0)
    let result = residual + projected
    eval(result)
    return result
  }

  private static func validateEncoderHeader(_ file: SafeTensorFile) throws {
    func require(_ name: String, _ shape: [Int]) throws {
      guard let record = file.tensors[name], record.dtype == "F16",
        record.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 video encoder tensor: \(name)")
      }
    }
    func convolution(_ name: String, input: Int, output: Int, kernel: Int) throws {
      try require(name + ".weight", [output, kernel, kernel, kernel, input])
      try require(name + ".bias", [output])
    }
    func normalization(_ name: String, channels: Int) throws {
      try require(name + ".weight", [channels])
      try require(name + ".bias", [channels])
    }
    try convolution("encoder.conv_in", input: 3, output: channels[0], kernel: 3)
    for index in channels.indices {
      let input = index == 0 ? channels[0] : channels[index - 1]
      let output = channels[index]
      for layer in 0..<2 {
        let source = layer == 0 ? input : output
        let name = "encoder.down.\(index).block.\(layer)"
        try normalization(name + ".norm1", channels: source)
        try convolution(name + ".conv1", input: source, output: output, kernel: 3)
        try normalization(name + ".norm2", channels: output)
        try convolution(name + ".conv2", input: output, output: output, kernel: 3)
        if source != output {
          try convolution(name + ".nin_shortcut", input: source, output: output, kernel: 1)
        }
      }
      if temporalFactors[index] * spatialFactors[index] > 1 {
        try convolution("encoder.down.\(index).downsample.conv",
          input: output, output: output, kernel: 3)
      }
    }
    try normalization("encoder.norm_out", channels: channels[5])
    try convolution("encoder.conv_out", input: channels[5], output: 48, kernel: 3)
    try convolution("quant_conv", input: 48, output: 48, kernel: 1)
  }
}
