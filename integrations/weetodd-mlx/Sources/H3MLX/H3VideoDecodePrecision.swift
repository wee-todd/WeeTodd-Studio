import Foundation
import MLX

/// Internal decoder arithmetic opt-in, independent of saved recipe memory mode.
/// FP32 remains the default; FP16 requires separate media-quality qualification.
public enum H3VideoDecodePrecision: String, Sendable {
  case float32 = "float32"
  case float16 = "float16"
  public static let defaultPrecision: Self = .float32
  var computeDType: DType { self == .float16 ? .float16 : .float32 }

  /// Conservatively reject values outside the finite half range rather than
  /// clamping, silently promoting, or publishing nonfinite pixels.
  func validateHalfCastInput(_ value: MLXArray) throws {
    try Task.checkCancellation()
    guard value.dtype == .float32 else {
      throw H3CheckpointError.invalid("H3 decoder precision boundary requires Float32 input.")
    }
    if self == .float16 {
      guard all(isFinite(value)).item(Bool.self),
        all(abs(value) .<= Float(65_504)).item(Bool.self) else {
        throw H3CheckpointError.invalid("FP16 H3 video input is nonfinite or exceeds the finite half range.")
      }
      try Task.checkCancellation()
    }
  }

  func transformerInput(_ value: MLXArray) throws -> MLXArray {
    try validateHalfCastInput(value)
    return self == .float16 ? value.asType(.float16) : value
  }

  func validatePixels(_ pixels: MLXArray) throws {
    try Task.checkCancellation()
    if self == .float16 {
      guard pixels.dtype == .float32, all(isFinite(pixels)).item(Bool.self) else {
        throw H3CheckpointError.invalid("FP16 H3 video decoding produced nonfinite Float32 pixels.")
      }
      try Task.checkCancellation()
    }
  }

  /// Product FP16 is admitted only with the independently qualified lower-memory policy.
  /// It never changes memory mode or silently falls back to another precision.
  public func validate(memoryMode: H3VideoDecodeMemoryMode?) throws {
    guard self != .float16 || memoryMode == .lowMemoryBF16 else {
      throw H3CheckpointError.invalid("FP16 H3 video decoding requires Swift MLX and low_memory_bf16; normal mode is not qualified.")
    }
  }

  public static func executionDiagnostics(requested: Self, applied: Self?) -> [String:Any] {
    var report = requested.diagnostics
    report["requestedComputePrecision"] = requested.rawValue
    report["computePrecision"] = applied?.rawValue ?? "not_applied"
    report["precisionApplied"] = applied != nil
    return report
  }

  public var diagnostics: [String:Any] {
    ["computePrecision":rawValue,"latentAndPostQuantPrecision":"float32",
      "rmsAndQKNormPrecision":"float32","finalLayerNormPrecision":"float32",
      "headPrecision":rawValue,"pixelAndBlendPrecision":"float32",
      "finiteHalfRangeAdmission":self == .float16,
      "overflowPolicy":"error_no_clamp_no_fallback",
      "qualityQualification":"separate_from_byte_exact_FP32_default",
      "defaultComputePrecision":Self.defaultPrecision.rawValue]
  }
}
