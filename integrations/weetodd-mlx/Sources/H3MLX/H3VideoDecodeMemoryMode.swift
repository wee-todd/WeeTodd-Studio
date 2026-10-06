import Foundation

/// Existing saved-recipe memory modes. Nil direct API requests preserve eager decoding.
public enum H3VideoDecodeMemoryMode: String, Sendable {
  case normal
  case lowMemoryBF16 = "low_memory_bf16"

  static func materializesProjection(for mode: Self?) -> Bool {
    mode == nil
  }

  static func materializesFirstResidual(for mode: Self?) -> Bool {
    mode != .normal
  }

  static func materializesBlockOutput(for mode: Self?) -> Bool {
    mode == nil
  }

  public static func diagnostics(for mode: Self?) -> [String: Any] {
    ["memoryMode": mode?.rawValue ?? "direct_eager",
      "materializationPolicy": mode == .normal ? "defer_projections_and_residual"
        : mode == .lowMemoryBF16 ? "defer_projections" : "eager",
      "packedWeightsResidentWithinVideoStage": true,
      "materializesBlockOutput": materializesBlockOutput(for: mode),
      "allocationCacheLimitBytes": 128 * 1024 * 1024,
      "spatialBatch": 4,
      "scope": "video-only residency; released before audio; no whole-generation speed guarantee"]
  }
}
