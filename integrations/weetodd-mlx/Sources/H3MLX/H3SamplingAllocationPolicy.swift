import Foundation

/// Temporary Metal buffer reuse is separate from retained model weights. MLX's
/// allocation-cache limit is soft: a retired buffer can temporarily exceed it.
/// Every evaluation drains and clears the pool before the next weighted stage.
enum H3SamplingAllocationPolicy {
  static let minimum = 128 * 1024 * 1024
  static let maximum = 4 * 1024 * 1024 * 1024
  static let decodeRows = 1024

  static func blockLimit(previous: Int, prepared: Bool) -> Int {
    prepared ? previous : minimum
  }

  static func limit(previous: Int, physical: Int, recommended: Int,
    available: Int, eligible: Bool) -> Int {
    guard previous > 0 else { return 0 }
    let baseline = min(previous, minimum)
    guard eligible, physical >= 64 * 1024 * 1024 * 1024,
      recommended > 0, available > 0 else { return baseline }
    let proposed = min(previous, min(maximum, min(physical / 64, recommended / 48)))
    let reserve = max(16 * 1024 * 1024 * 1024, physical / 8)
    // Leave a second pool-sized allowance for MLX's soft-limit overshoot.
    guard proposed >= baseline, available >= reserve,
      proposed <= (available - reserve) / 2 else { return baseline }
    return proposed
  }
}

public struct H3SamplingAllocationReport: Sendable, Equatable {
  public let softLimitBytes: Int
  public let maximumObservedCachedBytes: Int
  public let weightPreparationSeconds: Double
  public let clearedAtEvaluationBoundary: Bool

  var metadata: [String: Any] {
    ["softLimitBytes": softLimitBytes,
      "maximumObservedCachedBytes": maximumObservedCachedBytes,
      "weightPreparationSeconds": weightPreparationSeconds,
      "clearedAtEvaluationBoundary": clearedAtEvaluationBoundary,
      "decodeRowWindow": H3SamplingAllocationPolicy.decodeRows,
      "scope": "temporary Metal allocation pool; excludes retained weights and active arrays; MLX limit is soft"]
  }
}
