import Foundation

/// Nonoverlapping inference-thread durations. Preparation includes source reads,
/// reconstruction and source validation; background preparation is reported separately.
public struct ExecutionMetrics: Sendable, Codable {
  public internal(set) var preparationSeconds = 0.0
  public internal(set) var validationSeconds = 0.0
  public internal(set) var installationSeconds = 0.0
  public internal(set) var computeSeconds = 0.0
  public internal(set) var healthCheckSeconds = 0.0
  public internal(set) var backgroundPreparationSeconds = 0.0
  public internal(set) var preparationWaitSeconds = 0.0
  public internal(set) var preparedWeightBytes: UInt64 = 0
  public internal(set) var graphBuildSeconds = 0.0
  public internal(set) var decodedWeightBytes: UInt64 = 0
  public init() {}
}
