import Foundation

/// Scalar diagnostics around existing preparation completions. These exclusive
/// phase intervals are nested within the caller's whole preparation interval.
/// No MLX operation, tensor retention or additional completion is introduced.
enum H3PreparationObservation {
  enum Stage: String, Sendable, CaseIterable {
    case conditionProjection = "condition_projection"
    case tokenRefinement = "token_refinement"
    case timeAndSmallMetadata = "time_and_small_metadata"
    case adalnTables = "adaln_tables"
    case terminalCleanup = "terminal_cleanup"
  }

  struct Measurement: Sendable, Equatable {
    let stage: Stage
    let seconds: Double
    /// True only if the measured body returned normally. A failed phase is
    /// reported before the original error propagates; it is not a completion.
    let succeeded: Bool
  }

  typealias Observer = (Measurement) -> Void

  static func measure<T>(_ stage: Stage, observer: Observer?,
    clock: () -> UInt64 = { DispatchTime.now().uptimeNanoseconds },
    body: () throws -> T) rethrows -> T {
    // Ordinary generation reads no clock and invokes the body exactly once.
    guard let observer else { return try body() }
    let started = clock()
    var succeeded = false
    defer {
      let ended = clock()
      observer(Measurement(stage: stage,
        seconds: ended >= started ? Double(ended - started) / 1_000_000_000 : 0,
        succeeded: succeeded))
    }
    let result = try body()
    succeeded = true
    return result
  }
}
