import Foundation

/// Scalar execution evidence. The child timer is nested within bridge time;
/// neither is added to the whole renderer wall time.
public struct H3NativeBlockReport: Sendable, Equatable {
  public let evaluations: Int
  public let predictionSeconds: Double
  public let bridgeSeconds: Double
  public let childReaped: Bool
  let physicalMemory: H3NativeMemoryMonitor.Report?

  init(evaluations: Int, predictionSeconds: Double, bridgeSeconds: Double,
    childReaped: Bool, physicalMemory: H3NativeMemoryMonitor.Report? = nil) {
    self.evaluations = evaluations
    self.predictionSeconds = predictionSeconds
    self.bridgeSeconds = bridgeSeconds
    self.childReaped = childReaped
    self.physicalMemory = physicalMemory
  }

  var metadata: [String: Any] {
    var result: [String: Any] = [
      "evaluations": evaluations,
      "predictionSeconds": predictionSeconds,
      "inclusiveBridgeSeconds": bridgeSeconds,
      "transferAndManagementSeconds": max(0, bridgeSeconds - predictionSeconds),
      "timingScope": "child prediction is nested within inclusive parent bridge time; never add these timers to whole renderer wall time",
      "childReaped": childReaped,
      "precision": "fp16_projections_fp32_attention_and_residual",
      "physicalObservationStatus": physicalMemory == nil ? "not_sampled" : "observed"]
    if let memory = physicalMemory {
      func component(_ value: H3NativeMemoryMonitor.ProcessSummary) -> [String: Any] {
        var data: [String: Any] = ["pid": value.pid, "status": value.status.rawValue,
          "attempts": value.attempts, "availableSamples": value.availableSamples,
          "failedSamples": value.failedSamples, "zeroValueSamples": value.zeroValueSamples,
          "currentAboveLifetimeSamples": value.currentAboveLifetimeSamples,
          "identityMismatchSamples": value.identityMismatchSamples]
        if let peak = value.sampledCurrentPeakBytes { data["sampledCurrentPeakBytes"] = peak }
        if let peak = value.reportedLifetimeHighWaterBytes { data["reportedLifetimeHighWaterBytes"] = peak }
        if let code = value.lastErrorCode { data["lastErrorCode"] = code }
        return data
      }
      var physical: [String: Any] = ["lifecycle": memory.lifecycle.rawValue,
        "intervalSeconds": memory.intervalSeconds, "pairAttempts": memory.pairAttempts,
        "combinedSamples": memory.combinedSamples, "parent": component(memory.parent),
        "child": component(memory.child), "overflowSamples": memory.overflowSamples,
        "clockRegressionSamples": memory.clockRegressionSamples,
        "scope": "sequential parent/owned-child charged physical-footprint samples; shared memory can be charged twice; combined samples are non-atomic estimates, neither exact simultaneous peaks nor rigorous lower bounds; independent lifetime envelopes include earlier parent stages and are not simultaneous measured peaks"]
      if let peak = memory.sampledNonAtomicCombinedCurrentPeakEstimateBytes {
        physical["sampledNonAtomicCombinedCurrentPeakEstimateBytes"] = peak
      }
      if let peak = memory.sumIndependentHighWaterUpperBoundBytes {
        physical["sumIndependentHighWaterEnvelopeBytes"] = peak
      }
      result["physicalMemory"] = physical
    }
    return result
  }
}
