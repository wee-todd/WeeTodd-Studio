import Foundation

/// Temporal expansion and recovery only. No weighted stage is entered here.
public struct H3MotionFidelitySettings: Sendable, Codable {
  public enum Mode: String, Sendable, Codable { case adaptive, uniform }
  public let mode: Mode
  public let strength: Double
  public let maxHold: Int
  public let sensitivity: Double
  public let seed: UInt64
  public let maxFrames: Int
  public let evaluations: Int?

  public init(mode: Mode = .adaptive, strength: Double = 0.5,
    maxHold: Int = 2, sensitivity: Double = 0.5, seed: UInt64 = 42,
    maxFrames: Int = 345, evaluations: Int? = nil) throws {
    guard strength.isFinite, strength > 0, strength <= 1,
      sensitivity.isFinite, (0...1).contains(sensitivity),
      (2...4).contains(maxHold), (73...345).contains(maxFrames),
      seed <= UInt64(UInt32.max), evaluations == nil || (1...64).contains(evaluations!) else {
      throw H3CheckpointError.invalid("Invalid H3 Motion Fidelity settings.")
    }
    self.mode = mode; self.strength = strength; self.maxHold = maxHold
    self.sensitivity = sensitivity; self.seed = seed; self.maxFrames = maxFrames
    self.evaluations = evaluations
  }
  public func validate() throws {
    _ = try Self(mode: mode, strength: strength, maxHold: maxHold,
      sensitivity: sensitivity, seed: seed, maxFrames: maxFrames, evaluations: evaluations)
  }
}

public struct H3MotionFidelityPlan: Sendable, Encodable {
  public let format = "weetodd-motion-plan-v1"
  public let refinementSchedule = "explicit-video-noise-v1"
  public let settings: H3MotionFidelitySettings
  public let sourceFrames: Int
  public let expandedFrames: Int
  public let paddedFrames: Int
  public let holds: [Int]
  public let recovery: [Int]
  public let scores: [Double]
  public var noop: Bool { holds.allSatisfy { $0 == 1 } }
  public let fps = 24
  public var expandedSeconds: Double { Double(paddedFrames) / 24 }
  public var adaptiveAnalysisPerformed: Bool { settings.mode == .adaptive }
  private enum CodingKeys: String, CodingKey {
    case format, refinementSchedule, settings, sourceFrames, expandedFrames,
      paddedFrames, holds, recovery, scores, noop, fps, expandedSeconds, adaptiveAnalysisPerformed
  }
  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(format, forKey: .format)
    try values.encode(refinementSchedule, forKey: .refinementSchedule)
    try values.encode(settings, forKey: .settings)
    try values.encode(sourceFrames, forKey: .sourceFrames)
    try values.encode(expandedFrames, forKey: .expandedFrames)
    try values.encode(paddedFrames, forKey: .paddedFrames)
    try values.encode(holds, forKey: .holds)
    try values.encode(recovery, forKey: .recovery)
    try values.encode(scores, forKey: .scores)
    try values.encode(noop, forKey: .noop)
    try values.encode(fps, forKey: .fps)
    try values.encode(expandedSeconds, forKey: .expandedSeconds)
    try values.encode(adaptiveAnalysisPerformed, forKey: .adaptiveAnalysisPerformed)
  }

  public static func alignedFrames(_ count: Int) throws -> Int {
    guard (1...1380).contains(count) else {
      throw H3CheckpointError.invalid("H3 motion frame alignment exceeds its bounded timeline.")
    }
    return count + ((5 - count) % 17 + 17) % 17
  }
  public static func latentIndex(_ frame: Int) -> Int {
    let group = frame / 17, offset = frame % 17
    return group * 5 + (offset == 0 ? 0 : 1 + (offset - 1) / 4)
  }

  /// Jerk is the third temporal difference reduced over normalized latent
  /// channels and spatial locations. It is supplied only by the shared VAE path.
  public init(sourceFrames: Int, settings: H3MotionFidelitySettings,
    temporalJerk: [Float]? = nil) throws {
    try settings.validate(); try Task.checkCancellation()
    guard (60...345).contains(sourceFrames) else {
      throw H3CheckpointError.invalid("H3 Motion Fidelity requires 60–345 source frames at 24 fps.")
    }
    var scores = [Double](repeating: 0, count: sourceFrames)
    var holds: [Int]
    if settings.mode == .uniform {
      holds = [Int](repeating: settings.maxHold, count: sourceFrames)
    } else {
      guard let temporalJerk, !temporalJerk.isEmpty,
        temporalJerk.allSatisfy({ $0.isFinite && $0 >= 0 }),
        temporalJerk.count + 3 > Self.latentIndex(sourceFrames - 1) else {
        throw H3CheckpointError.invalid("Motion analysis requires complete finite H3 temporal jerk.")
      }
      let jerk = [temporalJerk[0], temporalJerk[0]] + temporalJerk + [temporalJerk.last!]
      var baseline = [Float](repeating: 0, count: 5)
      for phase in 0..<5 {
        let values = stride(from: phase, to: jerk.count, by: 5).map { jerk[$0] }.sorted()
        guard !values.isEmpty else { throw H3CheckpointError.invalid("Insufficient H3 temporal phases.") }
        baseline[phase] = values.count % 2 == 1 ? values[values.count / 2] :
          (values[values.count / 2 - 1] + values[values.count / 2]) / 2
      }
      guard baseline.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("Motion phase baseline is non-finite.")
      }
      let residual = jerk.indices.map { max(Float(0), jerk[$0] - baseline[$0 % 5]) }
      let scale = residual.max()!
      if scale > 1e-6 {
        for frame in 0..<sourceFrames {
          // Owned Python does the ratio in float32 before the threshold step.
          scores[frame] = Double(residual[Self.latentIndex(frame)] / scale)
        }
      }
      let threshold = Float(0.85 - 0.7 * settings.sensitivity)
      holds = scores.map { 1 + Int(ceil(Double(min(Float(1), max(Float(0),
        (Float($0) - threshold) / (Float(1) - threshold))) * Float(settings.maxHold - 1)))) }
      for frame in 1..<sourceFrames { holds[frame] = max(holds[frame], holds[frame - 1] - 1) }
      for frame in stride(from: sourceFrames - 2, through: 0, by: -1) {
        holds[frame] = max(holds[frame], holds[frame + 1] - 1)
      }
    }
    let expanded = holds.reduce(0, +), padded = try Self.alignedFrames(expanded)
    guard padded <= settings.maxFrames else {
      throw H3CheckpointError.invalid("Motion expansion exceeds the requested frame budget.")
    }
    var cursor = 0, recovery: [Int] = []
    for hold in holds { recovery.append(cursor); cursor += hold }
    self.settings = settings; self.sourceFrames = sourceFrames
    expandedFrames = expanded; paddedFrames = padded; self.holds = holds
    self.recovery = recovery
    self.scores = scores.map { Double((Float($0) * Float(100_000)).rounded(.toNearestOrEven) / Float(100_000)) }
  }

  public var expansionIndices: [Int] {
    var indices: [Int] = []; indices.reserveCapacity(paddedFrames)
    for frame in holds.indices { indices.append(contentsOf: repeatElement(frame, count: holds[frame])) }
    indices.append(contentsOf: repeatElement(sourceFrames - 1, count: paddedFrames - expandedFrames))
    return indices
  }
  public static func audioSampleBoundary(frame: Int) -> Int {
    Int((Double(frame) * 32_000 / 24).rounded(.toNearestOrEven))
  }

  public var audioFilter: String {
    var runs: [(Int, Int, Int)] = [], start = 0
    for end in 1...holds.count {
      if end == holds.count || holds[end] != holds[start] {
        runs.append((start, end, holds[start])); start = end
      }
    }
    var filters = ["[0:a]asplit=\(runs.count)" + runs.indices.map { "[s\($0)]" }.joined()]
    var expanded = 0
    for (index, run) in runs.enumerated() {
      let (start, end, hold) = run, count = (end - start) * hold
      let samples = Self.audioSampleBoundary(frame: expanded + count) - Self.audioSampleBoundary(frame: expanded)
      let tempo = hold == 4 ? "atempo=0.5,atempo=0.5" : hold == 3 ?
        "atempo=0.5,atempo=0.666666666667" : String(format: "atempo=%.12f", locale: Locale(identifier: "en_US_POSIX"), 1 / Double(hold))
      filters.append("[s\(index)]atrim=start_sample=\(Self.audioSampleBoundary(frame: start)):end_sample=\(Self.audioSampleBoundary(frame: end)),asetpts=PTS-STARTPTS,\(tempo),apad,atrim=end_sample=\(samples)[a\(index)]")
      expanded += count
    }
    filters.append(runs.indices.map { "[a\($0)]" }.joined() +
      "concat=n=\(runs.count):v=0:a=1,apad,atrim=end_sample=\(Self.audioSampleBoundary(frame: paddedFrames))[out]")
    return filters.joined(separator: ";")
  }
}
