import Foundation
import MLX
import LTX25Engine

/// Immutable auto intent; never changes the old manual request defaults.
public struct MLXAutomaticDurationPolicy: Codable, Sendable, Equatable {
  public let headCheckpointPath: String
  public let minimumSeconds: Double
  public let maximumSeconds: Double
  public let headHeaderSHA256: String?
  enum CodingKeys: String, CodingKey, CaseIterable {
    case headCheckpointPath = "head_checkpoint_path"
    case minimumSeconds = "minimum_seconds", maximumSeconds = "maximum_seconds"
    case headHeaderSHA256 = "head_header_sha256"
  }
  private struct AnyKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  public init(from decoder: Decoder) throws {
    let keys = try decoder.container(keyedBy: AnyKey.self)
    let actualKeys = Set(keys.allKeys.map(\.stringValue))
    let allowedKeys = Set(CodingKeys.allCases.map(\.rawValue))
    guard allowedKeys.subtracting(["head_header_sha256"]).isSubset(of:actualKeys),
      actualKeys.isSubset(of:allowedKeys) else {
      throw LTXError.invalid("Automatic duration has missing or unsupported fields.")
    }
    let container = try decoder.container(keyedBy: CodingKeys.self)
    headCheckpointPath = try container.decode(String.self, forKey: .headCheckpointPath)
    minimumSeconds = try container.decode(Double.self, forKey: .minimumSeconds)
    maximumSeconds = try container.decode(Double.self, forKey: .maximumSeconds)
    headHeaderSHA256 = try container.decodeIfPresent(String.self, forKey:.headHeaderSHA256)
    try validate()
  }
  private func validate() throws {
    guard headCheckpointPath.hasPrefix("/"), headCheckpointPath.utf8.count <= 4096,
      !headCheckpointPath.utf8.contains(0), minimumSeconds.isFinite,
      maximumSeconds.isFinite, (0.25...30).contains(minimumSeconds),
      maximumSeconds >= minimumSeconds, maximumSeconds <= 30,
      headHeaderSHA256 == nil || (headHeaderSHA256!.utf8.count == 64 &&
        headHeaderSHA256!.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })) else {
      throw LTXError.invalid("Automatic duration needs a bounded absolute head path and finite ordered bounds within 0.25–30 seconds.")
    }
  }
  public init(headCheckpointPath: String, minimumSeconds: Double = 1,
    maximumSeconds: Double = 20, headHeaderSHA256: String? = nil) {
    self.headCheckpointPath = headCheckpointPath
    self.minimumSeconds = minimumSeconds; self.maximumSeconds = maximumSeconds
    self.headHeaderSHA256 = headHeaderSHA256
  }
  public func maximumFrames(fps: Double) throws -> Int {
    try validate()
    return try MLXDurationHead.frames(seconds: maximumSeconds, fps: fps,
      minimumSeconds: minimumSeconds, maximumSeconds: maximumSeconds)
  }
}

/// Bound to the original frozen recipe, not the effective request after frame resolution.
public struct MLXTextPreparationBinding: Sendable, Equatable {
  public let originalRecipeSHA256: String
  public let prompt: String
  public let negativePrompt: String?
  public let gemmaRoot: String
  public let connectorCheckpoint: String
  public init(originalRecipeSHA256: String, prompt: String, negativePrompt: String?,
    gemmaRoot: String, connectorCheckpoint: String) throws {
    guard originalRecipeSHA256.utf8.count == 64,
      originalRecipeSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      gemmaRoot.hasPrefix("/"), connectorCheckpoint.hasPrefix("/"),
      gemmaRoot.utf8.count <= 4096, connectorCheckpoint.utf8.count <= 4096,
      !gemmaRoot.utf8.contains(0), !connectorCheckpoint.utf8.contains(0) else {
      throw LTXError.invalid("Prepared text requires the frozen recipe SHA and absolute component paths.")
    }
    self.originalRecipeSHA256 = originalRecipeSHA256
    self.prompt = prompt; self.negativePrompt = negativePrompt
    self.gemmaRoot = URL(fileURLWithPath: gemmaRoot).resolvingSymlinksInPath().standardizedFileURL.path
    self.connectorCheckpoint = URL(fileURLWithPath: connectorCheckpoint).resolvingSymlinksInPath().standardizedFileURL.path
  }
}

public struct MLXAutomaticDurationResolution: Sendable, Equatable {
  public let predictedDurationSeconds: Double
  public let resolvedFrames: Int
  public let fps: Double
  public var effectiveDurationSeconds: Double { Double(resolvedFrames) / fps }
}

/// Process-local, single-use ownership. No model weights or serialized array cache.
/// Intentionally not Sendable: the synchronous worker consumes it on its execution path.
public final class MLXPreparedTextLease {
  public struct Text {
    public let positive: MLXTextEncoder.Output
    public let negative: MLXTextEncoder.Output?
  }
  public let binding: MLXTextPreparationBinding
  public let resolution: MLXAutomaticDurationResolution
  private let gate = NSLock()
  private var storage: Text?
  init(binding: MLXTextPreparationBinding, resolution: MLXAutomaticDurationResolution,
    positive: MLXTextEncoder.Output, negative: MLXTextEncoder.Output?) {
    self.binding = binding; self.resolution = resolution
    storage = Text(positive: positive, negative: negative)
  }
  public var isReleased: Bool { gate.lock(); defer { gate.unlock() }; return storage == nil }
  public func release() { gate.lock(); storage = nil; gate.unlock() }
  /// Invalid binding, geometry, or cancellation fails closed and drops owned contexts.
  public func consume(expectedBinding: MLXTextPreparationBinding, resolvedFrames: Int,
    fps: Double) throws -> Text {
    gate.lock(); defer { gate.unlock() }
    let value = storage; storage = nil
    try Task.checkCancellation()
    guard expectedBinding == binding, resolvedFrames == resolution.resolvedFrames,
      fps == resolution.fps else { throw LTXError.invalid("Prepared text does not match the frozen recipe or resolved geometry.") }
    guard let value else { throw LTXError.invalid("Prepared text was already consumed or released.") }
    return value
  }
}

/// The caller performs complete worst-case header/geometry admission before encoding.
public enum MLXAutomaticTextPreparation {
  public struct Progress {
    public let stage: String
    public let text: MLXTextEncoder.Progress?
    public let resolution: MLXAutomaticDurationResolution?
  }
  /// Encodes positive once, resolves duration, then encodes negative only if requested.
  /// The local encoder/head are destroyed before the returned context lease reaches VAE.
  public static func prepare(binding: MLXTextPreparationBinding,
    policy: MLXAutomaticDurationPolicy, fps: Double,
    expectedHeadHeaderSHA256: String? = nil,
    admitResolvedFrames: (Int) throws -> Void,
    progress: (Progress) throws -> Void = { _ in }) throws -> MLXPreparedTextLease {
    _ = try policy.maximumFrames(fps: fps)
    try Task.checkCancellation()
    return try autoreleasepool {
      let head = try MLXDurationHead(checkpoint: URL(fileURLWithPath: policy.headCheckpointPath))
      if let pinned = policy.headHeaderSHA256, pinned != head.headerSHA256 {
        throw LTXError.invalid("Duration head changed before text encoding. Re-export the frozen automatic-duration request.")
      }
      if let expectedHeadHeaderSHA256 {
        guard expectedHeadHeaderSHA256 == head.headerSHA256 else {
          throw LTXError.invalid("Duration head changed before text encoding. Re-export the frozen automatic-duration request.")
        }
      }
      let encoder = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: binding.gemmaRoot),
        connectorURL: URL(fileURLWithPath: binding.connectorCheckpoint))
      let result = try prepare(binding: binding, policy: policy, fps: fps,
        encode: { prompt, negative in
          try encoder.encode(prompt: prompt) {
            try progress(Progress(stage: negative ? "text:unconditional" : "text", text: $0, resolution: nil))
          }
        }, predict: { try head.predict(video: $0.video, audio: $0.audio) },
        admitResolvedFrames: admitResolvedFrames, resolved: {
          try progress(Progress(stage: "duration_resolved", text: nil, resolution: $0))
        })
      do {
        try progress(Progress(stage: "text_weights_released", text: nil, resolution: result.resolution))
        try Task.checkCancellation()
        return result
      } catch { result.release(); throw error }
    }
  }
  /// Internal numeric-free seam exercises ownership and ordering on CPU fixtures.
  static func prepare(binding: MLXTextPreparationBinding, policy: MLXAutomaticDurationPolicy,
    fps: Double, encode: (String, Bool) throws -> MLXTextEncoder.Output,
    predict: (MLXTextEncoder.Output) throws -> Double,
    admitResolvedFrames: (Int) throws -> Void,
    resolved: (MLXAutomaticDurationResolution) throws -> Void = { _ in }) throws -> MLXPreparedTextLease {
    _ = try policy.maximumFrames(fps: fps)
    try Task.checkCancellation()
    let positive = try encode(binding.prompt, false)
    try Task.checkCancellation()
    let seconds = try predict(positive)
    let frames = try MLXDurationHead.frames(seconds: seconds, fps: fps,
      minimumSeconds: policy.minimumSeconds, maximumSeconds: policy.maximumSeconds)
    let resolution = MLXAutomaticDurationResolution(predictedDurationSeconds: seconds,
      resolvedFrames: frames, fps: fps)
    try admitResolvedFrames(frames)
    try resolved(resolution)
    try Task.checkCancellation()
    let negative = try binding.negativePrompt.map { try encode($0, true) }
    try Task.checkCancellation()
    return MLXPreparedTextLease(binding: binding, resolution: resolution, positive: positive, negative: negative)
  }
}
