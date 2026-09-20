import CryptoKit
import Foundation

public enum RippleAudioPolicy: String, Codable, CaseIterable {
  case preserve, silent
}

public struct RippleReference: Codable, Equatable, Identifiable {
  public var id = UUID()
  /// Zero-based frame relative to the captured source interval, sampled at draft.frameRate.
  public var frame: Int
  public var path: String
  public var originalPath = ""
  public var strength: Double = 1
  public init(frame: Int = 0, path: String = "") { self.frame = frame; self.path = path }
}

public struct RippleDraft: Codable, Equatable {
  public var sourcePath: String
  public var sourceSHA256: String?
  public var sourceIn: Double
  public var duration: Double
  public var sourceInspected: Bool?
  /// Presentation timestamp of the first decoded frame after the editorial trim.
  public var sourcePreviewStart: Double?
  public var frameRate: Double
  public var width = 768
  public var height = 448
  public var seed = 42
  public static let defaultPrompt = "Use the reference video for motion, timing, camera movement, composition, and unchanged scene content, while consistently propagating the visual edit established in the first frame throughout the video."
  public var prompt = RippleDraft.defaultPrompt
  public var loraStrength: Double = 1.35
  public var audioPolicy: RippleAudioPolicy = .preserve
  public var references = [RippleReference()]
  public init(clip: Clip, frameRate: Double) {
    sourcePath = clip.sourcePath; sourceIn = clip.sourceIn; duration = clip.duration
    self.frameRate = frameRate.isFinite && (1...60).contains(frameRate) ? frameRate : 24
    let scale = min(1, 1920 / Double(max(32, clip.generationWidth, clip.generationHeight)))
    width = max(32, Int((Double(clip.generationWidth) * scale / 32).rounded()) * 32)
    height = max(32, Int((Double(clip.generationHeight) * scale / 32).rounded()) * 32)
    seed = clip.seed
  }
  public var frameCount: Int {
    let count = ceil(duration * frameRate - 1e-7)
    guard count.isFinite, count > 0, count < Double(Int.max) else { return 0 }
    return Int(count)
  }
  /// Identity for accepting the initial submitted draft without retaining mutable external paths.
  public var inputFingerprint: String {
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    return SHA256.hash(data: (try? encoder.encode(self)) ?? Data()).map { String(format: "%02x", $0) }.joined()
  }
  public func sourceMatches(_ clip: Clip) -> Bool {
    sourcePath == clip.sourcePath && sourceIn == clip.sourceIn && duration == clip.duration
  }
  public func validate(requireReferences: Bool = true) throws {
    guard !sourcePath.isEmpty, sourceIn.isFinite, sourceIn >= 0,
      duration.isFinite, duration > 0, frameRate.isFinite, (1...60).contains(frameRate),
      frameCount > 0, duration <= 30, [width, height].allSatisfy({ (32...1920).contains($0) && $0 % 32 == 0 }),
      seed >= 0, seed < 4_294_967_296, loraStrength.isFinite, loraStrength > 0, loraStrength <= 3 else {
      throw StudioError.invalid("Ripple needs a source clip, a valid interval and frame rate, dimensions divisible by 32, and valid seed and adapter strength.")
    }
    guard requireReferences else { return }
    guard (1...9).contains(references.count) else {
      throw StudioError.invalid("Add between one and nine edited images.")
    }
    guard references.contains(where: { $0.frame == 0 }) else {
      throw StudioError.invalid("The first source frame (frame 0) is required. Add up to eight other distinct frames.")
    }
    guard Set(references.map(\.frame)).count == references.count else {
      throw StudioError.invalid("Assign each edited image to a different source frame.")
    }
    guard references.allSatisfy({ $0.frame >= 0 && $0.frame < frameCount && !$0.path.isEmpty
      && $0.strength.isFinite && (0...1).contains($0.strength) }) else {
      throw StudioError.invalid("Each reference needs an edited image, a frame inside the selected interval, and strength from 0 to 1.")
    }
  }
  public func bridgeObject(requireReferences: Bool = true) throws -> [String: Any] {
    try validate(requireReferences: requireReferences)
    var result: [String: Any] = ["source_path": sourcePath, "source_start": sourceIn, "duration": duration,
      "frame_rate": frameRate, "width": width, "height": height, "seed": seed,
      "prompt": prompt, "lora_strength": loraStrength, "audio_policy": audioPolicy.rawValue,
      "references": (requireReferences ? references : []).map { ["frame": $0.frame, "path": $0.path, "strength": $0.strength] as [String: Any] }]
    if let sourceSHA256 { result["source_sha256"] = sourceSHA256 }
    return result
  }
  public mutating func mapPaths(_ transform: (String) throws -> String) rethrows {
    sourcePath = try transform(sourcePath)
    for i in references.indices {
      references[i].path = try transform(references[i].path)
      references[i].originalPath = try transform(references[i].originalPath)
    }
  }
}

public struct RippleTake: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var created = Date()
  public var draft: RippleDraft
  public var submittedDraftFingerprint: String?
  public var path: String
  public var receiptPath: String
  public var artifactsDirectory: String
  public var hasAudio: Bool
  public init(draft: RippleDraft, path: String, receiptPath: String, artifactsDirectory: String, hasAudio: Bool,
    submittedDraftFingerprint: String? = nil) {
    self.draft = draft; self.path = path; self.receiptPath = receiptPath
    self.submittedDraftFingerprint = submittedDraftFingerprint
    self.artifactsDirectory = artifactsDirectory; self.hasAudio = hasAudio
  }
}

extension StudioProject {
  public mutating func mapRipplePaths(_ transform: (String) throws -> String) rethrows {
    for i in clips.indices {
      try clips[i].rippleDraft?.mapPaths(transform)
      if clips[i].rippleTakes != nil {
        for j in clips[i].rippleTakes!.indices {
          try clips[i].rippleTakes![j].draft.mapPaths(transform)
          clips[i].rippleTakes![j].path = try transform(clips[i].rippleTakes![j].path)
          clips[i].rippleTakes![j].receiptPath = try transform(clips[i].rippleTakes![j].receiptPath)
          clips[i].rippleTakes![j].artifactsDirectory = try transform(clips[i].rippleTakes![j].artifactsDirectory)
        }
      }
    }
  }
}
