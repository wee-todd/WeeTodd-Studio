import CoreFoundation
import Foundation

/// Explicit global augmentation levels, separate from per-media strength.
/// Absence retains each released runner's historical arithmetic.
public struct H3ReferenceNoiseControls: Sendable, Equatable {
  public let visual: Float
  public let audio: Float
  public init(visual: Float = 0.999, audio: Float = 1) throws {
    guard visual.isFinite, audio.isFinite, (0...1).contains(visual), (0...1).contains(audio) else {
      throw H3CheckpointError.invalid("H3 reference noise strengths must be finite values from zero to one.")
    }
    self.visual = visual; self.audio = audio
  }
  static func parse(_ config: [String: Any]) throws -> Self? {
    let visual = config["visual_condition_strength"], audio = config["audio_condition_strength"]
    guard visual != nil || audio != nil else { return nil }
    func value(_ item: Any?, fallback: Float) throws -> Float {
      guard let item else { return fallback }
      guard let number = item as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue.isFinite, (0...1).contains(number.doubleValue) else {
        throw H3CheckpointError.invalid("H3 reference noise strengths must be numeric values from zero to one.")
      }
      return number.floatValue
    }
    return try Self(visual: value(visual, fallback: 0.999), audio: value(audio, fallback: 1))
  }
}

/// Weight-free media placement. The producer may resolve editorial endpoints
/// to numeric positions; a headless `last` follows its padded generation end.
enum H3ReferencePlacement {
  static func frame(_ value: Any?, frames: Int) throws -> Int? {
    guard let value else { return nil }
    if let text = value as? String, text == "last" { return frames - 1 }
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
      (0..<Double(frames)).contains(number.doubleValue) else {
      throw H3CheckpointError.invalid("H3 reference placement needs a generated-frame index or last.")
    }
    return number.intValue
  }
  static func placing(_ reference: H3Ref2VAReference, frame: Int?) throws -> H3Ref2VAReference {
    guard let frame else { return reference }
    switch reference {
    case .image(let value): return .timedImage(value, frame: frame)
    case .audio(let value): return .timedAudio(value, frame: frame)
    case .video(let value): return .timedVideo(value, frame: frame)
    case .timedImage, .timedAudio, .timedVideo:
      throw H3CheckpointError.invalid("The H3 media resolver must not pre-place a reference.")
    }
  }
}
