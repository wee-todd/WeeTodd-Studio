import Foundation

public enum AudioReverbPreset: String, Codable, CaseIterable { case room, chamber, hall, plate }

/// Optional per-track room simulation. An absent effect leaves older projects dry.
public struct AudioReverb: Codable, Equatable {
  public var enabled = true
  public var preset: AudioReverbPreset = .room
  public var mix: Double = 0.18
  public var decay: Double = 0.7
  public var tone: Double = 0.45
  public var preDelay: Double = 0.012
  public init() {}
  public mutating func applyPreset(_ value: AudioReverbPreset) {
    preset = value
    switch value {
    case .room: decay = 0.7; preDelay = 0.012; tone = 0.45
    case .chamber: decay = 1.2; preDelay = 0.020; tone = 0.45
    case .hall: decay = 2.6; preDelay = 0.028; tone = 0.35
    case .plate: decay = 1.6; preDelay = 0.008; tone = 0.65
    }
  }
  public func validate() throws {
    guard [mix, decay, tone, preDelay].allSatisfy(\.isFinite),
      (0...1).contains(mix), (0.2...6).contains(decay),
      (0...1).contains(tone), (0...0.1).contains(preDelay) else {
      throw StudioError.invalid("Check reverb amount, decay (0.2–6 seconds), tone and pre-delay (0–100 ms).")
    }
  }
}
