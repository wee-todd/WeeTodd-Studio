import Foundation

/// Free-form Fish S2 performance guidance; these descriptions do not guarantee a voice identity.
public struct FishVoiceDirection: Codable, Equatable {
  public var pitch = ""
  public var pace = ""
  public var timbre = ""
  public var accent = ""
  public var description = ""
  public init() {}

  public static let pitchOptions = ["", "low voice", "pitch up"]
  public static let paceOptions = ["", "slow delivery", "fast delivery"]
  public static let timbreOptions = ["", "warm voice", "breathy voice", "raspy voice", "clear resonant voice"]

  public func tags() throws -> [String] {
    guard Self.pitchOptions.contains(pitch), Self.paceOptions.contains(pace),
      Self.timbreOptions.contains(timbre) else {
      throw StudioError.invalid("Choose a supported Fish pitch, pace, and timbre.")
    }
    let accent = try Self.validated(accent, maximum: 80)
    let description = try Self.validated(description, maximum: 120)
    var result = [pitch, pace, timbre].filter { !$0.isEmpty }
    if !accent.isEmpty { result.append("with \(accent) accent") }
    if !description.isEmpty { result.append(description) }
    guard result.count <= 8 else { throw StudioError.invalid("Use no more than eight Fish voice directions.") }
    for tag in result {
      guard !(try Self.validated(tag, maximum: 120)).isEmpty else {
        throw StudioError.invalid("Fish voice directions must contain 1–120 characters.")
      }
    }
    return result
  }

  private static func validated(_ value: String, maximum: Int) throws -> String {
    guard !value.unicodeScalars.contains(where: {
        CharacterSet(charactersIn: "[]<>").contains($0)
          || CharacterSet.controlCharacters.contains($0) || CharacterSet.newlines.contains($0)
      }) else {
      throw StudioError.invalid("Use plain Fish voice descriptions without brackets, markup, or control characters.")
    }
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.unicodeScalars.count <= maximum else {
      throw StudioError.invalid("Keep this Fish voice description within \(maximum) characters.")
    }
    return trimmed
  }
}
