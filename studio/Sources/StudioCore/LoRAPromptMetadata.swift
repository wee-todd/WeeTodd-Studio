import Foundation

/// Library guidance is copied with an applied LoRA. Only triggers affect prompt text.
public struct LoRAPromptMetadata: Codable, Equatable {
  public var triggers: [String]
  public var notes: String
  public var sourceURL: String?
  public init(triggers: [String] = [], notes: String = "", sourceURL: String? = nil) {
    self.triggers = triggers; self.notes = notes; self.sourceURL = sourceURL
  }
}

public enum LoRAPromptKey {
  public static func native(_ path: String) -> String {
    "native:" + URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
  }
  public static func drawThings(profileID: String, modelID: String) -> String {
    "drawthings:\(profileID.utf8.count):\(profileID)\(modelID)"
  }
}

public enum LoRAPromptInsertion {
  private static func normalized(_ text: String) -> String {
    text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
      .precomposedStringWithCanonicalMapping.lowercased(with: Locale(identifier: "en_US_POSIX"))
  }
  public static func applying(_ metadata: [LoRAPromptMetadata], to prompt: String) -> String {
    let present = normalized(prompt)
    var missing: [String] = []
    for text in metadata.flatMap(\.triggers) {
      let trigger = text.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trigger.isEmpty else { continue }
      let key = normalized(trigger)
      let pattern = "(?<![\\p{L}\\p{N}_])" + NSRegularExpression.escapedPattern(for: key)
        + "(?![\\p{L}\\p{N}_])"
      guard present.range(of: pattern, options: .regularExpression) == nil,
        missing.allSatisfy({ normalized($0).range(of: pattern, options: .regularExpression) == nil })
      else { continue }
      missing.append(trigger)
    }
    guard !missing.isEmpty else { return prompt }
    let prefix = missing.joined(separator: ", ")
    let field = try! NSRegularExpression(pattern:
      "^[\\t ]*(?:integrated_multimodal_description|detailed_description)[\\t ]*:",
      options: [.anchorsMatchLines, .caseInsensitive])
    let full = NSRange(prompt.startIndex..., in: prompt)
    if let match = field.firstMatch(in: prompt, range: full) {
      let start = NSMaxRange(match.range)
      let tail = NSRange(location: start, length: full.length - start)
      let boundary = try! NSRegularExpression(pattern:
        "^[\\t ]*(?:overall_soundscape|non_diegetic_music|subject_definitions|summary|retention_analysis)[\\t ]*:",
        options: [.anchorsMatchLines, .caseInsensitive])
      let end = boundary.firstMatch(in: prompt, range: tail)?.range.location ?? full.length
      let shot = try! NSRegularExpression(pattern: "\\[Shot[\\t ]+1\\](?:[\\t ]*:)?", options: .caseInsensitive)
      let marker = shot.firstMatch(in: prompt, range: NSRange(location: start, length: end - start))
      let offset = marker.map { NSMaxRange($0.range) } ?? start
      let index = String.Index(utf16Offset: offset, in: prompt)
      let space = index == prompt.endIndex || !prompt[index].isWhitespace ? " " : ""
      return String(prompt[..<index]) + " " + prefix + space + prompt[index...]
    }
    return prefix + (prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : "\n" + prompt)
  }
}
