import Foundation

public enum Drama3Text {
  public static func validate(_ text: String) throws {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw Drama3Error.invalid("Enter the text to speak.")
    }
    for token in [
      "<|im_start", "<|im_end", "<|voice", "<|text", "<|interleave", "<|semantic:", "<|fim_",
      "<|audio_separator", "<|endofturn",
    ] {
      if text.contains(token) {
        throw Drama3Error.invalid("The script contains reserved conversation structure: \(token).")
      }
    }
    let regex = try NSRegularExpression(pattern: "<\\|speaker[^>]*>?")
    for m in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
      let s = String(text[Range(m.range, in: text)!])
      guard (0..<16).contains(where: { s == "<|speaker:\($0)|>" }) else {
        throw Drama3Error.invalid("Speaker tags must use IDs 0–15 without leading zeros.")
      }
    }
  }
  public static func tagged(_ text: String, speaker: Int = 0) -> String {
    text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<|speaker:")
      ? text : "<|speaker:\(speaker)|> \(text)"
  }
  public static func chunks(_ text: String, limit: Int = 300) throws -> [String] {
    try validate(text)
    let protected = try NSRegularExpression(
      pattern: "<\\|[^>]*\\|>|<([A-Za-z][A-Za-z0-9_-]*)>.*?</\\1>|\\[[^\\[\\]]*\\]",
      options: .dotMatchesLineSeparators)
    let ns = text as NSString
    var units: [String] = []
    var cursor = 0
    func appendPlain(_ s: String) {
      var word = ""
      for c in s {
        word.append(c)
        if c.isWhitespace || ".!?;。！？；,:，：".contains(c) {
          units.append(word)
          word = ""
        }
      }
      if !word.isEmpty { units.append(word) }
    }
    for m in protected.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
      appendPlain(ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor)))
      units.append(ns.substring(with: m.range))
      cursor = NSMaxRange(m.range)
    }
    appendPlain(ns.substring(from: cursor))
    var result: [String] = []
    var current = ""
    var speaker = ""
    var speakers = Set<String>()
    for unit in units {
      let newSpeaker = unit.hasPrefix("<|speaker:") && !speakers.contains(unit)
      if !current.isEmpty
        && (current.utf8.count + unit.utf8.count > max(16, limit)
          || (newSpeaker && speakers.count >= 3))
      {
        result.append(current)
        current = newSpeaker ? "" : (speaker.isEmpty ? "" : speaker + " ")
        speakers = current.isEmpty ? [] : [speaker]
      }
      if unit.hasPrefix("<|speaker:") {
        speaker = unit
        speakers.insert(unit)
      }
      current += unit
    }
    if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { result.append(current) }
    return result
  }
}
public struct Drama3PromptPart: Codable, Equatable, Sendable {
  public var text: String?
  public var codes: Drama3Codes?
  public init(_ text: String) { self.text = text }
  public init(_ codes: Drama3Codes) { self.codes = codes }
}
public enum Drama3Prompt {
  public static func build(
    text: String, references: [(String, Drama3Codes)], history: [Drama3History],
    mode: Drama3Mode = .speak,
    prefix: Drama3Codes? = nil, suffix: Drama3Codes? = nil, direction: String = "",
    scene: String = "", characters: [Drama3Character] = [], nextText: String? = nil
  ) throws -> [Drama3PromptPart] {
    try Drama3Text.validate(text)
    let separated = mode == .scene || mode == .lineSeparated
    let castByName = mode == .scene && references.isEmpty && !characters.isEmpty
    var parts: [Drama3PromptPart] = [.init("<|im_start|>system\n")]
    if mode == .infill {
      parts.append(
        .init(
          "fill in the missing part of the speech; the suffix follows <|fim_suffix|> and the prefix follows <|fim_prefix|>"
        ))
    } else {
      if mode == .streamingText {
        parts.append(
          .init("streaming speech continuation with one-chunk lookahead split by <|endofturn|>"))
      } else if !castByName {
        parts.append(
          .init(
            references.isEmpty
              ? "convert the provided text to speech"
              : "convert the provided text to speech reference to the following:"))
      }
      for (i, r) in references.enumerated() {
        parts += [
          .init(
            "\n\n[Reference \(i+1)]\nText: \(Drama3Text.tagged(r.0).trimmingCharacters(in:.whitespacesAndNewlines))\nSpeech: "
          ), .init(r.1),
        ]
      }
      if !scene.isEmpty { parts.append(.init("\nScene: \(scene)")) }
      if !characters.isEmpty { parts.append(.init("\nCharacters:")) }
      for c in characters {
        parts.append(
          .init("\n- \(castByName ? c.name : "<|speaker:\(c.token)|> "+c.name): \(c.description)"))
      }
      if separated {
        parts.append(
          .init(
            "\nVoice every line of the script in order, with <|audio_separator|> between consecutive lines."
          ))
      }
      if separated && !direction.isEmpty { parts.append(.init("\nDirection: \(direction)")) }
    }
    parts.append(.init("<|im_end|>\n"))
    for h in history {
      parts += [
        .init("<|im_start|>user\n"),
        .init(
          Drama3Text.tagged(h.text)
            + (mode == .streamingText ? "<|endofturn|>" + (h.nextText ?? "") : "")),
        .init("<|im_end|>\n<|im_start|>assistant\n<|voice|>"), .init(h.codes),
        .init("<|im_end|>\n"),
      ]
    }
    var target = Drama3Text.tagged(text)
    if separated {
      var lines = target.components(separatedBy: "\n").filter {
        !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      }
      if castByName {
        lines = lines.map { line in
          var result = line
          for c in characters {
            result = result.replacingOccurrences(of: "<|speaker:\(c.token)|>", with: c.name + ":")
          }
          return result
        }
      }
      target = lines.joined(separator: "\n<|audio_separator|>\n")
    } else if !direction.isEmpty {
      let regex = try NSRegularExpression(pattern: "^<\\|speaker:[0-9]+\\|>\\s*")
      let range = NSRange(target.startIndex..., in: target)
      if let match = regex.firstMatch(in: target, range: range),
        let r = Range(match.range, in: target)
      {
        let tag = target[r].trimmingCharacters(in: .whitespacesAndNewlines)
        target = tag + " Direction: \(direction)\nLine: " + target[r.upperBound...]
      }
    }
    if mode == .streamingText { target += "<|endofturn|>" + (nextText ?? "") }
    parts += [.init("<|im_start|>user\n"), .init(target)]
    if mode == .infill {
      guard let prefix, let suffix else {
        throw Drama3Error.invalid("Infilling requires source codes on both sides.")
      }
      parts += [.init("<|fim_suffix|>"), .init(suffix), .init("<|fim_prefix|>"), .init(prefix)]
    }
    parts.append(.init("<|im_end|>\n<|im_start|>assistant\n<|voice|>"))
    return parts
  }
}
extension Drama3Text {
  public static func scoped(_ tag: String, text: String, selection: NSRange) throws -> String {
    guard tag.range(of: "^[A-Za-z][A-Za-z0-9_-]{0,31}$", options: .regularExpression) != nil,
      selection.length > 0, let range = Range(selection, in: text), !text[range].contains("<|")
    else {
      throw Drama3Error.invalid("Select the spoken words to direct, then choose a scoped cue.")
    }
    return String(text[..<range.lowerBound]) + "<\(tag)>" + text[range] + "</\(tag)>"
      + text[range.upperBound...]
  }
}
