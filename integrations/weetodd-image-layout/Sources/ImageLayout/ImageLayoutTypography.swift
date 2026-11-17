import Foundation

public enum ImageLayoutTypography {
  public static let colors = ["black", "white", "gray", "red", "orange", "yellow", "gold", "green", "blue", "purple", "pink", "brown", "cream", "silver", "mixed"]
  public static let fonts = ["serif", "sans-serif", "script", "blackletter", "typewriter", "decorative", "cutout", "mixed", "unclear"]
  public static let weights = ["thin", "regular", "bold", "mixed"]
  public static let cases = ["uppercase", "lowercase", "title-case", "mixed"]
  public static func editableStyle(_ caption: String) -> String {
    caption.replacingOccurrences(of: #"['\"‘’“”][\p{L}\p{N}]['\"‘’“”]"#, with: "glyph", options: .regularExpression)
      .replacingOccurrences(of: #"\b(letter|glyph|character)\s+[A-Z0-9](?![\p{L}\p{N}])"#, with: "$1", options: .regularExpression)
  }
  static func describesOnlyLettering(_ caption: String) -> Bool {
    guard caption.range(of: #"\b(text|lettering|letters|headline|masthead|words|title)\b"#, options: [.regularExpression, .caseInsensitive]) != nil else { return false }
    let withoutLiterals = caption.replacingOccurrences(of: #"["“‘][^"”’]*["”’]|'[^']*'"#, with: "", options: .regularExpression)
    let words = withoutLiterals.lowercased().components(separatedBy: CharacterSet.letters.inverted).filter { !$0.isEmpty }
    let qualifiers = Set(colors + ["a", "an", "the", "golden", "grey", "navy", "teal", "maroon", "dark", "light", "pale", "muted",
      "serif", "sans", "script", "thin", "regular", "bold", "uppercase", "lowercase", "large", "small", "medium", "tiny", "wide", "tall"])
    guard let subject = words.first(where: { !qualifiers.contains($0) }),
      ["text", "lettering", "letters", "headline", "headlines", "masthead", "title", "words", "word"].contains(subject) else { return false }
    if let block = words.firstIndex(of: "block"), block == 0 || words[block - 1] != "text" { return false }
    let typography = Set(colors + ["golden", "grey", "navy", "teal", "maroon", "dark", "light", "pale", "muted",
      "serif", "sans", "script", "thin", "regular", "bold", "uppercase", "lowercase", "capitals", "block", "text", "line", "lines",
      "type", "typeface", "font", "size", "tall", "wide", "spaced", "spacing", "letter", "letters", "lettering", "headline", "headlines",
      "masthead", "title", "words", "word", "reading", "spelling", "says", "saying", "written", "style", "with", "and", "or",
      "large", "small", "medium", "tiny", "a", "an", "the", "in", "on", "against", "at", "top", "bottom", "left", "right",
      "center", "centered", "middle", "of", "background", "above", "below"])
    // Unknown/material/subject words retain the complete physical region for review.
    return !words.isEmpty && words.allSatisfy { typography.contains($0) }
  }
}
