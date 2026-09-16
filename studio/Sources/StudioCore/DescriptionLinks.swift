import Foundation

public struct DescriptionLinkTarget: Equatable {
  public var id: String
  public var name: String
  public var aliases: [String]
  public var description: String
  public init(id: String, name: String, aliases: [String] = [], description: String) {
    self.id = id; self.name = name; self.aliases = aliases; self.description = description
  }
}
public struct DescriptionLinkRange: Equatable {
  public var range: NSRange
  public var targetID: String
  public var tooltip: String
}
public enum DescriptionLinks {
  /// Presentation only: never inserts markup or changes the saved description.
  /// Names shared by linked targets are ambiguous; explicit IDs remain available.
  public static func ranges(in text: String, targets: [DescriptionLinkTarget], mentions: [DescriptionMention] = [], sourceDescription: String? = nil) -> [DescriptionLinkRange] {
    var terms: [String: [String: DescriptionLinkTarget]] = [:]
    for target in targets {
      for raw in [target.id, target.name] + target.aliases {
        let term = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !term.isEmpty { terms[term, default: [:]][target.id] = target }
      }
    }
    var candidates: [DescriptionLinkRange] = []
    for (term, owners) in terms where owners.count == 1 {
      guard let target = owners.values.first else { continue }
      let pattern = "(?<![\\p{L}\\p{N}_])" + NSRegularExpression.escapedPattern(for: term) + "(?![\\p{L}\\p{N}_])"
      guard let expression = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
      for match in expression.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
        candidates.append(DescriptionLinkRange(range: match.range, targetID: target.id,
          tooltip: target.name + "\n\n" + (target.description.isEmpty ? "No description yet." : target.description)))
      }
    }
    var explicit: [DescriptionLinkRange] = []
    if sourceDescription == text {
      for mention in mentions where !mention.phrase.isEmpty && mention.occurrence >= 0 {
        guard let target = targets.first(where: { $0.id == mention.targetID }) else { continue }
        let ns = text as NSString; var cursor = 0; var count = 0
        while cursor < ns.length {
          let range = ns.range(of: mention.phrase, range: NSRange(location: cursor, length: ns.length - cursor))
          if range.location == NSNotFound { break }
          if count == mention.occurrence {
            if !explicit.contains(where: { NSIntersectionRange($0.range, range).length > 0 }) {
              explicit.append(DescriptionLinkRange(range: range, targetID: target.id,
                tooltip: target.name + "\n\n" + (target.description.isEmpty ? "No description yet." : target.description)))
            }
            break
          }
          count += 1; cursor = NSMaxRange(range)
        }
      }
    }
    // Resolve longer names first, then restore reading order. UTF-16 ranges match NSTextStorage.
    candidates.sort { $0.range.length == $1.range.length ? $0.range.location < $1.range.location : $0.range.length > $1.range.length }
    var accepted: [DescriptionLinkRange] = explicit
    for candidate in candidates where !accepted.contains(where: { NSIntersectionRange($0.range, candidate.range).length > 0 }) {
      accepted.append(candidate)
    }
    return accepted.sorted { $0.range.location < $1.range.location }
  }
}
