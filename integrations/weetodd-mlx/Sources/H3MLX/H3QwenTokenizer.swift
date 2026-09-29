import Darwin
import Foundation

/// Qwen3-VL's byte-level BPE for H3 prompt conditioning. The model reads raw
/// prompt tokens without a chat template or automatic BOS/EOS tokens.
public final class H3QwenTokenizer {
  private struct Pair: Hashable { let left: String; let right: String }
  private let vocabulary: [String: Int32]
  private let ranks: [Pair: Int]
  private let splitRegex: NSRegularExpression
  private let specialRegex: NSRegularExpression?
  private let specialIDs: [String: Int32]

  // The installed tokenizer's legacy split expression is corrected by the
  // reference tokenizer's fix_mistral_regex option. Pin the corrected form so
  // punctuation/case boundaries match the current H3 renderer.
  private static let correctedSplit = #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
  private static let byteGlyphs: [String] = {
    let direct = Array(33...126) + Array(161...172) + Array(174...255)
    var result = [String](repeating: "", count: 256)
    for value in direct { result[value] = String(UnicodeScalar(value)!) }
    var codepoint = 256
    for value in 0..<256 where result[value].isEmpty {
      result[value] = String(UnicodeScalar(codepoint)!)
      codepoint += 1
    }
    return result
  }()

  init(vocabulary: [String: Int32], merges: [String], regex: String,
    specialIDs: [String: Int32] = [:]) throws {
    guard !vocabulary.isEmpty, vocabulary.count <= 200_000,
      merges.count <= 200_000, specialIDs.count <= 512 else {
      throw H3CheckpointError.invalid("H3 Qwen vocabulary or merge count is invalid.")
    }
    self.vocabulary = vocabulary
    self.specialIDs = specialIDs
    splitRegex = try NSRegularExpression(pattern: regex)
    let patterns = specialIDs.keys.sorted { $0.count > $1.count }
      .map(NSRegularExpression.escapedPattern(for:))
    specialRegex = patterns.isEmpty ? nil : try NSRegularExpression(pattern: patterns.joined(separator: "|"))
    var map: [Pair: Int] = [:]
    map.reserveCapacity(merges.count)
    for (rank, merge) in merges.enumerated() {
      let parts = merge.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
      guard parts.count == 2 else { throw H3CheckpointError.invalid("Malformed H3 Qwen BPE merge.") }
      let pair = Pair(left: String(parts[0]), right: String(parts[1]))
      guard map.updateValue(rank, forKey: pair) == nil else {
        throw H3CheckpointError.invalid("Duplicate H3 Qwen BPE merge.")
      }
    }
    ranks = map
  }

  public convenience init(url: URL) throws {
    guard url.isFileURL else { throw H3CheckpointError.invalid("Qwen tokenizer must be a local JSON file.") }
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw H3CheckpointError.invalid("Cannot read Qwen tokenizer.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size > 0, status.st_size <= 32 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Qwen tokenizer is not a bounded regular JSON file.")
    }
    let bytes = try handle.read(upToCount: 32 * 1024 * 1024 + 1) ?? Data()
    guard bytes.count <= 32 * 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
      let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
      let rawVocabulary = model["vocab"] as? [String: Int],
      let merges = model["merges"] as? [String],
      let pre = root["pre_tokenizer"] as? [String: Any],
      pre["type"] as? String == "Sequence",
      let steps = pre["pretokenizers"] as? [[String: Any]], steps.count == 2,
      steps[0]["type"] as? String == "Split",
      let pattern = steps[0]["pattern"] as? [String: String],
      pattern["Regex"] != nil,
      steps[1]["type"] as? String == "ByteLevel",
      let normalizer = root["normalizer"] as? [String: Any],
      normalizer["type"] as? String == "NFC" else {
      throw H3CheckpointError.invalid("Unsupported H3 Qwen tokenizer JSON layout.")
    }
    var specials: [String: Int32] = [:]
    for entry in root["added_tokens"] as? [[String: Any]] ?? [] {
      guard let content = entry["content"] as? String, let id = entry["id"] as? Int,
        id >= 0, id <= Int(Int32.max) else {
        throw H3CheckpointError.invalid("Invalid H3 Qwen added token.")
      }
      specials[content] = Int32(id)
    }
    var vocab: [String: Int32] = [:]
    vocab.reserveCapacity(rawVocabulary.count)
    for (token, id) in rawVocabulary {
      guard id >= 0, id <= Int(Int32.max) else {
        throw H3CheckpointError.invalid("Invalid H3 Qwen vocabulary ID.")
      }
      vocab[token] = Int32(id)
    }
    try self.init(vocabulary: vocab, merges: merges, regex: Self.correctedSplit,
      specialIDs: specials)
  }

  public func encode(_ prompt: String) throws -> [Int32] {
    guard prompt.utf8.count <= 65_536 else {
      throw H3CheckpointError.invalid("H3 prompt exceeds the tokenizer input limit.")
    }
    let normalized = prompt.precomposedStringWithCanonicalMapping
    let text = normalized as NSString
    let full = NSRange(location: 0, length: text.length)
    var output: [Int32] = []
    var cursor = 0
    for match in specialRegex?.matches(in: normalized, range: full) ?? [] {
      if cursor < match.range.location {
        try encodePlain(text.substring(with: NSRange(location: cursor,
          length: match.range.location - cursor)), into: &output)
      }
      guard let id = specialIDs[text.substring(with: match.range)] else {
        throw H3CheckpointError.invalid("Unknown H3 Qwen special token.")
      }
      output.append(id)
      cursor = NSMaxRange(match.range)
    }
    if cursor < text.length { try encodePlain(text.substring(from: cursor), into: &output) }
    return output
  }

  private func encodePlain(_ text: String, into output: inout [Int32]) throws {
    let utf16 = text as NSString
    let matches = splitRegex.matches(in: text, range: NSRange(location: 0, length: utf16.length))
    var cursor = 0
    for match in matches {
      guard match.range.location == cursor else {
        throw H3CheckpointError.invalid("Qwen pretokenizer left unmatched text.")
      }
      let token = utf16.substring(with: match.range)
      var parts = token.utf8.map { Self.byteGlyphs[Int($0)] }
      while parts.count > 1 {
        var bestRank = Int.max, bestIndex = -1
        for index in 0..<(parts.count - 1) {
          if let rank = ranks[Pair(left: parts[index], right: parts[index + 1])], rank < bestRank {
            bestRank = rank; bestIndex = index
          }
        }
        if bestIndex < 0 { break }
        parts[bestIndex] += parts[bestIndex + 1]
        parts.remove(at: bestIndex + 1)
      }
      for part in parts {
        guard let id = vocabulary[part] else {
          throw H3CheckpointError.invalid("H3 Qwen BPE piece has no vocabulary ID.")
        }
        output.append(id)
      }
      cursor = NSMaxRange(match.range)
    }
    guard cursor == utf16.length else {
      throw H3CheckpointError.invalid("Qwen pretokenizer did not consume the prompt.")
    }
  }
}
