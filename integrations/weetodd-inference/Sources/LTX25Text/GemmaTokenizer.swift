import Foundation
import TensorIO

/// The released Gemma4-LTX tokenizer: literal-space normalization, Unicode BPE,
/// byte fallback and embedded special tokens. No network or external tokenizer.
public final class GemmaTokenizer {
  private struct Merge { let rank: Int; let token: Int }
  private struct EncodedModel: Decodable {
    struct Model: Decodable { let merges: [[String]] }
    let model: Model
  }
  private let vocabulary: [Data: Int]
  private let merges: [UInt64: Merge]
  private let specials: [(String, Int)]
  public let bosTokenID: Int
  public let padTokenID: Int

  public convenience init(fixed: SafeTensorFile) throws {
    let data = try fixed.withTensorBytes(named: "tokenizer_json") { Data($0) }
    let config = try fixed.withTensorBytes(named: "hf_asset__tokenizer_config.json") { Data($0) }
    try self.init(tokenizerJSON: data, configurationJSON: config)
  }
  public init(tokenizerJSON: Data, configurationJSON: Data) throws {
    guard tokenizerJSON.count <= 64 * 1024 * 1024 else { throw TextEncodingError.invalid("Tokenizer JSON exceeds its limit.") }
    // JSONSerialization/NSString strips a leading BOM from string values on
    // macOS. JSONDecoder preserves bytes; Data keys also distinguish NFC/NFD.
    let encoded = try JSONDecoder().decode(EncodedModel.self,from: tokenizerJSON)
    let vocab = try ExactJSONVocabulary.decode(tokenizerJSON), pairs = encoded.model.merges
    func tokenID(_ text: String) -> Int? { vocab[Data(text.utf8)] }
    guard
      let root = try JSONSerialization.jsonObject(with: tokenizerJSON) as? [String: Any],
      let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
      model["byte_fallback"] as? Bool == true, model["ignore_merges"] as? Bool == false,
      let normalizer = root["normalizer"] as? [String: Any],
      normalizer["type"] as? String == "Replace", normalizer["content"] as? String == "▁",
      (normalizer["pattern"] as? [String: String])?["String"] == " ",
      let pre = root["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Split",
      (pre["pattern"] as? [String: String])?["String"] == " ",
      pre["behavior"] as? String == "MergedWithPrevious", pre["invert"] as? Bool == false,
      let config = try JSONSerialization.jsonObject(with: configurationJSON) as? [String: Any],
      let bos = config["bos_token"] as? String, let pad = config["pad_token"] as? String,
      let bosID = tokenID(bos), let padID = tokenID(pad),
      let added = root["added_tokens"] as? [[String: Any]],
      let post = root["post_processor"] as? [String: Any], post["type"] as? String == "TemplateProcessing",
      let template = post["single"] as? [[String: Any]], template.count == 1,
      (template[0]["Sequence"] as? [String: Any])?["id"] as? String == "A"
    else { throw TextEncodingError.invalid("Unsupported embedded Gemma tokenizer contract.") }
    guard vocab.values.allSatisfy({ $0 >= 0 && $0 < Int(UInt32.max) }),
      Set(vocab.values).count == vocab.count else { throw TextEncodingError.invalid("Invalid tokenizer IDs.") }
    var ranks: [UInt64: Merge] = [:]
    for (rank, pair) in pairs.enumerated() {
      guard pair.count == 2, let a = tokenID(pair[0]), let b = tokenID(pair[1]),
        let combined = tokenID(pair[0] + pair[1]) else { throw TextEncodingError.invalid("Invalid BPE merge at rank \(rank): \(pair.debugDescription)") }
      let key = Self.key(a, b)
      if ranks[key] == nil { ranks[key] = Merge(rank: rank, token: combined) }
    }
    var specialTokens: [(String, Int)] = []
    for token in added {
      guard let content = token["content"] as? String, !content.isEmpty,
        let id = token["id"] as? Int, tokenID(content) == id,
        token["normalized"] as? Bool == false, token["single_word"] as? Bool == false,
        token["lstrip"] as? Bool == false, token["rstrip"] as? Bool == false else {
        throw TextEncodingError.invalid("Unsupported added-token matching behavior.")
      }
      specialTokens.append((content, id))
    }
    vocabulary = vocab; merges = ranks; specials = specialTokens.sorted { $0.0.count > $1.0.count }
    bosTokenID = bosID; padTokenID = padID
  }
  private static func key(_ a: Int, _ b: Int) -> UInt64 { UInt64(a) << 32 | UInt64(b) }
  public func encode(_ prompt: String, maxLength: Int = 1024) throws -> [Int] {
    guard (1...1024).contains(maxLength), prompt.utf8.count <= 128 * 1024 else {
      throw TextEncodingError.invalid("Gemma prompt exceeds its token or UTF-8 input limit.")
    }
    var whitespace = CharacterSet.whitespacesAndNewlines
    whitespace.insert(charactersIn: "\u{1c}\u{1d}\u{1e}\u{1f}")
    let text = prompt.trimmingCharacters(in: whitespace)
    var ids: [Int] = [], cursor = text.startIndex
    while cursor < text.endIndex {
      try Task.checkCancellation()
      var match: (Range<String.Index>, Int)?
      for (special, id) in specials {
        if let range = text.range(of: special, range: cursor..<text.endIndex),
          match == nil || range.lowerBound < match!.0.lowerBound {
          match = (range, id)
        }
      }
      let end = match?.0.lowerBound ?? text.endIndex
      ids += try encodeSegment(String(text[cursor..<end]).replacingOccurrences(of: " ", with: "▁"))
      if let match { ids.append(match.1); cursor = match.0.upperBound }
      else { cursor = text.endIndex }
    }
    if ids.first != bosTokenID { ids.insert(bosTokenID, at: 0) }
    return Array(ids.prefix(maxLength))
  }
  private func encodeSegment(_ text: String) throws -> [Int] {
    var ids: [Int] = []
    for scalar in text.unicodeScalars {
      let piece = String(scalar)
      if let id = vocabulary[Data(piece.utf8)] { ids.append(id) }
      else {
        for byte in piece.utf8 {
          guard let id = vocabulary[Data(String(format: "<0x%02X>", Int(byte)).utf8)] else {
            throw TextEncodingError.invalid("Missing byte fallback token.")
          }
          ids.append(id)
        }
      }
    }
    // A lazy candidate heap keeps long prompts O(n log n). Original character
    // positions provide stable leftmost tie breaking when ranks are equal.
    struct Candidate {
      let rank: Int, left: Int, right: Int, merged: Int, leftID: Int, rightID: Int
    }
    guard ids.count > 1 else { return ids }
    var previous = ids.indices.map { $0-1 }
    var next = ids.indices.map { $0+1 < ids.count ? $0+1 : -1 }
    var alive = [Bool](repeating: true, count: ids.count)
    var heap: [Candidate] = []
    func earlier(_ a: Candidate, _ b: Candidate) -> Bool {
      a.rank == b.rank ? a.left < b.left : a.rank < b.rank
    }
    func insert(_ left: Int) {
      guard left >= 0, alive[left], next[left] >= 0 else { return }
      let right = next[left]
      guard let merge = merges[Self.key(ids[left],ids[right])] else { return }
      heap.append(Candidate(rank: merge.rank, left: left, right: right, merged: merge.token,
        leftID: ids[left], rightID: ids[right]))
      var index = heap.count-1
      while index > 0 {
        let parent = (index-1)/2
        guard earlier(heap[index],heap[parent]) else { break }
        heap.swapAt(index,parent); index = parent
      }
    }
    func pop() -> Candidate? {
      guard !heap.isEmpty else { return nil }
      let result = heap[0], last = heap.removeLast()
      if !heap.isEmpty {
        heap[0] = last; var index = 0
        while index*2+1 < heap.count {
          var child = index*2+1
          if child+1 < heap.count && earlier(heap[child+1],heap[child]) { child += 1 }
          guard earlier(heap[child],heap[index]) else { break }
          heap.swapAt(child,index); index = child
        }
      }
      return result
    }
    for index in 0..<(ids.count-1) { insert(index) }
    var iteration = 0
    while let candidate = pop() {
      iteration += 1
      if iteration % 1024 == 0 { try Task.checkCancellation() }
      let left = candidate.left, right = candidate.right
      guard alive[left], alive[right], next[left] == right,
        ids[left] == candidate.leftID, ids[right] == candidate.rightID else { continue }
      ids[left] = candidate.merged; alive[right] = false; next[left] = next[right]
      if next[right] >= 0 { previous[next[right]] = left }
      insert(previous[left]); insert(left)
    }
    return ids.indices.filter { alive[$0] }.map { ids[$0] }
  }
}
