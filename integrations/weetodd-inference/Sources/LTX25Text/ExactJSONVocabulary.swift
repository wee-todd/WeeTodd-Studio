import Foundation

/// JSONDecoder's keyed containers use canonically equivalent String keys;
/// JSONSerialization also drops leading BOMs. Neither is suitable for BPE vocab.
/// This bounded scanner retains the UTF-8 identity of each JSON object key.
enum ExactJSONVocabulary {
  static func decode(_ data: Data) throws -> [Data: Int] {
    let bytes = Array(data)
    func fail() -> TextEncodingError { .invalid("Malformed tokenizer vocabulary JSON.") }
    func whitespace(_ i: inout Int) { while i < bytes.count && [9,10,13,32].contains(bytes[i]) { i += 1 } }
    func string(_ i: inout Int) throws -> Data {
      guard i < bytes.count, bytes[i] == 34 else { throw fail() }
      let start = i; i += 1; var escaped = false
      while i < bytes.count {
        if bytes[i] == 34 {
          let end = i; i += 1
          if !escaped { return Data(bytes[(start+1)..<end]) }
          let value = try JSONDecoder().decode(String.self,from: Data(bytes[start..<i]))
          return Data(value.utf8)
        }
        if bytes[i] == 92 { escaped = true; i += 1 }
        i += 1
      }
      throw fail()
    }
    func skip(_ i: inout Int) throws {
      whitespace(&i)
      guard i < bytes.count else { throw fail() }
      if bytes[i] == 34 { _ = try string(&i); return }
      if bytes[i] == 123 || bytes[i] == 91 {
        var depth = 1; i += 1
        while i < bytes.count && depth > 0 {
          if bytes[i] == 34 { _ = try string(&i); continue }
          if bytes[i] == 123 || bytes[i] == 91 { depth += 1 }
          if bytes[i] == 125 || bytes[i] == 93 { depth -= 1 }
          i += 1
        }
        guard depth == 0 else { throw fail() }; return
      }
      while i < bytes.count && ![9,10,13,32,44,125,93].contains(bytes[i]) { i += 1 }
    }
    func field(_ name: String, at start: Int) throws -> Int {
      var i = start; whitespace(&i)
      guard i < bytes.count, bytes[i] == 123 else { throw fail() }; i += 1
      while i < bytes.count {
        whitespace(&i); let key = try string(&i); whitespace(&i)
        guard i < bytes.count, bytes[i] == 58 else { throw fail() }; i += 1; whitespace(&i)
        if key == Data(name.utf8) { return i }
        try skip(&i); whitespace(&i)
        guard i < bytes.count, bytes[i] == 44 else { throw fail() }; i += 1
      }
      throw fail()
    }
    let model = try field("model",at: 0)
    var i = try field("vocab",at: model)
    guard bytes[i] == 123 else { throw fail() }; i += 1
    var result: [Data:Int] = [:]
    while i < bytes.count {
      whitespace(&i)
      guard i < bytes.count else { throw fail() }
      if bytes[i] == 125 { break }
      let key = try string(&i); whitespace(&i)
      guard i < bytes.count, bytes[i] == 58 else { throw fail() }; i += 1; whitespace(&i)
      let start = i
      while i < bytes.count && (48...57).contains(bytes[i]) { i += 1 }
      guard i > start, let value = Int(String(decoding: bytes[start..<i],as: UTF8.self)),
        result.updateValue(value,forKey: key) == nil else { throw fail() }
      whitespace(&i)
      guard i < bytes.count else { throw fail() }
      if bytes[i] == 125 { break }
      guard bytes[i] == 44 else { throw fail() }; i += 1
    }
    return result
  }
}
