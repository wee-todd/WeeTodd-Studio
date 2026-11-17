import Foundation

/// Small bounded parser: Foundation's dictionary decoding discards duplicate keys and number spelling.
/// Import must reject duplicate fields and distinguish integer dimensions from bools/fractional numbers.
indirect enum LayoutJSON {
  case object([String: LayoutJSON]), array([LayoutJSON]), string(String), number(String), bool(Bool), null
  var object: [String: LayoutJSON]? { if case .object(let v) = self { v } else { nil } }
  var array: [LayoutJSON]? { if case .array(let v) = self { v } else { nil } }
  var string: String? { if case .string(let v) = self { v } else { nil } }
  var double: Double? { if case .number(let v) = self { Double(v) } else { nil } }
  var integer: Int? { if case .number(let v) = self { Int(v) } else { nil } }
  subscript(_ key: String) -> LayoutJSON? { object?[key] }
  static func parse(_ data: Data) throws -> LayoutJSON {
    guard data.count <= 2 * 1024 * 1024 else { throw ImageLayoutError.invalid("Layout JSON exceeds 2 MiB.") }
    var parser = Parser(bytes: Array(data)); let value = try parser.value(depth: 0)
    parser.space(); guard parser.i == parser.bytes.count else { throw ImageLayoutError.invalid("Unexpected trailing JSON.") }
    return value
  }
  private struct Parser {
    let bytes: [UInt8]
    var i = 0
    mutating func space() { while i < bytes.count && [9, 10, 13, 32].contains(bytes[i]) { i += 1 } }
    mutating func take(_ byte: UInt8) -> Bool { space(); if i < bytes.count && bytes[i] == byte { i += 1; return true }; return false }
    mutating func require(_ byte: UInt8) throws { guard take(byte) else { throw ImageLayoutError.invalid("Malformed layout JSON.") } }
    mutating func text() throws -> String {
      space(); let start = i; try require(34)
      while i < bytes.count {
        let c = bytes[i]; i += 1
        if c == 34 { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<i])) }
        if c == 92 { guard i < bytes.count else { break }; i += 1 }
      }
      throw ImageLayoutError.invalid("Unterminated JSON string.")
    }
    mutating func value(depth: Int) throws -> LayoutJSON {
      space(); guard depth <= 64, i < bytes.count else { throw ImageLayoutError.invalid("Invalid or excessively nested JSON.") }
      if bytes[i] == 34 { return .string(try text()) }
      if take(123) {
        var entries: [String: LayoutJSON] = [:]
        if take(125) { return .object(entries) }
        repeat {
          let key = try text()
          guard entries[key] == nil else { throw ImageLayoutError.invalid("Duplicate JSON field: \(key)") }
          try require(58); entries[key] = try value(depth: depth + 1)
          if take(125) { return .object(entries) }; try require(44)
        } while true
      }
      if take(91) {
        var entries: [LayoutJSON] = []
        if take(93) { return .array(entries) }
        repeat {
          entries.append(try value(depth: depth + 1))
          if take(93) { return .array(entries) }; try require(44)
        } while true
      }
      let start = i
      while i < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[i]) { i += 1 }
      let token = String(decoding: bytes[start..<i], as: UTF8.self)
      if token == "true" { return .bool(true) }; if token == "false" { return .bool(false) }; if token == "null" { return .null }
      guard let number = Double(token), number.isFinite,
        (try? JSONSerialization.jsonObject(with: Data(token.utf8), options: .fragmentsAllowed)) is NSNumber else {
        throw ImageLayoutError.invalid("Invalid JSON value.")
      }
      return .number(token)
    }
  }
}
