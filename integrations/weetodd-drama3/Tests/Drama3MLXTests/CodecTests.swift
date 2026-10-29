import MLX
import XCTest

@testable import Drama3MLX

final class CodecTests: XCTestCase {
  func testCodeArchiveRoundTripsAndRejectsInvalidCodebooks() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let codes = MLXArray((0..<30).map { Int32($0) }).reshaped(10, 3)
    let file = root.appendingPathComponent("codes.safetensors")
    try Drama3AudioCodec.saveCodes(codes, to: file)
    XCTAssertEqual(
      try Drama3AudioCodec.loadCodes(file).asArray(Int32.self), Array(0..<30).map(Int32.init))
    XCTAssertThrowsError(
      try Drama3AudioCodec.saveCodes(MLXArray.zeros([9, 3], dtype: .int32), to: file))
  }
}
