import Drama3Contracts
import MLX
import XCTest

@testable import Drama3MLX

final class SessionTests: XCTestCase {
  func testInfillRetainsUnchangedCodeSpans() throws {
    let codes = MLXArray((0..<40).map(Int32.init)).reshaped(10, 4)
    let parts = try Drama3AudioEdit.partition(codes, start: 2048.0 / 44100, end: 4096.0 / 44100)
    XCTAssertEqual(parts.prefix.shape, [10, 1])
    XCTAssertEqual(parts.suffix.shape, [10, 2])
    XCTAssertEqual(parts.prefix[0, 0].item(Int.self), 0)
    XCTAssertEqual(parts.suffix[0, 0].item(Int.self), 2)
    XCTAssertThrowsError(try Drama3AudioEdit.partition(codes, start: 1, end: 0))
  }
  func testInsertionAndDeletionKeepExactCodeBoundaries() throws {
    let codes = MLXArray((0..<60).map(Int32.init)).reshaped(10, 6)
    let boundary = 2 * 2048.0 / 44100
    let insertion = try Drama3AudioEdit.partition(codes, start: boundary, end: boundary)
    XCTAssertEqual(
      concatenated([insertion.prefix, insertion.suffix], axis: 1).asArray(Int32.self),
      codes.asArray(Int32.self))
    let deletion = try Drama3AudioEdit.partition(codes, start: boundary, end: 4 * 2048.0 / 44100)
    XCTAssertEqual(deletion.prefix.dim(1) + deletion.suffix.dim(1), 4)
    XCTAssertEqual(deletion.suffix[0, 0].item(Int.self), 4)
  }
  func testMissingReceiptIsNeverACompleteTake() throws {
    XCTAssertThrowsError(
      try Drama3TakeBundle.verify(
        at: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
  }
  func testInsertionBetweenFramesPreservesEveryOriginalFrame() throws {
    let codes = MLXArray((0..<60).map(Int32.init)).reshaped(10, 6)
    let position = 2.7 * 2048 / 44100
    let insertion = try Drama3AudioEdit.partition(codes, start: position, end: position)
    XCTAssertEqual(
      concatenated([insertion.prefix, insertion.suffix], axis: 1).asArray(Int32.self),
      codes.asArray(Int32.self))
  }
}
