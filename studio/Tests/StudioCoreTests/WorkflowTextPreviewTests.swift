import XCTest
@testable import StudioCore

final class WorkflowTextPreviewTests: XCTestCase {
  func testLargeOutputIsBoundedWhileFullTextRemainsAvailable() {
    let original = String(repeating: "{\"beat\": 1}\n", count: 50_000)
    let preview = WorkflowTextPreview(original)
    XCTAssertLessThanOrEqual(preview.text.utf8.count, 32_000)
    XCTAssertTrue(preview.isTruncated)
    XCTAssertTrue(original.hasPrefix(preview.text))
    XCTAssertEqual(preview.fullText, original)
  }

  func testShortOutputAndExactBoundaryRemainUnchanged() {
    for original in ["", "Model response failed.\nRetry this step.", String(repeating: "x", count: 8_000)] {
      let preview = WorkflowTextPreview(original)
      XCTAssertEqual(preview.text, original)
      XCTAssertFalse(preview.isTruncated)
    }
  }

  func testLongUnbrokenLineAndCombiningSequenceAreBothBounded() {
    for original in [String(repeating: "x", count: 500_000), "e" + String(repeating: "\u{301}", count: 500_000)] {
      let preview = WorkflowTextPreview(original)
      XCTAssertLessThanOrEqual(preview.text.unicodeScalars.count, 8_000)
      XCTAssertTrue(preview.isTruncated)
      XCTAssertEqual(preview.fullText, original)
      XCTAssertFalse(preview.text.contains("\u{FFFD}"))
    }
  }
}
