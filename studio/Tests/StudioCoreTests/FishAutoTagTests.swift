import Foundation
import XCTest
@testable import StudioCore

final class FishAutoTagTests: XCTestCase {
  func testNumberedSegmentsResolveRepeatedWordsWithoutModelOccurrenceCounting() throws {
    let request = try FishAutoTagRequest(text: "Hello again. Hello again. Goodbye.")
    XCTAssertEqual(request.segments.count, 3)
    let result = try request.applying(#"{"tags":[{"segment":2,"tag":"excited"}]}"#)
    XCTAssertEqual(result, "Hello again. [excited] Hello again. Goodbye.")
  }
  func testRepairsOnlyMissingOuterCloserAndAcceptsJSONFence() throws {
    let request = try FishAutoTagRequest(text: "Hello again.")
    XCTAssertEqual(try request.applying(#"{"tags":[{"segment":1,"tag":"whisper"}]"#), "[whisper] Hello again.")
    XCTAssertEqual(try request.applying("```json\n{\"tags\":[]}\n```"), "Hello again.")
    for malformed in [#"{"tags":[{"segment":1,"tag":"whisper"}"#, #"{"tags":[{"segment":1,"tag":"whisper"#, "Commentary {\"tags\":[]}", "{\"tags\":[]} trailing"] {
      XCTAssertThrowsError(try request.applying(malformed))
    }
  }
  func testUnicodeExistingTagsAndWhitespaceStayByteForByte() throws {
    let source = "  [sad] Hello 👩🏽‍🚀!\n\n世界、こんにちは。"
    let request = try FishAutoTagRequest(text: source)
    XCTAssertEqual(try request.applying(#"{"tags":[{"segment":1,"tag":"whisper"}]}"#), "  [sad] [whisper] Hello 👩🏽‍🚀!\n\n世界、こんにちは。")
    XCTAssertEqual(try request.applying(#"{"tags":[]}"#), source)
  }
  func testUntrustedResponseCannotRewriteScriptOrUseUnknownAnchors() throws {
    let request = try FishAutoTagRequest(text: "Hello. Goodbye.")
    for output in [#"{"tags":[{"segment":0,"tag":"sad"}]}"#, #"{"tags":[{"segment":99,"tag":"sad"}]}"#, #"{"tags":[{"segment":1,"tag":"evil]rewrite["}]}"#, #"{"tags":[{"segment":1,"tag":"sad"},{"segment":1,"tag":"excited"}]}"#] {
      XCTAssertThrowsError(try request.applying(output))
    }
    XCTAssertThrowsError(try FishAutoTagRequest(text: String(repeating: "a", count: 12001)))
    XCTAssertThrowsError(try FishAutoTagRequest(text: "   "))
  }
  func testLongScriptRemainsFullyRepresentedWithBoundedSegments() throws {
    let text = (1...90).map { "Sentence \($0)." }.joined(separator: " ")
    let request = try FishAutoTagRequest(text: text)
    XCTAssertLessThanOrEqual(request.segments.count, 64)
    XCTAssertTrue(request.prompt.contains("Sentence 90."))
    XCTAssertEqual(try request.applying(#"{"tags":[]}"#), text)
  }
  func testSentencePunctuationInsideExistingTagsNeverCreatesAnInteriorAnchor() throws {
    let source = "[Speak softly. Whisper loudly.] Hello."
    let request = try FishAutoTagRequest(text: source)
    XCTAssertEqual(request.segments.map(\.text), ["Hello."])
    XCTAssertEqual(try request.applying(#"{"tags":[{"segment":1,"tag":"whisper"}]}"#),
      "[Speak softly. Whisper loudly.] [whisper] Hello.")
  }
}
