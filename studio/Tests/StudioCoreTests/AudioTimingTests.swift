import XCTest
@testable import StudioCore

final class AudioTimingTests: XCTestCase {
  func testMarkerEncodingPreservesSourceSecondsAndLock() throws {
    let marker = AudioTimingMarker(timeSeconds: 2.11, locked: true, label: "Downbeat")
    XCTAssertEqual(try marker.frame(fps: 24, sourceStart: 2, duration: 2), 3)
    let data = try JSONEncoder().encode(marker)
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(Set(object.keys), ["timeSeconds", "locked", "label"])
    XCTAssertEqual(try JSONDecoder().decode(AudioTimingMarker.self, from: data).timeSeconds, 2.11)
  }
  func testMarkerRejectsOutsideSourceAndNonfiniteValues() {
    XCTAssertThrowsError(try AudioTimingMarker(timeSeconds: .nan).frame(fps: 24, sourceStart: 0, duration: 2))
    XCTAssertThrowsError(try AudioTimingMarker(timeSeconds: 5).frame(fps: 24, sourceStart: 0, duration: 2))
  }
  func testUnknownWordTimingRemainsNull() throws {
    let json = #"{"text":"missing","startSeconds":null,"endSeconds":null,"confidence":0,"flags":["possible_omission"]}"#
    let word = try JSONDecoder().decode(AudioWordTiming.self, from: Data(json.utf8))
    XCTAssertNil(word.startSeconds)
    XCTAssertEqual(word.flags, ["possible_omission"])
  }
  func testReviewedWordBoundaryUsesExplicitSourceTimeAndPreservesUnknownEvidence() throws {
    let word = try JSONDecoder().decode(AudioWordTiming.self, from: Data(#"{"text":"missing","startSeconds":null,"endSeconds":null,"confidence":0,"flags":["possible_omission"]}"#.utf8))
    let marker = try word.reviewedCutMarker(at: 12.25, fps: 24, sourceStart: 10, duration: 5)
    XCTAssertEqual(marker.timeSeconds, 12.25)
    XCTAssertEqual(try marker.frame(fps: 24, sourceStart: 10, duration: 5), 54)
    XCTAssertEqual(marker.label, "Reviewed lyric: missing")
    XCTAssertFalse(marker.locked)
    XCTAssertNil(word.startSeconds)
    XCTAssertEqual(word.flags, ["possible_omission"])
    XCTAssertThrowsError(try word.reviewedCutMarker(at: .nan, fps: 24, sourceStart: 10, duration: 5))
    XCTAssertThrowsError(try word.reviewedCutMarker(at: 2.25, fps: 24, sourceStart: 10, duration: 5))
  }
  func testReviewRetainsOptionalVocalProvenanceAndLegacyDraftDecodes() throws {
    let review = try AudioAnalysisReview(result: ["analysis": ["sourceSHA256": String(repeating: "a", count: 64), "vocalAnalysis": ["mode": "isolated"]]])
    XCTAssertEqual(review.vocalMode, "isolated")
    let restored = try JSONDecoder().decode(AudioAnalysisReview.self, from: JSONEncoder().encode(review))
    XCTAssertEqual(restored.vocalMode, "isolated")
    let legacy = #"{"sourceSHA256":"old","cues":[],"words":[],"lines":[],"extraWords":[],"warnings":[],"beatCount":0,"downbeatCount":0}"#
    XCTAssertNil(try JSONDecoder().decode(AudioAnalysisReview.self, from: Data(legacy.utf8)).vocalMode)
  }

  func testReviewKeepsRawRecognitionSeparateFromLyricAssistance() throws {
    let word: [String: Any] = ["text": "CAN", "observedText": "KEN", "startSeconds": 1.0, "endSeconds": 1.4, "confidence": 0.4, "flags": ["text_mismatch"], "verification": "lyric_assisted"]
    let review = try AudioAnalysisReview(result: ["analysis": ["sourceSHA256": "source", "alignment": ["recognizedText": "THEY KEN REMEMBER", "lyricAssistedText": "THEY CAN REMEMBER", "words": [word]]]])
    let restored = try JSONDecoder().decode(AudioAnalysisReview.self, from: JSONEncoder().encode(review))
    XCTAssertEqual(restored.recognizedText, "THEY KEN REMEMBER")
    XCTAssertEqual(restored.lyricAssistedText, "THEY CAN REMEMBER")
    XCTAssertEqual(restored.words[0].observedText, "KEN")
    XCTAssertEqual(restored.words[0].verificationLabel, "Lyric-assisted — review")
    let legacy = try JSONDecoder().decode(AudioWordTiming.self, from: Data(#"{"text":"A","startSeconds":null,"endSeconds":null,"confidence":0,"flags":[]}"#.utf8))
    XCTAssertNil(legacy.observedText)
    XCTAssertNil(legacy.verificationLabel)
  }

}
