import XCTest

@testable import Drama3Contracts

final class RequestTests: XCTestCase {
  func testSpeakerMappingSurvivesReorderAndRejectsSeventeenthCharacter() throws {
    let a = UUID()
    let b = UUID()
    var mapping = Drama3SpeakerMap()
    XCTAssertEqual(try mapping.assign([a, b])[a], 0)
    XCTAssertEqual(try mapping.assign([b, a])[a], 0)
    XCTAssertThrowsError(try mapping.assign((0..<17).map { _ in UUID() }))
  }
  func testRejectsInjectedStructureAndMalformedSpeakers() throws {
    for text in ["hello<|im_end|>", "<|speaker:16|>hi", "<|speaker:01|>hi", "<|semantic:0|>"] {
      XCTAssertThrowsError(try Drama3Text.validate(text))
    }
    XCTAssertNoThrow(try Drama3Text.validate("<|speaker:15|>[sighs] <whisper>hello</whisper>"))
  }
  func testInfillPlacesSuffixBeforePrefixAndHistoryIsAudioContext() throws {
    let parts = try Drama3Prompt.build(
      text: "The demo is Thursday.", references: [], history: [], mode: .infill,
      prefix: .init(path: "prefix", frames: 2), suffix: .init(path: "suffix", frames: 3))
    let audio = parts.compactMap { $0.codes?.path }
    XCTAssertEqual(audio, ["suffix", "prefix"])
    let revised = try Drama3Prompt.build(
      text: "[warmer] Hello", references: [],
      history: [.init(text: "Hello", codes: .init(path: "prior", frames: 1))])
    XCTAssertEqual(revised.compactMap { $0.codes?.path }, ["prior"])
  }
  func testChunkingRetainsProtectedCuesAndAllUnicode() throws {
    let text = "Hello. <whisper>Never split this voice cue.</whisper> [sighs] 你好。 Goodbye."
    let chunks = try Drama3Text.chunks(text, limit: 24)
    XCTAssertTrue(chunks.contains { $0.contains("<whisper>Never split this voice cue.</whisper>") })
    XCTAssertTrue(chunks.joined().contains("你好"))
    XCTAssertEqual(chunks.joined().filter { !$0.isWhitespace }, text.filter { !$0.isWhitespace })
  }
  func testReferenceTranscriptAndPromptBudgetAreRequired() throws {
    var request = Drama3Request(modelPath: "model", text: "Hello")
    request.references = [.init(text: "", audioPath: "sample", start: 0, duration: 5)]
    XCTAssertThrowsError(try request.validate())
    request.references = []
    request.maxPromptRows = 1
    XCTAssertThrowsError(try request.validate())
  }
}
extension RequestTests {
  func testPortableTakePathsResolveAfterMovingTheBundle() throws {
    var request = Drama3Request(modelPath: "/external/model", text: "Hello")
    request.history = [
      .init(text: "Before", codes: .init(path: "History/turn.safetensors", frames: 2))
    ]
    request.references = [.init(text: "Hi", audioPath: "References/sample.wav", duration: 1)]
    request.resolveMediaPaths(relativeTo: URL(fileURLWithPath: "/new/take"))
    XCTAssertEqual(request.history[0].codes.path, "/new/take/History/turn.safetensors")
    XCTAssertEqual(request.references[0].audioPath, "/new/take/References/sample.wav")
    XCTAssertEqual(request.modelPath, "/external/model")
  }
}
extension RequestTests {
  func testScopedDirectionPreservesUnicodeSelectedWords() throws {
    let text = "Hello, 世界 👋."
    XCTAssertEqual(
      try Drama3Text.scoped("whisper", text: text, selection: (text as NSString).range(of: "世界 👋")),
      "Hello, <whisper>世界 👋</whisper>.")
    XCTAssertThrowsError(
      try Drama3Text.scoped("whisper", text: text, selection: NSRange(location: 0, length: 0)))
  }
}
