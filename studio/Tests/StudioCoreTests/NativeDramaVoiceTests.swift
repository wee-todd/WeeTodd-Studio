import Drama3Contracts
import XCTest

@testable import StudioCore

final class NativeDramaVoiceTests: XCTestCase {
  func testSpeakDoesNotCarryHiddenInfillMediaOrRanges() throws {
    var draft = VoiceDraft()
    draft.text = "Hello"
    draft.dramaMode = "speak"
    draft.dramaSourceAudio = "/removed-parent.wav"
    draft.dramaEditStart = 1
    draft.dramaEditEnd = 2
    let request = try draft.drama3Request()
    XCTAssertNil(request.sourceAudio)
    XCTAssertNil(request.editStart)
    XCTAssertNil(request.editEnd)
  }
  func testConversationHonorsEditedSampleAndKeepsAdditionalSavedSamples() throws {
    var draft = VoiceDraft()
    draft.modelPath = "/native/model"
    draft.text = "Hello"
    draft.referenceMode = .audioAndTranscript
    let original = VoiceReference(path: "/saved.wav", duration: 2, transcript: "Original")
    let extra = VoiceReference(path: "/extra.wav", duration: 1, transcript: "Extra")
    draft.reference = original
    draft.voiceProfile = .init(name: "Alex", samples: [original, extra])
    draft.startDialogue()
    draft.dialogue?.speakers[0].reference = .init(
      path: "/edited.wav", start: 1, duration: 1, transcript: "Edited")
    let request = try draft.drama3Request()
    XCTAssertEqual(request.references.map(\.audioPath), ["/edited.wav", "/extra.wav"])
    XCTAssertEqual(request.references[0].text, "Edited")
    XCTAssertEqual(request.references[0].start, 1)
  }
  func testNewVoiceAndConversationUseExplicitNativeDefaults() throws {
    var draft = VoiceDraft()
    draft.modelPath = "/native/model"
    draft.text = "Ready?"
    XCTAssertEqual(draft.engine, .fishDrama3)
    XCTAssertEqual(draft.referenceMode, .synthetic)
    draft.startDialogue()
    XCTAssertEqual(draft.dramaMode, "lineContinuation")
    let request = try draft.request()
    XCTAssertEqual(request.drama3?.mode, .lineContinuation)
    XCTAssertTrue(request.drama3?.text.hasPrefix("<|speaker:0|>") == true)
    XCTAssertEqual(request.drama3?.sampling.topK, 30)
  }
  func testConversationRevisionPreservesExplicitDirectModeAndSpeakerTags() throws {
    var draft = VoiceDraft()
    draft.text = "Ready?"
    draft.startDialogue()
    draft.dramaMode = "direct"
    draft.dramaDirection = "More quietly."
    let request = try draft.drama3Request()
    XCTAssertEqual(request.mode, .direct)
    XCTAssertEqual(request.text, "<|speaker:0|> Ready?")
    XCTAssertEqual(request.direction, "More quietly.")
  }
  func testConversationKeepsUnsavedManualSamplesAndCollectsTheirPaths() throws {
    var draft = VoiceDraft()
    draft.text = "Hello"
    draft.referenceMode = .audioAndTranscript
    draft.reference = .init(path: "/first.wav", duration: 1, transcript: "First")
    draft.dramaReferences = [.init(path: "/second.wav", duration: 1, transcript: "Second"),
      .init(path: "/third.wav", duration: 1, transcript: "Third")]
    draft.startDialogue()
    draft = try JSONDecoder().decode(VoiceDraft.self, from: JSONEncoder().encode(draft))
    try draft.dialogue?.mapPaths { "/collected/" + URL(fileURLWithPath: $0).lastPathComponent }
    let request = try draft.drama3Request()
    XCTAssertEqual(request.references.map(\.audioPath), ["/collected/first.wav", "/collected/second.wav", "/collected/third.wav"])
    XCTAssertEqual(request.references.map(\.text), ["First", "Second", "Third"])
  }
  func testSavedVoiceDescriptionAndMemoryBudgetReachNativeRequest() throws {
    var draft = VoiceDraft()
    draft.modelPath = "/native/model"
    draft.text = "Hello"
    draft.dramaMemoryGiB = 24
    draft.voiceProfile = .init(
      name: "Warm Alex", samples: [.init(path: "/retained/sample.wav", transcript: "Hi")],
      description: "Warm and steady")
    let request = try draft.drama3Request()
    XCTAssertEqual(request.maxResidentBytes, 24 * 1_073_741_824)
    XCTAssertEqual(request.characters.first?.description, "Warm and steady")
    draft.dramaMemoryGiB = .nan
    XCTAssertThrowsError(try draft.drama3Request())
  }
}
