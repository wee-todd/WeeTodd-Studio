import Foundation
import XCTest
@testable import StudioCore

final class VoiceDialogueTests: XCTestCase {
  func testConversationResolvesSpeakersAndPerLinePerformance() throws {
    var draft = VoiceDraft(); draft.engine = .qwen3TTS; draft.modelPath = "/model"
    let alice = VoiceSpeaker(name: "Alice", reference: VoiceReference(path: "/alice.wav", transcript: "Hello"))
    let bob = VoiceSpeaker(name: "Bob", reference: VoiceReference(path: "/bob.wav", transcript: "Hi"))
    var second = VoiceTurn(speakerID: bob.id, text: "How are you?")
    second.reference = VoiceReference(path: "/bob-happy.wav", transcript: "Wonderful!")
    draft.dialogue = VoiceDialogue(speakers: [alice, bob], turns: [VoiceTurn(speakerID: alice.id, text: "Hello."), second])
    draft.usesDialogue = true
    let request = try draft.request()
    XCTAssertEqual(request.turns?.map(\.speakerName), ["Alice", "Bob"])
    XCTAssertEqual(request.turns?.map { $0.reference?.path }, ["/alice.wav", "/bob-happy.wav"])
    XCTAssertEqual(request.turns?.map(\.text), ["Hello.", "How are you?"])
    draft.dialogue?.speakers.removeLast()
    XCTAssertThrowsError(try draft.request())
  }
  func testDialoguePersistenceAndPortableReferences() throws {
    var draft = VoiceDraft(); draft.reference = VoiceReference(path: "/single.wav")
    draft.startDialogue()
    draft.dialogue?.turns[0].reference = VoiceReference(path: "/delivery.wav")
    let decoded = try JSONDecoder().decode(VoiceDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertEqual(decoded, draft)
    var project = StudioProject(); project.voiceDraft = draft
    project.mapVoicePaths { "portable" + $0 }
    XCTAssertEqual(project.voiceDraft?.dialogue?.speakers.first?.reference?.path, "portable/single.wav")
    XCTAssertEqual(project.voiceDraft?.dialogue?.turns.first?.reference?.path, "portable/delivery.wav")
  }
  func testTagsPreserveWordsAndRejectUnsafeOrAmbiguousSuggestions() throws {
    let source = "Hello, 世界! Hello again."
    let output = try FishVoiceTags.applyingSuggestions("{\"tags\":[{\"before\":\"Hello\",\"occurrence\":2,\"tag\":\"excited\"}]}", to: source)
    XCTAssertEqual(output, "Hello, 世界! [excited] Hello again.")
    XCTAssertThrowsError(try FishVoiceTags.applyingSuggestions("{\"tags\":[{\"before\":\"missing\",\"occurrence\":1,\"tag\":\"sad\"}]}", to: source))
    XCTAssertThrowsError(try FishVoiceTags.applyingSuggestions("{\"tags\":[{\"before\":\"Hello\",\"occurrence\":1,\"tag\":\"bad]new words[\"}]}", to: source))
    XCTAssertThrowsError(try FishVoiceTags.applyingSuggestions("{\"tags\":[{\"before\":\"sad\",\"occurrence\":1,\"tag\":\"excited\"}]}", to: "[sad] Hello"))
    XCTAssertEqual(try FishVoiceTags.inserting("whisper", into: "Hello", at: NSRange(location: 0, length: 5)).text, "[whisper] Hello")
  }
  func testDialogueRejectsInvalidBudgetsAndUnsupportedTagsForQwen() throws {
    var draft = VoiceDraft(); draft.modelPath = "/model"; draft.text = "Hello"
    draft.reference = VoiceReference(path: "/ref.wav", transcript: "Hello")
    draft.startDialogue(); draft.dialogue?.turns[0].gapAfter = .nan
    XCTAssertThrowsError(try draft.request())
    draft.dialogue?.turns[0].gapAfter = 0
    draft.dialogue?.turns[0].text = "[excited] Hello"
    XCTAssertThrowsError(try draft.request())
    draft.engine = .fishS2Pro
    XCTAssertNoThrow(try draft.request())
  }
  func testAutoTagProposalRefusesChangedLineMovieOrVoiceMode() throws {
    var project = StudioProject(); var draft = VoiceDraft(); draft.engine = .fishS2Pro; draft.text = "Hello"
    draft.startDialogue(); project.voiceDraft = draft
    let session = UUID(), lineID = try XCTUnwrap(draft.dialogue?.turns.first?.id)
    let context = try VoiceTagContext(project: project, documentSessionID: session, lineID: lineID)
    XCTAssertNoThrow(try context.validate(project: project, documentSessionID: session))
    XCTAssertThrowsError(try context.validate(project: project, documentSessionID: UUID()))
    project.voiceDraft?.dialogue?.turns[0].text = "Edited"
    XCTAssertThrowsError(try context.validate(project: project, documentSessionID: session))
    project.voiceDraft = draft; project.voiceDraft?.usesDialogue = false
    XCTAssertThrowsError(try context.validate(project: project, documentSessionID: session))
    project.voiceDraft = draft; project.voiceDraft?.engine = .qwen3TTS
    XCTAssertThrowsError(try context.validate(project: project, documentSessionID: session))
  }

  func testAutoTagCannotSplitAnExistingWord() {
    XCTAssertThrowsError(try FishVoiceTags.applyingSuggestions("{\"tags\":[{\"before\":\"he\",\"occurrence\":1,\"tag\":\"excited\"}]}", to: "The hero arrives."))
  }
  func testChangingSpeakerClearsOldCharactersPerformanceOverride() {
    let first = VoiceSpeaker(name: "A"), second = VoiceSpeaker(name: "B")
    var turn = VoiceTurn(speakerID: first.id, text: "Hello")
    turn.reference = VoiceReference(path: "/a.wav")
    var dialogue = VoiceDialogue(speakers: [first, second], turns: [turn])
    dialogue.assignSpeaker(first.id, to: turn.id)
    XCTAssertNotNil(dialogue.turns[0].reference)
    dialogue.assignSpeaker(second.id, to: turn.id)
    XCTAssertEqual(dialogue.turns[0].speakerID, second.id)
    XCTAssertNil(dialogue.turns[0].reference)
  }

}
