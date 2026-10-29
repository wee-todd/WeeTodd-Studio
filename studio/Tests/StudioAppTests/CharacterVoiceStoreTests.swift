import AVFoundation
import StudioCore
import XCTest

@testable import WeeToddStudio

final class CharacterVoiceStoreTests: XCTestCase {
  @MainActor func testChoosingCharacterResolvesPinnedVoiceAndStableIdentity() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var character = PlanningSubject(name: "Alex", kind: .character)
    let voice = VoiceProfileSnapshot(
      name: "Alex's voice",
      samples: [.init(path: "/saved/sample.wav", duration: 2, transcript: "Hello")])
    character.voiceBinding = .init(profile: voice)
    var plan = ProjectPlanning()
    plan.subjects = [character]
    store.project.planning = plan
    store.openVoice()
    store.useCharacterVoice(character.id)
    XCTAssertEqual(store.project.voiceDraft?.voiceProfile, voice)
    XCTAssertEqual(store.project.voiceDraft?.characterID, character.id)
    XCTAssertEqual(store.project.voiceDraft?.engine, .fishDrama3)
    XCTAssertEqual(store.project.voiceDraft?.reference?.path, "/saved/sample.wav")
  }
  @MainActor func testSavingCloneAppliesItsRetainedProfile() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("sample.wav")
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4410)!
    buffer.frameLength = 4410
    buffer.floatChannelData![0].initialize(repeating: 0.1, count: 4410)
    do {
      let file = try AVAudioFile(forWriting: source, settings: format.settings)
      try file.write(from: buffer)
    }
    let store = StudioStore(
      dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    store.openVoice()
    store.change {
      $0.voiceDraft?.reference = .init(path: source.path, duration: 0.1, transcript: "Hi")
    }
    await store.saveClonedVoice(name: "Saved Alex", description: "Warm")
    XCTAssertNil(store.error)
    XCTAssertNotNil(store.project.voiceDraft?.voiceProfile)
    XCTAssertNotEqual(store.project.voiceDraft?.reference?.path, source.path)
    // An embedded profile from a portable character must become a usable,
    // independently retained voice in a clean receiving app.
    let receiving = StudioStore(
      dataDirectory: root.appendingPathComponent("receiving"), restoreSession: false)
    receiving.openVoice()
    receiving.change {
      $0.voiceDraft?.text = "Hello"
      $0.voiceDraft?.startDialogue()
      $0.voiceDraft?.dramaParentTakeID = UUID()
      $0.voiceDraft?.dramaSourceAudio = "/removed-parent.wav"
    }
    let embedded = try XCTUnwrap(store.project.voiceDraft?.voiceProfile)
    var characterDocument = CharacterSheetDocument(title: "Portable Alex")
    characterDocument.voiceBinding = .init(profile: embedded)
    let portable = root.appendingPathComponent("portable-character")
    let sourceDocuments = CharacterSheetDocumentStore(
      root: root.appendingPathComponent("source-documents"))
    try sourceDocuments.export(characterDocument, to: portable)
    try FileManager.default.removeItem(atPath: embedded.samples[0].path)
    let receivingDocuments = CharacterSheetDocumentStore(
      root: receiving.dataDirectory.appendingPathComponent("characters"))
    let imported = try receivingDocuments.importDocument(from: portable)
    let importedVoice = try XCTUnwrap(imported.voiceBinding?.profile)
    let opened = await receiving.retainAndUseCharacterVoice(importedVoice)
    XCTAssertTrue(opened)
    XCTAssertNil(receiving.error)
    XCTAssertEqual(receiving.project.voiceDraft?.voiceProfile?.reference, embedded.reference)
    XCTAssertTrue(receiving.showVoice)
    XCTAssertEqual(receiving.project.voiceDraft?.usesDialogue, false)
    XCTAssertNil(receiving.project.voiceDraft?.dramaParentTakeID)
    XCTAssertNil(receiving.project.voiceDraft?.dramaSourceAudio)
    let retained = try XCTUnwrap(receiving.project.voiceDraft?.reference)
    XCTAssertTrue(retained.path.hasPrefix(receiving.dataDirectory.path))
    XCTAssertEqual(
      try receiving.project.voiceDraft?.drama3Request().references.first?.audioPath, retained.path)
    try FileManager.default.removeItem(atPath: importedVoice.samples[0].path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: retained.path))
    XCTAssertEqual(receiving.voiceLibraryProfiles.count, 1)
  }

  @MainActor func testMissingCharacterSampleKeepsTheEditorClosed() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let missing = VoiceProfileSnapshot(
      name: "Unavailable",
      samples: [
        .init(
          path: root.appendingPathComponent("missing.wav").path,
          duration: 1, transcript: "Hello")
      ])
    let opened = await store.retainAndUseCharacterVoice(missing)
    XCTAssertFalse(opened)
    XCTAssertFalse(store.showVoice)
    XCTAssertNotNil(store.error)
    XCTAssertFalse(store.voiceLibraryBusy)
  }

}
extension CharacterVoiceStoreTests {
  @MainActor func testPlanningVoiceSurvivesDirectorVisualUndoAndReopening() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let character = PlanningSubject(name: "Alex", kind: .character)
    store.project.planning = ProjectPlanning()
    store.project.planning?.subjects = [character]
    var document = CharacterSheetDocument(title: "Alex")
    document.subjectKey = store.project.id.uuidString + ":" + character.id.uuidString
    let storage = CharacterSheetDocumentStore(root: root.appendingPathComponent("characters"))
    let controller = CharacterSheetSessionController(
      document: document, store: store, storage: storage)
    controller.edit { $0.title = "Visual edit" }
    let voice = VoiceProfileSnapshot(
      name: "New voice", samples: [.init(path: "/saved.wav", duration: 1, transcript: "Hello")])
    store.changePlanning { $0.subjects[0].voiceBinding = .init(profile: voice) }
    XCTAssertEqual(controller.document.voiceBinding?.profile, voice)
    controller.undo()
    XCTAssertEqual(controller.document.title, "Alex")
    XCTAssertEqual(controller.document.voiceBinding?.profile, voice)
    XCTAssertEqual(store.planning.subjects[0].voiceBinding?.profile, voice)
    controller.redo()
    XCTAssertEqual(store.planning.subjects[0].voiceBinding?.profile, voice)
    let reopened = CharacterSheetSessionController(
      document: document, store: store, storage: storage)
    XCTAssertEqual(reopened.document.voiceBinding?.profile, voice)
  }
  @MainActor func testCharacterDirectorVoiceAssignmentSynchronizesWithoutChangingVisualDefinition()
    throws
  {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let character = PlanningSubject(name: "Alex", kind: .character)
    var plan = ProjectPlanning()
    plan.subjects = [character]
    store.project.planning = plan
    var document = CharacterSheetDocument(title: "Alex")
    document.subjectKey = store.project.id.uuidString + ":" + character.id.uuidString
    let visual = document.definition
    let controller = CharacterSheetSessionController(
      document: document, store: store,
      storage: CharacterSheetDocumentStore(root: root.appendingPathComponent("characters")))
    let voice = VoiceProfileSnapshot(
      name: "Alex", samples: [.init(path: "/retained/sample.wav", duration: 1, transcript: "Hello")]
    )
    controller.assignVoice(.init(profile: voice))
    XCTAssertEqual(controller.document.definition, visual)
    XCTAssertEqual(store.planning.subjects[0].voiceBinding?.profile, voice)
    controller.undo()
    XCTAssertNil(store.planning.subjects[0].voiceBinding)
  }
}
