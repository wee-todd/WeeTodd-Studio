import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class VoiceStoreTests: XCTestCase {
  func directory() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }; return folder
  }
  @MainActor func testGenerationRetainsTakeWithoutAutomaticPlacement() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { command, _, body, _ in
      XCTAssertEqual(command, "voice-generate")
      let request = try XCTUnwrap(body["voice"] as? [String: Any])
      XCTAssertEqual(request["model_path"] as? String, "/model")
      XCTAssertEqual(request["reference_mode"] as? String, "audioAndTranscript")
      XCTAssertEqual((request["reference"] as? [String: Any])?["transcript"] as? String, "Original words")
      return ["audio": "/take.wav", "duration": 3.0, "sample_rate": 24000, "frames": 72000,
              "channels": 1, "artifacts": "/take", "truncated": false]
    })
    store.openVoice()
    store.change { $0.voiceDraft?.modelPath = "/model"; $0.voiceDraft?.text = "New words"
      $0.voiceDraft?.reference = VoiceReference(path: "/reference.wav", transcript: "Original words") }
    await store.generateVoice()
    XCTAssertNil(store.error); XCTAssertEqual(store.voiceTakes.count, 1); XCTAssertTrue(store.project.audio.isEmpty)
    let clip = Clip(); store.project.clips = [clip]; store.selectedClipID = clip.id
    store.placeVoiceTake(store.voiceTakes[0])
    XCTAssertEqual(store.project.audio.first?.anchor?.clipID, clip.id)
    XCTAssertEqual(store.project.audioTracks.last?.role, .voice)
  }
  @MainActor func testLateVoiceDoesNotMutateNewDocument() async throws {
    let entered = expectation(description: "voice started")
    var continuation: CheckedContinuation<[String: Any], Error>?
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { _, _, _, _ in
      try await withCheckedThrowingContinuation { continuation = $0; entered.fulfill() }
    })
    store.openVoice(); store.change { $0.voiceDraft?.engine = .fishS2Pro; $0.voiceDraft?.referenceMode = .synthetic
      $0.voiceDraft?.modelPath = "/model"; $0.voiceDraft?.text = "Hello" }
    let task = Task { await store.generateVoice() }; await fulfillment(of: [entered], timeout: 2)
    store.project = StudioProject()
    continuation?.resume(returning: ["audio": "/take.wav", "duration": 3.0, "sample_rate": 44100,
      "frames": 132300, "channels": 1, "artifacts": "/take", "truncated": false])
    await task.value
    XCTAssertTrue(store.project.assets.isEmpty); XCTAssertTrue(store.notice.contains("/take.wav"))
  }
  @MainActor func testAudioEditInvalidatesDriverButFutureUnrelatedTrackDoesNot() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var clip = Clip(); clip.audioDriverSelection = AudioDriverSelection(mode: .voice); clip.audioDriverMixKey = "prepared"
    let voice = AudioTrack(name: "Voice", role: .voice)
    var region = AudioRegion(assetID: UUID(), path: "/sample.wav"); region.trackID = voice.id
    store.project.clips = [clip]; store.project.audioTracks.append(voice); store.project.audio = [region]
    store.change { $0.audioTracks[0].gainDb = -9 }
    XCTAssertEqual(store.project.clips[0].audioDriverMixKey, "prepared")
    store.change { $0.audio[0].volume = 0.25 }
    XCTAssertNil(store.project.clips[0].audioDriverMixKey)
  }
}

extension VoiceStoreTests {
  @MainActor func testVoiceModelPersistsAcrossMoviesAndAppSessions() throws {
    let folder = try directory()
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    var draft = VoiceDraft(); draft.engine = .fishS2Pro; draft.modelPath = "/fish-model"
    store.project.voiceDraft = draft; store.openVoice()
    let model = try XCTUnwrap(store.selectedVoiceModel)
    XCTAssertEqual(model.path, "/fish-model")
    XCTAssertEqual(store.project.voiceDraft?.modelPath, "")
    store.project = StudioProject(); store.openVoice()
    XCTAssertEqual(store.project.voiceDraft?.modelID, model.id)
    XCTAssertEqual(store.project.voiceDraft?.engine, .fishS2Pro)
    let restored = StudioStore(dataDirectory: folder, restoreSession: false)
    restored.openVoice()
    XCTAssertEqual(restored.selectedVoiceModel?.path, "/fish-model")
    XCTAssertEqual(restored.project.voiceDraft?.modelPath, "")
    XCTAssertEqual(restored.project.voiceDraft?.engine, .fishS2Pro)
  }

  @MainActor func testOpeningSavedVoiceDraftPreservesItsModelAndReference() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var first = VoiceDraft(); first.modelPath = "/qwen-model"
    store.project.voiceDraft = first; store.openVoice()
    var saved = VoiceDraft(); saved.engine = .fishS2Pro; saved.modelPath = "/fish-model"
    saved.text = "Keep my script"; saved.reference = VoiceReference(path: "/sample.wav", transcript: "Keep my reference")
    store.project = StudioProject(); store.project.voiceDraft = saved; store.openVoice()
    XCTAssertEqual(store.selectedVoiceModel?.path, saved.modelPath)
    saved.modelPath = ""; saved.modelID = store.selectedVoiceModel?.id
    XCTAssertEqual(store.project.voiceDraft, saved)
  }

  @MainActor func testFailedRuntimeSavePreservesLegacyDraftLocation() throws {
    let file = try directory().appendingPathComponent("not-a-directory")
    try Data("file".utf8).write(to: file)
    let store = StudioStore(dataDirectory: file, restoreSession: false)
    var draft = VoiceDraft(); draft.modelPath = "/preserve-me"
    store.project.voiceDraft = draft; store.openVoice()
    XCTAssertEqual(store.project.voiceDraft, draft)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testFamilySelectionRestoresEachVariantAndPreservesScriptAndSample() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let small = InstalledVoiceModel(engine: .qwen3TTS, name: "Qwen3-TTS Base · 0.6B · 8-bit", path: "/qwen-small")
    let large = InstalledVoiceModel(engine: .qwen3TTS, name: "Qwen3-TTS Base · 1.7B · 8-bit", path: "/qwen-large")
    let fish = InstalledVoiceModel(engine: .fishS2Pro, name: "Fish S2 Pro · BF16", path: "/fish")
    var models = VoiceModelSettings(); models.register(small); models.register(large); models.register(fish)
    store.runtime.voiceModels = models; store.openVoice()
    store.change { $0.voiceDraft?.text = "Keep these words"; $0.voiceDraft?.reference = VoiceReference(path: "/sample.wav", transcript: "Sample words") }
    store.selectVoiceModel(large.id)
    store.selectVoiceFamily(.fishS2Pro)
    XCTAssertEqual(store.familyVoiceModels.map(\.id), [fish.id])
    XCTAssertEqual(store.selectedVoiceModel?.id, fish.id)
    store.selectVoiceFamily(.qwen3TTS)
    XCTAssertEqual(store.selectedVoiceModel?.id, large.id)
    XCTAssertEqual(store.project.voiceDraft?.text, "Keep these words")
    XCTAssertEqual(store.project.voiceDraft?.reference?.transcript, "Sample words")
    XCTAssertEqual(store.project.voiceDraft?.modelPath, "")
    let restored = RuntimeSettings.restoring(try Data(contentsOf: store.dataDirectory.appendingPathComponent("runtime.json")), defaults: store.runtime)
    XCTAssertEqual(restored.voiceModels?.preferred(for: .qwen3TTS)?.id, large.id)
  }

  @MainActor func testRemovedModelDoesNotSilentlySubstituteAnotherVariant() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { _, _, _, _ in
      XCTFail("A missing selected model must not launch inference"); return [:]
    })
    var settings = VoiceModelSettings()
    let first = settings.register(InstalledVoiceModel(engine: .qwen3TTS, name: "Small", path: "/small"))
    settings.register(InstalledVoiceModel(engine: .qwen3TTS, name: "Large", path: "/large"))
    store.runtime.voiceModels = settings; store.openVoice(); store.removeVoiceModel(first)
    XCTAssertNil(store.selectedVoiceModel)
    await store.generateVoice()
    XCTAssertTrue(store.error?.contains("installed speech model") == true)
  }

  @MainActor func testLateModelSetupUpdatesRuntimeWithoutChangingNewMovie() async throws {
    let entered = expectation(description: "inspection started")
    var continuation: CheckedContinuation<[String: Any], Error>?
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { command, _, body, _ in
      XCTAssertEqual(command, "voice-inspect")
      XCTAssertNil((body["voice"] as? [String: Any])?["engine"])
      return try await withCheckedThrowingContinuation { continuation = $0; entered.fulfill() }
    })
    store.openVoice()
    let task = Task { await store.addVoiceModel(at: URL(fileURLWithPath: "/qwen")) }
    await fulfillment(of: [entered], timeout: 2)
    store.project = StudioProject(); let newProject = store.project
    continuation?.resume(returning: ["model": ["path":"/qwen", "engine":"qwen3TTS", "name":"Qwen3-TTS Base · 1.7B · 8-bit"]])
    await task.value
    XCTAssertEqual(store.project, newProject)
    XCTAssertEqual(store.runtime.voiceModels?.models.first?.path, "/qwen")
  }

  @MainActor func testSelectingAnotherClipDuringDriverDoesNotPrepareOrRenderIt() async throws {
    let entered = expectation(description: "driver pending")
    var continuation: CheckedContinuation<[String: Any], Error>?
    var commands: [String] = []
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { command, _, _, _ in
      commands.append(command)
      if command == "audio-driver" {
        return try await withCheckedThrowingContinuation { continuation = $0; entered.fulfill() }
      }
      return [:]
    })
    var a = Clip(); a.audioDriverSelection = AudioDriverSelection(mode: .voice)
    let b = Clip(); store.project.clips = [a,b]; store.selectedClipID = a.id
    let task = Task { await store.generateSelected() }
    await fulfillment(of: [entered], timeout: 2)
    store.selectedClipID = b.id
    continuation?.resume(returning: ["path":"/driver.wav", "mix_key":"key", "duration":5.0])
    await task.value
    XCTAssertFalse(commands.contains("prepare")); XCTAssertFalse(commands.contains("describe-generation"))
    XCTAssertFalse(commands.contains("render"))
  }
}


extension VoiceStoreTests {
  @MainActor func testDialogueGenerationUsesRuntimeModelAndKeepsResolvedReferences() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { command, _, body, _ in
      XCTAssertEqual(command, "voice-dialogue")
      let request = try XCTUnwrap(body["voice"] as? [String: Any])
      XCTAssertEqual(request["model_path"] as? String, "/model")
      let turns = try XCTUnwrap(request["turns"] as? [[String: Any]])
      XCTAssertEqual(turns.count, 2)
      XCTAssertEqual(turns[1]["speaker_name"] as? String, "Alex")
      XCTAssertEqual((turns[1]["reference"] as? [String: Any])?["path"] as? String, "/alex.wav")
      return ["audio": "/dialogue.wav", "duration": 5.0, "sample_rate": 24000, "frames": 120000,
              "channels": 1, "artifacts": "/dialogue", "truncated": false]
    })
    store.openVoice()
    store.change {
      $0.voiceDraft?.modelPath = "/model"; $0.voiceDraft?.text = "Hello"
      $0.voiceDraft?.reference = VoiceReference(path: "/first.wav", transcript: "Hello")
      $0.voiceDraft?.startDialogue()
      let alex = VoiceSpeaker(name: "Alex", reference: VoiceReference(path: "/alex.wav", transcript: "Hi"))
      $0.voiceDraft?.dialogue?.speakers.append(alex)
      $0.voiceDraft?.dialogue?.turns.append(VoiceTurn(speakerID: alex.id, text: "Hi there"))
    }
    await store.generateVoice()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.voiceTakes.first?.voiceGeneration?.request.turns?.count, 2)
    XCTAssertEqual(store.voiceTakes.first?.voiceGeneration?.draft.modelPath, "")
  }
}


extension VoiceStoreTests {
  @MainActor func testCustomVoiceVariantKeepsSampledReferencesAndResolvesPerLineInstructions() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var settings = VoiceModelSettings()
    let base = settings.register(InstalledVoiceModel(engine: .qwen3TTS, name: "Base", path: "/base"))
    let custom = settings.register(InstalledVoiceModel(engine: .qwen3TTS, name: "CustomVoice", path: "/custom", kind: "custom_voice"))
    store.runtime.voiceModels = settings; store.openVoice()
    store.change {
      $0.voiceDraft?.text = "Hello"; $0.voiceDraft?.reference = VoiceReference(path: "/sample.wav", transcript: "Hi")
      $0.voiceDraft?.startDialogue()
      $0.voiceDraft?.dialogue?.speakers[0].presetVoice = "aiden"
      $0.voiceDraft?.dialogue?.speakers[0].instructions = "Happy"
      $0.voiceDraft?.dialogue?.turns[0].instructions = "Sad"
    }
    store.selectVoiceModel(custom.id)
    var draft = try XCTUnwrap(store.project.voiceDraft); draft.modelPath = custom.path
    let request = try draft.request()
    XCTAssertEqual(request.turns?.first?.speaker, "aiden")
    XCTAssertEqual(request.turns?.first?.instruct, "Sad")
    XCTAssertNil(request.turns?.first?.reference)
    XCTAssertEqual(draft.dialogue?.speakers.first?.reference?.path, "/sample.wav")
    store.selectVoiceModel(base.id)
    draft = try XCTUnwrap(store.project.voiceDraft); draft.modelPath = base.path
    let clone = try draft.request()
    XCTAssertNil(clone.turns?.first?.instruct)
    XCTAssertEqual(clone.turns?.first?.reference?.path, "/sample.wav")
  }
}


extension VoiceStoreTests {
  @MainActor func testFirstCustomVoiceRegistrationGeneratesWithoutReopeningDraft() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false, invocation: { command, _, body, _ in
      XCTAssertEqual(command, "voice-generate")
      let voice = try XCTUnwrap(body["voice"] as? [String: Any])
      XCTAssertEqual(voice["reference_mode"] as? String, "customVoice")
      XCTAssertEqual(voice["speaker"] as? String, "ryan")
      XCTAssertNil(voice["reference"])
      return ["audio":"/custom.wav", "duration":1.0, "frames":24000, "sample_rate":24000, "channels":1, "artifacts":"/custom"]
    })
    store.openVoice(); store.change { $0.voiceDraft?.text = "Hello" }
    try store.registerVoiceModel(["model":["engine":"qwen3TTS", "path":"/custom", "name":"CustomVoice", "kind":"custom_voice"]])
    XCTAssertNil(store.project.voiceDraft?.modelID)
    await store.generateVoice()
    XCTAssertNil(store.error); XCTAssertEqual(store.voiceTakes.count, 1)
  }
}
