import AVFoundation
import Drama3Contracts
import StudioCore
import XCTest

@testable import WeeToddStudio

final class Drama3RoutingTests: XCTestCase {
  @MainActor func testInstalledNativeSpeechRetainsAndReusesCharacterWithoutPython() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let modelPath = env["DRAMA3_STUDIO_MODEL"], let worker = env["DRAMA3_STUDIO_WORKER"],
      let dataRoot = env["DRAMA3_STUDIO_DATA"]
    else {
      throw XCTSkip(
        "Opt-in real native Studio voice/character qualification requires DRAMA3_STUDIO_MODEL, DRAMA3_STUDIO_WORKER and DRAMA3_STUDIO_DATA."
      )
    }
    let root = URL(fileURLWithPath: dataRoot).appendingPathComponent(UUID().uuidString)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime.pythonPath = "/missing/python"
    store.runtime.root = "/missing/python-runtime"
    store.runtime.drama3WorkerPath = worker
    var settings = VoiceModelSettings()
    _ = settings.register(.init(engine: .fishDrama3, name: "Qualified Drama 3", path: modelPath))
    store.runtime.voiceModels = settings
    store.openVoice()
    let script = "A reusable character voice, made natively in WeeTodd Studio."
    store.change {
      $0.voiceDraft?.text = script
      $0.voiceDraft?.seed = 42
    }
    await store.generateVoice()
    XCTAssertNil(store.error)
    let first = try XCTUnwrap(store.voiceTakes.last)
    XCTAssertEqual(first.voiceGeneration?.truncated, false)
    store.change {
      $0.voiceDraft?.reference = .init(
        path: first.path, duration: first.duration, transcript: script)
    }
    await store.saveClonedVoice(name: "Retained Alex", description: "Warm and steady")
    XCTAssertNil(store.error)
    let profile = try XCTUnwrap(store.project.voiceDraft?.voiceProfile)
    let character = PlanningSubject(name: "Alex", kind: .character)
    store.changePlanning {
      var assigned = character
      assigned.voiceBinding = .init(profile: profile)
      $0.subjects.append(assigned)
    }
    try FileManager.default.removeItem(atPath: first.path)
    store.useCharacterVoice(character.id)
    XCTAssertEqual(store.project.voiceDraft?.characterID, character.id)
    store.change { $0.voiceDraft?.text = "These samples belong to the character now." }
    await store.generateVoice()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.voiceTakes.count, 2)
    let reused = try XCTUnwrap(store.voiceTakes.last)
    XCTAssertEqual(reused.voiceGeneration?.truncated, false)
    XCTAssertTrue(FileManager.default.fileExists(atPath: profile.samples[0].path))
    XCTAssertEqual(reused.voiceGeneration?.draft.characterID, character.id)
    print("Real Studio native voice/retention/character qualification: \(root.path)")
  }
  @MainActor func testDirectingConversationSendsVerifiedParentAudioHistory() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let take = root.appendingPathComponent("parent")
    try FileManager.default.createDirectory(at: take, withIntermediateDirectories: true)
    var captured: [String: Any]?
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false) {
      command, _, body, _ in
      XCTAssertEqual(command, "drama3-render")
      captured = body
      throw StudioError.invalid("Captured native directing request.")
    }
    var settings = VoiceModelSettings()
    let model = settings.register(.init(engine: .fishDrama3, name: "Native", path: "/native/model"))
    store.runtime.voiceModels = settings
    var draft = VoiceDraft()
    draft.modelID = model.id
    draft.modelPath = model.path
    draft.text = "Ready?"
    draft.startDialogue()
    let original = try draft.drama3Request()
    try JSONEncoder().encode(original).write(to: take.appendingPathComponent("request.json"))
    let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1))
    let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 2048))
    pcm.frameLength = 2048
    pcm.floatChannelData![0].initialize(repeating: 0.1, count: 2048)
    do { try AVAudioFile(forWriting: take.appendingPathComponent("speech.wav"), settings: format.settings).write(from: pcm) }
    var header = try JSONSerialization.data(withJSONObject: ["codes": ["dtype": "I32", "shape": [10, 1], "data_offsets": [0, 40]]])
    while header.count % 8 != 0 { header.append(32) }
    var length = UInt64(header.count).littleEndian
    var codes = withUnsafeBytes(of: &length) { Data($0) }
    codes.append(header); codes.append(Data(repeating: 0, count: 40))
    try codes.write(to: take.appendingPathComponent("codes.safetensors"))
    let codec = String(repeating: "c", count: 64)
    let receipt = Drama3Receipt(id: original.id, frames: 2048, codeFrames: 1, truncated: false,
      audioSHA256: try Drama3AudioPreparation.fileDigest(take.appendingPathComponent("speech.wav").path),
      codesSHA256: try Drama3AudioPreparation.fileDigest(take.appendingPathComponent("codes.safetensors").path),
      modelFingerprint: String(repeating: "a", count: 64), segmentFrames: [1], elapsedSeconds: 0,
      peakResidentBytes: 0, codecFingerprint: codec)
    try JSONEncoder().encode(receipt).write(to: take.appendingPathComponent("receipt.json"))
    var asset = MediaAsset(name: "Conversation", kind: .audio, path: take.appendingPathComponent("speech.wav").path)
    asset.duration = receipt.duration
    asset.voiceGeneration = VoiceGeneration(draft: draft, request: try draft.request(), artifacts: take.path,
      sampleRate: 44100, frames: 2048, truncated: false)
    store.project.assets = [asset]
    store.prepareDramaRevision(asset, infill: false)
    store.project.voiceDraft?.dramaDirection = "More quietly."
    await store.generateVoice()
    let body = try XCTUnwrap(captured)
    XCTAssertEqual(body["mode"] as? String, "direct")
    XCTAssertEqual(body["contextCodecFingerprint"] as? String, codec)
    let history = try XCTUnwrap(body["history"] as? [[String: Any]])
    XCTAssertEqual(history.count, 1)
    let turn = try XCTUnwrap(history.first)
    XCTAssertEqual(turn["text"] as? String, "<|speaker:0|> Ready?")
    let prior = try XCTUnwrap(turn["codes"] as? [String: Any])
    XCTAssertEqual(prior["path"] as? String, take.appendingPathComponent("codes.safetensors").path)
    XCTAssertEqual(prior["sha256"] as? String, receipt.codesSHA256)
  }
  @MainActor func testFreshVoiceUsesDrama3AndPreservesSavedEngineChoice() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.openVoice()
    XCTAssertEqual(store.project.voiceDraft?.engine, .fishDrama3)
    store.selectVoiceFamily(.fishS2Pro)
    store.project = StudioProject()
    store.openVoice()
    XCTAssertEqual(store.project.voiceDraft?.engine, .fishS2Pro)
  }
  @MainActor func testSceneControlCompletionUsesNativeWorkerWithoutPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = root.appendingPathComponent("worker")
    try Data("""
      #!/bin/sh
      [ "$1" = supplement-head ] || exit 23
      printf '%s\\n' '{"status":"ok","result":{"native":true}}'

      """.utf8).write(to: worker)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    var runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
    runtime.drama3WorkerPath = worker.path
    let result = try await Bridge().invoke("drama3-supplement-head", runtime: runtime,
      payload: ["modelPath": "/installed/model", "source": "/retained/original.safetensors"],
      output: root.appendingPathComponent("setup"))
    XCTAssertEqual(result["native"] as? Bool, true)
  }
  @MainActor func testNativeSpeechPreflightNeverRequiresPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = root.appendingPathComponent("worker")
    try Data("#!/bin/sh\nprintf '%s\\n' '{\"status\":\"ok\",\"result\":{\"native\":true}}'\n".utf8)
      .write(to: worker)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    var runtime = RuntimeSettings(
      root: "/missing", pythonPath: "/missing/python", profilesDirectory: root.path)
    runtime.drama3WorkerPath = worker.path
    let result = try await Bridge().invoke(
      "drama3-preflight", runtime: runtime, payload: ["modelPath": "/model"],
      output: root.appendingPathComponent("take"))
    XCTAssertEqual(result["native"] as? Bool, true)
  }
}
