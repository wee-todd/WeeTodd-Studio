import AVFoundation
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeHeadlessExportTests: XCTestCase {
  private func directory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NativeExport-\(UUID())")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    return directory
  }
  @MainActor func testStudioNativeExporterPreparesThroughSharedServicesWithoutPython() async throws {
    for engine in [Engine.h3, .ltx25] {
      let root = try directory()
      var commands: [String] = [], preparedBytes = Data()
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, runtime, payload, output in
        commands.append(command)
        XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
        if command.hasSuffix("-prepare") {
          let target = try XCTUnwrap(output)
          try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
          let recipe = target.appendingPathComponent("recipe.json")
          preparedBytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2",
            "engine": engine.rawValue, "prompt": "Exact composed AV prompt", "components": [:],
            "config": ["seed": 7744, "width": 64, "height": 64]])
          try preparedBytes.write(to: recipe)
          return ["recipePath": recipe.path, "report": ["nativeRuntime": "swift-mlx"]]
        }
        XCTAssertTrue(command.hasSuffix("-preflight")); return ["nativeRuntime": "swift-mlx"]
      })
      store.runtime.root = "/unavailable"; store.runtime.pythonPath = "/unavailable/python"
      store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
      store.runtime.h3WorkerPath = "/usr/bin/true"; store.runtime.ltx25WorkerPath = "/usr/bin/true"
      store.runtime.ffmpegPath = "/usr/bin/true"
      var clip = Clip(engine: engine); clip.duration = 1; clip.seed = 7744; clip.prompt = "Original editor prompt"
      clip.generationWidth = 64; clip.generationHeight = 64
      store.project.clips = [clip]; store.selectedClipID = clip.id
      XCTAssertTrue(store.nativeHeadlessEligible)
      var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
      let url = root.appendingPathComponent("clip.weetodd-job.json")
      try await store.exportNativeHeadlessJob(body: body, to: url, clipOnly: true)
      let job = try NativeHeadlessJob.read(from: url)
      XCTAssertEqual(commands, [engine == .h3 ? "h3-native-prepare" : "ltx-native-prepare",
        engine == .h3 ? "h3-native-preflight" : "ltx-native-preflight"])
      XCTAssertEqual(job.recipes[clip.id.uuidString]?.bytes, preparedBytes)
      XCTAssertEqual(job.project.clips[0].prompt, clip.prompt); XCTAssertEqual(job.project.clips[0].seed, 7744)
      XCTAssertEqual(store.project.clips, [clip]); XCTAssertNil(store.error)
    }
  }
  @MainActor func testNativeExporterRejectsUnsupportedFinishingBeforePreparation() async throws {
    let root = try directory()
    var calls = 0
    let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { _, _, _, _ in calls += 1; return [:] })
    store.runtime.nativeLTX25Enabled = true; store.runtime.ltx25WorkerPath = "/usr/bin/true"
    store.runtime.ffmpegPath = "/usr/bin/true"
    var clip = Clip(); clip.sourcePan = 0.5; store.project.clips = [clip]; store.selectedClipID = clip.id
    var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
    do { try await store.exportNativeHeadlessJob(body: body, to: root.appendingPathComponent("job.json"), clipOnly: false); XCTFail("Pan cannot disappear") }
    catch { XCTAssertTrue(error.localizedDescription.contains("pan")) }
    XCTAssertEqual(calls, 0)
  }
  @MainActor func testNativeExporterPreservesMissingLastFrameAdmissionDuringCleanup() async throws {
    for (preparationCreatedInputs, scene) in [(false, false), (false, true), (true, false), (true, true)] {
      let root = try directory(), profile = root.appendingPathComponent("model.json")
      let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
        "prompt": "Original prompt", "components": ["transformer_path": "/models/transformer", "loras": []],
        "config": ["pipeline_mode": "distilled", "stage1_steps": 8, "stage2_steps": 3,
          "frame_rate": 24, "width": 64, "height": 64, "seed": 43, "duration_seconds": 1],
        "conditioning": ["version": 1, "task": "fflf", "inputs": []]]
      try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
      let image = root.appendingPathComponent("first.png")
      try Data([1]).write(to: image) // Admission rejects the missing last frame before image decoding.
      let asset = MediaAsset(name: "First only", kind: .image, path: image.path)
      var clip = Clip(engine: .ltx25); clip.prompt = "Preserve the original action"
      clip.duration = 1; clip.generationWidth = 64; clip.generationHeight = 64
      clip.profileID = profile.path; clip.generationSelection = GenerationSelection(task: "fflf")
      clip.attachments = [Attachment(assetID: asset.id, role: .first)]
      var commands: [String] = []
      let invocation: Bridge.Invocation? = preparationCreatedInputs ? { command, runtime, payload, output in
        commands.append(command); XCTAssertEqual(command, "ltx-native-prepare")
        XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
        if preparationCreatedInputs {
          try FileManager.default.createDirectory(at: try XCTUnwrap(output), withIntermediateDirectories: true)
        }
        var request = payload; request["runtime"] = try runtime.object()
        return try NativeLTXPreparation.compose(request: request)
      } : nil
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: invocation)
      store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: root.path)
      store.runtime.nativeLTX25Enabled = true; store.runtime.ltx25WorkerPath = "/usr/bin/true"
      store.runtime.ffmpegPath = "/usr/bin/true"
      store.project.clips = [clip]; store.project.assets = [asset]; store.selectedClipID = clip.id
      if scene {
        var next = Clip(engine: .ltx25); next.prompt = "Continue the same action"
        next.duration = 1; next.generationWidth = 64; next.generationHeight = 64; next.profileID = profile.path
        next.generationSelection = GenerationSelection(task: "t2v")
        next.continuity = ClipContinuity(mode: "scene", sourceClipID: clip.id)
        store.project.clips.append(next)
      }
      let original = store.project, destination = root.appendingPathComponent("job.json")
      var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
      do {
        try await store.exportNativeHeadlessJob(body: body, to: destination, clipOnly: !scene)
        XCTFail("Missing last frame must fail before worker preflight")
      } catch {
        XCTAssertEqual(error.localizedDescription, "First and last frames requires First frame and Last frame images.")
      }
      XCTAssertEqual(commands, preparationCreatedInputs ? ["ltx-native-prepare"] : [])
      XCTAssertEqual(store.project, original); XCTAssertFalse(store.bridge.busy)
      XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
      XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix("job.json.inputs-") })
    }
  }
  @MainActor func testNativeExporterPreservesH3ProfileAdmissionBeforeInputsExist() async throws {
    for cleanupMode in ["absent", "partial", "denied"] {
      let root = try directory(), profile = root.appendingPathComponent("model.json")
      defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path) }
      let image = root.appendingPathComponent("first.png"); try Data([1]).write(to: image)
      let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "Reference action",
        "components": ["task": "ref2va", "transformer": "/models/transformer", "text_encoder": "/models/text",
          "tokenizer": "/models/tokenizer.json", "video_vae": "/models/video", "audio_vae": "/models/audio"],
        "config": ["width": 64, "height": 64, "duration_seconds": 2.5, "seed": 42, "steps": 5],
        // A render recipe with live inputs is not a parameterized Studio model profile.
        "conditioning": ["version": 1, "task": "ref2va", "audio_policy": "generated",
          "inputs": [["kind": "image", "role": "reference", "path": image.path]]]]
      try JSONSerialization.data(withJSONObject: recipe).write(to: profile)
      let invocation: Bridge.Invocation? = cleanupMode == "absent" ? nil : { _, runtime, payload, output in
        let target = try XCTUnwrap(output)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data([1]).write(to: target.appendingPathComponent("partial-input"))
        if cleanupMode == "denied" {
          // A readable parent without write permission makes removal of its child fail.
          try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: root.path)
        }
        var request = payload; request["runtime"] = try runtime.object()
        return try NativeH3Preparation.compose(request: request)
      }
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: invocation)
      store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: root.path)
      store.runtime.nativeH3Enabled = true; store.runtime.h3WorkerPath = "/usr/bin/false"
      store.runtime.ffmpegPath = "/usr/bin/true"
      let asset = MediaAsset(name: "Reference", kind: .image, path: image.path)
      var clip = Clip(engine: .h3); clip.prompt = "Preserve the reference action"; clip.duration = 2.5
      clip.generationWidth = 64; clip.generationHeight = 64; clip.profileID = profile.path
      clip.generationSelection = GenerationSelection(task: "ref2va")
      clip.attachments = [Attachment(assetID: asset.id, role: .reference)]
      store.project.clips = [clip]; store.project.assets = [asset]; store.selectedClipID = clip.id
      let original = store.project, destination = root.appendingPathComponent("job.json")
      var body = try store.payload(); body["generateIDs"] = [clip.id.uuidString]
      do {
        try await store.exportNativeHeadlessJob(body: body, to: destination, clipOnly: true)
        XCTFail("An inadmissible profile must fail before worker execution")
      } catch {
        XCTAssertEqual(error.localizedDescription, "Swift H3 generation is experimental: no compatible ref2va profile is installed.")
      }
      XCTAssertEqual(store.project, original); XCTAssertFalse(store.bridge.busy)
      XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
      let retainedInputs = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasPrefix("job.json.inputs-") }
      XCTAssertEqual(retainedInputs.count, cleanupMode == "denied" ? 1 : 0)
    }
  }
  @MainActor func testNativeExporterPreparesPendingRippleMovieAndLTXWithoutOrdinaryGeneration() async throws {
    for engine in [Engine.movie, .ltx25] {
      let root = try directory(), source = root.appendingPathComponent("source.mov"), edited = root.appendingPathComponent("edited.png")
      try Data("source identity".utf8).write(to: source); try Data("edited image".utf8).write(to: edited)
      var commands: [String] = [], bytes = Data(), frozenTake: URL?
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, runtime, payload, output in
        commands.append(command); XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
        if command == "ripple-native-prepare" {
          let target = try XCTUnwrap(output), take = try XCTUnwrap(payload["takeOutput"] as? String)
          frozenTake = URL(fileURLWithPath: take)
          let draft = try JSONDecoder().decode(RippleDraft.self, from: JSONSerialization.data(withJSONObject: payload["draft"]!))
          XCTAssertEqual(draft.sourceIn, 0.5); XCTAssertEqual(draft.seed, 47); XCTAssertEqual(draft.audioPolicy, .silent)
          try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
          let guide = target.appendingPathComponent("guide.rgb"), reference = target.appendingPathComponent("edited.png")
          try Data(repeating: 0, count: 25 * 64 * 64 * 3).write(to: guide); try FileManager.default.copyItem(at: edited, to: reference)
          let raw: [String: Any] = ["version": 1, "engine": "ltx25", "task": "ripple",
            "gemma_root": "/models/text", "transformer_root": "/models/transformer", "connector_checkpoint": "/models/fixed",
            "video_checkpoint": "/models/video", "audio_checkpoint": "/models/audio", "adapter_path": "/models/ripple",
            "adapter_strength": draft.loraStrength, "guide_path": guide.path, "first_reference_path": reference.path,
            "source_path": draft.sourcePath, "source_sha256": try NativeHeadlessJob.fileHash(source), "source_start": draft.sourceIn,
            "duration": draft.duration, "editorial_frames": draft.frameCount, "width": draft.width, "height": draft.height,
            "frames": 25, "fps": draft.frameRate, "seed": draft.seed, "prompt": draft.prompt,
            "reference_strength": draft.references[0].strength, "anchors": [], "audio_policy": draft.audioPolicy.rawValue,
            "ffmpeg_path": runtime.ffmpegPath, "output_directory": take]
          bytes = try JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys])
          let recipe = target.appendingPathComponent("ripple-request.json"); try bytes.write(to: recipe)
          return ["recipePath": recipe.path, "modelFrames": 25, "editorialFrames": draft.frameCount]
        }
        XCTAssertEqual(command, "ltx-native-preflight"); XCTAssertEqual(output, frozenTake)
        return ["nativeRuntime": "swift-mlx"]
      })
      store.runtime.nativeLTX25Enabled = true; store.runtime.nativeRippleEnabled = true
      store.runtime.ltx25WorkerPath = "/usr/bin/true"; store.runtime.ffmpegPath = "/usr/bin/true"
      store.runtime.pythonPath = "/unavailable/python"
      var clip = Clip(engine: engine); clip.sourcePath = source.path; clip.sourceIn = 0.5; clip.duration = 1
      var draft = RippleDraft(clip: clip, frameRate: 24); draft.width = 64; draft.height = 64
      draft.seed = 47; draft.audioPolicy = .silent; draft.references[0].path = edited.path; draft.references[0].strength = 0.7
      clip.rippleDraft = draft; store.project.clips = [clip]; store.selectedClipID = clip.id
      var body = try store.payload(); body["generateIDs"] = [] // Imported movie clips still have explicit pending Ripple work.
      let destination = root.appendingPathComponent("ripple.weetodd-job.json")
      try await store.exportNativeHeadlessJob(body: body, to: destination, clipOnly: true)
      let job = try NativeHeadlessJob.read(from: destination)
      XCTAssertEqual(commands, ["ripple-native-prepare", "ltx-native-preflight"])
      XCTAssertEqual(job.recipes[clip.id.uuidString]?.bytes, bytes)
      XCTAssertEqual(job.recipes[clip.id.uuidString]?.engine, "ltx25")
      XCTAssertEqual(job.project.clips, [clip]); XCTAssertEqual(store.project.clips, [clip])
      XCTAssertFalse(FileManager.default.fileExists(atPath: frozenTake!.path))
    }
  }
  @MainActor func testInstalledStudioNativeExportWithoutPython() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_HEADLESS_EXPORT"] else {
      throw XCTSkip("Opt-in native export/preflight using existing installed Studio inputs; no inference")
    }
    let options = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let root = URL(fileURLWithPath: try XCTUnwrap(options["output"]))
    if let reuseProject = options["reuseProject"] {
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      let url = URL(fileURLWithPath: reuseProject), expected = try ProjectStorage.read(url)
      var calls = 0
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { _, _, _, _ in
        calls += 1; throw StudioError.invalid("Read-only reopen qualification cannot invoke a worker.")
      })
      store.runtime.root = "/unavailable"; store.runtime.pythonPath = "/unavailable/python"
      store.load(url)
      XCTAssertNil(store.error); XCTAssertEqual(store.projectURL, url)
      XCTAssertEqual(store.project, expected); XCTAssertEqual(store.selectedClipID, expected.clips.first?.id)
      var observations: [[String: Any]] = []
      for clip in store.project.clips {
        XCTAssertTrue(FileManager.default.isReadableFile(atPath: clip.sourcePath))
        XCTAssertTrue(clip.versions.contains(where: { $0.path == clip.sourcePath }), "Accepted source must retain its take version")
        let asset = AVURLAsset(url: URL(fileURLWithPath: clip.sourcePath))
        let tracks = try await asset.loadTracks(withMediaType: .video)
        XCTAssertFalse(tracks.isEmpty)
        let duration = try await asset.load(.duration).seconds
        XCTAssertGreaterThanOrEqual(duration + 0.05, clip.sourceIn + clip.duration)
        var observation: [String: Any] = ["clipID": clip.id.uuidString, "source": clip.sourcePath,
          "sourceSHA256": try NativeHeadlessJob.fileHash(URL(fileURLWithPath: clip.sourcePath)), "sourceIn": clip.sourceIn,
          "duration": clip.duration, "versions": clip.versions.count]
        if let take = clip.rippleTakes?.first(where: { $0.path == clip.sourcePath }) {
          XCTAssertNotNil(take.submittedDraftFingerprint)
          XCTAssertTrue(FileManager.default.isReadableFile(atPath: take.draft.sourcePath))
          XCTAssertEqual(take.draft.sourceSHA256, try NativeHeadlessJob.fileHash(URL(fileURLWithPath: take.draft.sourcePath)))
          XCTAssertTrue(FileManager.default.isReadableFile(atPath: take.receiptPath))
          XCTAssertTrue(take.draft.references.allSatisfy { FileManager.default.isReadableFile(atPath: $0.path) })
          try await NativeRippleMedia.verifyPublishedTake(take.path, draft: take.draft, hasAudio: take.hasAudio)
          observation["rippleTakeID"] = take.id.uuidString; observation["rippleReceipt"] = take.receiptPath
          observation["replayReferences"] = take.draft.references.map(\.path)
          observation["originalSourceIn"] = take.draft.sourceIn
        }
        observations.append(observation)
      }
      XCTAssertEqual(calls, 0)
      try JSONSerialization.data(withJSONObject: ["project": url.path, "studioStoreReopened": true,
        "pythonAvailable": false, "inferenceExecuted": false, "workerInvocations": calls, "clips": observations],
        options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("reopen-qualification.json"))
      return
    }
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    let profile = URL(fileURLWithPath: try XCTUnwrap(options["recipe"]))
    let copied = profiles.appendingPathComponent("matched.json")
    if !FileManager.default.fileExists(atPath: copied.path) { try FileManager.default.copyItem(at: profile, to: copied) }
    let request = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(options["editorRequest"])))) as! [String: Any]
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project = try JSONDecoder().decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: request["project"]!))
    store.globalAssets = try JSONDecoder().decode([MediaAsset].self, from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    store.selectedClipID = UUID(uuidString: request["clipID"] as! String)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
    store.runtime.h3WorkerPath = options["h3Worker"]; store.runtime.ltx25WorkerPath = options["ltx25Worker"]
    store.runtime.ffmpegPath = options["ffmpeg"] ?? "/opt/homebrew/bin/ffmpeg"
    store.runtime.nativeRippleEnabled = options["nativeRippleEnabled"] == "true" || options["rippleAdapterPath"] != nil
    store.runtime.rippleProfileID = options["rippleProfileID"] ?? (options["rippleAdapterPath"] == nil ? nil : copied.path)
    store.runtime.rippleAdapterPath = options["rippleAdapterPath"]
    let selected = try XCTUnwrap(store.selectedClip)
    if let index = store.project.clips.firstIndex(where: { $0.id == selected.id }) { store.project.clips[index].profileID = copied.path }
    var body = try store.payload(); body["generateIDs"] = [selected.id.uuidString]
    let job = root.appendingPathComponent("exported.weetodd-job.json")
    if let expected = options["expectedAdmissionError"] {
      do {
        try await store.exportNativeHeadlessJob(body: body, to: job, clipOnly: options["clipOnly"] != "false")
        XCTFail("The retained inadmissible fixture must reject before worker execution")
      } catch {
        XCTAssertTrue(error is StudioError, "Admission must retain its actionable Studio error type: \(error)")
        XCTAssertEqual(error.localizedDescription, expected)
        try JSONSerialization.data(withJSONObject: ["error": error.localizedDescription,
          "errorType": String(reflecting: type(of: error)), "pythonAvailable": false,
          "inferenceExecuted": false], options: [.prettyPrinted, .sortedKeys])
          .write(to: root.appendingPathComponent("admission-error-qualification.json"))
      }
      XCTAssertFalse(FileManager.default.fileExists(atPath: job.path))
      return
    }
    do {
      try await store.exportNativeHeadlessJob(body: body, to: job, clipOnly: options["clipOnly"] != "false")
      let frozen = try NativeHeadlessJob.read(from: job)
      XCTAssertEqual(frozen.project.clips.first(where: { $0.id == selected.id })?.prompt, selected.prompt)
      let preflight = try await NativeHeadlessExecutor.run(job: frozen,
        output: root.appendingPathComponent("CLI-preflight"), preflightOnly: true)
      XCTAssertEqual(preflight["python_inference"] as? Bool, false)
      try JSONSerialization.data(withJSONObject: ["exportedJob": job.path, "pythonAvailable": false,
        "inferenceExecuted": false, "clipID": selected.id.uuidString], options: [.prettyPrinted, .sortedKeys])
        .write(to: root.appendingPathComponent("export-qualification.json"))
    } catch {
      // XCTest's uncaught async error report can name a later suppressed cleanup
      // NSError. Report the error received by the actual exporter caller instead.
      XCTFail("Native export/preflight failed (\(String(reflecting: type(of: error)))): \(error.localizedDescription)")
    }
  }
}
