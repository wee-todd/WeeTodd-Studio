import AVFoundation
import AppKit
import Combine
import CryptoKit
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class RippleStoreTests: XCTestCase {
  @MainActor func testRippleFrameImageUsesDrawThingsDimensionsAndKeepsEditAfterTrim() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.duration = 5.18
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    store.updateRipple {
      $0.width = 1376; $0.height = 768
      $0.references[0].originalPath = "/old-frame.png"
    }
    let original = try XCTUnwrap(store.rippleClip?.rippleDraft)
    let originalContext = RippleImageContext(clipID: clip.id, draft: original, reference: original.references[0])
    var image = store.makeRippleImageDraft(originalContext, width: original.width, height: original.height,
                                           previousDraft: nil)
    XCTAssertEqual(image.width, 1344)
    XCTAssertEqual(image.height, 768)
    image.prompt = "A red winter coat"
    store.imageWorkspaceLibrary.record(image, preview: "/old-result.png")

    store.rippleClipID = nil
    store.project.clips[0].duration = 5.16
    store.openRipple()
    let refreshed = try XCTUnwrap(store.rippleClip?.rippleDraft)
    XCTAssertEqual(refreshed.references[0].id, original.references[0].id)
    store.updateRipple { $0.references[0].originalPath = "/new-frame.png" }
    let current = try XCTUnwrap(store.rippleClip?.rippleDraft)
    let context = RippleImageContext(clipID: clip.id, draft: current, reference: current.references[0])
    let restored = store.makeRippleImageDraft(context, width: current.width, height: current.height,
                                               previousDraft: nil)
    XCTAssertEqual(restored.prompt, "A red winter coat")
    XCTAssertEqual(restored.canvas?.path, "/new-frame.png")
    XCTAssertEqual(restored.width, 1344)
    XCTAssertEqual(restored.height, 768)
    XCTAssertEqual(restored.rippleReference, context)
    XCTAssertNotEqual(restored, image, "The old generated preview must not be reused for a changed trim")
  }

  @MainActor func testReopeningUneditedRippleDraftUsesChangedTimelineTrim() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.duration = 5.18
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    store.updateRipple {
      $0.prompt = "Keep the coat blue"
      $0.references[0].originalPath = "/old-frame.png"
    }
    store.rippleClipID = nil
    store.project.clips[0].duration = 5.16
    store.openRipple()
    let draft = try XCTUnwrap(store.rippleClip?.rippleDraft)
    XCTAssertEqual(draft.duration, 5.16)
    XCTAssertEqual(draft.frameCount, 124)
    XCTAssertEqual(draft.prompt, "Keep the coat blue")
    XCTAssertEqual(draft.references[0].originalPath, "")
  }

  private func nativeSource(_ directory: URL, times: [Double] = [0, 0.5, 1, 1.5]) async throws -> URL {
    let url = directory.appendingPathComponent("source.mov")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    writer.add(input)
    XCTAssertTrue(writer.startWriting()); writer.startSession(atSourceTime: .zero)
    for frame in 0..<4 {
      while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
      var buffer: CVPixelBuffer?
      XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32BGRA, nil, &buffer), kCVReturnSuccess)
      let pixels = try XCTUnwrap(buffer)
      CVPixelBufferLockBaseAddress(pixels, [])
      let bytes = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: UInt8.self)
      for y in 0..<64 { for x in 0..<64 {
        let offset = y * CVPixelBufferGetBytesPerRow(pixels) + x * 4
        bytes[offset] = frame == 2 ? 255 : 0
        bytes[offset + 1] = frame == 1 ? 255 : 0
        bytes[offset + 2] = frame == 0 ? 255 : 0
        bytes[offset + 3] = 255
      } }
      CVPixelBufferUnlockBaseAddress(pixels, [])
      XCTAssertTrue(adaptor.append(pixels, withPresentationTime: CMTime(seconds: times[frame], preferredTimescale: 600)))
    }
    input.markAsFinished(); writer.endSession(atSourceTime: CMTime(value: 4, timescale: 2))
    await writer.finishWriting(); XCTAssertEqual(writer.status, .completed)
    return url
  }

  func testNativeRipplePreparationFreezesReferencesAndWritesWorkerContract() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let movie = try await nativeSource(root)
    let reference = try await NativeRippleMedia.extractFrame([
      "source_path": movie.path, "source_start": 0.0, "duration": 2.0,
      "frame_rate": 2.0, "width": 64, "height": 64, "frame": 0],
      into: root.appendingPathComponent("reference"))
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = movie.path; clip.duration = 2.0
    var draft = RippleDraft(clip: clip, frameRate: 2)
    draft.width = 64; draft.height = 64
    draft.references[0].path = try XCTUnwrap(reference["image_path"] as? String)
    let transformer = root.appendingPathComponent("transformer")
    let text = root.appendingPathComponent("text")
    try FileManager.default.createDirectory(at: transformer, withIntermediateDirectories: false)
    try FileManager.default.createDirectory(at: text, withIntermediateDirectories: false)
    let video = root.appendingPathComponent("video.safetensors")
    let audio = root.appendingPathComponent("audio.safetensors")
    let adapter = root.appendingPathComponent("adapter.safetensors")
    for path in [video, audio, adapter] { try Data([1]).write(to: path) }
    let profile = root.appendingPathComponent("profile.json")
    try JSONSerialization.data(withJSONObject: [
      "format": "weetodd-headless-v2", "engine": "ltx25",
      "config": ["pipeline_mode": "distilled"],
      "components": ["transformer_path": transformer.path,
        "text_encoder_path": text.path, "video_vae_path": video.path,
        "audio_vae_path": audio.path, "loras": [], "ic_loras": []]
    ]).write(to: profile)
    let inputs = root.appendingPathComponent("inputs")
    let take = root.appendingPathComponent("take")
    let result = try await NativeRipplePreparation.prepare(draft: draft,
      runtime: ["rippleAdapterPath": adapter.path, "rippleProfileID": profile.path,
        "profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"],
      into: inputs, output: take)
    let recipe = try XCTUnwrap(result["recipePath"] as? String)
    let values = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf:
      URL(fileURLWithPath: recipe))) as? [String: Any])
    XCTAssertEqual(values["task"] as? String, "ripple")
    XCTAssertEqual(values["frames"] as? Int, 9)
    XCTAssertEqual(values["editorial_frames"] as? Int, 4)
    XCTAssertEqual(values["output_directory"] as? String, take.path)
    XCTAssertEqual((values["anchors"] as? [Any])?.count, 0)
    XCTAssertEqual((values["source_sha256"] as? String)?.count, 64)
    XCTAssertTrue(FileManager.default.fileExists(atPath: values["first_reference_path"] as! String))
    XCTAssertFalse(FileManager.default.fileExists(atPath: take.path))
  }

  func testNativeRipplePreparationRejectsOversizedProfileBeforeReadingMedia() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mp4")
    let adapter = root.appendingPathComponent("adapter.safetensors")
    let profile = root.appendingPathComponent("profile.json")
    try Data([1]).write(to: source)
    try Data([1]).write(to: adapter)
    try Data(repeating: 32, count: 1024 * 1024 + 1).write(to: profile)
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = source.path; clip.duration = 1
    var draft = RippleDraft(clip: clip, frameRate: 24)
    draft.references[0].path = "/edited.png"
    do {
      _ = try await NativeRipplePreparation.prepare(draft: draft,
        runtime: ["rippleAdapterPath": adapter.path, "rippleProfileID": profile.path,
          "profilesDirectory": root.path, "ffmpegPath": "/usr/bin/true"],
        into: root.appendingPathComponent("inputs"), output: root.appendingPathComponent("take"))
      XCTFail("Oversized profile must fail before source inspection")
    } catch {
      XCTAssertTrue(error.localizedDescription.contains("plain installed LTX 2.5 distilled profile"))
    }
  }

  func testNativeRipplePublishedTakeChecksActualFramesAndAudio() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let movie = try await nativeSource(root)
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = movie.path; clip.duration = 2
    var draft = RippleDraft(clip: clip, frameRate: 2)
    draft.width = 64; draft.height = 64
    try await NativeRippleMedia.verifyPublishedTake(movie.path,
      draft: draft, hasAudio: false)
    do {
      try await NativeRippleMedia.verifyPublishedTake(movie.path,
        draft: draft, hasAudio: true)
      XCTFail("An absent audio stream must fail verification")
    } catch { XCTAssertTrue(error.localizedDescription.contains("audio")) }
    draft.duration = 1
    do {
      try await NativeRippleMedia.verifyPublishedTake(movie.path,
        draft: draft, hasAudio: false)
      XCTFail("A two-second take must not satisfy a one-second draft")
    } catch { XCTAssertTrue(error.localizedDescription.contains("timing")) }
  }

  @MainActor func testOptedInRippleUsesNativePreparationAndWorker() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let movie = try await nativeSource(root)
    let take = root.appendingPathComponent("take.mov")
    try FileManager.default.copyItem(at: movie, to: take)
    var calls: [String] = []
    let store = StudioStore(dataDirectory: root, restoreSession: false,
      invocation: { command, _, payload, _ in
        if command == "audio-mix" { throw CancellationError() }
        calls.append(command)
        switch command {
        case "ripple-native-prepare":
          XCTAssertNotNil(payload["draft"] as? [String: Any])
          return ["recipePath": root.appendingPathComponent("request.json").path]
        case "ltx-native-render":
          XCTAssertNotNil(payload["recipePath"] as? String)
          return ["video_path": take.path, "duration": 2.0, "frames": 4,
            "frame_rate": 2.0, "width": 64, "height": 64, "has_audio": false,
            "receipt_path": root.appendingPathComponent("receipt.json").path,
            "artifacts_directory": root.path,
            "frozen_references": [["frame": 0,
              "path": root.appendingPathComponent("frozen.png").path, "strength": 1.0]],
            "source_sha256": String(repeating: "a", count: 64)]
        default:
          XCTFail("Unexpected Ripple command: \(command)")
          throw StudioError.invalid("Unexpected Ripple command")
        }
      })
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    store.runtime.nativeRippleEnabled = true
    store.runtime.pythonPath = "/no-python-for-ripple-route-test"
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = movie.path; clip.duration = 2
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    store.updateRipple { draft in
      draft.width = 64; draft.height = 64; draft.frameRate = 2
      draft.references[0].path = "/edited.png"
    }
    await store.generateRipple()
    XCTAssertNil(store.error)
    XCTAssertEqual(calls, ["ripple-native-prepare", "ltx-native-render"])
    XCTAssertEqual(store.selectedClip?.rippleTakes?.first?.path, take.path)
  }

  @MainActor func testInstalledNativeRippleStudioLifecycle() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_RIPPLE_LIFECYCLE"] else {
      throw XCTSkip("Opt-in installed LTX 2.5 Ripple Studio lifecycle")
    }
    let options = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let root = URL(fileURLWithPath: try XCTUnwrap(options["output"]))
    if let reuseProject = options["reuseProject"] {
      // Inspect a completed take after a test assertion failure without repeating inference.
      let reopened = StudioStore(dataDirectory: root, restoreSession: false)
      reopened.load(URL(fileURLWithPath: reuseProject))
      let clip = try XCTUnwrap(reopened.project.clips.first)
      let take = try XCTUnwrap(clip.rippleTakes?.last)
      XCTAssertEqual(clip.sourcePath, take.path)
      try await NativeRippleMedia.verifyPublishedTake(take.path, draft: take.draft,
        hasAudio: take.hasAudio)
      let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf:
        URL(fileURLWithPath: take.receiptPath))) as! [String: Any]
      XCTAssertEqual(receipt["editorial_frames"] as? Int, 72)
      let digest = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: take.path)))
        .map { String(format: "%02x", $0) }.joined()
      let evidence: [String: Any] = ["video": take.path, "sha256": digest,
        "pythonPath": "/unavailable/python", "reopenedWithoutRerender": true]
      try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
        .write(to: root.appendingPathComponent("ripple-studio-qualification.json"), options: .atomic)
      return
    }
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    let profile = profiles.appendingPathComponent("ripple.json")
    let components: [String: Any] = [
      "transformer_path": try XCTUnwrap(options["transformer"]),
      "text_encoder_path": try XCTUnwrap(options["text"]),
      "video_vae_path": try XCTUnwrap(options["video"]),
      "audio_vae_path": try XCTUnwrap(options["audio"]),
      "loras": [], "ic_loras": []]
    let profileValue: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "config": ["pipeline_mode": "distilled"], "components": components]
    try JSONSerialization.data(withJSONObject: profileValue, options: [.prettyPrinted, .sortedKeys])
      .write(to: profile, options: .atomic)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: profiles.path)
    store.runtime.nativeRippleEnabled = true
    store.runtime.ltx25WorkerPath = try XCTUnwrap(options["worker"])
    store.runtime.rippleProfileID = profile.path
    store.runtime.rippleAdapterPath = try XCTUnwrap(options["adapter"])
    store.runtime.ffmpegPath = try XCTUnwrap(options["ffmpeg"])
    var clip = Clip(name: "Kitten", engine: .movie)
    clip.sourcePath = try XCTUnwrap(options["source"])
    clip.duration = 3
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    store.openRipple()
    await store.inspectRipple()
    XCTAssertNil(store.error)
    store.updateRipple { draft in
      draft.prompt = "The kitten has fluffy white fur. Preserve the original kitten motion, hanging toy, background, camera, composition and timing."
      draft.seed = 42
      draft.width = 768
      draft.height = 448
      draft.frameRate = 24
      draft.references[0].path = options["edit"]!
    }
    var previews = Set<Int>()
    let observer = store.bridge.$livePreview.sink { if let revision = $0?.previewRevision { previews.insert(revision) } }
    defer { observer.cancel() }
    let started = Date()
    await store.generateRipple()
    XCTAssertNil(store.error)
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.last)
    XCTAssertGreaterThan(previews.count, 0)
    XCTAssertTrue(store.canApplyRipple(take, to: try XCTUnwrap(store.selectedClip)))
    XCTAssertEqual(store.selectedClip?.sourcePath, options["source"])
    store.applyRipple(take)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.sourcePath, take.path)
    let saved = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: saved)
    let reopened = StudioStore(dataDirectory: root, restoreSession: false)
    reopened.load(saved)
    XCTAssertEqual(reopened.project.clips.first?.sourcePath, take.path)
    try await NativeRippleMedia.verifyPublishedTake(take.path, draft: take.draft,
      hasAudio: take.hasAudio)
    let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf:
      URL(fileURLWithPath: take.receiptPath))) as! [String: Any]
    XCTAssertEqual(receipt["editorial_frames"] as? Int, 72)
    let digest = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: take.path)))
      .map { String(format: "%02x", $0) }.joined()
    if let expected = options["expectedSHA256"] { XCTAssertEqual(digest, expected) }
    let evidence: [String: Any] = ["video": take.path, "sha256": digest,
      "renderAndAcceptanceSeconds": Date().timeIntervalSince(started),
      "decodedPreviewCount": previews.count, "pythonPath": store.runtime.pythonPath]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("ripple-studio-qualification.json"), options: .atomic)
  }

  @MainActor func testNativeRippleInspectionAndFrameExtractionWorkWithoutPython() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = try await nativeSource(directory)
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    store.runtime.nativeLTX25Enabled = false
    store.runtime.nativeRippleEnabled = true
    store.runtime.pythonPath = "/missing-python-for-native-ripple-test"
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = movie.path; clip.sourceIn = 0.5; clip.duration = 1
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    await store.inspectRipple()
    XCTAssertNil(store.error)
    let draft = try XCTUnwrap(store.rippleClip?.rippleDraft)
    XCTAssertEqual(draft.frameRate, 2)
    XCTAssertEqual(draft.sourcePreviewStart ?? -1, 0.5, accuracy: 0.001)
    XCTAssertEqual(draft.frameCount, 2)
    let reference = try XCTUnwrap(draft.references.first)
    await store.extractRippleFrame(referenceID: reference.id)
    XCTAssertNil(store.error)
    let extracted = try XCTUnwrap(store.rippleClip?.rippleDraft?.references.first?.originalPath)
    let bitmap = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: URL(fileURLWithPath: extracted))))
    let color = try XCTUnwrap(bitmap.colorAt(x: 32, y: 32)?.usingColorSpace(.deviceRGB))
    XCTAssertGreaterThan(color.greenComponent, 0.75)
    XCTAssertLessThan(color.redComponent, 0.4)
  }

  @MainActor func testNativeRippleRejectsVariableCadenceBeforeExtractingAnEditFrame() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = try await nativeSource(directory, times: [0, 0.5, 1.15, 1.65])
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    addTeardownBlock { await MainActor.run { store.invalidateTimelinePlayback() } }
    store.runtime.nativeLTX25Enabled = true
    store.runtime.pythonPath = "/missing-python-for-native-ripple-test"
    var clip = Clip(name: "Variable", engine: .movie)
    clip.sourcePath = movie.path; clip.sourceIn = 0; clip.duration = 1.5
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    await store.inspectRipple()
    XCTAssertTrue(store.error?.contains("constant frame rate") == true)
    XCTAssertNil(store.rippleClip?.rippleDraft?.sourcePreviewStart)
  }
  func testNativeRippleStreamsEditedFirstFrameAndSourceFramesInOrder() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = try await nativeSource(directory)
    let request: [String: Any] = ["source_path": movie.path, "source_start": 0.0,
      "duration": 2.0, "frame_rate": 2.0, "width": 64, "height": 64, "frame": 1]
    let edited = try await NativeRippleMedia.extractFrame(request,
      into: directory.appendingPathComponent("edit"))
    let guide = try await NativeRippleMedia.prepareGuide(request,
      editedFirstFrame: URL(fileURLWithPath: edited["image_path"] as! String),
      into: directory.appendingPathComponent("guide"))
    XCTAssertEqual(guide["frames"] as? Int, 9)
    let raw = try Data(contentsOf: URL(fileURLWithPath: guide["rgb_path"] as! String))
    XCTAssertEqual(raw.count, 9 * 64 * 64 * 3)
    func color(_ frame: Int) -> [UInt8] {
      let offset = (frame * 64 * 64 + 32 * 64 + 32) * 3
      return Array(raw[offset..<offset + 3])
    }
    XCTAssertGreaterThan(color(0)[1], 180) // edited green first
    XCTAssertGreaterThan(color(1)[0], 180) // source red follows, not replaced
    XCTAssertGreaterThan(color(2)[1], 180)
    XCTAssertGreaterThan(color(3)[2], 180)
    XCTAssertEqual(color(4), color(8)) // terminal frame is cloned for VAE padding
  }
  @MainActor private func store(invocation: Bridge.Invocation? = nil) throws -> StudioStore {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, runtime, payload, output in
      // Draft edits schedule a separate timeline preview mix. This fixture has no
      // playable media; keep that background command outside Ripple assertions.
      if command == "audio-mix" { throw CancellationError() }
      guard ["ripple-inspect", "ripple-frame", "ripple-generate"].contains(command),
        let invocation else {
        XCTFail("Unexpected bridge command in Ripple fixture: \(command)")
        throw StudioError.invalid("Unexpected test bridge command")
      }
      return try await invocation(command, runtime, payload, output)
    })
    addTeardownBlock {
      await MainActor.run { store.invalidateTimelinePlayback() }
    }
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.sourceIn = 2.5; clip.duration = 4.8
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    store.updateRipple { $0.references[0].path = "/edited.png" }
    return store
  }
  private static var result: [String: Any] {
    ["video_path": "/new-take.mp4", "duration": 4.8, "frames": 116,
      "frame_rate": 24.0, "width": 768, "height": 448, "has_audio": false,
      "receipt_path": "/take/receipt.json", "artifacts_directory": "/take",
      "frozen_references": [["frame": 0, "path": "/take/frozen-frame0.png", "strength": 1.0]],
      "source_sha256": String(repeating: "a", count: 64)]
  }
  @MainActor func testGenerationRetainsSourceUntilExplicitApplyAndRestoresOriginalInterval() async throws {
    let store = try store { command, _, payload, _ in
      XCTAssertEqual(command, "ripple-generate")
      let request = try XCTUnwrap(payload["ripple"] as? [String: Any])
      XCTAssertEqual(request["source_start"] as? Double, 2.5)
      XCTAssertEqual(request["audio_policy"] as? String, "preserve")
      return Self.result
    }
    await store.generateRipple()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2.5)
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertFalse(take.hasAudio, "Silent inputs must remain valid")
    XCTAssertEqual(take.draft.references[0].path, "/take/frozen-frame0.png")
    XCTAssertEqual(take.submittedDraftFingerprint, store.selectedClip?.rippleDraft?.inputFingerprint)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    XCTAssertEqual(store.project.assets.last?.path, take.path)
    store.applyRipple(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/new-take.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 0)
    XCTAssertEqual(store.selectedClip?.duration, 4.8)
    XCTAssertTrue(store.selectedClip?.hasReviewedReusedTake == true,
      "Explicit Ripple application must be a usable reviewed take for native as well as imported clips")
    XCTAssertEqual(store.selectedClip?.versions.first?.path, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.versions.first?.usableSourceIn, 2.5)
    store.restoreRippleTakeInputs(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2.5)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    XCTAssertEqual(store.selectedClip?.rippleDraft?.references[0].path, "/take/frozen-frame0.png")
    XCTAssertEqual(try store.selectedClip?.rippleDraft?.bridgeObject()["source_sha256"] as? String,
      String(repeating: "a", count: 64))
  }
  @MainActor func testChangedDraftRetainsTakeButCannotApplyIt() async throws {
    weak var active: StudioStore?
    let store = try store { _, _, _, _ in
      active?.updateRipple { $0.seed += 1 }
      return Self.result
    }
    active = store
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertNil(store.rippleSelectedTakeID)
    XCTAssertFalse(store.canApplyRipple(take, to: store.selectedClip!))
    store.applyRipple(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
  }
  @MainActor func testReopenedIdenticalDocumentCannotReceiveLateTake() async throws {
    weak var active: StudioStore?
    let store = try store { _, _, _, _ in
      if let active { try active.replaceDocument(active.project, url: nil, isDirty: false) }
      return Self.result
    }
    active = store
    await store.generateRipple()
    XCTAssertNil(store.selectedClip?.rippleTakes)
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.notice.contains("destination movie changed"))
  }
  @MainActor func testRestoredTakeUsesFrozenImageAfterExternalEditIsDeleted() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let external = directory.appendingPathComponent("external-edit.png")
    let frozen = directory.appendingPathComponent("frozen-edit.png")
    try Data("edited image snapshot".utf8).write(to: external)
    try FileManager.default.copyItem(at: external, to: frozen)
    let store = try store { _, _, _, _ in
      var result = Self.result
      result["frozen_references"] = [["frame": 0, "path": frozen.path, "strength": 1.0]]
      return result
    }
    store.updateRipple { $0.references[0].path = external.path }
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    try FileManager.default.removeItem(at: external)
    store.restoreRippleTakeInputs(take)
    let request = try XCTUnwrap(store.selectedClip?.rippleDraft).bridgeObject()
    let references = try XCTUnwrap(request["references"] as? [[String: Any]])
    XCTAssertEqual(references[0]["path"] as? String, frozen.path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: frozen.path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: external.path))
  }
  @MainActor func testMissingFrozenReferencesCannotCreateUnreplayableTake() async throws {
    let store = try store { _, _, _, _ in
      var result = Self.result; result.removeValue(forKey: "frozen_references"); return result
    }
    await store.generateRipple()
    XCTAssertNil(store.selectedClip?.rippleTakes)
    XCTAssertTrue(store.error?.contains("frozen replay") == true)
  }
  @MainActor func testChangedTrimPreventsApply() async throws {
    let store = try store { _, _, _, _ in Self.result }
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    store.project.clips[0].sourceIn += 1
    XCTAssertFalse(store.canApplyRipple(take, to: store.project.clips[0]))
    store.applyRipple(take)
    XCTAssertEqual(store.project.clips[0].sourceIn, 3.5)
  }
  @MainActor func testExtractionUsesExactRelativeFrameAndRejectsChangedDraft() async throws {
    weak var active: StudioStore?
    let store = try store { command, _, payload, _ in
      XCTAssertEqual(command, "ripple-frame")
      XCTAssertEqual((payload["ripple"] as? [String: Any])?["frame"] as? Int, 17)
      active?.updateRipple { $0.references[0].frame = 18 }
      return ["image_path": "/frame17.png"]
    }
    active = store
    store.updateRipple { $0.references[0].frame = 17 }
    await store.extractRippleFrame(referenceID: store.rippleClip!.rippleDraft!.references[0].id)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[0].originalPath, "")
    XCTAssertTrue(store.notice.contains("inputs changed"))
  }
  @MainActor func testAddReferenceBindsPlayheadExactlyAndReusesExistingFrame() throws {
    let store = try store()
    let id = try store.addRippleReference(frame: 23)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.last?.frame, 23)
    XCTAssertEqual(try store.addRippleReference(frame: 23), id)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 2)
    let firstID = store.rippleClip!.rippleDraft!.references[0].id
    store.removeRippleReference(firstID)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 2)
    for frame in 1...7 { try store.addRippleReference(frame: frame) }
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 9)
    XCTAssertThrowsError(try store.addRippleReference(frame: 99))
    XCTAssertThrowsError(try store.addRippleReference(frame: -1))
  }
  @MainActor func testRequiredFirstFrameCannotMoveAndMovingOtherFrameClearsOldImages() throws {
    let store = try store()
    let first = store.rippleClip!.rippleDraft!.references[0].id
    XCTAssertThrowsError(try store.setRippleReferenceFrame(first, frame: 2))
    let second = try store.addRippleReference(frame: 12)
    store.updateRipple { $0.references[1].path = "/old-edit.png"; $0.references[1].originalPath = "/old-source.png" }
    XCTAssertThrowsError(try store.setRippleReferenceFrame(second, frame: 0))
    try store.setRippleReferenceFrame(second, frame: 18)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].frame, 18)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].path, "")
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].originalPath, "")
  }
  @MainActor func testInitialInspectAdoptsSourceCadenceAndAspectButAssignedImagesKeepTheirGrid() async throws {
    let store = try store { _, _, _, _ in
      ["source_frame_rate": 30.0, "source_preview_start": 2.5333333333,
       "source": ["width": 1080, "height": 1920], "has_audio": true]
    }
    store.updateRipple { $0.references[0].path = "" }
    await store.inspectRipple()
    XCTAssertEqual(store.rippleClip?.rippleDraft?.frameRate, 30)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.sourcePreviewStart, 2.5333333333)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.width, 1088)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.height, 1920)
    store.updateRipple { $0.references[0].path = "/edited.png"; $0.frameRate = 24 }
    await store.inspectRipple()
    XCTAssertEqual(store.rippleClip?.rippleDraft?.frameRate, 24)
  }
  @MainActor func testRuntimePathsRemainOutOfClipAndOrdinaryGenerationSettings() throws {
    var settings = RuntimeSettings.defaults()
    settings.rippleAdapterPath = "/weights/ripple.safetensors"; settings.rippleProfileID = "/profiles/ltx25.json"
    let decoded = try JSONDecoder().decode(RuntimeSettings.self, from: JSONEncoder().encode(settings))
    XCTAssertEqual(decoded.rippleAdapterPath, settings.rippleAdapterPath)
    XCTAssertNil(settings.generationSettings.rippleAdapterPath)
    XCTAssertNil(settings.generationSettings.rippleProfileID)
  }
}
