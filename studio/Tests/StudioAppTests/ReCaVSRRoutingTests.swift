import AVFoundation
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class ReCaVSRRoutingTests: XCTestCase {
  @MainActor func testCancelledPreflightCannotPublishSuccess() async throws {
    var entered: CheckedContinuation<Void, Never>?, release: CheckedContinuation<Void, Never>?
    let store = StudioStore(restoreSession: false, invocation: { _, _, _, _ in
      entered?.resume(); entered = nil
      await withCheckedContinuation { release = $0 }
      return ["preflight": true]
    })
    var request: Task<[String: Any], Error>?
    await withCheckedContinuation { continuation in
      entered = continuation
      request = Task { try await store.invokeReCaVSRWorker("recavsr-native-preflight", runtime: store.runtime,
        recipe: "/unused.json", output: FileManager.default.temporaryDirectory) }
    }
    request?.cancel(); release?.resume()
    do { _ = try await request?.value; XCTFail("Cancelled preflight returned success") }
    catch { XCTAssertTrue(error is CancellationError) }
  }
  func testUpscaleCopyClampsRoundedRelayDurationToSourceEnd() throws {
    var clip = Clip(name: "Rounded source", engine: .movie)
    clip.sourcePath = "/source.mov"
    clip.duration = 1.005
    var media = MediaAsset(name: "Source", kind: .video, path: clip.sourcePath, scope: .project)
    media.duration = 1
    media.fps = 24
    let recavsr = try NativeLTXMovieClipCopy.create(source: clip, media: media, method: .reCaVSR)
    XCTAssertEqual(recavsr.clip.duration, 1)
    XCTAssertEqual(recavsr.clip.attachments.first?.sourceDurationSeconds, 1)
    let ltx = try NativeLTXMovieClipCopy.create(source: clip, media: media)
    XCTAssertEqual(ltx.clip.duration, clip.duration)
  }

  @MainActor func testReCaVSRUsesNativeRouteWithAnEmptyPromptAndLTXDisabled() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let recipe = directory.appendingPathComponent("recipe.json")
    try Data("{}".utf8).write(to: recipe)
    var calls: [String] = []
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { command, runtime, _, _ in
        calls.append(command)
        XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
        switch command {
        case "recavsr-native-describe": return ["fingerprint": "recavsr-test"]
        case "recavsr-native-prepare": return ["recipePath": recipe.path, "report": [:]]
        case "recavsr-native-render": throw StudioError.invalid("Stopped before rendering")
        default: throw StudioError.invalid("Unexpected backend: \(command)")
        }
      })
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: directory.path, nativeLTX25Enabled: false)
    store.addClip()
    store.editClip {
      $0.engine = .ltx25
      $0.prompt = ""
      $0.generationSelection = GenerationSelection(task: "video_upscale")
      $0.generationSelection?.movieUpscaleMethod = .reCaVSR
    }
    await store.generateSelected()
    XCTAssertEqual(calls, ["recavsr-native-describe", "recavsr-native-prepare", "recavsr-native-render"])
    XCTAssertEqual(store.error, "Stopped before rendering")
    XCTAssertFalse(store.actionItems.contains { ["runtime", "profiles"].contains($0.id) })
    XCTAssertFalse(store.issues(for: try XCTUnwrap(store.selectedClip)).contains("Add a prompt"))
  }

  @MainActor func testPackagedReCaVSRRendersTrimmedMovieThroughStudio() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let worker = environment["WEETODD_TEST_RECAVSR_WORKER"],
      let models = environment["WEETODD_TEST_RECAVSR_MODELS"],
      let source = environment["WEETODD_TEST_RECAVSR_SOURCE"] else {
      throw XCTSkip("Opt in with the packaged ReCaVSR worker, checkpoint folder and source fixture.")
    }
    let qaDirectory = environment["WEETODD_TEST_RECAVSR_QA_DIRECTORY"]
    let directory = qaDirectory.map { URL(fileURLWithPath: $0) }
      ?? FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { if qaDirectory == nil { try? FileManager.default.removeItem(at: directory) } }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: directory.path, reCaVSRWorkerPath: worker,
      reCaVSRModelDirectory: models, nativeLTX25Enabled: false)
    let fps = 24_000.0 / 1001
    var sourceClip = Clip(name: "Source", engine: .movie)
    sourceClip.sourcePath = source
    sourceClip.sourceIn = 2 / fps
    sourceClip.duration = 8 / fps
    store.project.clips = [sourceClip]
    store.select(sourceClip.id)
    await store.createMovieUpscaleCopy(method: .reCaVSR)
    XCTAssertEqual(store.selectedClip?.usesReCaVSR, true, store.error ?? "")
    await store.generateSelected()
    let selected = try XCTUnwrap(store.selectedClip)
    let version = try XCTUnwrap(selected.versions.last, store.error ?? "")
    let renderedAsset = try XCTUnwrap(store.project.assets.first { $0.path == version.path })
    XCTAssertEqual(renderedAsset.upscaleReceipt, version.path + ".json")
    let reopened = StudioStore(dataDirectory: directory.appendingPathComponent("reopened"), restoreSession: false)
    reopened.project = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(store.project))
    reopened.selectedAssetID = renderedAsset.id
    reopened.openVideoUpscale()
    XCTAssertEqual(reopened.videoUpscaleTool, .reCaVSR)
    XCTAssertEqual(reopened.videoUpscaleSourcePath, source)
    XCTAssertEqual(reopened.reCaVSRResult?.output, version.path)
    XCTAssertEqual(store.project.clips.first(where: { $0.id == sourceClip.id })?.sourcePath, source)
    XCTAssertEqual(selected.sourcePath, version.path)
    XCTAssertEqual(selected.duration, 8 / fps, accuracy: 0.0001)
    XCTAssertEqual(store.clipState(selected), .generated)
    store.generationDescriptions.removeAll()
    try await store.revalidateExistingNativeTakes()
    XCTAssertEqual(store.clipState(selected), .generated)
    XCTAssertTrue(store.nativeHeadlessEligible)
    let output = AVURLAsset(url: URL(fileURLWithPath: version.path))
    let video = try await output.loadTracks(withMediaType: .video)
    let audio = try await output.loadTracks(withMediaType: .audio)
    let track = try XCTUnwrap(video.first)
    let outputFPS = try await track.load(.nominalFrameRate)
    let size = try await track.load(.naturalSize)
    XCTAssertEqual(Double(outputFPS), fps, accuracy: 0.0001)
    XCTAssertEqual(size, CGSize(width: 576, height: 416))
    XCTAssertEqual(audio.count, 1)
    let audioRange = try await XCTUnwrap(audio.first).load(.timeRange)
    XCTAssertEqual(audioRange.start.seconds, 0, accuracy: 0.0001)
    XCTAssertEqual(audioRange.duration.seconds, 8 / fps, accuracy: 0.0001)

    store.videoUpscaleSourcePath = source
    store.videoUpscaleTool = .reCaVSR
    store.runReCaVSRUpscale()
    await store.videoUpscaleTask?.value
    let whole = try XCTUnwrap(store.reCaVSRResult, store.videoUpscaleStatus)
    let asset = try XCTUnwrap(store.project.assets.first { $0.path == whole.output })
    XCTAssertEqual(store.project.clips.count, 2)
    XCTAssertEqual(asset.width, 576)
    XCTAssertEqual(asset.height, 416)
    XCTAssertEqual(asset.fps, fps, accuracy: 0.0001)
    if qaDirectory != nil {
      try ProjectStorage.write(store.project, to: directory.appendingPathComponent("Autosave.weetodd"))
      try JSONEncoder().encode(store.runtime).write(to: directory.appendingPathComponent("runtime.json"))
      store.saveVideoUpscalePreferences()
    }
  }
}
