import AVFoundation
import Combine
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

/// Optional real scene execution; the normal suite never starts a model worker.
final class NativeLTXSceneLifecycleTests: XCTestCase {
  private struct Source: Codable { let path: String; let sha256: String }
  private struct Manifest: Codable {
    let recipe: String
    let recipeSHA256: String
    let editorRequest: String
    let editorRequestSHA256: String
    let expectedPreparedRecipe: String
    let expectedPreparedRecipeSHA256: String
    let worker: String
    let workerSHA256: String
    let output: String
    let ffmpeg: String
    let decodeMode: String
    let clipCount: Int
    let frames: Int
    let audioChannels: Int
    let sources: [Source]
  }
  private func require(_ valid: Bool, _ message: String) throws {
    guard valid else { throw StudioError.invalid(message) }
  }
  private func object(_ url: URL) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
  }
  private func admit(_ project: StudioProject, expected: [String: Any], count: Int,
    decodeMode: String, frames: Int) throws {
    let scene = try XCTUnwrap(expected["scene"] as? [String: Any])
    let segments = try XCTUnwrap(scene["segments"] as? [[String: Any]])
    let config = try XCTUnwrap(expected["config"] as? [String: Any])
    let fps = try XCTUnwrap(config["frame_rate"] as? Double)
    try require(count >= 2 && project.clips.count == count && segments.count == count,
      "Keep every frozen scene member; do not shorten the scene for qualification.")
    try require(scene["decode_mode"] as? String == decodeMode
      && project.clips.first?.continuity?.sceneDecodeMode == decodeMode,
      "Preserve the actual scene decode mode.")
    for (index, clip) in project.clips.enumerated() {
      let segment = segments[index]
      try require(clip.engine == .ltx25 && clip.id.uuidString == segment["clip_id"] as? String
        && clip.prompt == segment["prompt"] as? String && clip.seed == segment["seed"] as? Int
        && clip.duration == segment["duration_seconds"] as? Double
        && clip.generationWidth == config["width"] as? Int
        && clip.generationHeight == config["height"] as? Int,
        "Preserve scene order, IDs, prompts, seeds, duration and geometry.")
      try require(clip.sourcePath.isEmpty && clip.versions.isEmpty && clip.sourceIn == 0
        && clip.rippleDraft == nil && clip.extensionSource.isEmpty,
        "Start from fresh unaccepted editor shots, not a replay of an accepted movie.")
      if index > 0 {
        try require(clip.continuityMode == "scene"
          && clip.continuity?.sourceClipID == project.clips[index - 1].id,
          "Retain the actual predecessor chain.")
      }
    }
    let duration = project.clips.reduce(0) { $0 + $1.duration }
    try require(duration == config["duration_seconds"] as? Double
      && Double(frames) == duration * fps, "Preserve the complete scene duration and frame count.")
  }

  func testSceneAdmissionRejectsShorteningAndChangedSettings() throws {
    for count in [2, 6] {
      var project = StudioProject()
      project.clips = (0..<count).map { index in
        var clip = Clip(name: "Shot \(index)", engine: .ltx25)
        clip.prompt = "Frozen shot \(index)"; clip.seed = 42 + index; clip.duration = 5
        clip.generationWidth = 768; clip.generationHeight = 448
        clip.continuity = ClipContinuity(sceneDecodeMode: "windowed")
        return clip
      }
      for index in 1..<count {
        project.clips[index].continuity = ClipContinuity(mode: "scene",
          sourceClipID: project.clips[index - 1].id, sceneDecodeMode: "windowed")
      }
      let expected: [String: Any] = ["scene": ["decode_mode": "windowed", "segments": project.clips.map {
        ["clip_id": $0.id.uuidString, "prompt": $0.prompt, "seed": $0.seed,
          "duration_seconds": $0.duration] as [String: Any]
      }], "config": ["frame_rate": 24.0, "width": 768, "height": 448, "duration_seconds": Double(count * 5)] as [String: Any]]
      try admit(project, expected: expected, count: count, decodeMode: "windowed", frames: count * 120)
      var changed = project; changed.clips.removeLast()
      XCTAssertThrowsError(try admit(changed, expected: expected, count: count, decodeMode: "windowed", frames: count * 120))
      changed = project; changed.clips[count - 1].seed += 1
      XCTAssertThrowsError(try admit(changed, expected: expected, count: count, decodeMode: "windowed", frames: count * 120))
      changed = project; changed.clips[0].continuity?.sceneDecodeMode = "single"
      XCTAssertThrowsError(try admit(changed, expected: expected, count: count, decodeMode: "windowed", frames: count * 120))
      changed = project; changed.clips[count - 1].sourcePath = "/old/accepted.mp4"
      XCTAssertThrowsError(try admit(changed, expected: expected, count: count, decodeMode: "windowed", frames: count * 120))
    }
  }

  private func verifySources(_ sources: [Source]) throws {
    try require(!sources.isEmpty, "Pin the actual scene source media.")
    for source in sources {
      try require(try NativeHeadlessJob.fileHash(URL(fileURLWithPath: source.path)) == source.sha256,
        "The scene source changed: \(source.path)")
    }
  }
  private func normalized(_ value: [String: Any]) throws -> NSDictionary {
    func input(_ value: [String: Any]) throws -> [String: Any] {
      var value = value
      for key in ["path", "source_path"] {
        if let path = value[key] as? String {
          value[key] = "sha256:" + (try NativeHeadlessJob.fileHash(URL(fileURLWithPath: path)))
        }
      }
      return value
    }
    var recipe = value
    var conditioning = try XCTUnwrap(recipe["conditioning"] as? [String: Any])
    conditioning["inputs"] = try XCTUnwrap(conditioning["inputs"] as? [[String: Any]]).map(input)
    recipe["conditioning"] = conditioning
    var scene = try XCTUnwrap(recipe["scene"] as? [String: Any])
    var segments = try XCTUnwrap(scene["segments"] as? [[String: Any]])
    for index in segments.indices {
      if let image = segments[index]["image_input"] as? [String: Any] {
        segments[index]["image_input"] = try input(image)
      }
    }
    scene["segments"] = segments; recipe["scene"] = scene
    return NSDictionary(dictionary: recipe)
  }

  func testSceneParityAllowsOnlyIdenticalMediaRelocation() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source"), copied = root.appendingPathComponent("copied")
    try Data([1, 2, 3]).write(to: source); try Data([1, 2, 3]).write(to: copied)
    let expected: [String: Any] = ["conditioning": ["inputs": [["path": source.path, "strength": 1]]],
      "scene": ["segments": [["seed": 42, "image_input": ["path": source.path]]]],
      "config": ["width": 768, "seed": 42]]
    var relocated = expected
    relocated["conditioning"] = ["inputs": [["path": copied.path, "strength": 1]]]
    relocated["scene"] = ["segments": [["seed": 42, "image_input": ["path": copied.path]]]]
    XCTAssertTrue(try normalized(expected).isEqual(normalized(relocated)))
    relocated["config"] = ["width": 768, "seed": 43]
    XCTAssertFalse(try normalized(expected).isEqual(normalized(relocated)))
    relocated["config"] = expected["config"]
    try Data([1, 2, 4]).write(to: copied)
    XCTAssertFalse(try normalized(expected).isEqual(normalized(relocated)))
  }

  @MainActor func testInstalledNativeSceneLifecycleWithoutPython() async throws {
    guard let manifestPath = ProcessInfo.processInfo.environment["WEETODD_NATIVE_LTX_TYPED_SCENE_LIFECYCLE"] else {
      throw XCTSkip("Explicit scene lifecycle manifest required; no weighted work by default.")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifestPath)))
    let profileURL = URL(fileURLWithPath: manifest.recipe), editorURL = URL(fileURLWithPath: manifest.editorRequest)
    let expectedURL = URL(fileURLWithPath: manifest.expectedPreparedRecipe), workerURL = URL(fileURLWithPath: manifest.worker)
    let pinned = [(profileURL, manifest.recipeSHA256), (editorURL, manifest.editorRequestSHA256),
      (expectedURL, manifest.expectedPreparedRecipeSHA256), (workerURL, manifest.workerSHA256)]
    for (url, hash) in pinned {
      try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen scene fixture or worker changed.")
    }
    try verifySources(manifest.sources)
    let expected = try object(expectedURL), request = try object(editorURL)
    let project = try JSONDecoder().decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: request["project"]!))
    let global = try JSONDecoder().decode([MediaAsset].self, from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    let clipID = try XCTUnwrap(UUID(uuidString: try XCTUnwrap(request["clipID"] as? String)))
    try require(project.clips.contains(where: { $0.id == clipID }), "Retain the original selected scene member.")
    try admit(project, expected: expected, count: manifest.clipCount, decodeMode: manifest.decodeMode, frames: manifest.frames)
    let attachedIDs = Set(project.clips.flatMap { $0.attachments.map(\.assetID) })
    let attachedPaths = Set((project.assets + global).filter { attachedIDs.contains($0.id) }.map(\.path))
    try require(attachedPaths.isSubset(of: Set(manifest.sources.map(\.path))),
      "Freeze every attached scene source, including any later shot image or audio driver.")
    let root = URL(fileURLWithPath: manifest.output)
    try require(!FileManager.default.fileExists(atPath: root.path), "Use a fresh scene lifecycle destination.")
    let profiles = root.appendingPathComponent("Profiles"), copied = profiles.appendingPathComponent("matched.json")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    try Data(contentsOf: profileURL).write(to: copied)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project = project; store.globalAssets = global; store.selectedClipID = clipID
    for index in store.project.clips.indices { store.project.clips[index].profileID = copied.path }
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeLTX25Enabled = true; store.runtime.ltx25WorkerPath = manifest.worker
    store.runtime.ffmpegPath = manifest.ffmpeg
    await store.reloadProfiles(); await store.prepareSelected()
    try require(store.error == nil, store.error ?? "Scene preparation failed.")
    let prepared = URL(fileURLWithPath: try XCTUnwrap(store.preparedRecipe))
    try require(try normalized(object(prepared)).isEqual(normalized(expected)),
      "Scene preparation differs from the frozen export; stop before weighted work.")
    try require(store.project.assets == project.assets && store.globalAssets == global
      && store.project.clips.allSatisfy({ $0.sourcePath.isEmpty && $0.versions.isEmpty }),
      "Preparation changed sources or accepted a shot.")
    try verifySources(manifest.sources)
    for (url, hash) in pinned {
      try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen scene fixture or worker changed during preparation.")
    }
    let preparedSHA = try NativeHeadlessJob.fileHash(prepared)
    var evidence: [String: Any] = ["workerSHA256": manifest.workerSHA256, "pythonAvailable": false,
      "profileSHA256": manifest.recipeSHA256, "editorRequestSHA256": manifest.editorRequestSHA256,
      "preparedRecipe": prepared.path, "preparedRecipeSHA256": preparedSHA,
      "expectedPreparedRecipe": expectedURL.path, "exactPreparedRecipeParityAllowingOnlyHashIdenticalMediaRelocation": true,
      "sourcesUnchanged": true, "clipCount": manifest.clipCount, "frames": manifest.frames,
      "decodeMode": manifest.decodeMode, "productionQualified": false]
    if ProcessInfo.processInfo.environment["WEETODD_NATIVE_LTX_TYPED_SCENE_PREPARE_ONLY"] == "1" {
      evidence["proofLevel"] = "studio-scene-preparation-and-native-preflight-only"
      evidence["inferenceExecuted"] = false; evidence["studioAccepted"] = false
      try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
        .write(to: root.appendingPathComponent("scene-preparation-qualification.json"))
      return
    }
    let beforeRender = store.project
    var previews = Set<Int>(), progress = Set<Double>()
    let observer = store.bridge.$livePreview.sink { if let revision = $0?.previewRevision { previews.insert(revision) } }
    let progressObserver = store.bridge.$fraction.sink { if $0 > 0 && $0 < 1 { progress.insert($0) } }
    defer { observer.cancel(); progressObserver.cancel() }
    await store.renderPrepared()
    try require(store.error == nil, store.error ?? "Scene render failed.")
    let pending = try XCTUnwrap(store.pendingContinuousScene)
    let publication = manifest.decodeMode == "windowed" ? "windowed_decode_native_latent_chain" : "single_decode_native_latent_chain"
    try require(store.project == beforeRender && pending.report.members.map(\.clipID) == project.clips.map(\.id)
      && pending.report.publicationMode == publication && store.canAcceptContinuousScene
      && !previews.isEmpty && !progress.isEmpty, "A new scene must remain pending for review and report progress/previews.")
    var start = 0.0
    for (member, clip) in zip(pending.report.members, project.clips) {
      try require(member.sourceIn == start && member.duration == clip.duration, "The pending scene ranges changed.")
      start += clip.duration
    }
    await store.acceptContinuousScene()
    try require(store.error == nil && store.pendingContinuousScene == nil, store.error ?? "Scene acceptance failed.")
    for (index, clip) in store.project.clips.enumerated() {
      let version = try XCTUnwrap(clip.versions.last), member = pending.report.members[index]
      try require(clip.id == project.clips[index].id && clip.sourcePath == pending.video
        && clip.sourceIn == member.sourceIn && clip.duration == member.duration
        && version.sceneTakeID == pending.id && version.sceneMembers == pending.report.members
        && version.seed == project.clips[index].seed && version.recipePath == prepared.path
        && version.path == pending.video && version.usableSourceIn == member.sourceIn
        && version.usableDuration == member.duration && version.sceneFrameRate == pending.report.frameRate
        && clip.renderedSignature == store.signature(for: clip), "All shots must accept the same scene take and exact source ranges.")
    }
    let inspected = try await StudioStore.inspectNativeMovie(pending.video)
    let config = try XCTUnwrap(expected["config"] as? [String: Any])
    let fps = try XCTUnwrap(config["frame_rate"] as? Double)
    try require(inspected["width"] as? Int == config["width"] as? Int
      && inspected["height"] as? Int == config["height"] as? Int && inspected["fps"] as? Double == fps
      && abs(try XCTUnwrap(inspected["duration"] as? Double) - start) <= 1 / fps,
      "The published movie must retain the full scene geometry, duration and frame rate.")
    let movie = AVURLAsset(url: URL(fileURLWithPath: pending.video))
    let audio = try await movie.loadTracks(withMediaType: .audio)
    let video = try await movie.loadTracks(withMediaType: .video)
    try require(video.count == 1 && audio.count == 1, "Publish exactly one video track and one synchronized audio track.")
    let audioRange = try await XCTUnwrap(audio.first).load(.timeRange)
    let videoRange = try await XCTUnwrap(video.first).load(.timeRange)
    let descriptions = try await XCTUnwrap(audio.first).load(.formatDescriptions)
    let channels = try XCTUnwrap(descriptions.first.flatMap { CMAudioFormatDescriptionGetStreamBasicDescription($0) }).pointee.mChannelsPerFrame
    try require(Int(channels) == manifest.audioChannels && abs(audioRange.duration.seconds - start) <= 1 / fps
      && abs(videoRange.duration.seconds - audioRange.duration.seconds) <= 1 / fps
      && abs(videoRange.start.seconds - audioRange.start.seconds) <= 1 / fps,
      "The complete scene must retain synchronized audio and its original channel count.")
    let take = URL(fileURLWithPath: pending.video).deletingLastPathComponent()
    let reportURL = take.appendingPathComponent("report.json"), report = try object(reportURL)
    try require(report["frames"] as? Int == manifest.frames && report["python_inference"] as? Bool == false
      && report["nativeRuntime"] as? String == "swift-mlx" && report["publication_mode"] as? String == publication,
      "The actual scene receipt must prove full frame count and native inference.")
    try require(try NativeHeadlessJob.fileHash(prepared) == preparedSHA, "The frozen scene recipe changed during execution.")
    try require(store.globalAssets == global && project.assets.allSatisfy { original in
      store.project.assets.first(where: { $0.id == original.id }) == original
    }, "Scene acceptance changed the original assets.")
    try verifySources(manifest.sources)
    for (url, hash) in pinned {
      try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen scene fixture or worker changed during execution.")
    }
    let saved = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: saved)
    let reopened = StudioStore(dataDirectory: root, restoreSession: false); reopened.load(saved)
    try require(reopened.error == nil && reopened.project == store.project,
      "Every accepted scene member and shared take must survive actual Store.load.")
    evidence["proofLevel"] = "new-studio-scene-generation-pending-review-acceptance-save-reopen"
    evidence["inferenceExecuted"] = true; evidence["studioAccepted"] = true
    evidence["video"] = pending.video; evidence["videoSHA256"] = try NativeHeadlessJob.fileHash(URL(fileURLWithPath: pending.video))
    evidence["acceptedProject"] = saved.path; evidence["sceneTakeID"] = pending.id.uuidString
    evidence["sourceIns"] = pending.report.members.map(\.sourceIn); evidence["decodedPreviewCount"] = previews.count
    evidence["progressUpdateCount"] = progress.count; evidence["audioChannels"] = Int(channels)
    evidence["audioDuration"] = audioRange.duration.seconds; evidence["workerReport"] = reportURL.path
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("scene-studio-qualification.json"))
  }
}
