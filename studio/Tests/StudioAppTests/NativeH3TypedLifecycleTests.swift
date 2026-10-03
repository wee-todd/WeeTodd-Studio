import AVFoundation
import Combine
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeH3TypedLifecycleTests: XCTestCase {
  private enum Route: String, Codable, CaseIterable {
    case movieReference, audioReference, soundtrackReference, audioToVideo, afterExtension, funControl
    var componentTask: String { self == .funControl ? "t2va" : "ref2va" }
    var task: String {
      switch self {
      case .audioToVideo: return "a2v"
      case .afterExtension: return "extension"
      case .funControl: return "control"
      default: return "ref2va"
      }
    }
    var kinds: [AssetKind] {
      switch self {
      case .movieReference, .soundtrackReference: return [.image, .video]
      case .audioReference: return [.image, .audio]
      case .audioToVideo: return [.audio]
      case .funControl: return [.video]
      case .afterExtension: return []
      }
    }
    var roles: [MediaRole] {
      switch self {
      case .audioToVideo: return [.audioDriver]
      case .funControl: return [.control]
      case .afterExtension: return []
      default: return [.reference, .reference]
      }
    }
  }
  private struct Source: Codable { let path: String; let sha256: String }
  private struct Manifest: Codable {
    let engine: String
    let route: Route
    let task: String
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
    let sources: [Source]
  }
  private func require(_ valid: Bool, _ message: String) throws {
    guard valid else { throw StudioError.invalid(message) }
  }
  private func admit(route: Route, task: String, profile: [String: Any], clip: Clip,
    assets: [MediaAsset]) throws {
    try require(task == route.task && clip.engine == .h3 && clip.generationSelection?.task == route.task,
      "The typed H3 route must match the actual editor task.")
    let components = try XCTUnwrap(profile["components"] as? [String: Any])
    try require(profile["engine"] as? String == "h3" && components["task"] as? String == route.componentTask,
      "The typed H3 route must retain its actual component partition.")
    let config = try XCTUnwrap(profile["config"] as? [String: Any])
    try require(config["width"] as? Int == clip.generationWidth && config["height"] as? Int == clip.generationHeight
      && config["duration_seconds"] as? Double == clip.duration && config["seed"] as? Int == clip.seed
      && profile["prompt"] as? String == clip.prompt, "Preserve frozen H3 settings and prompt.")
    let kinds = try clip.attachments.map { attachment in
      try XCTUnwrap(assets.first(where: { $0.id == attachment.assetID })).kind
    }
    try require(kinds == route.kinds && clip.attachments.map(\.role) == route.roles,
      "Preserve typed media roles and ordering; never substitute still images for movie or audio inputs.")
    try require(clip.sourcePath.isEmpty && clip.versions.isEmpty && clip.continuityMode == "independent"
      && clip.rippleDraft == nil, "Start from the actual unaccepted editor request.")
    if route == .afterExtension {
      try require(clip.extensionDirection == "after" && !clip.extensionSource.isEmpty,
        "External after-extension requires its original movie source.")
    } else {
      try require(clip.extensionSource.isEmpty && clip.extensionDirection.isEmpty,
        "Do not turn another H3 route into an extension.")
    }
    if route == .funControl {
      try require(components["fun_controlnet"] is String && clip.attachments[0].controlType == "canny_edges",
        "The Fun fixture requires its actual control component and Canny guide.")
    }
    if route == .audioToVideo {
      let driver = clip.attachments[0]
      try require(driver.audioSourceStart != nil && driver.audioSourceDuration != nil,
        "Independent A2V must retain its explicit source interval.")
    }
  }

  func testTypedAdmissionPreservesEveryRouteAndRejectsImageSubstitution() throws {
    for route in Route.allCases {
      var clip = Clip(engine: .h3)
      clip.duration = 2.5; clip.generationWidth = 384; clip.generationHeight = 256
      clip.seed = 42; clip.prompt = "Frozen AV prompt"
      clip.generationSelection = GenerationSelection(task: route.task)
      let assets = route.kinds.map { MediaAsset(name: "Typed source", kind: $0, path: "/unread/source") }
      clip.attachments = zip(assets, route.roles).map { Attachment(assetID: $0.0.id, role: $0.1) }
      if route == .afterExtension { clip.extensionDirection = "after"; clip.extensionSource = "/unread/source.mp4" }
      if route == .audioToVideo { clip.attachments[0].audioSourceStart = 0.3; clip.attachments[0].audioSourceDuration = 2.5 }
      var components: [String: Any] = ["task": route.componentTask]
      if route == .funControl { components["fun_controlnet"] = "/unread/control.safetensors" }
      let config: [String: Any] = ["width": 384, "height": 256, "duration_seconds": 2.5, "seed": 42]
      let profile: [String: Any] = ["engine": "h3", "components": components, "prompt": clip.prompt,
        "config": config]
      try admit(route: route, task: route.task, profile: profile, clip: clip, assets: assets)
      XCTAssertThrowsError(try admit(route: route, task: "t2va", profile: profile, clip: clip, assets: assets))
      var changed = clip; changed.seed += 1
      XCTAssertThrowsError(try admit(route: route, task: route.task, profile: profile, clip: changed, assets: assets))
      if !assets.isEmpty {
        var wrong = assets; wrong[wrong.count - 1].kind = .image
        XCTAssertThrowsError(try admit(route: route, task: route.task, profile: profile, clip: clip, assets: wrong))
      }
    }
  }

  private func verifySources(_ sources: [Source]) throws {
    try require(!sources.isEmpty, "Freeze actual H3 media source identities before qualification.")
    for source in sources {
      try require(try NativeHeadlessJob.fileHash(URL(fileURLWithPath: source.path)) == source.sha256,
        "An H3 source changed: \(source.path)")
    }
  }
  private func normalizedRecipe(_ value: [String: Any]) throws -> NSDictionary {
    var recipe = value
    var conditioning = try XCTUnwrap(value["conditioning"] as? [String: Any])
    var inputs = try XCTUnwrap(conditioning["inputs"] as? [[String: Any]])
    for index in inputs.indices {
      // Only location-bearing media fields may move between preparation roots.
      // Their bytes, all other fields and the entire model/config contract stay exact.
      for key in ["path", "source_path"] {
        if let path = inputs[index][key] as? String {
          inputs[index][key] = "sha256:" + (try NativeHeadlessJob.fileHash(URL(fileURLWithPath: path)))
        }
      }
    }
    conditioning["inputs"] = inputs; recipe["conditioning"] = conditioning
    return NSDictionary(dictionary: recipe)
  }

  @MainActor func testInstalledTypedH3StudioLifecycleWithoutPython() async throws {
    guard let manifestPath = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_TYPED_LIFECYCLE"] else {
      throw XCTSkip("Explicit typed H3 lifecycle manifest required; no weighted work by default.")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifestPath)))
    try require(manifest.engine == "h3", "Typed lifecycle requires the H3 engine.")
    let profileURL = URL(fileURLWithPath: manifest.recipe), editorURL = URL(fileURLWithPath: manifest.editorRequest)
    let expectedURL = URL(fileURLWithPath: manifest.expectedPreparedRecipe), workerURL = URL(fileURLWithPath: manifest.worker)
    for (url, hash) in [(profileURL, manifest.recipeSHA256), (editorURL, manifest.editorRequestSHA256),
      (expectedURL, manifest.expectedPreparedRecipeSHA256), (workerURL, manifest.workerSHA256)] {
      try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen H3 fixture/worker identity changed.")
    }
    try verifySources(manifest.sources)
    let profileBytes = try Data(contentsOf: profileURL)
    let profile = try JSONSerialization.jsonObject(with: profileBytes) as! [String: Any]
    let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: expectedURL)) as! [String: Any]
    let request = try JSONSerialization.jsonObject(with: Data(contentsOf: editorURL)) as! [String: Any]
    let project = try JSONDecoder().decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: request["project"]!))
    let global = try JSONDecoder().decode([MediaAsset].self, from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    let clipID = try XCTUnwrap(UUID(uuidString: try XCTUnwrap(request["clipID"] as? String)))
    let original = try XCTUnwrap(project.clips.first(where: { $0.id == clipID }))
    try admit(route: manifest.route, task: manifest.task, profile: profile, clip: original, assets: project.assets + global)
    let conditioning = try XCTUnwrap(expected["conditioning"] as? [String: Any])
    try require(conditioning["task"] as? String == manifest.task && conditioning["audio_policy"] as? String == "generated",
      "Preserve the frozen prepared task and generated-audio policy.")
    let mediaPath = manifest.route == .afterExtension ? original.extensionSource
      : try XCTUnwrap((project.assets + global).first(where: { $0.id == original.attachments.last?.assetID })).path
    let sourceAsset = AVURLAsset(url: URL(fileURLWithPath: mediaPath))
    let sourceVideos = try await sourceAsset.loadTracks(withMediaType: .video)
    let sourceAudio = try await sourceAsset.loadTracks(withMediaType: .audio)
    switch manifest.route {
    case .movieReference: try require(!sourceVideos.isEmpty && sourceAudio.isEmpty, "The movie fixture must retain its silent source.")
    case .soundtrackReference: try require(!sourceVideos.isEmpty && !sourceAudio.isEmpty, "The soundtrack fixture must retain its actual movie/audio tracks.")
    case .audioReference, .audioToVideo: try require(sourceVideos.isEmpty && !sourceAudio.isEmpty, "The standalone audio fixture must remain audio.")
    case .afterExtension, .funControl: try require(!sourceVideos.isEmpty, "The route requires its actual source movie/guide.")
    }
    let root = URL(fileURLWithPath: manifest.output)
    try require(!FileManager.default.fileExists(atPath: root.path), "Use a fresh typed lifecycle output directory.")
    let profiles = root.appendingPathComponent("Profiles"), copied = profiles.appendingPathComponent("matched.json")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    try profileBytes.write(to: copied)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project = project; store.globalAssets = global; store.selectedClipID = clipID
    store.project.clips[try XCTUnwrap(store.project.clips.firstIndex(where: { $0.id == clipID }))].profileID = copied.path
    let others = store.project.clips.filter { $0.id != clipID }, originalAssets = store.project.assets
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true; store.runtime.h3WorkerPath = manifest.worker; store.runtime.ffmpegPath = manifest.ffmpeg
    await store.reloadProfiles(); await store.prepareSelected()
    try require(store.error == nil, store.error ?? "Preparation failed.")
    let prepared = URL(fileURLWithPath: try XCTUnwrap(store.preparedRecipe))
    let actual = try JSONSerialization.jsonObject(with: Data(contentsOf: prepared)) as! [String: Any]
    try require(try normalizedRecipe(actual).isEqual(normalizedRecipe(expected)),
      "Typed preparation differs from the frozen native export; stop before weighted work.")
    try require(store.project.clips.filter({ $0.id != clipID }) == others && store.project.assets == originalAssets,
      "Preparation changed source assets or another clip.")
    try verifySources(manifest.sources)
    let preparedSHA = try NativeHeadlessJob.fileHash(prepared)
    if ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_TYPED_PREPARE_ONLY"] == "1" {
      try require(store.globalAssets == global, "Preparation changed global source assets.")
      for (url, hash) in [(profileURL, manifest.recipeSHA256), (editorURL, manifest.editorRequestSHA256),
        (expectedURL, manifest.expectedPreparedRecipeSHA256), (workerURL, manifest.workerSHA256)] {
        try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen preparation fixture/worker changed.")
      }
      let evidence: [String: Any] = ["proofLevel": "typed-studio-preparation-and-native-preflight-only",
        "route": manifest.route.rawValue, "task": manifest.task, "inferenceExecuted": false,
        "studioAccepted": false, "productionQualified": false, "pythonAvailable": false,
        "workerSHA256": manifest.workerSHA256, "editorRequestSHA256": manifest.editorRequestSHA256,
        "profileSHA256": manifest.recipeSHA256, "preparedRecipe": prepared.path,
        "preparedRecipeSHA256": preparedSHA, "expectedPreparedRecipe": expectedURL.path,
        "expectedPreparedRecipeSHA256": manifest.expectedPreparedRecipeSHA256,
        "exactPreparedRecipeParityAllowingOnlyHashIdenticalMediaRelocation": true,
        "sourcesUnchanged": true, "otherClipsUnchanged": true,
        "sourceVideoTrackCount": sourceVideos.count, "sourceAudioTrackCount": sourceAudio.count,
        "scope": "Admission, media/source guards and actual Studio preparation only; no render, preview, acceptance or quality claim."]
      try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
        .write(to: root.appendingPathComponent("typed-preparation-qualification.json"))
      return
    }
    var previews = Set<Int>()
    let observer = store.bridge.$livePreview.sink { if let revision = $0?.previewRevision { previews.insert(revision) } }
    defer { observer.cancel() }
    let started = Date()
    await store.renderPrepared()
    try require(store.error == nil, store.error ?? "Render failed.")
    let accepted = try XCTUnwrap(store.selectedClip), version = try XCTUnwrap(accepted.versions.last)
    try require(accepted.id == clipID && accepted.sourcePath == version.path && version.seed == original.seed
      && version.recipePath == prepared.path && accepted.renderedSignature == store.signature(for: accepted)
      && accepted.duration == original.duration && previews.count > 0,
      "Actual typed H3 preview and timeline acceptance must complete.")
    try require(store.project.clips.filter({ $0.id != clipID }) == others && store.globalAssets == global
      && originalAssets.allSatisfy({ asset in store.project.assets.first(where: { $0.id == asset.id }) == asset }),
      "Generation changed source assets or another clip.")
    try verifySources(manifest.sources)
    let movie = try await StudioStore.inspectNativeMovie(version.path)
    let asset = AVURLAsset(url: URL(fileURLWithPath: version.path))
    let videoTracks = try await asset.loadTracks(withMediaType: .video), audioTracks = try await asset.loadTracks(withMediaType: .audio)
    let videoRange = try await XCTUnwrap(videoTracks.first).load(.timeRange)
    let audioRange = try await XCTUnwrap(audioTracks.first).load(.timeRange)
    try require(abs(videoRange.start.seconds - audioRange.start.seconds) <= 0.1
      && abs(videoRange.duration.seconds - audioRange.duration.seconds) <= 0.1,
      "Published H3 audio/video tracks must agree within 0.1 seconds.")
    var frames = Int((original.duration * 24).rounded(.toNearestOrEven))
    while frames % 17 != 5 { frames += 1 }
    try require(movie["width"] as? Int == original.generationWidth && movie["height"] as? Int == original.generationHeight
      && movie["fps"] as? Double == 24 && abs(try XCTUnwrap(movie["duration"] as? Double) - Double(frames) / 24) <= 0.05,
      "Published media differs from frozen geometry/aligned duration.")
    let take = URL(fileURLWithPath: version.path).deletingLastPathComponent(), receiptURL = take.appendingPathComponent("result.json")
    let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
    try require(receipt["status"] as? String == "complete" && receipt["task"] as? String == manifest.task
      && receipt["nativeRuntime"] as? String == "swift-mlx" && receipt["frames"] as? Int == frames,
      "The published worker receipt must prove this typed execution.")
    for (url, hash) in [(prepared, preparedSHA), (take.appendingPathComponent("studio-recipe.json"), preparedSHA),
      (profileURL, manifest.recipeSHA256), (editorURL, manifest.editorRequestSHA256), (workerURL, manifest.workerSHA256)] {
      try require(try NativeHeadlessJob.fileHash(url) == hash, "Frozen typed execution bytes changed.")
    }
    let saved = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: saved)
    let reopened = StudioStore(dataDirectory: root, restoreSession: false); reopened.load(saved)
    try require(reopened.error == nil && reopened.project == store.project
      && reopened.project.clips.first(where: { $0.id == clipID })?.sourcePath == version.path,
      "The typed H3 accepted take must survive save/reopen.")
    let evidence: [String: Any] = ["proofLevel": "new-typed-studio-generation-preview-acceptance-save-reopen",
      "route": manifest.route.rawValue, "task": manifest.task, "studioAccepted": true, "productionQualified": false,
      "pythonAvailable": false, "inferenceExecuted": true, "workerSHA256": manifest.workerSHA256,
      "editorRequestSHA256": manifest.editorRequestSHA256, "profileSHA256": manifest.recipeSHA256,
      "preparedRecipe": prepared.path, "preparedRecipeSHA256": preparedSHA,
      "expectedPreparedRecipe": expectedURL.path, "expectedPreparedRecipeSHA256": manifest.expectedPreparedRecipeSHA256,
      "video": version.path, "videoSHA256": try NativeHeadlessJob.fileHash(URL(fileURLWithPath: version.path)),
      "workerReceipt": receiptURL.path, "workerReceiptSHA256": try NativeHeadlessJob.fileHash(receiptURL),
      "acceptedProject": saved.path, "decodedPreviewCount": previews.count, "otherClipsAndSourcesUnchanged": true,
      "requestedDuration": original.duration, "modelFrames": frames, "modelDuration": Double(frames) / 24,
      "videoTrackDuration": videoRange.duration.seconds, "audioTrackDuration": audioRange.duration.seconds,
      "syncToleranceSeconds": 0.1, "renderAndAcceptanceSeconds": Date().timeIntervalSince(started),
      "scope": "Actual typed Studio execution only; historical direct/graph output is not acceptance or quality approval."]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("studio-qualification.json"))
  }
}
