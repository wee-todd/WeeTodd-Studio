import AVFoundation
import Combine
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeH3T2VALifecycleTests: XCTestCase {
  private func admit(options: [String: String], profile: [String: Any], clip: Clip) throws {
    guard options["engine"] == "h3", options["task"] == "t2va",
      profile["engine"] as? String == "h3",
      let components = profile["components"] as? [String: Any],
      components["task"] as? String == "t2va",
      let conditioning = profile["conditioning"] as? [String: Any],
      conditioning["task"] as? String == "t2v",
      let inputs = conditioning["inputs"] as? [[String: Any]], inputs.isEmpty,
      clip.engine == .h3, clip.generationSelection?.task == "t2v",
      clip.attachments.isEmpty, clip.continuityMode == "independent",
      clip.extensionSource.isEmpty, clip.extensionDirection.isEmpty,
      clip.sourcePath.isEmpty, clip.versions.isEmpty, clip.rippleDraft == nil else {
      throw StudioError.invalid("Select an explicit ordinary H3 T2VA manifest and a new unaccepted clip without references or continuation.")
    }
    guard let config = profile["config"] as? [String: Any],
      config["width"] as? Int == clip.generationWidth,
      config["height"] as? Int == clip.generationHeight,
      config["duration_seconds"] as? Double == clip.duration,
      config["seed"] as? Int == clip.seed,
      profile["prompt"] as? String == clip.prompt else {
      throw StudioError.invalid("The new Studio clip must preserve the original T2VA geometry, duration, seed and prompt.")
    }
  }

  func testManifestAdmissionRequiresExplicitOrdinaryTaskAndOriginalSettings() throws {
    var clip = Clip(engine: .h3)
    clip.generationSelection = GenerationSelection(task: "t2v")
    clip.generationWidth = 768; clip.generationHeight = 448
    clip.duration = 5; clip.seed = 20260929; clip.prompt = "Original AV prompt"
    let profile: [String: Any] = ["engine": "h3", "components": ["task": "t2va"],
      "conditioning": ["task": "t2v", "inputs": []], "prompt": clip.prompt,
      "config": ["width": 768, "height": 448, "duration_seconds": 5.0, "seed": clip.seed]]
    let options = ["engine": "h3", "task": "t2va"]
    try admit(options: options, profile: profile, clip: clip)
    XCTAssertThrowsError(try admit(options: ["engine": "h3"], profile: profile, clip: clip))
    XCTAssertThrowsError(try admit(options: ["engine": "h3", "task": "ref2va"], profile: profile, clip: clip))
    var changed = clip; changed.seed += 1
    XCTAssertThrowsError(try admit(options: options, profile: profile, clip: changed))
    changed = clip; changed.sourcePath = "/unread/previous-take.mp4"
    XCTAssertThrowsError(try admit(options: options, profile: profile, clip: changed))
    changed = clip; changed.continuity = ClipContinuity(mode: "frame")
    XCTAssertThrowsError(try admit(options: options, profile: profile, clip: changed))
  }

  @MainActor func testInstalledOrdinaryT2VAStudioLifecycleWithoutPython() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_NATIVE_H3_T2VA_LIFECYCLE"] else {
      throw XCTSkip("Opt-in new ordinary H3 T2VA Studio generation/acceptance; no inference by default.")
    }
    let options = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifest))) as! [String: String]
    let profileURL = URL(fileURLWithPath: try XCTUnwrap(options["recipe"]))
    let profileBytes = try Data(contentsOf: profileURL)
    let profile = try JSONSerialization.jsonObject(with: profileBytes) as! [String: Any]
    let editorURL = URL(fileURLWithPath: try XCTUnwrap(options["editorRequest"]))
    let request = try JSONSerialization.jsonObject(with: Data(contentsOf: editorURL)) as! [String: Any]
    let project = try JSONDecoder().decode(StudioProject.self,
      from: JSONSerialization.data(withJSONObject: request["project"]!))
    let clipID = try XCTUnwrap(UUID(uuidString: try XCTUnwrap(request["clipID"] as? String)))
    let original = try XCTUnwrap(project.clips.first(where: { $0.id == clipID }))
    try admit(options: options, profile: profile, clip: original)
    let worker = try XCTUnwrap(options["worker"])
    let expectedWorkerSHA = try XCTUnwrap(options["workerSHA256"])
    guard try NativeHeadlessJob.fileHash(URL(fileURLWithPath: worker)) == expectedWorkerSHA else {
      throw StudioError.invalid("The selected H3 worker differs from the frozen lifecycle manifest.")
    }
    let root = URL(fileURLWithPath: try XCTUnwrap(options["output"]))
    guard !FileManager.default.fileExists(atPath: root.path) else {
      throw StudioError.invalid("Use a fresh lifecycle output directory; never overwrite an earlier qualification.")
    }
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    let copied = profiles.appendingPathComponent("matched-t2va.json")
    try profileBytes.write(to: copied)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.project = project
    store.globalAssets = try JSONDecoder().decode([MediaAsset].self,
      from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    store.selectedClipID = clipID
    let index = try XCTUnwrap(store.project.clips.firstIndex(where: { $0.id == clipID }))
    store.project.clips[index].profileID = copied.path
    let otherClips = store.project.clips.filter { $0.id != clipID }
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: profiles.path)
    store.runtime.nativeH3Enabled = true; store.runtime.h3WorkerPath = worker
    store.runtime.ffmpegPath = options["ffmpeg"] ?? "/opt/homebrew/bin/ffmpeg"
    await store.reloadProfiles()
    XCTAssertEqual(store.profiles.count, 1)
    await store.prepareSelected()
    guard store.error == nil else { throw StudioError.invalid(store.error!) }
    let prepared = URL(fileURLWithPath: try XCTUnwrap(store.preparedRecipe))
    let preparedRecipe = try JSONSerialization.jsonObject(with: Data(contentsOf: prepared)) as! [String: Any]
    for key in ["config", "components", "conditioning"] {
      guard NSDictionary(dictionary: try XCTUnwrap(preparedRecipe[key] as? [String: Any]))
        .isEqual(to: try XCTUnwrap(profile[key] as? [String: Any])) else {
        throw StudioError.invalid("T2VA preparation changed the frozen \(key); stop before generation.")
      }
    }
    guard preparedRecipe["prompt"] as? String == original.prompt else {
      throw StudioError.invalid("T2VA preparation changed the frozen prompt; stop before generation.")
    }
    XCTAssertTrue(store.preparedReport.contains("swift-mlx"))
    guard store.project.clips.filter({ $0.id != clipID }) == otherClips else {
      throw StudioError.invalid("T2VA preparation changed another clip; stop before generation.")
    }
    let preparedSHA = try NativeHeadlessJob.fileHash(prepared)
    var previews = Set<Int>()
    let observation = store.bridge.$livePreview.sink { event in
      if let revision = event?.previewRevision { previews.insert(revision) }
    }
    defer { observation.cancel() }
    let started = Date()
    await store.renderPrepared()
    guard store.error == nil else { throw StudioError.invalid(store.error!) }
    let accepted = try XCTUnwrap(store.selectedClip)
    let version = try XCTUnwrap(accepted.versions.last)
    guard accepted.id == clipID, accepted.sourcePath == version.path,
      version.seed == original.seed, version.recipePath == prepared.path,
      accepted.renderedSignature == store.signature(for: accepted),
      accepted.duration == original.duration,
      store.project.clips.filter({ $0.id != clipID }) == otherClips,
      previews.count > 0 else {
      throw StudioError.invalid("The new T2VA take did not complete preview/acceptance without changing other clips.")
    }
    let saved = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: saved)
    let reopened = StudioStore(dataDirectory: root, restoreSession: false)
    reopened.load(saved)
    guard reopened.error == nil, reopened.project == store.project,
      reopened.project.clips.first(where: { $0.id == clipID })?.sourcePath == version.path else {
      throw StudioError.invalid("The newly accepted T2VA take did not survive save/reopen.")
    }
    let movie = try await StudioStore.inspectNativeMovie(version.path)
    let audioTracks = try await AVURLAsset(url: URL(fileURLWithPath: version.path))
      .loadTracks(withMediaType: .audio)
    var modelFrames = Int((original.duration * 24).rounded(.toNearestOrEven))
    while modelFrames % 17 != 5 { modelFrames += 1 }
    guard !audioTracks.isEmpty, movie["width"] as? Int == original.generationWidth,
      movie["height"] as? Int == original.generationHeight, movie["fps"] as? Double == 24,
      abs(try XCTUnwrap(movie["duration"] as? Double) - Double(modelFrames) / 24) <= 0.05 else {
      throw StudioError.invalid("The new T2VA movie is missing synchronized media or differs from the frozen geometry/frame alignment.")
    }
    let takeDirectory = URL(fileURLWithPath: version.path).deletingLastPathComponent()
    let receiptURL = takeDirectory.appendingPathComponent("result.json")
    let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
    guard receipt["status"] as? String == "complete", receipt["task"] as? String == "t2va",
      receipt["nativeRuntime"] as? String == "swift-mlx", receipt["frames"] as? Int == modelFrames,
      (receipt["referenceImages"] as? [[String: Any]])?.isEmpty == true,
      try NativeHeadlessJob.fileHash(prepared) == preparedSHA,
      try NativeHeadlessJob.fileHash(takeDirectory.appendingPathComponent("studio-recipe.json")) == preparedSHA,
      try NativeHeadlessJob.fileHash(URL(fileURLWithPath: worker)) == expectedWorkerSHA else {
      throw StudioError.invalid("The published T2VA receipt does not match this frozen Studio execution.")
    }
    let evidence: [String: Any] = ["proofLevel": "new-studio-generation-preview-acceptance-save-reopen",
      "studioAccepted": true, "productionQualified": false, "inferenceExecuted": true,
      "pythonPath": store.runtime.pythonPath, "runtimeRoot": store.runtime.root,
      "worker": worker, "workerSHA256": expectedWorkerSHA, "clipID": clipID.uuidString,
      "originalEditorRequest": editorURL.path, "originalEditorRequestSHA256": try NativeHeadlessJob.fileHash(editorURL),
      "originalRecipe": profileURL.path, "originalRecipeSHA256": try NativeHeadlessJob.fileHash(profileURL),
      "preparedRecipe": prepared.path, "preparedRecipeSHA256": preparedSHA,
      "workerReceipt": receiptURL.path, "workerReceiptSHA256": try NativeHeadlessJob.fileHash(receiptURL),
      "audioTrackCount": audioTracks.count,
      "video": version.path, "videoSHA256": try NativeHeadlessJob.fileHash(URL(fileURLWithPath: version.path)),
      "acceptedProject": saved.path, "decodedPreviewCount": previews.count,
      "requestedDuration": original.duration, "modelFrames": modelFrames,
      "modelDuration": Double(modelFrames) / 24, "seed": original.seed,
      "renderAndAcceptanceSeconds": Date().timeIntervalSince(started), "otherClipsUnchanged": true,
      "scope": "Actual new ordinary T2VA Studio lifecycle only; no inferred Comfy acceptance or visual quality approval."]
    try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("studio-qualification.json"))
  }
}
