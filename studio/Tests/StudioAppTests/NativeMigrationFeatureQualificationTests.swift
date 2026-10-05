import AVFoundation
import Combine
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

/// Explicit opt-in fixtures. Ordinary test runs never start installed inference.
final class NativeMigrationFeatureQualificationTests:XCTestCase {
  private func object(_ url:URL) throws -> [String:Any] {
    try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
  }
  private func write(_ value:Any,_ url:URL) throws {
    try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:url)
  }

  @MainActor func testPrepareFrozenSceneV2StudioFixture() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_SCENE_V2_FIXTURE"] else {
      throw XCTSkip("Explicit original scene-v2 fixture preparation; no inference.")
    }
    let options = try object(URL(fileURLWithPath:manifest)) as! [String:String]
    let original = try object(URL(fileURLWithPath:options["recipe"]!))
    let root = URL(fileURLWithPath:options["output"]!)
    guard !FileManager.default.fileExists(atPath:root.path) else { throw StudioError.invalid("Fresh fixture output required.") }
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at:profiles,withIntermediateDirectories:true)
    var profile = original;profile.removeValue(forKey:"scene")
    profile["conditioning"] = ["version":1,"task":"a2v","inputs":[],"audio_policy":"source"]
    let profileURL = profiles.appendingPathComponent("scene.json");try write(profile,profileURL)
    let config = original["config"] as! [String:Any],scene = original["scene"] as! [String:Any]
    let segments = scene["segments"] as! [[String:Any]]
    let inputs = (original["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    let audio = inputs.first { $0["kind"] as? String == "audio" }!
    let fps = (config["frame_rate"] as! NSNumber).doubleValue
    var project = StudioProject(),elapsed = 0.0
    var sourceAssets:[String:MediaAsset] = [:]
    for input in inputs {
      let path = input["path"] as! String
      if sourceAssets[path] == nil {
        sourceAssets[path] = MediaAsset(name:"Frozen source",kind:input["kind"] as? String == "image" ? .image:.audio,path:path)
      }
    }
    project.assets = sourceAssets.values.sorted { $0.path < $1.path }
    for (index,segment) in segments.enumerated() {
      var clip = Clip(name:"Scene shot \(index+1)",engine:.ltx25)
      clip.id = UUID(uuidString:segment["clip_id"] as! String)!
      clip.prompt = segment["prompt"] as! String;clip.soundscape = "";clip.music = "N/A"
      clip.duration = (segment["duration_seconds"] as! NSNumber).doubleValue;clip.seed = segment["seed"] as! Int
      clip.generationWidth = config["width"] as! Int;clip.generationHeight = config["height"] as! Int
      clip.profileID = profileURL.path;clip.generationSelection = .init(task:"a2v")
      clip.continuity = .init(mode:index == 0 ? "independent":"scene",sourceClipID:index == 0 ? nil:project.clips[index-1].id,boundaryImagePolicy:scene["boundary_image_policy"] as! String,sceneDecodeMode:scene["decode_mode"] as! String)
      var driver = Attachment(assetID:sourceAssets[audio["path"] as! String]!.id,role:.audioDriver)
      driver.id = index == 0 ? UUID(uuidString:audio["id"] as! String)!:UUID()
      driver.audioSourceStart = (audio["source_start_seconds"] as! NSNumber).doubleValue+elapsed
      driver.audioSourceDuration = clip.duration;clip.attachments = [driver]
      for input in inputs where input["kind"] as? String == "image" {
        let frame = input["frame_index"] as! Int,start = Int((elapsed*fps).rounded())
        guard frame >= start,frame < start+Int((clip.duration*fps).rounded()) else { continue }
        var attachment = Attachment(assetID:sourceAssets[input["path"] as! String]!.id,role:.keyframe)
        attachment.time = Double(frame-start)/fps;attachment.strength = (input["strength"] as! NSNumber).doubleValue
        clip.attachments.append(attachment)
      }
      project.clips.append(clip);elapsed += clip.duration
    }
    let runtime:[String:Any] = ["profilesDirectory":profiles.path,"ffmpegPath":options["ffmpeg"]!]
    let request:[String:Any] = ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "globalAssets":[],"clipID":project.clips[0].id.uuidString,"runtime":runtime]
    let composed = try NativeLTXPreparation.compose(request:request),actual = composed["recipe"] as! [String:Any]
    // Compare the complete numerical/media workload; editor attachment UUIDs and
    // canonical order are transport metadata, not a sampler/conditioning change.
    func numerical(_ value:[String:Any]) throws -> Data {
      var result = value,conditioning = result["conditioning"] as! [String:Any]
      conditioning["inputs"] = (conditioning["inputs"] as! [[String:Any]]).map { input -> [String:Any] in
        var input = input;input.removeValue(forKey:"id");return input
      }.sorted { String(describing:$0["frame_index"] ?? -1) < String(describing:$1["frame_index"] ?? -1) }
      result["conditioning"] = conditioning
      var scene = result["scene"] as! [String:Any]
      scene["decode_mode"] = scene["decode_mode"] ?? "single"
      result["scene"] = scene
      return try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys,.withoutEscapingSlashes])
    }
    try write(actual,root.appendingPathComponent("actual-composed.json"))
    XCTAssertEqual(try numerical(actual),try numerical(original),"Do not change the proven scene workload to make Studio qualification easier.")
    guard try numerical(actual) == numerical(original) else { throw StudioError.invalid("Scene workload differs; stop before inference.") }
    let prepared = try NativeLTXPreparation.prepare(request:request,destination:root.appendingPathComponent("Expected-inputs"))
    let expected = URL(fileURLWithPath:prepared["recipePath"] as! String)
    let editor = root.appendingPathComponent("editor.json");try write(request,editor)
    let worker = URL(fileURLWithPath:options["worker"]!)
    let evidence:[String:Any] = ["recipe":profileURL.path,"recipeSHA256":try NativeHeadlessJob.fileHash(profileURL),
      "editorRequest":editor.path,"editorRequestSHA256":try NativeHeadlessJob.fileHash(editor),
      "expectedPreparedRecipe":expected.path,"expectedPreparedRecipeSHA256":try NativeHeadlessJob.fileHash(expected),
      "worker":worker.path,"workerSHA256":try NativeHeadlessJob.fileHash(worker),"output":root.appendingPathComponent("Studio-lifecycle").path,
      "ffmpeg":options["ffmpeg"]!,"decodeMode":scene["decode_mode"]!,"clipCount":segments.count,
      "frames":Int((elapsed*fps).rounded()),"audioChannels":2,
      "sources":try project.assets.map { ["path":$0.path,"sha256":try NativeHeadlessJob.fileHash(URL(fileURLWithPath:$0.path))] }]
    try write(evidence,root.appendingPathComponent("lifecycle.json"))
    try write(["inferenceExecuted":false,"originalNumericalWorkloadUnchanged":true,
      "allowedTransportDifferences":["new editor attachment UUIDs","canonical input ordering","omitted explicit single decode default"],"anchors":inputs.compactMap { $0["frame_index"] as? Int }],root.appendingPathComponent("fixture-parity.json"))
  }

  @MainActor func testInstalledMovieUpscaleStudioLifecycleWithoutPython() async throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_MOVIE_UPSCALE_LIFECYCLE"] else {
      throw XCTSkip("Explicit installed movie upscale Studio lifecycle; no inference by default.")
    }
    let options = try object(URL(fileURLWithPath:manifest)) as! [String:String]
    let original = try object(URL(fileURLWithPath:options["recipe"]!))
    let root = URL(fileURLWithPath:options["output"]!),profiles = root.appendingPathComponent("Profiles")
    guard !FileManager.default.fileExists(atPath:root.path) else { throw StudioError.invalid("Fresh lifecycle output required.") }
    try FileManager.default.createDirectory(at:profiles,withIntermediateDirectories:true)
    let profile = profiles.appendingPathComponent("movie.json")
    try Data(contentsOf:URL(fileURLWithPath:options["profile"]!)).write(to:profile)
    let source = original["source"] as! [String:Any],audio = original["audio_source"] as! [String:Any]
    var project = try ProjectStorage.read(URL(fileURLWithPath:options["pendingProject"]!))
    let originalClip = project.clips[0]
    var movie = MediaAsset(name:"Source movie",kind:.video,path:source["path"] as! String)
    movie.duration = (source["duration_seconds"] as! NSNumber).doubleValue;movie.fps = (source["fps"] as! NSNumber).doubleValue
    let sound = MediaAsset(name:"Original PCM",kind:.audio,path:audio["path"] as! String)
    project.assets = [movie,sound]
    var reference = Attachment(assetID:movie.id,role:.reference)
    reference.sourceStartSeconds = (source["start_seconds"] as! NSNumber).doubleValue
    reference.sourceDurationSeconds = Double(source["frames"] as! Int)/movie.fps
    var sidecar = Attachment(assetID:sound.id,role:.audioDriver)
    sidecar.audioSourceStart = (audio["start_seconds"] as! NSNumber).doubleValue
    sidecar.audioSourceDuration = (audio["duration_seconds"] as! NSNumber).doubleValue
    project.clips[0].attachments = [reference,sidecar];project.clips[0].profileID = profile.path
    let worker = URL(fileURLWithPath:options["worker"]!),workerSHA = try NativeHeadlessJob.fileHash(worker)
    let store = StudioStore(dataDirectory:root,restoreSession:false)
    store.project = project;store.selectedClipID = originalClip.id
    store.runtime = RuntimeSettings(root:"/unavailable",pythonPath:"/unavailable/python",profilesDirectory:profiles.path)
    store.runtime.nativeLTX25Enabled = true;store.runtime.ltx25WorkerPath = worker.path;store.runtime.ffmpegPath = options["ffmpeg"]!
    await store.reloadProfiles();await store.prepareSelected()
    guard store.error == nil else { throw StudioError.invalid(store.error!) }
    let prepared = URL(fileURLWithPath:try XCTUnwrap(store.preparedRecipe))
    let wrapper = try object(prepared),actual = wrapper["movie_upscale"] as! [String:Any]
    var expected = original,expectedSource = source
    expected["output_directory"] = actual["output_directory"]
    expectedSource["rgb_path"] = (actual["source"] as! [String:Any])["rgb_path"];expected["source"] = expectedSource
    func canonical(_ value:Any) throws -> Data { try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes]) }
    guard try canonical(actual) == canonical(expected) else { throw StudioError.invalid("Movie preparation changed the frozen request; stop before inference.") }
    let preparedSHA = try NativeHeadlessJob.fileHash(prepared)
    if ProcessInfo.processInfo.environment["WEETODD_MOVIE_UPSCALE_PREPARE_ONLY"] == "1" {
      try write(["inferenceExecuted":false,"prepared":prepared.path,"frozenWorkloadUnchanged":true],root.appendingPathComponent("preparation.json"));return
    }
    var previews = Set<Int>(),progress = Set<Double>()
    let preview = store.bridge.$livePreview.sink { if let revision = $0?.previewRevision { previews.insert(revision) } }
    let status = store.bridge.$fraction.sink { if $0 > 0 && $0 < 1 { progress.insert($0) } }
    defer { preview.cancel();status.cancel() }
    let start = Date();await store.renderPrepared()
    guard store.error == nil else { throw StudioError.invalid(store.error!) }
    let accepted = try XCTUnwrap(store.selectedClip),version = try XCTUnwrap(accepted.versions.last)
    XCTAssertEqual(accepted.id,originalClip.id);XCTAssertEqual(accepted.duration,originalClip.duration)
    XCTAssertEqual(accepted.sourcePath,version.path);XCTAssertEqual(version.seed,originalClip.seed)
    XCTAssertEqual(version.recipePath,prepared.path);XCTAssertEqual(accepted.renderedSignature,store.signature(for:accepted))
    XCTAssertFalse(previews.isEmpty);XCTAssertFalse(progress.isEmpty)
    let saved = root.appendingPathComponent("accepted.weetodd");try ProjectStorage.write(store.project,to:saved)
    let reopened = StudioStore(dataDirectory:root,restoreSession:false);reopened.load(saved)
    XCTAssertNil(reopened.error);XCTAssertEqual(reopened.project,store.project)
    XCTAssertEqual(try NativeHeadlessJob.fileHash(worker),workerSHA);XCTAssertEqual(try NativeHeadlessJob.fileHash(prepared),preparedSHA)
    let inspected = try await StudioStore.inspectNativeMovie(version.path)
    XCTAssertEqual(inspected["width"] as? Int,2*(source["width"] as! Int))
    XCTAssertEqual(inspected["height"] as? Int,2*(source["height"] as! Int))
    XCTAssertEqual(inspected["fps"] as? Double,movie.fps)
    XCTAssertEqual(try XCTUnwrap(inspected["duration"] as? Double),originalClip.duration,accuracy:1/movie.fps)
    try write(["proofLevel":"new-studio-generation-preview-acceptance-save-reopen","inferenceExecuted":true,
      "pythonInference":false,"worker":worker.path,"workerSHA256":workerSHA,"prepared":prepared.path,"preparedSHA256":preparedSHA,
      "video":version.path,"acceptedProject":saved.path,"previews":previews.count,"progressUpdates":progress.count,
      "seconds":Date().timeIntervalSince(start),"frozenWorkloadUnchanged":true],root.appendingPathComponent("qualification.json"))
  }
}
