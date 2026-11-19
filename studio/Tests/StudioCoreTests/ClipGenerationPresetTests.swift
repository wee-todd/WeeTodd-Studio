import Foundation
import XCTest
@testable import StudioCore

final class ClipGenerationPresetTests: XCTestCase {
  func testSceneRecipeAppliesAcrossMembersWithDifferentInputTasks() throws {
    var source = project(.ltx25), second = source.clips[0]
    second.id = UUID(); second.generationSelection?.task = "i2v"; second.continuity = .init(mode: "scene", sourceClipID: source.clips[0].id)
    source.clips.append(second)
    let snapshot = try ClipPresetRecipe.capture(from: source.clips[0], project: source, resolvedLibrary: library())
    let result = try ClipPresetRecipe.candidate(applying: snapshot, to: second, project: source, resolvedLibrary: library())
    XCTAssertEqual(result.clip.generationSelection?.task, "i2v"); XCTAssertEqual(result.clip.continuity, second.continuity)
  }
  func testSpecializedLTXAdapterRoundTripsAsLogicalIdentityWithoutAPath() throws {
    var source = project(.ltx25), models = library()
    source.clips[0].generationSelection?.ltx25AdapterBinding = .init(kind: .dfr, path: "/models/detail.safetensors")
    source.clips[0].generationSelection?.ltx25DFR = .init(experimentalEnabled: true)
    let ref = GenerationLoRARef(id: "detail", backend: "native", family: "ltx25")
    models.loraEntries = [.init(reference: ref, name: "DFR", compatibleFamilies: ["ltx25"], availability: .installed,
      binding: .nativeFile(path: "/models/detail.safetensors", profile: "dfr"))]
    let snapshot = try ClipPresetRecipe.capture(from: source.clips[0], project: source, resolvedLibrary: models)
    XCTAssertNotNil(snapshot.fields["specializedAdapter"])
    XCTAssertFalse(String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self).contains("/models/"))
    var target = source.clips[0]; target.generationSelection?.ltx25AdapterBinding = nil; target.generationSelection?.ltx25DFR = nil
    let result = try ClipPresetRecipe.candidate(applying: snapshot, to: target, project: source, resolvedLibrary: models)
    XCTAssertEqual(result.clip.generationSelection?.ltx25AdapterBinding, source.clips[0].generationSelection?.ltx25AdapterBinding)
  }
  func testRelayMoviePresetKeepsMethodScaleSourceAndCanvas() throws {
    var project = StudioProject()
    var clip = Clip(name: "Source upscale", engine: .ltx25)
    clip.generationSelection = .init(task: "video_upscale")
    clip.generationSelection?.movieUpscaleMethod = .relayVSR
    let ref = GenerationModelRef(id: "relay", backend: "restoration", family: "relayVSR", variant: "x2")
    project.generationLibraryBindings = ["clip-model:" + clip.id.uuidString: ref]
    let library = GenerationLibrarySnapshot(models: [.init(reference: ref, name: "RelayVSR 2×", workspaces: [.clip], tasks: ["video_upscale"], supportsLoRAs: false,
      availability: .installed, binding: .restoration(tool: "relayVSR", components: ["modelDirectory": "/models/relay", "scale": "2"]))])
    let snapshot = try ClipPresetRecipe.capture(from: clip, project: project, resolvedLibrary: library)
    let result = try ClipPresetRecipe.candidate(applying: snapshot, to: clip, project: project, resolvedLibrary: library)
    XCTAssertEqual(result.clip, clip); XCTAssertTrue(result.assets.isEmpty)
    var mismatch = snapshot; mismatch.fields["scale"] = .integer(4)
    XCTAssertThrowsError(try ClipPresetRecipe.candidate(applying: mismatch, to: clip, project: project, resolvedLibrary: library))
  }
  func library() -> GenerationLibrarySnapshot {
    .init(models: ["h3", "ltx23", "ltx25", "drawThings"].map { engine in
      .init(reference: .init(id: engine, backend: engine, family: engine), name: engine, workspaces: [.clip], tasks: ["t2v", "i2v", "fflf", "ref2va", "video_upscale"],
        availability: .installed, binding: engine == "drawThings" ? .drawThings(connectionID: "local", modelID: "video") : .nativeVideo(profileID: "/models/" + engine + ".json"))
    })
  }
  func project(_ engine: Engine) -> StudioProject {
    var project = StudioProject(), clip = Clip(name: "My shot", engine: engine)
    clip.profileID = "/models/" + engine.rawValue + ".json"; clip.prompt = "Keep my actors"; clip.soundscape = "My sound"; clip.music = "My music"
    clip.generationWidth = 1536; clip.generationHeight = 1024; clip.duration = 7; clip.sourceIn = 2; clip.sourcePath = "/source.mov"
    clip.settingsOverride = MovieSettings(); clip.settingsOverride?.fps = 24
    clip.generationSelection = .init(task: "t2v"); clip.generationSelection?.steps = engine == .ltx25 ? nil : 8
    if engine == .drawThings { clip.drawThings = .init(profileID: "local", modelID: "video", modelFamily: "drawThings", configuration: ["steps": .number(8), "fps": .number(24), "strength": .number(0.4), "width": .number(1536)]) }
    project.clips = [clip]; return project
  }
  func testEveryTimelineBackendAppliesOwnedRecipeWithoutEditingProjectInputs() throws {
    for engine in [Engine.h3, .ltx23, .ltx25, .drawThings] {
      let source = project(engine), snapshot = try ClipPresetRecipe.capture(from: source.clips[0], project: source, resolvedLibrary: library())
      var target = source; target.clips[0].prompt = "New actors"; target.clips[0].duration = 9; target.clips[0].sourceIn = 3
      target.clips[0].generationWidth = 1024; target.clips[0].generationHeight = 1536
      let application = try ClipPresetRecipe.candidate(applying: snapshot, to: target.clips[0], project: target, resolvedLibrary: library())
      let candidate = application.clip
      XCTAssertEqual(candidate.prompt, target.clips[0].prompt); XCTAssertEqual(candidate.sourcePath, target.clips[0].sourcePath)
      XCTAssertEqual(candidate.sourceIn, 3); XCTAssertEqual(candidate.duration, 9)
      XCTAssertEqual(candidate.generationWidth, 1024); XCTAssertEqual(candidate.generationHeight, 1536)
      XCTAssertEqual(candidate.settingsOverride, target.clips[0].settingsOverride); XCTAssertEqual(candidate.soundscape, target.clips[0].soundscape)
      XCTAssertEqual(candidate.music, target.clips[0].music); XCTAssertEqual(candidate.versions, target.clips[0].versions)
      XCTAssertEqual(candidate.drawThings?.configuration["strength"], target.clips[0].drawThings?.configuration["strength"])
      XCTAssertEqual(try ClipPresetRecipe.capture(from: candidate, project: target, resolvedLibrary: library()), snapshot)
      let encoded = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
      XCTAssertFalse(encoded.contains("/source.mov")); XCTAssertFalse(encoded.contains("/models/")); XCTAssertFalse(encoded.contains("Keep my actors"))
    }
  }
  func testTaskClockAndUnknownFieldChangesRejectRatherThanCoercing() throws {
    let source = project(.ltx25), snapshot = try ClipPresetRecipe.capture(from: source.clips[0], project: source, resolvedLibrary: library())
    var target = source; target.clips[0].generationSelection?.task = "fflf"
    XCTAssertThrowsError(try ClipPresetRecipe.candidate(applying: snapshot, to: target.clips[0], project: target, resolvedLibrary: library()))
    var bad = snapshot; bad.fields["duration"] = .number(3)
    XCTAssertThrowsError(try ClipPresetRecipe.candidate(applying: bad, to: source.clips[0], project: source, resolvedLibrary: library()))
    let h3 = project(.h3), h3Preset = try ClipPresetRecipe.capture(from: h3.clips[0], project: h3, resolvedLibrary: library())
    target = source; target.clips[0].settingsOverride?.fps = 30
    XCTAssertThrowsError(try ClipPresetRecipe.candidate(applying: h3Preset, to: target.clips[0], project: target, resolvedLibrary: library()))
    XCTAssertEqual(target.clips[0].settingsOverride?.fps, 30)
  }
  func testNativeLoRAApplicationReturnsOwnedAssetsAndNeverInsertsPromptWords() throws {
    var library = library(), source = project(.ltx25)
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/models/style.safetensors", scope: .project); asset.loraModel = .ltx25
    source.assets = [asset]; var attachment = Attachment(assetID: asset.id, role: .lora); attachment.strength = 0.75; attachment.enabled = false
    source.clips[0].attachments = [attachment]
    let ref = GenerationLoRARef(id: "style", backend: "native", family: "ltx25")
    library.loraEntries = [.init(reference: ref, name: "Style", compatibleFamilies: ["ltx25"], availability: .installed, binding: .nativeFile(path: asset.path))]
    let snapshot = try ClipPresetRecipe.capture(from: source.clips[0], project: source, resolvedLibrary: library)
    var target = source; target.assets = []; target.clips[0].attachments = []
    let application = try ClipPresetRecipe.candidate(applying: snapshot, to: target.clips[0], project: target, resolvedLibrary: library)
    XCTAssertEqual(application.clip.prompt, target.clips[0].prompt); XCTAssertEqual(application.assets.count, 1)
    XCTAssertEqual(application.clip.attachments.first?.assetID, application.assets.first?.id)
    XCTAssertEqual(application.clip.attachments.first?.strength, 0.75); XCTAssertEqual(application.clip.attachments.first?.isEnabled, false)
  }
}
