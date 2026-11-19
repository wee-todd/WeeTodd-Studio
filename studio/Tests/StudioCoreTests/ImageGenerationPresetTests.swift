import Foundation
import ImageLayout
import XCTest
@testable import StudioCore

final class ImageGenerationPresetTests: XCTestCase {
  func testReapplyingNativeStackPreservesExactMemberAndAssetIdentity() throws {
    for provider in [ImageExecutionProvider.nativeMOSS, .nativeH3] {
      var library = library(), value = draft(provider)
      if provider == .nativeMOSS { value.mossImage?.libraryDirectory = "/models" } else { value.h3Image?.imageVAEPath = "/models/vae.safetensors" }
      let family = provider == .nativeMOSS ? "moss_image" : "h3"
      let ref = GenerationLoRARef(id: "style", backend: "native", family: family)
      library.loraEntries = [.init(reference: ref, name: "Style", compatibleFamilies: [family], availability: .installed, binding: .nativeFile(path: "/style.safetensors"))]
      var asset = MediaAsset(name: "My style name", kind: .lora, path: "/style.safetensors", scope: .global); asset.loraModel = LoRAModel(rawValue: family)
      let member = LoRAMember(asset: asset, strength: 0.7)
      if provider == .nativeMOSS { value.mossImage?.loras = [member] } else { value.h3Image?.loras = [member] }
      let snapshot = try ImagePresetRecipe.capture(from: value, resolvedLibrary: library)
      XCTAssertEqual(try ImagePresetRecipe.candidate(applying: snapshot, to: value, resolvedLibrary: library), value)
    }
  }
  func testDrawThingsKeepsItsSupportedGuidanceAndZeroShiftRange() throws {
    var value = draft(.drawThings); value.guidance = 30; value.shift = 0
    XCTAssertNoThrow(try ImagePresetRecipe.capture(from: value, resolvedLibrary: library()))
  }
  func testMOSSRawSkipLayerIndicesAreValidated() throws {
    var value = draft(.nativeMOSS); value.mossImage?.skipLayers = [42]
    XCTAssertThrowsError(try ImagePresetRecipe.capture(from: value, resolvedLibrary: library()))
  }
  func library() -> GenerationLibrarySnapshot {
    let entries: [(String, String, ResolvedGenerationModel)] = [
      ("nativeMOSS", "moss_image", .nativeMOSS(transformer: "/models/moss", shared: "/models/shared", library: "/models")),
      ("nativeH3", "h3", .nativeH3Image(profileID: "/models/h3.json", imageVAE: "/models/vae.safetensors")),
      ("nativeMLX", "qwen_image", .nativeQwen(manifest: "/models/qwen.json")),
      ("drawThings", "flux", .drawThings(connectionID: "local", modelID: "flux"))]
    return .init(models: entries.map { backend, family, binding in
      .init(reference: .init(id: backend, backend: backend, family: family, variant: backend == "nativeMOSS" ? "sft" : nil,
        precision: backend == "nativeMOSS" ? "bf16" : nil), name: backend, workspaces: [.image], tasks: ["image"],
        supportsLoRAs: backend != "nativeMLX", availability: .installed, binding: binding)
    })
  }
  func draft(_ provider: ImageExecutionProvider) -> DrawThingsImageDraft {
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    draft.selectProvider(provider); draft.prompt = "Keep my subject and lettering"; draft.width = 1536; draft.height = 1024
    switch provider {
    case .nativeMOSS: draft.mossImage?.transformerDirectory = "/models/moss"; draft.mossImage?.sharedDirectory = "/models/shared"
    case .nativeH3: draft.modelID = "/models/h3.json"
    case .nativeMLX: draft.nativeImage?.manifestPath = "/models/qwen.json"
    case .drawThings: draft.profileID = "local"; draft.modelID = "flux"
    }
    return draft
  }
  func testAllFourProvidersRoundTripOnlyOwnedSettingsAndPreserveCreativeInputs() throws {
    for provider in ImageExecutionProvider.allCases {
      let original = draft(provider), snapshot = try ImagePresetRecipe.capture(from: original, resolvedLibrary: library())
      let data = try JSONEncoder().encode(snapshot), text = String(decoding: data, as: UTF8.self)
      XCTAssertFalse(text.contains(original.prompt)); XCTAssertFalse(text.contains("/models/")); XCTAssertNil(snapshot.fields["width"])
      var target = original; target.prompt = "A new creative prompt"; target.width = 1024; target.height = 1536
      target.name = "Current project name"; target.seed = 42
      target.canvas = .init(path: "/source.png"); target.canvas?.enabled = false
      target.moodboard = [.init(path: "/reference.png")]; target.moodboard[0].enabled = false
      let candidate = try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library())
      XCTAssertEqual(candidate.prompt, target.prompt); XCTAssertEqual(candidate.destination, target.destination)
      XCTAssertEqual(candidate.name, target.name); XCTAssertEqual(candidate.width, target.width); XCTAssertEqual(candidate.height, target.height)
      XCTAssertEqual(candidate.canvas, target.canvas); XCTAssertEqual(candidate.moodboard, target.moodboard)
      XCTAssertEqual(try ImagePresetRecipe.capture(from: candidate, resolvedLibrary: library()), snapshot)
    }
  }
  func testMOSSBoxesGuideRegionalControlAndRuntimePreferencesAreNotPresetOwned() throws {
    var target = draft(.nativeMOSS)
    target.mossLayout = .init(width: target.width, height: target.height, caption: "Layout", regions: [
      .init(type: .text, caption: "Blue letters", text: "MOVE ME", bounds: .init(x: 0.2, y: 0.1, width: 0.6, height: 0.2))])
    target.mossLayout?.guideImagePath = "/source.png"; target.mossImage?.regionalLayout = true; target.mossImage?.weightCacheGB = 8
    var snapshot = try ImagePresetRecipe.capture(from: target, resolvedLibrary: library())
    snapshot.fields["steps"] = .integer(6); snapshot.fields["guidance"] = .number(1)
    let candidate = try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library())
    XCTAssertEqual(candidate.mossLayout, target.mossLayout); XCTAssertEqual(candidate.mossImage?.regionalLayout, true)
    XCTAssertEqual(candidate.mossImage?.weightCacheGB, 8); XCTAssertEqual(candidate.steps, 6)
  }
  func testSwitchingAwayFromMOSSAndBackPreservesEditableLayoutAndGuide() throws {
    var original = draft(.nativeMOSS)
    original.mossLayout = .init(width: original.width, height: original.height, caption: "Layout", regions: [
      .init(type: .text, caption: "Blue letters", text: "MOVE ME", bounds: .init(x: 0.2, y: 0.1, width: 0.6, height: 0.2)),
      .init(type: .object, caption: "Person holding a red umbrella", bounds: .init(x: 0.1, y: 0.3, width: 0.4, height: 0.6))])
    original.mossLayout?.guideImagePath = "/source.png"
    original.mossImage?.regionalLayout = true
    original.mossImage?.weightCacheGB = 8
    let moss = try ImagePresetRecipe.capture(from: original, resolvedLibrary: library())
    for provider in [ImageExecutionProvider.nativeMLX, .nativeH3, .drawThings] {
      let other = try ImagePresetRecipe.capture(from: draft(provider), resolvedLibrary: library())
      let switched = try ImagePresetRecipe.candidate(applying: other, to: original, resolvedLibrary: library())
      XCTAssertEqual(switched.executionProvider, provider)
      XCTAssertEqual(switched.mossLayout, original.mossLayout)
      XCTAssertEqual(switched.mossImage, original.mossImage)
      XCTAssertEqual(switched.prompt, original.prompt)
      XCTAssertEqual(switched.width, original.width); XCTAssertEqual(switched.height, original.height)
      // The dormant layout also survives saving/reopening while another model is selected.
      let reopened = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(switched))
      let restored = try ImagePresetRecipe.candidate(applying: moss, to: reopened, resolvedLibrary: library())
      XCTAssertEqual(restored.executionProvider, .nativeMOSS)
      XCTAssertEqual(restored.mossLayout, original.mossLayout)
      XCTAssertEqual(restored.mossImage?.regionalLayout, true)
      XCTAssertEqual(restored.mossImage?.weightCacheGB, 8)
      XCTAssertEqual(restored.prompt, original.prompt); XCTAssertEqual(restored.destination, original.destination)
      XCTAssertEqual(restored.width, original.width); XCTAssertEqual(restored.height, original.height)
      XCTAssertEqual(try ImagePresetRecipe.capture(from: restored, resolvedLibrary: library()), moss)
    }
  }
  func testModelOnlySwitchWithStoredMOSSLayoutKeepsSourceCanvasDimensions() throws {
    var original = draft(.nativeMOSS); original.width = 1872; original.height = 1056
    original.mossLayout = .init(width: original.width, height: original.height, caption: "Saved layout", regions: [])
    original.mossImage?.regionalLayout = true
    let choice = GenerationPresetSnapshot(context: .init(workspace: .image, task: "image"),
      model: library().modelEntries[1].reference, fields: [:])
    XCTAssertThrowsError(try ImagePresetRecipe.candidate(applying: choice, to: original, resolvedLibrary: library()))
    let switched = try ImagePresetRecipe.selectingModel(XCTUnwrap(choice.model), to: original, resolvedLibrary: library())
    XCTAssertEqual(switched.executionProvider, .nativeH3)
    XCTAssertEqual(switched.width, 1872); XCTAssertEqual(switched.height, 1056)
    XCTAssertEqual(switched.mossLayout, original.mossLayout)
    XCTAssertEqual(switched.prompt, original.prompt)
  }
  func testOmittedFieldsPreserveButUnknownFieldsAndWrongContextsReject() throws {
    var target = draft(.nativeMOSS); target.steps = 30
    var snapshot = try ImagePresetRecipe.capture(from: target, resolvedLibrary: library()); snapshot.fields = [:]; snapshot.adapters = nil
    XCTAssertEqual(try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library()).steps, 30)
    snapshot.fields["width"] = .integer(512)
    XCTAssertThrowsError(try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library()))
    snapshot.fields = [:]; snapshot.context.task = "t2v"
    XCTAssertThrowsError(try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library()))
  }
  func testUnavailableModelAndQwenAdapterStackCannotApply() throws {
    let target = draft(.nativeMLX); var snapshot = try ImagePresetRecipe.capture(from: target, resolvedLibrary: library())
    snapshot.adapters = [.init(reference: .init(id: "a", backend: "native", family: "moss_image"))]
    XCTAssertThrowsError(try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library()))
    snapshot.adapters = []; var missing = library(); missing.modelEntries[2].availability = .driveDisconnected
    XCTAssertThrowsError(try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: missing))
  }
  func testEveryReleasedMOSSSamplerAndScheduleRoundTrips() throws {
    for sampler in ["euler", "er_sde"] { for schedule in ["release", "simple"] {
      var target = draft(.nativeMOSS); target.mossImage?.sampler = sampler; target.mossImage?.schedule = schedule
      let snapshot = try ImagePresetRecipe.capture(from: target, resolvedLibrary: library())
      let applied = try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library())
      XCTAssertEqual(applied.mossImage?.sampler, sampler); XCTAssertEqual(applied.mossImage?.schedule, schedule)
    } }
    var target = draft(.nativeMOSS); target.mossImage?.sampler = "heun"
    XCTAssertThrowsError(try ImagePresetRecipe.capture(from: target, resolvedLibrary: library()))
  }
  func testH3AdapterSettingsRoundTripAndAppearInUpdateReviewWithoutTriggerInsertion() throws {
    var library = library(), target = draft(.nativeH3)
    let ref = GenerationLoRARef(id: "style", backend: "native", family: "h3")
    library.loraEntries = [.init(reference: ref, name: "Style", compatibleFamilies: ["h3"], availability: .installed,
      binding: .nativeFile(path: "/models/style.safetensors", profile: "auto"))]
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/models/style.safetensors", scope: .global)
    asset.loraModel = .h3; asset.loraProfile = "auto"
    target.h3Image?.loras = [.init(asset: asset, strength: 0.8, enabled: false, h3Settings: .init(startAfterEvaluations: 3))]
    let snapshot = try ImagePresetRecipe.capture(from: target, resolvedLibrary: library)
    let applied = try ImagePresetRecipe.candidate(applying: snapshot, to: target, resolvedLibrary: library)
    XCTAssertEqual(applied.prompt, target.prompt); XCTAssertEqual(applied.h3Image?.loras.first?.h3Settings, target.h3Image?.loras.first?.h3Settings)
    var older = snapshot; older.adapters![0].fields.removeValue(forKey: "h3Settings")
    XCTAssertEqual(try ImagePresetRecipe.candidate(applying: older, to: target, resolvedLibrary: library).h3Image?.loras.first?.h3Settings, target.h3Image?.loras.first?.h3Settings)
    older.adapters![0].fields["h3Settings"] = .automatic
    XCTAssertNil(try ImagePresetRecipe.candidate(applying: older, to: target, resolvedLibrary: library).h3Image?.loras.first?.h3Settings)
    var changed = snapshot
    changed.adapters![0].fields["h3Settings"] = try .encoded(H3LoRASettings(startAfterEvaluations: 5))
    XCTAssertTrue(GenerationPresetDiff.changes(from: snapshot, to: changed).contains { $0.path == "adapters.style.h3Settings.startAfterEvaluations" })
  }
}
