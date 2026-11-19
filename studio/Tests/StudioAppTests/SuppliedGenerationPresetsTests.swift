import Foundation
import InferenceContracts
import StudioCore
import XCTest
@testable import WeeToddStudio

final class SuppliedGenerationPresetsTests: XCTestCase {
  func testGuidedSuppliedPresetCarriesNativePipelineIntent() throws {
    for mode in ["guided", "guided_hq"] {
      let raw: [String: Any] = ["pipelineMode": mode, "supportedTasks": ["t2v"], "presets": [], "controls": ["stepsEditable": true, "refinementStepsEditable": false, "cfgEditable": true, "shiftEditable": true, "stepsExplanation": "", "cfgExplanation": "", "shiftExplanation": ""]]
      let descriptor = try JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: raw))
      let entry = GenerationModelEntry(reference: .init(id: mode, backend: "ltx25", family: "ltx25"), name: mode, workspaces: [.clip], tasks: ["t2v"], availability: .installed, binding: .nativeVideo(profileID: "/models/dev.json"), generation: descriptor)
      let presets = SuppliedGenerationPresets.records(library: .init(models: [entry]))
      let snapshot = try XCTUnwrap(presets.first?.snapshot)
      guard case .object(let selection) = snapshot.fields["selection"], case .object(let guidance) = selection["ltx25Guidance"] else { return XCTFail("Guided default lost its native pipeline intent") }
      XCTAssertEqual(guidance["mode"], .string(mode)); XCTAssertEqual(guidance["experimentalEnabled"], .bool(true))
    }
  }
  private func library() -> GenerationLibrarySnapshot {
    var models: [GenerationModelEntry] = []
    func add(_ id: String, backend: String, family: String, variant: String? = nil, precision: String? = nil, workspaces: [GenerationWorkspaceKind], tasks: [String], binding: ResolvedGenerationModel) {
      models.append(.init(reference: .init(id: id, backend: backend, family: family, variant: variant, precision: precision), name: id,
        workspaces: workspaces, tasks: tasks, availability: .installed, binding: binding))
    }
    for variant in ["base", "sft"] {
      add("moss-" + variant, backend: "nativeMOSS", family: "moss_image", variant: variant, precision: "bf16", workspaces: [.image], tasks: ["image"], binding: .nativeMOSS(transformer: "/models/" + variant, shared: "/models/shared", library: "/models"))
    }
    add("h3-image", backend: "nativeH3", family: "h3", workspaces: [.image], tasks: ["image"], binding: .nativeH3Image(profileID: "/models/h3.json", imageVAE: ""))
    add("qwen-image", backend: "nativeMLX", family: "qwen_image", workspaces: [.image], tasks: ["image"], binding: .nativeQwen(manifest: "/models/qwen.json"))
    add("dt-image", backend: "drawThings", family: "flux", workspaces: [.image], tasks: ["image"], binding: .drawThings(connectionID: "local", modelID: "image"))
    for engine in ["h3", "ltx23", "ltx25"] {
      add(engine, backend: engine, family: engine, workspaces: [.clip], tasks: ["t2v", "i2v", "fflf", "a2v", "ref2va", "control", "extension"] + (engine == "ltx25" ? ["video_upscale"] : []), binding: .nativeVideo(profileID: "/models/" + engine + ".json"))
    }
    add("dt-video", backend: "drawThings", family: "ltx", workspaces: [.clip], tasks: ["t2v"], binding: .drawThings(connectionID: "local", modelID: "video"))
    add("ripple", backend: "ripple", family: "ltx25", variant: "distilled", workspaces: [.ripple], tasks: ["ripple"], binding: .ripple(profileID: "/models/ltx.json", adapterPath: "/models/ripple.safetensors"))
    for variant in ["turbo", "base"] {
      add("flash-" + variant, backend: "restoration", family: "flashVSR", variant: variant, workspaces: [.restoration, .finishing], tasks: ["flashVSR"], binding: .restoration(tool: "flashVSR", components: ["modelRoot": "/models/flash", "variant": variant, "decoder": "tiny"]))
    }
    add("h3-restore", backend: "restoration", family: "h3Regenerate", workspaces: [.restoration, .finishing], tasks: ["h3Regenerate"], binding: .restoration(tool: "h3Regenerate", components: ["baseProfilePath": "/models/h3.json"]))
    for scale in [2, 4] {
      add("relay-\(scale)", backend: "restoration", family: "relayVSR", variant: "x\(scale)", workspaces: [.restoration, .clip], tasks: ["relayVSR", "video_upscale"], binding: .restoration(tool: "relayVSR", components: ["modelDirectory": "/models/relay/x\(scale)", "scale": String(scale)]))
    }
    for precision in ["bf16", "int8ConvRot"] {
      add("reca-" + precision, backend: "restoration", family: "reCaVSR", precision: precision, workspaces: [.restoration, .clip], tasks: ["reCaVSR", "video_upscale"], binding: .restoration(tool: "reCaVSR", components: ["modelDirectory": "/models/reca", "precision": precision]))
    }
    add("pixel", backend: "restoration", family: "pixelCQ", workspaces: [.finishing], tasks: ["pixelCQ"], binding: .restoration(tool: "pixelCQ", components: ["profileID": "/models/ltx.json", "pixelAdapterPath": "/models/pixel.safetensors", "cqAdapterPath": "/models/cq.safetensors"]))
    let fast = GenerationLoRAEntry(reference: .init(id: "fast", backend: "native", family: "moss_image"), name: "Fast", compatibleFamilies: ["moss_image"], compatibleVariants: ["sft"], availability: .installed, binding: .nativeFile(path: "/models/fast.safetensors", profile: "moss_sft_4step_v2"))
    return .init(models: models, loras: [fast])
  }
  func testEverySuppliedRecipeCanApplyThroughItsCodecAndEveryWorkspaceHasDefaults() throws {
    let library = library(), presets = SuppliedGenerationPresets.records(library: library)
    XCTAssertTrue(presets.contains { $0.snapshot.context.workspace == .ripple })
    XCTAssertTrue(presets.contains { $0.snapshot.context.task == "video_upscale" && $0.snapshot.model?.backend == "ltx25" })
    XCTAssertTrue(presets.contains { $0.snapshot.context.workspace == .finishing && $0.name.contains("flash-base") })
    XCTAssertTrue(presets.contains { $0.name.contains("H3 → Flash") })
    XCTAssertTrue(presets.contains { $0.name.contains("Pixel + CQ → Flash") })
    XCTAssertEqual(Set(presets.map { $0.snapshot.context.workspace }), Set(GenerationWorkspaceKind.allCases))
    for preset in presets {
      do { try apply(preset.snapshot, library: library) }
      catch { XCTFail("\(preset.name): \(error)") }
    }
  }
  private func apply(_ snapshot: GenerationPresetSnapshot, library: GenerationLibrarySnapshot) throws {
    switch snapshot.context.workspace {
    case .image:
      let draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
      _ = try ImagePresetRecipe.candidate(applying: snapshot, to: draft, resolvedLibrary: library)
    case .clip:
      var project = StudioProject(), clip = Clip(engine: .ltx25)
      clip.generationSelection = .init(task: snapshot.context.task == "scene" ? "t2v" : snapshot.context.task)
      if snapshot.context.task == "video_upscale" {
        if let method = snapshot.fields["method"] { clip.generationSelection?.movieUpscaleMethod = try method.decoded(MovieUpscaleMethod.self) }
        if let scale = snapshot.fields["scale"] { clip.generationSelection?.relayVSRScale = try scale.decoded(NativeRelayVSRScale.self) }
        if snapshot.model?.backend == "ltx25" { var movie = LTX25MovieUpscaleSettings(); movie.experimentalEnabled = true
          if case .object(let fields) = snapshot.fields["movie"], let mode = fields["mode"] { movie.mode = try mode.decoded(LTX25MovieUpscaleSettings.Mode.self) }
          if movie.mode == .latentOnly { movie.anchors = .none }; clip.generationSelection?.ltx25MovieUpscale = movie }
      }
      project.clips = [clip]
      if snapshot.context.task == "scene" { var second = clip; second.id = UUID(); second.continuity = .init(mode: "scene", sourceClipID: clip.id); project.clips.append(second) }
      _ = try ClipPresetRecipe.candidate(applying: snapshot, to: clip, project: project, resolvedLibrary: library)
    case .ripple:
      var clip = Clip(engine: .movie); clip.sourcePath = "/source.mov"
      _ = try RipplePresetRecipe.candidate(applying: snapshot, to: .init(clip: clip, frameRate: 24), resolvedLibrary: library)
    case .restoration:
      var state = RestorationPresetState(tool: RestorationPresetTool(rawValue: snapshot.context.task)!); state.model = snapshot.model
      if let scale = snapshot.fields["scale"], state.tool == .relayVSR { state.relayScale = try scale.decoded(NativeRelayVSRScale.self) }
      _ = try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library)
    case .finishing:
      guard case .object(let fields) = snapshot.fields["stages"] else { return XCTFail("Missing stages") }
      let stages = try fields.keys.sorted().enumerated().map { index, key -> VideoProcessingStep in
        let inner = try fields[key]!.decoded(GenerationPresetSnapshot.self)
        var stage: VideoProcessingStep = inner.context.task == "pixelCQ" ? .init(pixelCQ: .init()) : inner.context.task == "h3Regenerate" ? .init(h3Regenerate: .init()) : .init(settings: .init())
        stage.id = "stage-\(index)"; if index > 0 { stage.input = "stage-\(index - 1)" }; return stage
      }
      _ = try FinishingPresetRecipe.candidate(applying: snapshot, to: .init(steps: stages), resolvedLibrary: library, sourceFPS: 24)
    }
  }
}
