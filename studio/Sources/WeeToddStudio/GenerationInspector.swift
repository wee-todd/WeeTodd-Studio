import StudioCore
import SwiftUI

struct GenerationInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var isFastH3:Bool { descriptor?.fasth3 == true || store.profiles.first(where: { $0.id==clip.profileID })?.generation?.fasth3 == true }
  var isVDN:Bool { descriptor?.vdn == true || store.profiles.first(where: { $0.id==clip.profileID })?.generation?.vdn == true }
  var descriptor: GenerationDescriptor? {
    guard let result = store.generationDescriptions[clip.id],
      result["studioEngine"] as? String == clip.engine.rawValue,
      result["studioTask"] as? String == clip.inferredTask,
      result["studioProfile"] as? String == clip.profileID,
      let object = result["generation"] else { return nil }
    return try? JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: object))
  }
  var ltxRendererSummary: String {
    guard store.runtime.usesNativeLTX25 else { return "Renderer: Python MLX" }
    if clip.inferredTask == "video_upscale" {
      return "Renderer: Swift MLX · source movie 2× · decoded previews"
    }
    let generation = descriptor ?? store.profiles.first(where: { $0.id == clip.profileID })?.generation
    let summary = (clip.generationSelection?.ltx25DFR?.enabled ?? generation?.dfrEnabled ?? false)
      ? "DFR refinement" : generation?.ltx25ExecutionSummary ?? "sampling settings pending"
    return "Renderer: Swift MLX · \(summary) · decoded previews"
  }
  var resolvedVideoDecodePrecision: NativeH3VideoDecodePrecision {
    .resolved(selection: clip.generationSelection?.h3VideoDecodePrecision,
      recipeValue: descriptor?.videoDecodePrecision ?? store.profiles.first(where: {
        $0.id == clip.profileID && $0.engine == clip.engine.rawValue
      })?.generation?.videoDecodePrecision)
  }
  var resolvedAttentionPolicy: NativeH3AttentionPolicy {
    .resolved(selection:clip.generationSelection?.h3AttentionPolicy,
      recipeValue:descriptor?.attentionPolicy ?? store.profiles.first(where: {
        $0.id == clip.profileID && $0.engine == clip.engine.rawValue
      })?.generation?.attentionPolicy)
  }
  var solAttentionAvailable: Bool {
    ["ref2va", "i2v", "fflf"].contains(clip.inferredTask) && clip.continuityMode == "independent"
      && clip.extensionDirection.isEmpty && !isFastH3 && !isVDN
      && clip.generationSelection?.h3Joint == nil && clip.generationSelection?.h3MotionFidelity == nil
      && clip.generationSelection?.transformerBackend != "nnc_experimental"
      && descriptor?.attentionPolicyEditable == true
  }
  var tasks: [String] {
    if let selected=store.profiles.first(where: { $0.id==clip.profileID && $0.engine==clip.engine.rawValue }),
      (selected.generation?.vdn == true || selected.generation?.fasth3 == true) { return ["t2v"] }
    let supported = store.profiles.filter { $0.engine == clip.engine.rawValue }
      .flatMap { $0.generation?.supportedTasks ?? [] }
    return Array(Set(supported + [clip.inferredTask])).sorted()
  }
  var needsAutomaticComponents: Bool {
    guard clip.profileID != "auto" else { return false }
    guard let profile = store.profiles.first(where: {
      $0.id == clip.profileID && $0.engine == clip.engine.rawValue
    }) else { return true }
    if clip.inferredTask == "video_upscale" { return false }
    if clip.engine == .ltx25, let dfr = clip.generationSelection?.ltx25DFR {
      if (profile.generation?.dfrEnabled ?? false) != dfr.enabled { return true }
      if dfr.enabled && dfr.temporalRounds > 0 && profile.generation?.dfrTemporalAvailable != true { return true }
    }
    if clip.engine == .ltx25, store.runtime.usesNativeLTX25,
      (profile.generation?.pipelineMode ?? "distilled") != (clip.generationSelection?.ltx25Guidance?.mode.rawValue ?? "distilled") { return true }
    return profile.generation.map { !$0.supportedTasks.contains(clip.inferredTask) } ?? false
  }
  func edit(_ body: @escaping (inout GenerationSelection) -> Void) {
    store.editClip {
      var value = $0.generationSelection ?? GenerationSelection(task: $0.inferredTask, preset: .custom)
      body(&value)
      $0.generationSelection = value
    }
  }
  func integer(_ key: WritableKeyPath<GenerationSelection, Int?>, default value: Int) -> Binding<Int> {
    Binding(get: { store.selectedClip?.generationSelection?[keyPath: key] ?? value },
      set: { number in edit { $0[keyPath: key] = number } })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Picker("Model", selection: Binding(get: { clip.engine }, set: { model in
        store.editClip { $0.selectLocalModel(model) }
      })) {
        ForEach([Engine.h3, .ltx23, .ltx25]) {
          Text($0.label).tag($0)
        }
      }
      if clip.reviewUsesSourceVideo || clip.inferredTask == "video_upscale" {
        LabeledContent("Task", value: clip.displayTask)
      } else {
        Picker("Task", selection: Binding(get: { clip.inferredTask }, set: { task in
          store.editClip { $0.selectGenerationTask(task) }
        })) {
          ForEach(tasks, id: \.self) { Text(GenerationSelection.taskLabel($0)).tag($0) }
        }
      }
      if clip.generationSelection?.isModified == true {
        HStack {
          Text("Modified").foregroundStyle(.secondary)
          Spacer()
          Button("Reset") { edit { $0.resetOverrides() } }
        }
      }
      if let descriptor {
        controls(descriptor.controls)
      } else {
        Text(store.validationErrors[clip.id] == nil ? "Loading model settings…" : "Choose a compatible model to view settings").font(.caption2).foregroundStyle(.secondary)
      }
      if clip.engine == .h3, store.runtime.usesNativeH3 {
        let vdn=isVDN || isFastH3
        Picker("Sampler", selection: Binding(get: {
          clip.generationSelection?.h3SamplingMethod?.rawValue
            ?? (store.generationDescriptions[clip.id]?["generation"] as? [String: Any])?["samplingMethod"] as? String ?? "euler"
        }, set: { value in edit { $0.h3SamplingMethod = NativeH3SamplingMethod(rawValue: value) } })) {
          ForEach(NativeH3SamplingMethod.allCases) { Text($0.label).tag($0.rawValue) }
        }
        .disabled(vdn)
        if vdn {
          Text(isFastH3 ? "FastH3 · Four Euler evaluations. Independent text-to-audiovisual only." : "VDN · Fixed Euler schedule and released adapters at strength 1. Text-to-video only.").font(.caption2).foregroundStyle(.secondary)
        } else { H3CreativeInspector(clip:clip) }
        if !vdn && (["t2v","t2va"].contains(clip.inferredTask) || clip.generationSelection?.h3MotionFidelity != nil) {
          H3MotionFidelityInspector(clip:clip)
        }
        if clip.generationSelection?.h3SamplingMethod == .resMultistep {
          Text("Experimental multistep sampling. Turbo adapters require Euler.").font(.caption2).foregroundStyle(.secondary)
        }
      }
      if clip.engine == .h3, store.runtime.usesNativeH3 {
        DisclosureGroup("Advanced attention") {
          Picker("Attention", selection:Binding(get: {
            clip.generationSelection?.h3AttentionPolicy?.rawValue ?? "recipe"
          }, set: { value in edit {
            $0.h3AttentionPolicy = value == "recipe" ? nil : NativeH3AttentionPolicy(rawValue:value)
          } })) {
            Text("Recipe default (Dense unless specified)").tag("recipe")
            Text(NativeH3AttentionPolicy.dense.label).tag(NativeH3AttentionPolicy.dense.rawValue)
            Text(NativeH3AttentionPolicy.solExperimental.label).tag(NativeH3AttentionPolicy.solExperimental.rawValue)
              .disabled(!solAttentionAvailable)
          }
          .accessibilityIdentifier("h3-attention-policy")
          Text("Resolved: \(resolvedAttentionPolicy.label)")
          Text("Sol approximates generated-video attention and may change video and audio. Only ordinary independent Swift MLX Ref2VA or FL2VA with one immediate strength-1 Turbo adapter, Euler, 4 evaluations and Drop AdaLN is supported. Checkpoint and packed-row eligibility are checked before rendering. The first two evaluations remain dense; tau is fixed at 0.5.")
            .font(.caption2).foregroundStyle(.secondary)
        }
        DisclosureGroup("Advanced video decoding") {
          Picker("Precision", selection: Binding(get: {
            clip.generationSelection?.h3VideoDecodePrecision?.rawValue ?? "recipe"
          }, set: { value in edit {
            $0.h3VideoDecodePrecision = value == "recipe" ? nil : NativeH3VideoDecodePrecision(rawValue:value)
          } })) {
            Text("Recipe default (FP32 unless specified)").tag("recipe")
            ForEach(NativeH3VideoDecodePrecision.allCases) { Text($0.label).tag($0.rawValue) }
          }
          .accessibilityIdentifier("h3-video-decode-precision")
          Text("Resolved: \(resolvedVideoDecodePrecision == .float16 ? "FP16" : "FP32")")
          Text("FP32 preserves the default decoder. FP16 may change pixels and requires Swift MLX with a lower-memory profile. Sampling and audio decoding are unchanged.")
            .font(.caption2).foregroundStyle(.secondary)
        }
      }
      if clip.engine == .ltx25, store.runtime.usesNativeLTX25 {
        DiffusionVAEInspector(clip:clip)
        if clip.inferredTask == "video_upscale" { MovieUpscaleInspector(clip:clip) }
        else {
          LTXDFRInspector(clip: clip, generation: descriptor ?? store.profiles.first(where: { $0.id == clip.profileID })?.generation)
          guidedControls; automaticDurationControls; singleStageControls; ordinaryKeyframeControls
        }
      }
      ForEach(store.generationDescriptions[clip.id]?["warnings"] as? [String] ?? [], id: \.self) { warning in
        Text(warning).font(.caption2).foregroundStyle(.secondary)
      }
      if needsAutomaticComponents {
        Button("Use automatic model components") {
          store.editClip { $0.selectAutomaticModelComponents() }
        }
      }
      if let error = store.validationErrors[clip.id] {
        Text(error).font(.caption2).foregroundStyle(.red).textSelection(.enabled)
      }
      Button("Set up or repair models…") { store.showRuntime = true }
      if clip.engine == .ltx25 {
        Text(ltxRendererSummary)
          .font(.caption2).foregroundStyle(.secondary)
      }
      if clip.engine == .h3 {
        if isVDN || isFastH3 {
          Text("This model uses the shared MLX worker with staged unloading and no retained page cache.").font(.caption2).foregroundStyle(.secondary)
          Button("Reset incompatible model overrides") {
            store.editClip { $0.generationSelection?.resetOverrides();$0.h3PagingCacheGB=nil }
          }
        } else {
        DisclosureGroup("Memory and execution") {
          Picker("Transformer", selection: Binding(get: {
            clip.generationSelection?.transformerBackend ?? "recipe"
          }, set: { value in edit { $0.transformerBackend = value == "recipe" ? nil : value } })) {
            Text("Recipe default").tag("recipe")
            Text("MLX").tag("mlx")
            Text("Native NNC (experimental)").tag("nnc_experimental")
          }
          if let generation = store.generationDescriptions[clip.id]?["generation"] as? [String: Any],
            generation["transformerBackend"] as? String == "nnc_experimental" {
            Text("Native NNC: Ref2VA with the ComfyUI BF16 Turbo LoRA at strength 1, four Euler evaluations and Drop AdaLN. Uses one GPU block plus one CPU prefetch, FP16 projections and FP32 attention. MLX handles conditioning and decoding. Native core buffers use a fixed policy; MLX chunk controls apply only outside the native core. Advanced cache/control/refinement settings are not supported.")
              .font(.caption2).foregroundStyle(.secondary)
          }
          if clip.engine == .h3 && store.runtime.usesNativeH3 {
            Picker("Transformer weight cache", selection:Binding(get:{
              clip.generationSelection?.h3TransformerWeightCacheGB ?? -1
            },set:{ value in edit { $0.h3TransformerWeightCacheGB = value < 0 ? nil : value } })) {
              Text("Recipe default").tag(-1)
              Text("Stream blocks · 0 GiB").tag(0)
              ForEach([8,16,32,48,64,96],id: \.self) { budget in Text("Up to \(budget) GiB").tag(budget) }
            }
            .disabled(descriptor?.transformerWeightCacheEditable != true || clip.generationSelection?.transformerBackend == "nnc_experimental")
            Text("Reuses a fixed set of block weights between sampling steps. The rest stream normally. All cached weights release before decoding. The worker checks available memory and reserves space for activations; this is a weight budget, not total app memory.")
              .font(.caption2).foregroundStyle(.secondary)
          } else {
          Picker("Sampling weights", selection: Binding(get: {
            clip.generationSelection?.memoryPolicy ?? "recipe"
          }, set: { value in edit { $0.memoryPolicy = value == "recipe" ? nil : value } })) {
            Text((clip.generationSelection?.preset ?? .custom) == .custom ? "Recipe default" : "App default").tag("recipe")
            Text("Paged · lower memory").tag("paged")
            Text("Paged · larger workspace").tag("pagedNormal")
            Text("Resident · experimental high RAM").tag("resident")
          }
          }
          Picker("Projection", selection: Binding(get: {
            clip.generationSelection?.projectionBackend ?? "recipe"
          }, set: { value in edit { $0.projectionBackend = value == "recipe" ? nil : value } })) {
            Text((clip.generationSelection?.preset ?? .custom) == .custom ? "Recipe default" : "App default").tag("recipe")
            Text("MLX").tag("mlx")
            Text("Automatic").tag("auto")
          }
          if let generation = store.generationDescriptions[clip.id]?["generation"] as? [String: Any],
            let acceleration = generation["acceleration"] as? [String: Any] {
            Text("Resolved: \(AccelerationSettings.memoryPolicyLabel(acceleration["memoryPolicy"] as? String ?? "recipe")) · \(acceleration["projectionBackend"] as? String ?? "auto")")
              .font(.caption2)
            Text(acceleration["explanation"] as? String ?? "")
              .font(.caption2).foregroundStyle(.secondary)
          }
          if !store.runtime.usesNativeH3 {
            Text("Paged · larger workspace retains checkpoint pagination with larger working buffers; fit on 36 GB hardware has not been qualified. Resident sampling requires high RAM and no page cache. Components still unload between stages. It has not been qualified on 36 GB hardware.")
              .font(.caption2).foregroundStyle(.secondary)
          }
        }
        }
      }
    }.font(.caption).textFieldStyle(.roundedBorder)
      .task(id: store.generationDescriptionTaskKey(for: clip)) {
        try? await Task.sleep(for: .milliseconds(350))
        guard !Task.isCancelled else { return }
        await store.describeGeneration()
      }
  }
  @ViewBuilder var singleStageControls:some View {
    let generation=store.generationDescriptions[clip.id]?["generation"] as? [String:Any] ?? [:]
    let available=generation["singleStageAvailable"] as? Bool ?? false
    let ingredients=generation["referenceFamily"] as? String == "ingredients"
    let methods=(generation["singleStageMethods"] as? [String])?.compactMap(LTX25SingleStageMethod.init(rawValue:)) ?? LTX25SingleStageMethod.allCases
    if available || clip.generationSelection?.ltx25SingleStage != nil {
      DisclosureGroup("Full-resolution single-stage sampling") {
        Toggle("Override with single-stage sampling",isOn:Binding(get:{ clip.generationSelection?.ltx25SingleStage != nil },set:{ enabled in
          edit { $0.ltx25SingleStage=enabled ? LTX25SingleStageSettings():nil }
        })).disabled(!available && clip.generationSelection?.ltx25SingleStage == nil)
        if clip.generationSelection?.ltx25SingleStage == nil,
          (store.generationDescriptions[clip.id]?["generation"] as? [String:Any])?["singleStageEnabled"] as? Bool == true {
          Text(ingredients ? "The Ingredients profile supplies its sampler. Enable the override to choose eight-step ancestral or full CFG++ sampling." : "The model profile uses single-stage sampling. Enable the override to change its sampler or negative schedule. Select a two-stage profile to use 8 + 3 sampling.")
            .font(.caption2).foregroundStyle(.secondary)
        }
        if let settings=clip.generationSelection?.ltx25SingleStage {
          Toggle("Enable experimental execution",isOn:Binding(get:{ settings.experimentalEnabled },set:{ enabled in edit { $0.ltx25SingleStage?.experimentalEnabled=enabled } }))
          Picker("Sampler",selection:Binding(get:{ settings.method },set:{ method in edit {
            $0.ltx25SingleStage?.method=method
            if ingredients || method != .cfgpp { $0.ltx25SingleStage?.negativeSchedule = .full }
          } })) {
            ForEach(methods) { Text($0.label).tag($0) }
          }
          if settings.method == .cfgpp {
            if ingredients {
              Text("Full unconditional passes — 16 evaluations").font(.caption2).foregroundStyle(.secondary)
            } else {
              Picker("Negative passes",selection:Binding(get:{ settings.negativeSchedule },set:{ schedule in edit { $0.ltx25SingleStage?.negativeSchedule=schedule } })) {
                ForEach(LTX25NegativeSchedule.allCases) { Text("\($0.label) — \($0.evaluationCount) evaluations").tag($0) }
              }
            }
            Text(ingredients ? "Ingredients CFG++ uses its fixed empty unconditional context on all eight steps. Clip negative prompts are unsupported." : "Uses the clip's negative prompt. CFG++ generates audio; it cannot freeze an A2V driver.").font(.caption2).foregroundStyle(.secondary)
          }
          Text(ingredients ? "Eight updates at the output resolution with one described Ingredients sheet and no second stage. Ancestral uses eight evaluations; CFG++ uses sixteen. Experimental; real-model quality is not qualified." : "Eight updates at the output resolution, with no spatial upscale or second stage. Supports ordinary timed images and generated slots. Experimental; real-model quality is not qualified.").font(.caption2).foregroundStyle(.secondary)
        }
      }
    }
  }
  @ViewBuilder var ordinaryKeyframeControls:some View {
    let available=(store.generationDescriptions[clip.id]?["generation"] as? [String:Any])?["ordinaryKeyframesAvailable"] as? Bool ?? false
    if available || clip.generationSelection?.ltx25Keyframes != nil {
      DisclosureGroup("Timed images and generated keyframes") {
        Toggle("Use experimental keyframes",isOn:Binding(get:{ clip.generationSelection?.ltx25Keyframes != nil },set:{ enabled in
          edit { $0.ltx25Keyframes=enabled ? LTX25KeyframeSettings() : nil }
        })).disabled(!available && clip.generationSelection?.ltx25Keyframes == nil)
        if let settings=clip.generationSelection?.ltx25Keyframes {
          Toggle("Enable experimental execution",isOn:Binding(get:{ settings.experimentalEnabled },set:{ enabled in
            edit { $0.ltx25Keyframes?.experimentalEnabled=enabled }
          }))
          Stepper("Generated keyframes: \(settings.generatedCount)",value:Binding(get:{ settings.generatedCount },set:{ count in
            edit { $0.ltx25Keyframes?.generatedCount=count }
          }),in:0...8).disabled(!settings.experimentalEnabled)
          Text("Attach up to eight images with First, Last or Keyframe roles; set each keyframe's time in its attachment controls. Generated slots apply to stage one only. Experimental; visual quality is not qualified.")
            .font(.caption2).foregroundStyle(.secondary)
          if clip.generationSelection?.ltx25AutomaticDuration != nil {
            Text("Automatic timing resolves Last frame and generated slots after duration prediction. Numeric keyframes must fit the predicted interval.").font(.caption2).foregroundStyle(.secondary)
          }
        }
      }
    }
  }
  @ViewBuilder func controls(_ controls: GenerationControls) -> some View {
    if let steps = controls.evaluations {
      HStack {
        Text(clip.engine == .ltx25 && controls.refinementSteps == 0
          ? "Evaluations" : controls.refinementSteps == nil ? "Steps" : "Stage one steps")
        TextField("Steps", value: controls.stepsEditable ? integer(\.steps, default: steps) : .constant(steps), format: .number.grouping(.never))
          .disabled(!controls.stepsEditable)
      }
    }
    if let refinement = controls.refinementSteps {
      HStack {
        Text("Refinement · evaluations")
        TextField("Refinement steps", value: integer(\.refinementSteps, default: refinement), format: .number.grouping(.never))
          .disabled(!controls.refinementStepsEditable)
      }
    }
    if !controls.stepsExplanation.isEmpty {
      Text(controls.stepsExplanation).font(.caption2).foregroundStyle(.secondary)
    }
    if controls.cfg != nil {
      HStack {
      Text("CFG")
      if controls.cfgEditable, let cfg = controls.cfg {
        TextField("CFG", value: Binding(get: { clip.generationSelection?.cfg ?? cfg },
          set: { value in edit { $0.cfg = value } }), format: .number)
      } else { Spacer(); Text(controls.cfg.map { String($0) } ?? "Unavailable").foregroundStyle(.secondary) }
      }.help(controls.cfgExplanation)
    } else if clip.engine == .h3 {
      Text("Guidance is built into H3.").font(.caption2).foregroundStyle(.secondary)
    }
    if controls.shift != nil {
      HStack {
      Text("Shift")
      if controls.shiftEditable, let shift = controls.shift {
        TextField("Shift", value: Binding(get: { clip.generationSelection?.shift ?? shift },
          set: { value in edit { $0.shift = value } }), format: .number)
      } else { Spacer(); Text(controls.shift.map { String($0) } ?? "Unavailable").foregroundStyle(.secondary) }
      }.help(controls.shiftExplanation)
    }
  }

  func guidanceNumber(_ key: WritableKeyPath<LTX25GuidanceSettings, Double?>, wireKey: String, fallback: Double) -> Binding<Double> {
    Binding(get: { clip.generationSelection?.ltx25Guidance?[keyPath: key]
      ?? ((store.generationDescriptions[clip.id]?["generation"] as? [String: Any])?["guidance"] as? [String: Any])?[wireKey] as? Double ?? fallback },
      set: { value in edit { $0.ltx25Guidance?[keyPath: key] = value } })
  }
  var inheritedGuidance: [String: Any] {
    ((store.generationDescriptions[clip.id]?["generation"] as? [String: Any])?["guidance"] as? [String: Any]) ?? [:]
  }
  @ViewBuilder var guidedControls: some View {
    Picker("Generation mode", selection: Binding(get: {
      clip.generationSelection?.ltx25Guidance?.mode.rawValue ?? "distilled"
    }, set: { value in edit {
      $0.ltx25Guidance = LTX25GuidanceMode(rawValue: value).map { LTX25GuidanceSettings(mode: $0) }
      $0.steps = nil; $0.refinementSteps = nil; $0.cfg = nil
    } })) {
      Text("Fast distilled").tag("distilled")
      ForEach(LTX25GuidanceMode.allCases) { Text($0.label).tag($0.rawValue) }
    }
    if let guidance = clip.generationSelection?.ltx25Guidance {
      Toggle("Enable experimental Dev guidance", isOn: Binding(get: { guidance.experimentalEnabled },
        set: { value in edit { $0.ltx25Guidance?.experimentalEnabled = value } }))
      Text("Requires a compatible Dev profile and the official distilled refinement adapter. Quality and performance are unqualified. The negative prompt is evaluated in this mode.")
        .font(.caption2).foregroundStyle(.secondary)
      DisclosureGroup("Advanced guidance") {
        HStack { Text("Audio CFG"); TextField("Audio CFG", value: guidanceNumber(\.audioCFG, wireKey: "audio_cfg_scale", fallback: 7), format: .number) }
        HStack { Text("STG"); TextField("STG", value: guidanceNumber(\.stgScale, wireKey: "stg_scale", fallback: guidance.mode == .guided ? 1 : 0), format: .number) }
        HStack { Text("Video rescale"); TextField("Video rescale", value: guidanceNumber(\.videoRescale, wireKey: "video_rescale_scale", fallback: guidance.mode == .guided ? 0.7 : 0.45), format: .number) }
        HStack { Text("Audio rescale"); TextField("Audio rescale", value: guidanceNumber(\.audioRescale, wireKey: "audio_rescale_scale", fallback: guidance.mode == .guided ? 0.7 : 1), format: .number) }
        HStack { Text("Modality guidance"); TextField("Modality guidance", value: guidanceNumber(\.modalityScale, wireKey: "modality_scale", fallback: 3), format: .number) }
        TextField("STG block indices, comma separated", text: Binding(get: {
          (guidance.stgBlocks ?? inheritedGuidance["stg_blocks"] as? [Int] ?? (guidance.mode == .guided ? [28] : [])).map(String.init).joined(separator: ",")
        }, set: { text in edit {
          let parts = text.split(separator: ",", omittingEmptySubsequences: false)
          let values = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
          $0.ltx25Guidance?.stgBlocks = text.isEmpty ? [] : values.count == parts.count ? values : [-1]
        } }))
        TextField("Custom sigmas (blank: adaptive)", text: Binding(get: {
          (guidance.sigmas ?? inheritedGuidance["stage1_sigmas"] as? [Double])?.map { String($0) }.joined(separator: ",") ?? ""
        }, set: { text in edit {
          let parts = text.split(separator: ",", omittingEmptySubsequences: false)
          let values = parts.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
          $0.ltx25Guidance?.sigmas = text.isEmpty ? [] : values.count == parts.count && values.allSatisfy(\.isFinite) ? values : [-1]
        } }))
        Text("Sigmas need updates + 1 points, strictly descending from (0,1] to zero. Invalid entries prevent preparation.").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
  @ViewBuilder var automaticDurationControls: some View {
    Picker("Duration", selection: Binding(get: {
      clip.generationSelection?.ltx25AutomaticDuration == nil ? "manual" : "automatic"
    }, set: { mode in edit {
      $0.ltx25AutomaticDuration = mode == "automatic" ? LTX25AutomaticDurationSettings() : nil
    } })) {
      Text("Manual shot duration").tag("manual")
      Text("Automatic (experimental)").tag("automatic")
    }
    if let automatic = clip.generationSelection?.ltx25AutomaticDuration {
      Toggle("Enable experimental automatic duration", isOn: Binding(get:{ automatic.experimentalEnabled },
        set:{ value in edit { $0.ltx25AutomaticDuration?.experimentalEnabled=value } }))
      HStack {
        Text("Minimum seconds")
        TextField("Minimum seconds",value:Binding(get:{automatic.minimumSeconds},
          set:{value in edit { $0.ltx25AutomaticDuration?.minimumSeconds=value }}),format:.number)
      }
      HStack {
        Text("Maximum seconds")
        TextField("Maximum seconds",value:Binding(get:{automatic.maximumSeconds},
          set:{value in edit { $0.ltx25AutomaticDuration?.maximumSeconds=value }}),format:.number)
      }
      Text("Requires the LTX 2.5 duration head. Bounds are 0.25–30 seconds on the 8k+1 frame grid. Only ordinary text/image/first-last shots are supported. The accepted take adopts its predicted video duration; source audio, scenes, extensions and specialized controls require manual timing.")
        .font(.caption2).foregroundStyle(.secondary)
    }
  }
}
