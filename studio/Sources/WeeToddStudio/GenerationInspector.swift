import StudioCore
import SwiftUI

struct GenerationInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var descriptor: GenerationDescriptor? {
    guard let result = store.generationDescriptions[clip.id],
      result["studioEngine"] as? String == clip.engine.rawValue,
      result["studioTask"] as? String == clip.inferredTask,
      result["studioProfile"] as? String == clip.profileID,
      let object = result["generation"] else { return nil }
    return try? JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: object))
  }
  var tasks: [String] {
    let supported = store.profiles.filter { $0.engine == clip.engine.rawValue }
      .flatMap { $0.generation?.supportedTasks ?? [] }
    return Array(Set(supported + [clip.inferredTask])).sorted()
  }
  var needsAutomaticComponents: Bool {
    guard clip.profileID != "auto" else { return false }
    guard let profile = store.profiles.first(where: {
      $0.id == clip.profileID && $0.engine == clip.engine.rawValue
    }) else { return true }
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
      if clip.reviewUsesSourceVideo {
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
      if clip.engine == .h3 {
        DisclosureGroup("Memory and execution") {
          Picker("Sampling weights", selection: Binding(get: {
            clip.generationSelection?.memoryPolicy ?? "recipe"
          }, set: { value in edit { $0.memoryPolicy = value == "recipe" ? nil : value } })) {
            Text((clip.generationSelection?.preset ?? .custom) == .custom ? "Recipe default" : "App default").tag("recipe")
            Text("Paged · lower memory").tag("paged")
            Text("Paged · larger workspace").tag("pagedNormal")
            Text("Resident · experimental high RAM").tag("resident")
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
          Text("Paged · larger workspace retains checkpoint pagination with larger working buffers; fit on 36 GB hardware has not been qualified. Resident sampling requires high RAM and no page cache. Components still unload between stages. It has not been qualified on 36 GB hardware.")
            .font(.caption2).foregroundStyle(.secondary)
        }
      }
    }.font(.caption).textFieldStyle(.roundedBorder)
      .task(id: store.generationRequestKey(for: clip)) {
        try? await Task.sleep(for: .milliseconds(350))
        guard !Task.isCancelled else { return }
        await store.describeGeneration()
      }
  }
  @ViewBuilder func controls(_ controls: GenerationControls) -> some View {
    if let steps = controls.evaluations {
      HStack {
        Text(controls.refinementSteps == nil ? "Steps" : "Stage one steps")
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
    if !controls.stepsEditable && !controls.stepsExplanation.isEmpty {
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
}
