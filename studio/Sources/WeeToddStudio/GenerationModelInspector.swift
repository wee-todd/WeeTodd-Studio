import StudioCore
import SwiftUI

/// Select installed components without changing the authored inputs or sampler overrides.
struct GenerationModelInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var selectedProfile: ModelProfile? {
    store.profiles.first { $0.id == clip.profileID && $0.engine == clip.engine.rawValue }
  }
  var family: String? {
    let result = store.generationDescriptions[clip.id]
    guard result?["studioProfile"] as? String == clip.profileID,
      result?["studioTask"] as? String == clip.inferredTask,
      result?["studioEngine"] as? String == clip.engine.rawValue else {
      return selectedProfile?.generation?.referenceFamily
    }
    return (result?["generation"] as? [String: Any])?["referenceFamily"] as? String
      ?? selectedProfile?.generation?.referenceFamily
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Picker("Model", selection: Binding(get: { clip.engine }, set: { engine in
        store.editClip { $0.selectLocalModel(engine) }
      })) {
        ForEach([Engine.h3, .ltx23, .ltx25]) { Text($0.label).tag($0) }
      }
      Picker("Model profile", selection: Binding(get: { clip.profileID }, set: { value in
        store.editClip { $0.selectGenerationProfile(value) }
      })) {
        Text("Automatic components").tag("auto")
        if clip.profileID != "auto" && selectedProfile == nil {
          Text("Saved profile · unavailable").tag(clip.profileID).disabled(true)
        }
        ForEach(store.profiles.filter { $0.engine == clip.engine.rawValue }) {
          Text($0.name).tag($0.id)
        }
      }
      .help("Select installed model components and required adapters. Your inputs and explicit sampling settings are retained.")
      Label(GenerationPresentation.methodLabel(for: clip, referenceFamily: family), systemImage: "sparkles")
        .font(.caption).foregroundStyle(.secondary)
      if let selectedProfile, !GenerationPresentation.canUseProfile(
        supportedTasks: selectedProfile.generation?.supportedTasks ?? [selectedProfile.task], for: clip) {
        Text("This profile needs different inputs. Add its supported references below, select another model profile, or use automatic components.")
          .font(.caption2).foregroundStyle(.orange)
      }
      if family == "ingredients" || clip.attachments.contains(where: { $0.role == .control && $0.controlType == "ingredients_reference_sheet" }) {
        Label("Ingredients adapter · supplied by model profile", systemImage: "lock")
          .font(.caption2).foregroundStyle(.secondary)
      } else if let family, family != "ordinary" {
        Label("Required \(family.uppercased()) adapter · supplied by model profile", systemImage: "lock")
          .font(.caption2).foregroundStyle(.secondary)
      }
      HStack {
        Spacer()
        Button("Model setup…") { store.showRuntime = true }.font(.caption2)
      }
      if selectedProfile?.generation?.fasth3Omni == true {
        Text(NativeH3OmniProfile.requirements).font(.caption2).foregroundStyle(.secondary)
        ForEach(NativeH3OmniProfile.conflicts(clip:clip),id: \.self) { issue in
          Text(issue).font(.caption2).foregroundStyle(.orange)
        }
        Button("Apply required Omni sampling settings") {
          store.editClip { NativeH3OmniProfile.applyRequiredSettings(to:&$0) }
        }.help("Explicitly clears Turbo/two-stage, refinement, reference-noise, attention and cache overrides. Keeps all inputs, LoRAs, prompt, canvas and duration. Disable extra LoRAs or remove incompatible inputs below.")
      }
      ClipLoRAInspector(clip: clip)
      if clip.generationSelection?.isModified == true || clip.h3PagingCacheGB != nil {
        Button("Use model defaults") { store.editClip { $0.useSelectedModelDefaults() } }
          .font(.caption)
          .help("Reset sampling, guidance and execution overrides to the selected profile. Keep the prompt, inputs, seed, canvas and duration.")
      }
    }.font(.caption).textFieldStyle(.roundedBorder)
  }
}
