import StudioCore
import SwiftUI

struct GenerationPresetControls: View {
  @EnvironmentObject var store: StudioStore
  let target: GenerationPresetTarget
  @State private var dialog: Dialog?
  @State private var message: String?
  @State private var applying = false
  private enum Dialog: Identifiable {
    case create(UUID, GenerationTargetToken, String), review(GenerationPresetReviewToken)
    var id: UUID { switch self { case .create(let id, _, _): return id; case .review(let value): return value.id } }
  }
  var body: some View {
    let selection = store.presetSelection(for: target)
    let selected = selection.flatMap { store.generationPresets.preset(id: $0.presetID) }
    let actions = store.presetActions(for: target)
    VStack(alignment: .leading, spacing: 7) {
      HStack {
        Text("Preset").font(.caption.bold())
        Menu {
          Button { perform { try store.clearPresetSelection(for: target) } } label: {
            Label("No preset", systemImage: selection == nil ? "checkmark" : "circle")
          }
          Divider()
          ForEach(store.presets(for: target)) { preset in
            Button {
              do {
                let token = try store.presetTargetToken(for: target)
                applying = true; message = nil
                Task { @MainActor in
                  defer { applying = false }
                  do { try await store.applyPreset(preset, for: target, expectedTarget: token) }
                  catch { message = error.localizedDescription }
                }
              } catch { message = error.localizedDescription }
            } label: {
              Label(preset.name + store.presetDependencyLabel(preset), systemImage: selection?.presetID == preset.id ? "checkmark" : preset.origin == .supplied ? "sparkles" : "slider.horizontal.3")
            }
          }
          if store.presets(for: target).isEmpty { Text("No presets saved for this task") }
        } label: {
          Text(selected?.name ?? (selection == nil ? "No preset" : "Missing preset")).lineLimit(1)
        }.accessibilityLabel("Generation preset").accessibilityIdentifier("generation-preset-menu")
          .disabled(applying || store.operationBusy)
        if applying { ProgressView().controlSize(.small) }
      }
      if actions.isModified {
        Text(actions.changeCount > 0 ? "Modified · \(actions.changeCount) change\(actions.changeCount == 1 ? "" : "s")" : "Modified · check settings")
          .font(.caption2).foregroundStyle(.orange).accessibilityIdentifier("generation-preset-modified")
      }
      HStack(spacing: 12) {
        if actions.hasSelection {
          Button("Update…") { perform { dialog = .review(try store.beginPresetUpdate(for: target)) } }
            .disabled(!actions.canUpdate || applying)
            .help(actions.canUpdate ? "Review changes before updating this preset" : actions.issue ?? "No changes to save")
            .accessibilityLabel("Update preset").accessibilityIdentifier("generation-preset-update")
        }
        Button("Save as new…") {
          perform { dialog = .create(UUID(), try store.presetTargetToken(for: target), selected.map { $0.name + " copy" } ?? "My preset") }
        }.disabled(!actions.canSaveNew || applying).accessibilityLabel("Save as new preset")
          .accessibilityIdentifier("generation-preset-save-new")
        if case .image = target, store.imagePresetUndo != nil {
          Button("Undo settings") { perform { try store.undoImagePreset() } }.disabled(applying)
            .help("Undo the last settings application when no later image edit has replaced it")
        }
        if case .restoration = target, store.restorationPresetUndo != nil {
          Button("Undo settings") { perform { try store.undoRestorationPreset() } }.disabled(applying || !store.canUndoRestorationPreset)
            .help("Restore the previous restoration recipe, model and preset selection")
        }
      }.font(.caption)
      if let issue = message ?? store.generationPresetError ?? actions.issue {
        Text(issue).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
      }
      if case .finishing = target {} else { Divider(); GenerationModelPicker(target: target) }
    }.frame(maxWidth: .infinity, alignment: .leading)
      .sheet(item: $dialog) { value in
        switch value {
        case .create(_, let token, let suggested):
          GenerationPresetNameView(target: target, expectedTarget: token, suggestedName: suggested,
            onCancel: { dialog = nil }, onSaved: { dialog = nil; message = nil }).environmentObject(store)
        case .review(let review):
          GenerationPresetReviewView(review: review, onCancel: { dialog = nil },
            onSaveAsNew: { dialog = .create(UUID(), review.targetToken, review.name + " copy") },
            onRefresh: { dialog = .review($0) }, onComplete: { dialog = nil; message = nil }).environmentObject(store)
        }
      }
      .task { await store.refreshGenerationLibrary() }
  }
  private func perform(_ action: () throws -> Void) {
    do { try action(); message = nil } catch { message = error.localizedDescription }
  }
}

@MainActor extension StudioStore {
  func presetDependencyLabel(_ preset: StudioGenerationPreset) -> String {
    if let ref = preset.snapshot.model {
      guard let entry = generationLibrary.modelEntries.first(where: { $0.reference == ref }) else { return " · Relink model" }
      if entry.availability != .installed { return " · " + entry.availability.label }
    }
    if (preset.snapshot.adapters ?? []).contains(where: { adapter in !generationLibrary.loraEntries.contains { $0.reference == adapter.reference && $0.availability == .installed } }) { return " · Check LoRAs" }
    return ""
  }
  func presetContext(for target: GenerationPresetTarget) -> GenerationPresetContext? {
    switch target {
    case .image: return .init(workspace: .image, task: "image")
    case .clip(let id): return project.clips.first { $0.id == id }.map { ClipPresetRecipe.presetContext($0, project: project) }
    case .ripple: return .init(workspace: .ripple, task: "ripple")
    case .restoration(let tool): return .init(workspace: .restoration, task: tool.rawValue)
    case .finishing: return .init(workspace: .finishing, task: "finishing")
    }
  }
  func presets(for target: GenerationPresetTarget) -> [StudioGenerationPreset] {
    generationPresets.presets.filter { $0.snapshot.context == presetContext(for: target) }
  }
}
