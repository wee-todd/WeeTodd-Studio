import StudioCore
import SwiftUI

struct GenerationPresetNameView: View {
  @EnvironmentObject var store: StudioStore
  let target: GenerationPresetTarget
  let expectedTarget: GenerationTargetToken
  let onCancel, onSaved: () -> Void
  @State private var name: String
  @State private var busy = false
  @State private var message: String?
  @FocusState private var focused: Bool
  init(target: GenerationPresetTarget, expectedTarget: GenerationTargetToken, suggestedName: String,
    onCancel: @escaping () -> Void, onSaved: @escaping () -> Void) {
    self.target = target; self.expectedTarget = expectedTarget; self.onCancel = onCancel; self.onSaved = onSaved
    _name = State(initialValue: suggestedName)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Save as new preset").font(.title2.bold())
      Text("Save the current model, LoRAs and generation settings.").foregroundStyle(.secondary)
      TextField("Preset name", text: $name).textFieldStyle(.roundedBorder).focused($focused)
        .accessibilityIdentifier("generation-preset-name")
      if let message { Text(message).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
      HStack {
        if busy { ProgressView().controlSize(.small) }
        Spacer()
        Button("Cancel", role: .cancel, action: onCancel).keyboardShortcut(.cancelAction).disabled(busy)
        Button("Save as new") {
          busy = true; message = nil
          Task { @MainActor in
            defer { busy = false }
            do { _ = try await store.saveNewPreset(name: name, for: target, expectedTarget: expectedTarget); onSaved() }
            catch { message = error.localizedDescription }
          }
        }.keyboardShortcut(.defaultAction).disabled(busy || name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      }
    }.padding(24).frame(width: 440).interactiveDismissDisabled(busy)
      .onAppear { focused = true }
  }
}

struct GenerationPresetReviewView: View {
  @EnvironmentObject var store: StudioStore
  let review: GenerationPresetReviewToken
  let onCancel, onSaveAsNew, onComplete: () -> Void
  let onRefresh: (GenerationPresetReviewToken) -> Void
  @State private var busy = false
  @State private var message: String?
  init(review: GenerationPresetReviewToken, onCancel: @escaping () -> Void, onSaveAsNew: @escaping () -> Void,
    onRefresh: @escaping (GenerationPresetReviewToken) -> Void, onComplete: @escaping () -> Void) {
    self.review = review; self.onCancel = onCancel; self.onSaveAsNew = onSaveAsNew; self.onRefresh = onRefresh; self.onComplete = onComplete
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Update “\(review.name)”").font(.title2.bold())
      Text("Review \(review.changes.count) change\(review.changes.count == 1 ? "" : "s") before replacing the saved settings.")
        .foregroundStyle(.secondary)
      ScrollView {
        Grid(alignment: .topLeading, horizontalSpacing: 18, verticalSpacing: 12) {
          GridRow {
            Text("Setting").frame(width: 215, alignment: .leading)
            Text("Saved value").frame(width: 230, alignment: .leading)
            Text("Current value").frame(width: 230, alignment: .leading)
          }.font(.headline)
          Divider().gridCellColumns(3)
          ForEach(review.changes) { change in
            let row = store.presetReviewRow(change, review: review)
            GridRow {
              Text(row.label).frame(width: 215, alignment: .leading)
              Text(row.saved).foregroundStyle(.secondary).frame(width: 230, alignment: .leading)
              Text(row.current).frame(width: 230, alignment: .leading)
            }.textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
          }
        }.font(.callout).padding(12)
      }.frame(minHeight: 230, idealHeight: 330).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
      if store.generationPresets.preset(id: review.presetID)?.origin == .supplied {
        Text("Your update is saved as a personal override of the supplied preset.").font(.caption).foregroundStyle(.secondary)
      }
      if let message {
        HStack {
          Text(message).font(.caption).foregroundStyle(.red).textSelection(.enabled)
          Button("Refresh review") {
            busy = true
            Task { @MainActor in
              defer { busy = false }
              await store.refreshGenerationLibrary(force: true)
              do { onRefresh(try store.beginPresetUpdate(for: review.target)) }
              catch { self.message = error.localizedDescription }
            }
          }.disabled(busy)
        }
      }
      HStack {
        if busy { ProgressView().controlSize(.small) }
        Spacer()
        Button("Cancel", role: .cancel, action: onCancel).keyboardShortcut(.cancelAction).disabled(busy)
        Button("Save as new…", action: onSaveAsNew).disabled(busy)
        Button("Update preset") {
          busy = true; message = nil
          Task { @MainActor in
            defer { busy = false }
            do { try await store.confirmPresetUpdate(review); onComplete() }
            catch { message = error.localizedDescription }
          }
        }.keyboardShortcut(.defaultAction).disabled(busy || review.changes.isEmpty)
          .accessibilityIdentifier("generation-preset-confirm-update")
      }
    }.padding(24).frame(width: 790).background(Theme.background).interactiveDismissDisabled(busy)
  }
}

@MainActor extension StudioStore {
  func presetReviewRow(_ change: GenerationPresetChange, review: GenerationPresetReviewToken) -> GenerationPresetChange {
    var result = change
    func modelTitle(_ ref: GenerationModelRef?, counterpart: GenerationModelRef?, previous: Bool) -> String {
      guard let ref else { return "Not specified" }
      let entry = generationLibrary.modelEntries.first { $0.id == ref.id }
      let sameVariant = entry?.reference.variant == ref.variant && entry?.reference.precision == ref.precision && entry?.reference.backend == ref.backend
      let name = sameVariant ? entry!.name : [ref.family, ref.variant, ref.precision].compactMap { $0 }.joined(separator: " · ")
      if ref.id == counterpart?.id, ref.sourceIdentity != counterpart?.sourceIdentity {
        return name + (previous ? " · previous source" : " · current source")
      }
      return name
    }
    if change.path == "model" {
      result.saved = modelTitle(review.before.model, counterpart: review.after.model, previous: true); result.current = modelTitle(review.after.model, counterpart: review.before.model, previous: false)
    } else if change.path.hasPrefix("fields.stages."), change.path.hasSuffix(".model") {
      let parts = change.path.split(separator: ".")
      if parts.count == 4, let index = Int(parts[2]) {
        func reference(_ snapshot: GenerationPresetSnapshot) -> GenerationModelRef? {
          guard case .object(let stages) = snapshot.fields["stages"], let value = stages[String(index)] else { return nil }
          return (try? value.decoded(GenerationPresetSnapshot.self))?.model
        }
        let before = reference(review.before), after = reference(review.after)
        result.label = "Stage \(index + 1) · Model / precision"
        result.saved = modelTitle(before, counterpart: after, previous: true); result.current = modelTitle(after, counterpart: before, previous: false)
      }
    } else if change.path == "adapters.order" {
      func names(_ stack: [GenerationPresetAdapter]?) -> String {
        stack?.map { item in generationLibrary.loraEntries.first { $0.id == item.reference.id }?.name ?? item.reference.family }.joined(separator: ", ") ?? "Not specified"
      }
      result.saved = names(review.before.adapters); result.current = names(review.after.adapters)
    }
    let labels = ["steps": "Steps", "seed": "Seed", "guidance": "Guidance (CFG)", "cfg": "Guidance (CFG)", "shift": "Shift", "sampler": "Sampler",
      "negativePrompt": "Negative guidance", "skipLayers": "Negative branch skip layers", "skipStart": "Skip start", "skipStop": "Skip stop",
      "schedule": "Schedule", "decoderMode": "Image decoder", "frameSelection": "Image frame", "temporalSigma": "Temporal sigma",
      "temporalCompression": "Motion sampling", "colorFix": "Color correction", "loraStrength": "Required adapter strength", "posteriorSeed": "Posterior seed"]
    if let last = change.path.split(separator: ".").last, let label = labels[String(last)] {
      result.label = label
      let parts = change.path.split(separator: ".")
      if parts.count > 3, parts[1] == "stages", let index = Int(parts[2]) { result.label = "Stage \(index + 1) · " + label }
    }
    for adapter in (review.before.adapters ?? []) + (review.after.adapters ?? []) where change.path.hasPrefix("adapters." + adapter.reference.id) {
      if let entry = generationLibrary.loraEntries.first(where: { $0.id == adapter.reference.id }) {
        result.label = result.label.replacingOccurrences(of: adapter.reference.id, with: entry.name)
      }
    }
    return result
  }
}
