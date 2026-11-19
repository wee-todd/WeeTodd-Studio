import StudioCore
import SwiftUI

struct GenerationModelPicker: View {
  @EnvironmentObject var store: StudioStore
  let target: GenerationPresetTarget
  var stageIndex: Int? = nil
  @State private var libraryOpen = false
  @State private var message: String?
  @State private var applying = false
  var body: some View {
    let current = store.currentGenerationLibraryModel(for: target, stageIndex: stageIndex)
    let context = try? store.generationLibraryContext(for: target, stageIndex: stageIndex)
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text("Model").font(.caption.bold())
        Menu {
          ForEach(context.map { store.generationLibrary.models(for: $0) } ?? []) { model in
            Button {
              do {
                let token = try store.presetTargetToken(for: target)
                applying = true; message = nil
                Task { @MainActor in
                  defer { applying = false }
                  do { try await store.useGenerationLibraryModel(model.reference, for: target, expectedTarget: token, stageIndex: stageIndex) }
                  catch { message = error.localizedDescription }
                }
              } catch { message = error.localizedDescription }
            } label: {
              Label(model.name + (model.availability == .installed ? "" : " · " + model.availability.label),
                systemImage: current?.id == model.id ? "checkmark" : "cube")
            }.disabled(!model.availability.canResolve)
          }
          Divider()
          Button("Models & LoRAs…") { libraryOpen = true }
        } label: { Text(current?.name ?? "Choose model…").lineLimit(2) }
        .accessibilityIdentifier("generation-model-picker").disabled(applying || store.operationBusy)
        if applying { ProgressView().controlSize(.small) }
      }
      HStack {
        if let current { Text(current.availability.label).foregroundStyle(current.availability == .installed ? Color.secondary : .orange) }
        Spacer()
        Button("Models & LoRAs…") { libraryOpen = true }.accessibilityIdentifier("generation-library-open")
      }.font(.caption2)
      if let message { Text(message).font(.caption2).foregroundStyle(.orange).textSelection(.enabled) }
    }
    .sheet(isPresented: $libraryOpen) {
      GenerationLibraryView(target: target, stageIndex: stageIndex, onClose: { libraryOpen = false }).environmentObject(store)
    }
  }
}
