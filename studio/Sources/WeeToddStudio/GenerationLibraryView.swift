import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

/// Links installed files in place. Browsing never downloads or starts generation.
struct GenerationLibraryView: View {
  @EnvironmentObject var store: StudioStore
  var target: GenerationPresetTarget? = nil
  var stageIndex: Int? = nil
  var onClose: (() -> Void)? = nil
  @Environment(\.dismissWindow) private var dismissWindow
  @State private var tab = 0
  @State private var search = ""
  @State private var compatibleOnly = true
  @State private var modelID: String?
  @State private var loraID: String?
  @State private var message: String?
  @State private var busy = false
  @State private var advancedOpen = false
  private var context: GenerationLibraryContext? { target.flatMap { try? store.generationLibraryContext(for: $0, stageIndex: stageIndex) } }
  private var current: GenerationModelEntry? { target.flatMap { store.currentGenerationLibraryModel(for: $0, stageIndex: stageIndex) }
    ?? store.generationLibrary.modelEntries.first { $0.id == modelID } }
  private var models: [GenerationModelEntry] {
    let all = compatibleOnly ? context.map { store.generationLibrary.models(for: $0) } ?? store.generationLibrary.modelEntries : store.generationLibrary.modelEntries
    return all.filter { search.isEmpty || ($0.name + " " + $0.reference.family + " " + $0.reference.backend).localizedCaseInsensitiveContains(search) }
      .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  private var loras: [GenerationLoRAEntry] {
    let all: [GenerationLoRAEntry]
    if compatibleOnly, let context, let model = current { all = store.generationLibrary.loras(for: context, model: model.reference) }
    else { all = store.generationLibrary.loraEntries }
    return all.filter { search.isEmpty || ($0.name + " " + $0.reference.family).localizedCaseInsensitiveContains(search) }
      .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
  }
  private func backendName(_ name: String) -> String {
    switch name {
    case "nativeMOSS", "moss_image": return "Local · MOSS Image"
    case "nativeH3", "h3": return "Local · H3"
    case "nativeMLX", "qwen_image": return "Local · Qwen Image"
    case "ltx25": return "Local · LTX 2.5"
    case "ltx23": return "Local · LTX 2.3"
    case "ripple": return "Local · Ripple"
    case "restoration": return "Local restoration"
    case "drawThings": return "Draw Things"
    default: return name
    }
  }
  private func status(_ entry: GenerationModelEntry) -> String { entry.reference.backend == "drawThings" && entry.availability == .installed ? "On connection" : entry.availability.label }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Models & LoRAs").font(.title2.bold())
        Spacer()
        if store.generationLibraryBusy || busy { ProgressView().controlSize(.small) }
        Button("Refresh") { Task { await store.refreshGenerationLibrary(force: true) } }.disabled(busy || store.generationLibraryBusy)
        Button("Done") { if let onClose { onClose() } else { dismissWindow(id: "generation-library") } }.keyboardShortcut(.defaultAction)
      }
      HStack {
        TextField("Search installed models and LoRAs", text: $search).textFieldStyle(.roundedBorder)
        Picker("Show", selection: $tab) { Text("Models").tag(0); Text("LoRAs").tag(1) }.pickerStyle(.segmented).frame(width: 190)
        if target != nil { Toggle("Compatible only", isOn: $compatibleOnly).toggleStyle(.checkbox) }
      }
      HSplitView {
        Group {
          if tab == 0 {
            List(models, selection: $modelID) { model in
              VStack(alignment: .leading, spacing: 4) {
                HStack { Text(model.name).font(.headline); if current?.id == model.id { Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint) } }
                Text(backendName(model.reference.backend) + " · " + status(model)).font(.caption).foregroundStyle(.secondary)
              }.padding(.vertical, 5).tag(model.id)
            }
          } else {
            List(loras, selection: $loraID) { lora in
              VStack(alignment: .leading, spacing: 4) {
                Text(lora.name).font(.headline)
                Text(backendName(lora.reference.family) + " · " + lora.availability.label).font(.caption).foregroundStyle(.secondary)
              }.padding(.vertical, 5).tag(lora.id)
            }
          }
        }.frame(minWidth: 340)
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            if tab == 0, let entry = store.generationLibrary.modelEntries.first(where: { $0.id == modelID }) {
              Text(entry.name).font(.title3.bold())
              Text(backendName(entry.reference.backend) + " · " + status(entry)).font(.subheadline)
              if let detail = entry.availability.detail ?? entry.detail { Text(detail).font(.caption).foregroundStyle(.secondary) }
              Text("Tasks: " + entry.tasks.joined(separator: ", ")).font(.caption)
              if let target {
                Button("Use model") { perform {
                  let token = try store.presetTargetToken(for: target)
                  try await store.useGenerationLibraryModel(entry.reference, for: target, expectedTarget: token, stageIndex: stageIndex)
                } }.disabled(busy || !entry.availability.canResolve || context.map { !entry.supports($0) } != false)
              }
              ForEach(entry.binding?.localPaths ?? [], id: \.self) { item in
                Text(item).font(.caption.monospaced()).textSelection(.enabled)
              }
            } else if tab == 1, let entry = store.generationLibrary.loraEntries.first(where: { $0.id == loraID }) {
              Text(entry.name).font(.title3.bold())
              Text(entry.availability.label).font(.subheadline)
              Text(entry.availability.detail ?? entry.detail ?? "Adjust strength and enable or disable it in the working LoRA stack.").font(.caption).foregroundStyle(.secondary)
              if let requirements = entry.recipeRequirements { Text(requirements).font(.callout).textSelection(.enabled) }
              if let target, let current {
                Button("Add to LoRA stack") { perform {
                  let token = try store.presetTargetToken(for: target)
                  try await store.addGenerationLibraryLoRA(entry.reference, for: target, expectedTarget: token)
                } }.disabled(busy || !entry.availability.canResolve || !entry.supports(current))
              }
              if case .nativeFile(let file, _, _, _) = entry.binding { Text(file).font(.caption.monospaced()).textSelection(.enabled) }
            } else {
              Text(tab == 0 ? "Select a model to see its installed components." : "Select a LoRA to see its compatibility.").foregroundStyle(.secondary)
            }
          }.frame(maxWidth: .infinity, alignment: .leading).padding(16)
        }.frame(minWidth: 260, idealWidth: 310)
      }
      HStack {
        Button("Link model library…") { choose(.libraryRoot, folder: true) }
        Menu("Add existing…") {
          Button("Model folder…") { chooseModel(folder: true, use: false) }
          Button("Model recipe or manifest…") { chooseModel(folder: false, use: false) }
          if let model = current { Button("LoRA file…") { chooseLoRA(model, use: false) } }
        }
        if target != nil {
          Menu("Add and use…") {
            Button("Model folder…") { chooseModel(folder: true, use: true) }
            Button("Model recipe or manifest…") { chooseModel(folder: false, use: true) }
            if let model = current { Button("LoRA file…") { chooseLoRA(model, use: true) } }
          }
        }
        Spacer()
        Text("Files stay in place").font(.caption).foregroundStyle(.secondary)
      }.disabled(busy)
      DisclosureGroup("Advanced setup") {
        HStack { Button("Studio model setup…") { advancedOpen = true }; Text("Component links, downloads and native conversions").font(.caption).foregroundStyle(.secondary) }
      }
      if !store.generationLibraryRootDirectory.isEmpty { Text("Library: " + store.generationLibraryRootDirectory).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled) }
      if let message = message ?? store.generationLibraryError { Text(message).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
    }.padding(22).frame(minWidth: 820, idealWidth: 940, minHeight: 520, idealHeight: 650).background(Theme.background)
      .task { await store.refreshGenerationLibrary(); if modelID == nil { modelID = current?.id } }
      .sheet(isPresented: $advancedOpen) { RuntimeView(onClose: { advancedOpen = false }).environmentObject(store) }
  }
  private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
    busy = true; message = nil
    Task { @MainActor in
      defer { busy = false }
      do { try await operation() } catch { message = error.localizedDescription }
    }
  }
  private func chooseModel(folder: Bool, use: Bool) {
    do { choose(.model(target: target, expectedTarget: try target.map { try store.presetTargetToken(for: $0) }, addAndUse: use, stageIndex: stageIndex), folder: folder) }
    catch { message = error.localizedDescription }
  }
  private func chooseLoRA(_ model: GenerationModelEntry, use: Bool) {
    do { choose(.lora(model: model.reference, target: target, expectedTarget: try target.map { try store.presetTargetToken(for: $0) }, addAndUse: use), folder: false) }
    catch { message = error.localizedDescription }
  }
  private func choose(_ intent: GenerationLibraryImportIntent, folder: Bool) {
    let panel = NSOpenPanel(); panel.canChooseFiles = !folder; panel.canChooseDirectories = folder; panel.allowsMultipleSelection = false
    panel.prompt = "Link existing"; panel.directoryURL = URL(fileURLWithPath: store.generationLibraryRootDirectory.isEmpty ? "/Volumes" : store.generationLibraryRootDirectory)
    if panel.runModal() == .OK, let url = panel.url {
      perform { let result = try await store.linkGenerationLibraryItem(url, intent: intent); message = result.detail ?? "Added to the library." }
    }
  }
}
