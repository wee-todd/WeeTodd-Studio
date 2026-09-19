import StudioCore
import SwiftUI

/// Reuses image generation while preserving the user's ordinary image workspace.
struct ReferenceSheetGenerator: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  let subject: ReferenceSheetContext
  let referencePaths: [String]
  let onUse: (MediaAsset) async -> Bool
  @State private var context: ReferenceSheetContext
  @State private var lease: ReferenceWorkspaceLease?
  @State private var started = false
  @State private var attaching = false
  @State private var failure: String?
  @State private var setupExpanded = false
  @State private var referencesExpanded = false
  init(context: ReferenceSheetContext, referencePaths: [String], onUse: @escaping (MediaAsset) async -> Bool) {
    subject = context; self.referencePaths = referencePaths; self.onUse = onUse
    _context = State(initialValue: context)
  }
  var body: some View {
    Group {
    if started {
    ImageGenerationEditor(onClose: { dismiss() }, onUseReference: { asset in
      Task { @MainActor in
        do { try lease?.validate(store: store) } catch { failure = error.localizedDescription; return }
        attaching = true
        if await onUse(asset) { dismiss() }
        else { failure = "Could not attach this candidate. It is preserved in Project Assets; return to review and check the error." }
        attaching = false
      }
    }, referenceTools: AnyView(tools))
    } else { ProgressView("Preparing reference editor…") }
    }
      .frame(minWidth: 980, idealWidth: 1480, maxWidth: .infinity, minHeight: 680, idealHeight: 960, maxHeight: .infinity)
      .background(ReferenceEditorWindow())
      .disabled(attaching)
      .interactiveDismissDisabled(store.operationBusy || attaching)
      .onAppear { begin() }
      .onDisappear { restore() }
      .alert("Reference image", isPresented: Binding(get: { failure != nil }, set: { if !$0 { failure = nil } })) {
        Button("OK") { failure = nil }
      } message: { Text(failure ?? "") }
  }
  private var tools: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text(subject.name).font(.headline)
        Button(setupExpanded ? "Hide reference setup" : "Reference setup") { setupExpanded.toggle() }
        if !referencePaths.isEmpty {
          Button("Existing references (\(referencePaths.count))") { referencesExpanded.toggle() }
        }
        Spacer()
        Menu("Candidates") {
          ForEach(Array(store.project.assets.filter { $0.generation?.referenceSheet?.subjectKey == subject.subjectKey }.enumerated()), id: \.element.id) { index, asset in
            Button("\(asset.name) · Candidate \(index + 1)") { store.imagePreviewPath = asset.path }
          }
        }.disabled(!store.project.assets.contains { $0.generation?.referenceSheet?.subjectKey == subject.subjectKey })
      }
      if setupExpanded {
      HStack(spacing: 16) {
        Picker("Template", selection: $context.template) {
          ForEach(ReferenceSheetTemplate.allCases) { Text($0.label).tag($0) }
        }.frame(width: 310)
        TextField("Visual style", text: $context.style).textFieldStyle(.roundedBorder)
        Spacer()
        Button("Apply template") {
          if var draft = store.imageDraft { context.apply(to: &draft); store.imageDraft = draft; store.imageEstimate = nil }
        }
        Menu("Starting settings") {
          Button("Krea Turbo · 8 steps") { store.imageDraft?.steps = 8; store.imageEstimate = nil }
          Button("FLUX.2 Klein 4B / 9B · 4 steps, CFG 1") {
            store.imageDraft?.steps = 4; store.imageDraft?.guidance = 1; store.imageEstimate = nil
          }
          Text("Qwen Edit / other models: choose model, then edit settings or import its DT config.")
        }
      }
      TextField("Pose / camera instructions (optional)", text: $context.direction, axis: .vertical)
        .lineLimit(1...3).textFieldStyle(.roundedBorder)
      Text("Apply template rebuilds the prompt. Starting settings change steps/CFG only; choose your model and LoRAs separately.")
        .font(.caption).foregroundStyle(.secondary)
      }
      if referencesExpanded && !referencePaths.isEmpty {
        ScrollView(.horizontal) {
          HStack {
            Text("Existing references").font(.caption.bold())
            ForEach(ImagePreviewSelection.references(paths: referencePaths, subject: subject.name)) { item in
              HStack {
                PreviewableImage(path: item.path, title: item.title).frame(width: 60, height: 48)
                VStack(alignment: .leading) {
                  Text(item.title).font(.caption2).lineLimit(2).help(item.path)
                  HStack {
                    Button("Canvas") { store.loadImageInputs([URL(fileURLWithPath: item.path)], canvas: true) }
                    Button("Mood board") { store.loadImageInputs([URL(fileURLWithPath: item.path)], canvas: false) }
                      .disabled(!canAddMoodboard)
                  }.font(.caption)
                }.frame(width: 180)
              }
            }
          }
        }.frame(height: 62)
        Text("Canvas uses image-to-image strength; mood-board references require a compatible model. Existing references are available here and are never enabled silently.")
          .font(.caption2).foregroundStyle(.secondary)
      }
    }.padding(.horizontal, 16).padding(.vertical, 10).disabled(store.operationBusy)
  }
  private var canAddMoodboard: Bool {
    guard let draft = store.imageDraft else { return false }
    let rules = store.drawThingsCatalogs[draft.profileID]?["capabilities"] as? [String: Any]
    let operations = (rules?[draft.modelID] as? [String: Any])?["operations"] as? [String: Any]
    let spec = operations?["image"] as? [String: Any]
    let count = draft.moodboard.filter { $0.enabled && $0.strength > 0 }.count + 1
    let roles = (draft.canvas?.enabled == true ? ["canvas"] : []) + Array(repeating: "moodboard", count: count)
    return (spec?["inputRoleCombinations"] as? [[String]] ?? []).contains { $0.sorted() == roles.sorted() }
  }
  private func begin() {
    guard !started else { return }; started = true
    lease = ReferenceWorkspaceLease(store: store, subjectKey: subject.subjectKey)
    let draft = store.makeReferenceImageDraft(context, previousDraft: lease?.previousDraft)
    context = draft.referenceSheet ?? context
    let saved = store.imageWorkspaceLibrary.sessions[draft.storageKey]
    store.referenceSheetOpen = true
    store.restoringImageWorkspace = true
    store.imagePreviewPath = saved?.draft == draft ? saved?.previewPath : nil
    store.imageDraft = draft; store.imageEstimate = nil
    store.restoringImageWorkspace = false; store.persistImageWorkspace()
  }
  private func restore() {
    guard started else { return }
    _ = lease?.restore(store: store)
  }
}
