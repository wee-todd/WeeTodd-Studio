import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct PromptAssistantView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  let context: PromptAssistantContext
  @AppStorage("qwen35PromptModelPath") private var modelPath = ""
  @State private var showModelSetup = false
  @State private var models: [URL] = []
  @State private var instructions = "Improve this prompt while preserving its intent."
  @State private var output = ""
  @State private var outputLimit = 512
  @State private var running = false
  @State private var error: String?
  @State private var summary = ""
  @State private var truncated = false
  @State private var repetition: String?
  @State private var showWorkflow = false
  @State private var images: [PromptAssistantImage] = []
  private var selectedImages: [PromptAssistantImage] { images.filter(\.enabled) }
  private var supportsVision: Bool { URL(fileURLWithPath: modelPath).lastPathComponent == LocalPromptModels.filenames[0] }

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Prompt Assistant").font(.title2)
        Spacer()
        Text("Local · Qwen3.5").foregroundStyle(.secondary)
        Button("Done") { dismiss() }.disabled(running)
      }
      Text("Plan and refine prompts locally with Qwen3.5. Reuse an installed model or set up the assistant in Studio.")
        .font(.callout).foregroundStyle(.secondary)
      HStack {
        Picker("Model", selection: $modelPath) {
          Text("Choose an installed model").tag("")
          ForEach(models, id: \.path) { url in Text(LocalPromptModels.label(for: url.path)).tag(url.path) }
        }.frame(maxWidth: 390)
        Button("Set up assistant…") { showModelSetup = true }
        Button("Rescan") { scan() }
        Spacer()
        Picker("Output tokens", selection: $outputLimit) {
          ForEach([128, 256, 512, 1024], id: \.self) { Text(String($0)).tag($0) }
        }.frame(width: 190)
      }.disabled(running)
      if modelPath.isEmpty {
        Text("Set up Qwen3.5 4B or 9B, or reuse a compatible installed checkpoint. H3's Qwen encoder is not a text generator.")
          .font(.caption).foregroundStyle(.secondary)
      } else {
        Text(modelPath).font(.caption2).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
      }
      HStack {
        Text("Images for the assistant · \(selectedImages.count)/8").font(.headline)
        Spacer()
        Button("Load images…") { loadImages() }.disabled(running)
        Button("Exclude all") { for i in images.indices { images[i].enabled = false } }.disabled(running || selectedImages.isEmpty)
      }
      if !images.isEmpty {
        ScrollView(.horizontal) {
          LazyHStack(alignment: .top, spacing: 12) {
            ForEach($images) { $image in
              VStack(alignment: .leading, spacing: 4) {
                WorkspaceImage(path: image.path).frame(width: 105, height: 68).clipped()
                Toggle(image.label, isOn: $image.enabled).font(.caption).lineLimit(2).frame(width: 140, alignment: .leading)
                  .help(image.path)
              }
            }
          }
        }.frame(height: 112).disabled(running)
      }
      if !selectedImages.isEmpty && !supportsVision {
        Text("Choose Qwen3.5 4B for image understanding, or exclude images for text-only generation.")
          .font(.caption).foregroundStyle(.orange)
      }
      HStack(alignment: .top, spacing: 16) {
        VStack(alignment: .leading, spacing: 8) {
          Text("Current prompt").font(.headline)
          ScrollView { Text(context.original.isEmpty ? "No prompt yet." : context.original)
              .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            .padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
          Text("Instructions").font(.headline)
          TextEditor(text: $instructions).font(.body).frame(height: 100).disabled(running)
          Text("Each run uses these instructions and the current prompt on the left. Apply a proposal to make it the source for a later session.")
            .font(.caption2).foregroundStyle(.secondary)
        }
        VStack(alignment: .leading, spacing: 8) {
          Text("Proposed text · editable").font(.headline)
          TextEditor(text: $output).font(.body).disabled(running)
          if truncated { Label("Output limit reached. Review the ending, or regenerate with more tokens.", systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange) }
          if let repetition { Text(repetition).font(.caption).foregroundStyle(.orange) }
          Text(summary).font(.caption).foregroundStyle(.secondary)
        }
      }.frame(minHeight: 190)
      Text(selectedImages.isEmpty ? "No images selected. The assistant will use text only." : "Checked images are supplied in the order shown, at up to 512 pixels per side. Original files stay unchanged.")
        .font(.caption).foregroundStyle(.secondary)
      if let error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
      HStack {
        if running {
          ProgressView().controlSize(.small)
          Text(store.bridge.message).font(.caption)
          Button("Cancel") { store.bridge.cancel() }
        } else {
          Button("Step-by-step…") { showWorkflow = true }.disabled(store.operationBusy || !supportsVision)
          Button("Generate text") { Task { await generate() } }
            .disabled(modelPath.isEmpty || instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || store.operationBusy || selectedImages.count > 8 || (!selectedImages.isEmpty && !supportsVision))
        }
        Spacer()
        Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(output, forType: .string) }
          .disabled(output.isEmpty || running)
        Button("Apply to prompt") { apply() }.buttonStyle(.borderedProminent)
          .disabled(output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || running)
      }
    }.padding(24).frame(width: 960, height: 720)
      .sheet(isPresented: $showModelSetup) {
        AssistantModelSetupView(store: store, currentModelPath: modelPath) { modelPath = $0; scan() }
      }
      .sheet(isPresented: $showWorkflow) {
        WorkflowView(sessionScope: context.persistenceScope, initialBuiltin: "weetodd.staged-prompt-editing",
          initialInputs: ["source": .string(context.original), "instructions": .string(instructions)],
          initialImages: selectedImages, onPrompt: { output = $0; summary = "Staged workflow proposal · review before applying" })
          .environmentObject(store)
      }
      .interactiveDismissDisabled(running).onAppear {
        images = context.images.enumerated().map { index, image in
          var value = image; value.enabled = index < 8; return value
        }
        scan()
      }
      .onDisappear { if running { store.bridge.cancel() } }
  }
  private func scan() {
    models = LocalPromptModels.discover()
    if !modelPath.isEmpty, !models.contains(where: { $0.path == modelPath }) {
      models.append(URL(fileURLWithPath: modelPath))
    }
    if modelPath.isEmpty { modelPath = models.first?.path ?? "" }
  }
  private func locate() {
    let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
    panel.message = "Select qwen_3.5_4b_i8x.ckpt or qwen_3.5_9b_i5x.ckpt in your Draw Things model store."
    panel.directoryURL = modelPath.isEmpty ? LocalPromptModels.drawThingsDirectory : URL(fileURLWithPath: modelPath).deletingLastPathComponent()
    guard panel.runModal() == .OK, let url = panel.url else { return }
    guard LocalPromptModels.filenames.contains(url.lastPathComponent) else {
      error = "Choose a supported Draw Things Qwen3.5 checkpoint. Other Qwen versions and H3 encoders are not supported."; return
    }
    modelPath = url.path; error = nil; scan()
  }
  private func loadImages() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
    panel.message = "Choose images for local prompt assistance. Originals are referenced in place."
    guard panel.runModal() == .OK else { return }
    for url in panel.urls where !images.contains(where: { $0.path == url.path }) {
      var item = PromptAssistantImage(path: url.path, label: "Reference · " + String(url.deletingPathExtension().lastPathComponent.prefix(80)))
      item.enabled = selectedImages.count < 8; images.append(item)
    }
  }
  private func generate() async {
    let currentInstructions = instructions
    let previous = output
    running = true; error = nil; summary = ""; truncated = false; repetition = nil
    output = ""
    defer { running = false }
    do {
      let response = try await store.bridge.invoke("assist-prompt", runtime: store.runtime, payload: [
        "textRequest": ["modelPath": modelPath, "systemPrompt": context.systemPrompt(imageCount: selectedImages.count),
          "prompt": context.userPrompt(instructions: currentInstructions),
          "maxTokens": outputLimit, "images": selectedImages.map(\.request)]])
      guard let text = response["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw StudioError.invalid("The assistant returned no text.")
      }
      repetition = PromptAssistantContext.repetitionNotice(output: text, original: context.original, previous: previous)
      output = text; truncated = response["truncated"] as? Bool ?? false
      let seconds = response["totalSeconds"] as? Double ?? 0
      summary = String(format: "%.1f seconds · %d output tokens · %d images · model unloaded", seconds, response["outputTokens"] as? Int ?? 0, response["imagesUsed"] as? Int ?? 0)
    } catch { self.error = error.localizedDescription }
  }
  private func apply() {
    do {
      try context.validate(project: store.project, image: store.imageDraft, documentSessionID: store.documentSessionID, assets: store.allAssets)
      if let id = context.clipID, let index = store.project.clips.firstIndex(where: { $0.id == id }) {
        store.change { $0.clips[index].prompt = output }
      } else { store.imageDraft?.prompt = output; store.imageEstimate = nil }
      store.notice = "Assistant text applied. Review the prompt before generation."
      dismiss()
    } catch { self.error = error.localizedDescription }
  }
}
