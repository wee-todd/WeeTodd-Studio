import AppKit
import ImageLayout
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct MOSSLayoutAnalysisView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @AppStorage("mossLayoutAnalyzerDirectory") private var modelDirectory = ""
  @State private var source: URL?
  @State private var analysis: CapturedMOSSLayoutAnalysis?
  @State private var job: Task<Void, Never>?
  @State private var message = ""
  var onApplied: (CapturedMOSSLayoutAnalysis) -> Void
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        Text("Layout from image").font(.title2)
        Spacer()
        Button("Done") { dismiss() }.disabled(job != nil).keyboardShortcut(.cancelAction)
      }
      Text("Analyze composition and lettering locally, then edit the proposed boxes and prompt before generating.")
        .font(.callout).foregroundStyle(.secondary)
      HSplitView {
        VStack(alignment: .leading, spacing: 12) {
          Button("Choose source image…") { chooseSource() }.disabled(job != nil)
          if let source {
            Text(source.lastPathComponent).font(.caption).textSelection(.enabled)
            CachedImageThumbnail(path: source.path, maximumPixelSize: 640)
              .aspectRatio(contentMode: .fit).frame(maxWidth: .infinity, maxHeight: 280)
          }
          Text("Target · about 2 megapixels").font(.headline)
          Text("Preserves the whole source and its aspect ratio on MOSS’s 16-pixel canvas grid.")
            .font(.caption).foregroundStyle(.secondary)
          Text("The source is an analysis guide. Generation uses your editable descriptions, boxes and lettering.")
            .font(.caption).foregroundStyle(.secondary)
          Text("Subject-box control is enabled when you use this layout. Moving a whole subject also moves its detail boxes.").font(.caption2).foregroundStyle(.secondary)
          Divider()
          Text("Local vision model").font(.headline)
          Text(modelDirectory.isEmpty ? "Choose installed MLX Qwen3.5-4B or Gemma4-12B vision weights." : modelDirectory)
            .font(.caption).textSelection(.enabled).lineLimit(3)
          Button("Choose model folder…") { chooseModel() }.disabled(job != nil)
          if let recommended = MOSSLayoutAnalysisPreparation.discoverModel(libraryDirectory: store.imageDraft?.mossImage?.libraryDirectory,
            sharedDirectory: store.imageDraft?.mossImage?.sharedDirectory), recommended != modelDirectory,
            MOSSLayoutAnalysisPreparation.visionModel(URL(fileURLWithPath: recommended)) == .gemma4_12b {
            Button("Use installed Gemma4-12B") { modelDirectory = recommended; analysis = nil; message = "" }.disabled(job != nil)
          }
          if MOSSLayoutAnalysisPreparation.visionModel(URL(fileURLWithPath: modelDirectory)) != .gemma4_12b {
            Link("Qwen3.5-4B MLX model", destination: URL(string: "https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit")!).font(.caption)
          }
          Text("Swift MLX + Apple Vision. Analysis weights unload before image generation.")
            .font(.caption2).foregroundStyle(.secondary)
          Spacer()
        }.padding(.trailing, 14).frame(minWidth: 220, idealWidth: 270, maxWidth: 310)
        VStack(alignment: .leading, spacing: 10) {
          if let analysis {
            let layout = analysis.proposal.layout
            Text("\(layout.regions.count) regions · \(layout.width) × \(layout.height) · \(Double(layout.width * layout.height) / 1_000_000, specifier: "%.2f") MP")
              .font(.headline)
            MOSSLayoutProposalPreview(layout: layout).frame(minHeight: 300)
            ScrollView {
              VStack(alignment: .leading, spacing: 8) {
                Text(layout.summary.caption).font(.callout).textSelection(.enabled)
                ForEach(Array(analysis.proposal.warnings.enumerated()), id: \.offset) { _, warning in
                  Label(warning, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
              }.frame(maxWidth: .infinity, alignment: .leading)
            }.frame(maxHeight: 150)
          } else {
            ContentUnavailableView("Create an editable layout", systemImage: "rectangle.3.group",
              description: Text("Choose an image, then analyze its subjects, placement, colors, lighting and readable text."))
            Text("Review the proposed layout before applying. Generation starts only when you choose Generate.")
              .font(.caption).foregroundStyle(.secondary)
          }
        }.padding(.leading, 14).frame(minWidth: 470, maxWidth: .infinity)
      }
      if !message.isEmpty { Text(message).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
      HStack {
        if job != nil {
          MOSSLayoutAnalysisProgress(bridge: store.bridge)
          Button("Cancel analysis") { job?.cancel(); store.bridge.cancel() }
        }
        Spacer()
        Button(analysis == nil ? "Analyze image" : "Analyze again") { analyze() }
          .disabled(job != nil || source == nil || modelDirectory.isEmpty)
        Button("Use layout") { apply() }.buttonStyle(.borderedProminent).disabled(job != nil || analysis == nil)
      }
    }.padding(24).frame(minWidth: 1000, minHeight: 740)
      .interactiveDismissDisabled(job != nil)
      .onAppear {
        if modelDirectory.isEmpty || !MOSSLayoutAnalysisPreparation.compatibleModel(URL(fileURLWithPath: modelDirectory)) {
          let settings = store.imageDraft?.mossImage
          modelDirectory = MOSSLayoutAnalysisPreparation.discoverModel(libraryDirectory: settings?.libraryDirectory,
            sharedDirectory: settings?.sharedDirectory) ?? ""
        }
      }
      .onDisappear { if job != nil { job?.cancel(); store.bridge.cancel() } }
  }
  private func chooseSource() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    source = url; analysis = nil; message = ""
  }
  private func chooseModel() {
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    guard MOSSLayoutAnalysisPreparation.compatibleModel(url) else { message = "Choose complete MLX Qwen3.5-4B Q4/Q8 or Gemma4-12B unified Q8 vision weights."; return }
    modelDirectory = url.path; analysis = nil; message = ""
  }
  private func analyze() {
    guard let source else { return }
    let model = modelDirectory
    analysis = nil; message = ""
    job = Task {
      defer { job = nil }
      do { analysis = try await store.analyzeMOSSLayout(source: source, modelDirectory: model) }
      catch is CancellationError { message = "Image analysis cancelled. The current layout was kept." }
      catch { message = error.localizedDescription }
    }
  }
  private func apply() {
    guard let analysis else { return }
    do { try store.applyMOSSLayoutAnalysis(analysis); onApplied(analysis); dismiss() }
    catch { message = error.localizedDescription }
  }
}

private struct MOSSLayoutAnalysisProgress: View {
  @ObservedObject var bridge: Bridge
  var body: some View {
    HStack { ProgressView().controlSize(.small); Text(bridge.busy ? bridge.message : "Preparing source…").font(.caption).lineLimit(2) }
  }
}

private struct MOSSLayoutProposalPreview: View {
  let layout: ImageLayoutDraft
  var body: some View {
    GeometryReader { geometry in
      let scale = min(geometry.size.width / Double(layout.width), geometry.size.height / Double(layout.height))
      let width = Double(layout.width) * scale, height = Double(layout.height) * scale
      ZStack(alignment: .topLeading) {
        if let path = layout.guideImagePath {
          CachedImageThumbnail(path: path, maximumPixelSize: 1280).aspectRatio(contentMode: .fit)
            .frame(width: width, height: height)
        }
        ForEach(Array(layout.regions.enumerated()), id: \.element.id) { index, region in
          let color: Color = region.type == .text ? .blue : region.type == .character ? .orange : .green
          Rectangle().stroke(color, lineWidth: 1.5)
            .overlay(alignment: .topLeading) { Text("\(index + 1)").font(.system(size: 10, weight: .bold)).padding(2).background(color).foregroundStyle(.white) }
            .frame(width: region.bounds.width * width, height: region.bounds.height * height)
            .offset(x: region.bounds.x * width, y: region.bounds.y * height)
        }
      }.frame(width: width, height: height).clipped()
        .frame(maxWidth: .infinity, maxHeight: .infinity).background(.black.opacity(0.8))
    }
  }
}
