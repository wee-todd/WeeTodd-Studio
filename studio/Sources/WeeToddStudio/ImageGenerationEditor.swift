import AppKit
import ImageIO
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct ImageGenerationEditor: View {
  @EnvironmentObject var store: StudioStore
  var onClose: (() -> Void)? = nil
  var onUseReference: ((MediaAsset) -> Void)? = nil
  var referenceTools: AnyView? = nil
  var backLabel: String? = nil
  @State private var showResult = true
  @State private var zoom = 1.0
  @State private var loraLibraryOpen = false
  @State private var assistant: PromptAssistantContext?
  @State private var moodboardVisible = false
  @State private var configOpen = false
  @State private var connectionsOpen = false
  @State private var inspection: ImagePreviewSelection?
  @State private var promptExpanded = true
  var draft: DrawThingsImageDraft? { store.imageDraft }
  var livePreview: BridgeProgressEvent? { store.bridge.busy ? store.bridge.livePreview : nil }
  var connection: DrawThingsConnection? {
    isNative ? nil : store.drawThingsConnections.first { $0.id == draft?.profileID }
  }
  func binding<T>(_ key: WritableKeyPath<DrawThingsImageDraft, T>, fallback: T) -> Binding<T> {
    Binding(get: { store.imageDraft?[keyPath: key] ?? fallback }, set: { value in
      store.imageDraft?[keyPath: key] = value; store.imageEstimate = nil
    })
  }
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Button { if let onClose { onClose() } else { store.imageDraft = nil } } label: {
          Label(backLabel ?? (onClose == nil ? "Back to movie" : "Back to approval"), systemImage: "arrow.left")
        }.disabled(store.bridge.busy)
          .keyboardShortcut(.escape, modifiers: [])
        Divider().frame(height: 20)
        TextField("Image name", text: binding(\.name, fallback: "Generated image")).textFieldStyle(.roundedBorder)
          .frame(minWidth: 180, maxWidth: 380)
        Spacer()
        Button(moodboardVisible ? "Hide mood board" : "Mood board (\(draft?.moodboard.count ?? 0))") { moodboardVisible.toggle() }
        Menu("Tools") {
        Button("Prompt Assistant…") {
          if let draft { assistant = PromptAssistantContext(projectID: store.project.id, image: draft, documentSessionID: store.documentSessionID) }
        }.disabled(store.bridge.busy)
        Button("Import Config…") { store.configImportClipID = nil; configOpen = true }.disabled(isNative)
        Link("Draw Things presets", destination: DrawThingsConfigImport.presetsURL)
        Button("Export Headless Job…") { store.exportDrawThingsImageJob() }.disabled(store.bridge.busy || store.preparingImageRequest || draft?.modelID.isEmpty != false)
        }
      }.padding(16)
      Divider()
      if let referenceTools { referenceTools }
      if let error = store.referenceImageError {
        Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 16).padding(.vertical, 8)
      }
      HSplitView {
        settings.frame(minWidth: 260, idealWidth: 300, maxWidth: 330)
        VStack(spacing: 12) {
          HStack {
            Text(livePreview != nil ? "LIVE PREVIEW · APPROXIMATE" : showResult && store.imagePreviewPath != nil ? "RESULT" : "CANVAS").font(.caption.bold())
            if store.imagePreviewPath != nil {
              Picker("Display", selection: $showResult) { Text("Input").tag(false); Text("Result").tag(true) }.labelsHidden().pickerStyle(.segmented).frame(width: 130)
                .disabled(livePreview != nil)
            }
            Spacer()
            Button("Inspect image") {
              if let path = showResult ? store.imagePreviewPath ?? draft?.canvas?.path : draft?.canvas?.path {
                inspection = ImagePreviewSelection(path: path, title: draft?.name)
              }
            }.disabled(livePreview != nil || (store.imagePreviewPath == nil && draft?.canvas == nil))
            Button("Fit") { zoom = 1 }
            Slider(value: $zoom, in: 0.5...3).frame(width: 100).help("Canvas zoom")
          }.font(.caption)
          GeometryReader { geometry in
            let path = livePreview?.previewPath ?? (showResult ? store.imagePreviewPath ?? draft?.canvas?.path : draft?.canvas?.path)
            ZStack {
              Color.black.opacity(0.9).allowsHitTesting(false)
              if let path {
                let ratio = CGFloat(max(1, draft?.width ?? 512)) / CGFloat(max(1, draft?.height ?? 512))
                let width = min(geometry.size.width, geometry.size.height * ratio)
                ScrollView([.horizontal, .vertical]) {
                WorkspaceImage(path: path, fill: !(showResult && store.imagePreviewPath != nil) && draft?.canvas?.fit == "fill")
                  .id(livePreview.map { "live-\($0.previewRevision ?? 0)" } ?? path)
                  .frame(width: width * zoom, height: width / ratio * zoom).clipped()
                  .allowsHitTesting(false)
                  .frame(minWidth: geometry.size.width, minHeight: geometry.size.height)
                }.contentShape(Rectangle()).clipped().accessibilityLabel("Image viewport")
              } else {
                VStack(spacing: 12) {
                  Image(systemName: "photo.badge.plus").font(.largeTitle)
                  Text("Drop a canvas image here").font(.headline)
                  Text("Or leave it empty for text-to-image / mood-board generation.").font(.caption)
                  Button("Load image…") { store.chooseImageInputs(canvas: true) }
                }.foregroundStyle(.white)
              }
            }.frame(width: geometry.size.width, height: geometry.size.height).contentShape(Rectangle()).clipped()
              .onDrop(of: [UTType.fileURL.identifier, UTType.text.identifier], isTargeted: nil) { store.dropImageInputs($0, canvas: true) }
          }.frame(minHeight: 220)
          HStack {
            Button("Load canvas…") { store.chooseImageInputs(canvas: true); showResult = false }
            assetsMenu(canvas: true)
            if showResult, let path = store.imagePreviewPath {
              Button("Use result as canvas") { store.loadImageInputs([URL(fileURLWithPath: path)], canvas: true); showResult = false }
                .disabled(store.bridge.busy)
            }
            Spacer()
          }.font(.caption)
          if draft?.canvas != nil {
          HStack {
            if draft?.canvas != nil {
              Toggle("Use canvas", isOn: Binding(get: { draft?.canvas?.enabled ?? false }, set: { store.imageDraft?.canvas?.enabled = $0; store.imageEstimate = nil }))
              if !isNative { Picker("Placement", selection: Binding(get: { draft?.canvas?.fit ?? "fit" }, set: { store.imageDraft?.canvas?.fit = $0; store.imageEstimate = nil })) {
                Text("Fit · preserve image").tag("fit"); Text("Fill · crop edges").tag("fill")
              }.frame(maxWidth: 190) }
              Button("Clear") { store.imageDraft?.canvas = nil; store.imageEstimate = nil }
            }
            Spacer()
          }.font(.caption)
          }
          HStack {
            Text("Prompt").font(.caption.bold()); Spacer()
            Button(promptExpanded ? "Hide prompt" : "Show prompt") { promptExpanded.toggle() }.font(.caption)
          }
          if promptExpanded {
          TextEditor(text: binding(\.prompt, fallback: "")).font(.system(size: 14))
            .scrollContentBackground(.hidden).padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
            .frame(minHeight: 120, idealHeight: 180, maxHeight: 210).overlay(alignment: .topLeading) {
              if draft?.prompt.isEmpty != false { Text("Describe the image or edit…").foregroundStyle(.secondary).padding(15).allowsHitTesting(false) }
            }
          }
        }.padding(16).frame(minWidth: 470, maxWidth: .infinity)
        if moodboardVisible { moodboard.frame(minWidth: 205, idealWidth: 235, maxWidth: 290) }
      }
      Divider()
      HStack {
        if store.bridge.busy {
          ProgressView(value: store.bridge.fraction).frame(width: 90)
          Text(store.bridge.message).font(.caption)
          Button("Cancel") { store.bridge.cancel() }
        } else { Text("Ready · saves to \(draft?.destination.scope.rawValue ?? "project") assets").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        if let onUseReference {
          Button("Use as reference") {
            if let asset = usableReference { onUseReference(asset) }
          }.disabled(store.bridge.busy || usableReference == nil)
        }
        Button(isNative ? "Check Settings" : "Check Settings & CU") { Task { await store.prepareImageGeneration() } }.disabled(store.bridge.busy || store.preparingImageRequest || draft?.modelID.isEmpty != false)
        Button("Generate Image") { Task { await store.generateImageAsset() } }.buttonStyle(.borderedProminent)
          .disabled(store.bridge.busy || store.preparingImageRequest || store.imageEstimate?["eligibility"] as? String != "allowed")
      }.padding(16)
    }.background(Theme.background).frame(maxWidth: .infinity, maxHeight: .infinity)
      .sheet(item: $assistant) { PromptAssistantView(context: $0).environmentObject(store) }
      .sheet(item: $inspection) { ImagePreview(path: $0.path, title: $0.title) }
      .sheet(isPresented: $configOpen) { DrawThingsConfigImportView(onClose: { configOpen = false }).environmentObject(store) }
      .sheet(isPresented: $connectionsOpen) { DrawThingsSettings(onClose: { connectionsOpen = false }).environmentObject(store) }
      .sheet(isPresented: $loraLibraryOpen) { LoRALibrary(imageWorkspace: true).environmentObject(store) }
      .onAppear { moodboardVisible = draft?.moodboard.isEmpty == false }
      .onChange(of: draft?.moodboard.count) { _, count in if (count ?? 0) > 0 { moodboardVisible = true } }
      .onChange(of: showResult) { _, _ in zoom = 1 }
      .onChange(of: store.imagePreviewPath) { _, path in if path != nil { showResult = true; zoom = 1 } }
      .onChange(of: connection, initial: true) { _, connection in
        if let connection { Task { await store.discoverDrawThings(connection) } }
      }
  }
  private var usableReference: MediaAsset? {
    store.allAssets.first {
      $0.path == store.imagePreviewPath
        && $0.generation?.referenceSheet?.subjectKey == draft?.referenceSheet?.subjectKey
        && $0.generation?.rippleReference == draft?.rippleReference
    }
  }
  var settings: some View {
    Form {
      backendPicker
      if isNative { nativeSettings } else {
      Section("Draw Things") {
        Picker("Connection", selection: Binding(get: { draft?.profileID ?? "" }, set: { id in
          store.selectImageConnection(id)
        })) {
          Text("Choose a connection").tag("")
          ForEach(store.drawThingsConnections) { Text($0.name).tag($0.id) }
        }.disabled(store.bridge.busy)
        HStack {
          Button("Connections…") { connectionsOpen = true }
          Button("Refresh") {
            if let connection { Task { await store.discoverDrawThings(connection, force: true) } }
          }.disabled(connection == nil || store.drawThingsDiscovery.loading.contains(draft?.profileID ?? ""))
        }
        let models = store.drawThingsModels(draft?.profileID ?? "", operation: "image")
        let hasCatalog = store.drawThingsCatalogs[draft?.profileID ?? ""] != nil
        let loading = store.drawThingsDiscovery.loading.contains(draft?.profileID ?? "")
        let discoveryError = store.drawThingsDiscovery.errors[draft?.profileID ?? ""]
        if connection == nil {
          Text("Choose a connection to load its image models.").font(.caption).foregroundStyle(.secondary)
        } else if loading {
          HStack { ProgressView().controlSize(.small); Text("Loading image models…").font(.caption) }
        } else if let discoveryError {
          Text((hasCatalog ? "Refresh failed; showing the last loaded models. " : "Could not load image models. ") + discoveryError)
            .font(.caption).foregroundStyle(.orange).textSelection(.enabled)
          Button("Retry loading models") { if let connection { Task { await store.discoverDrawThings(connection, force: true) } } }
        } else if hasCatalog {
          Text(models.isEmpty ? "No supported image models were reported by this connection." : "\(models.count) image models available")
            .font(.caption).foregroundStyle(.secondary)
        }
        Picker("Model", selection: Binding(get: { draft?.modelID ?? "" }, set: { id in
          let compatible: Set<String>? = hasCatalog
            ? Set(store.drawThingsLoRAs(profileID: draft?.profileID ?? "", modelID: id).map(\.id)) : nil
          store.imageDraft?.selectModel(id, compatibleLoRAIDs: compatible); store.imageEstimate = nil
        })) {
          Text("Choose an image model").tag("")
          if let id = draft?.modelID, !id.isEmpty, !models.contains(where: { $0.id == id }) {
            Text("\(id) · \(hasCatalog ? "not in this catalog" : "saved selection; loading catalog")").tag(id).disabled(true)
          }
          ForEach(models, id: \.id) { Text($0.name).tag($0.id) }
        }.disabled(store.bridge.busy || connection == nil || (loading && models.isEmpty))
        if hasCatalog, let draft, !draft.modelID.isEmpty, !models.contains(where: { $0.id == draft.modelID }) {
          Text("The saved model was not reported by this connection. Refresh its inventory or choose another image model.")
            .font(.caption).foregroundStyle(.orange)
        } else if hasCatalog, let draft, !draft.modelID.isEmpty, !store.imageModelsForInputs().contains(where: { $0.id == draft.modelID }) {
          Text("This model does not support the enabled image inputs. Disable them, move one to the canvas, or choose another model.")
            .font(.caption).foregroundStyle(.orange)
        }
      }
      }
      Section("Generation") {
        TextField("Width", value: binding(\.width, fallback: 512), format: .number.grouping(.never))
        TextField("Height", value: binding(\.height, fallback: 512), format: .number.grouping(.never))
        Text(isNative ? "Dimensions: multiples of 32" : "Dimensions: multiples of 64").font(.caption2).foregroundStyle(.secondary)
        TextField("Steps", value: binding(\.steps, fallback: 4), format: .number.grouping(.never))
        TextField("CFG", value: binding(\.guidance, fallback: 1), format: .number).disabled(isNative)
        if !isNative, draft?.canvas?.enabled == true {
          Text("Generation strength · \(Int((draft?.strength ?? 1) * 100))%")
          Slider(value: binding(\.strength, fallback: 1), in: 0...1)
          Text("Higher values allow more regeneration. Editing models also use the canvas as a reference.").font(.caption2).foregroundStyle(.secondary)
        }
        Picker("Seed mode", selection: binding(\.randomSeedEachGeneration, fallback: true)) {
          Text("Random each generation").tag(true)
          Text("Fixed seed").tag(false)
        }
        if draft?.randomSeedEachGeneration != false {
          Text("A fresh seed is chosen for every generation. Saved settings keep −1; each result records its actual seed.").font(.caption2).foregroundStyle(.secondary)
        } else {
          TextField("Seed", value: binding(\.seed, fallback: 0), format: .number.grouping(.never))
          Text("The same seed and settings reproduce the same image. Enter −1 or choose Random each generation for variations.").font(.caption2).foregroundStyle(.secondary)
          Button("New fixed seed") { store.imageDraft?.seed = Int.random(in: 0...Int(UInt32.max)); store.imageEstimate = nil }
        }
        if isNative { Text("Euler · automatic shift").font(.caption) } else {
        Picker("Sampler", selection: binding(\.sampler, fallback: nil)) {
          Text("Server default").tag(nil as Int?)
          ForEach(Array(Self.samplers.enumerated()), id: \.offset) { index, name in Text(name).tag(Optional(index)) }
        }
        Toggle("Override Shift", isOn: Binding(get: { draft?.shift != nil }, set: { store.imageDraft?.shift = $0 ? 1 : nil; store.imageEstimate = nil }))
        if draft?.shift != nil { TextField("Shift", value: Binding(get: { draft?.shift ?? 1 }, set: { store.imageDraft?.shift = $0; store.imageEstimate = nil }), format: .number) }
        DisclosureGroup("Negative prompt") { TextField("Negative prompt", text: binding(\.negativePrompt, fallback: ""), axis: .vertical) }
        }
      }
      if !isNative { Section("LoRAs & Groups") { loraControls } }
      Section("Preflight") {
        if let estimate = store.imageEstimate {
          if !isNative { Text("Estimated CU: \(number(estimate["estimateCU"]))")
          Text(estimate["limitMode"] as? String == "notApplicable" ? "Self-hosted · no cloud CU limit" : "CU eligibility checked for this request").font(.caption) }
          if isNative, let memory = estimate["memorySummary"] as? String { Text(memory).font(.caption) }
          ForEach(Array(store.imagePreflightIssues.enumerated()), id: \.offset) { _, message in Text(message).font(.caption) }
        } else { Text(isNative ? "Check settings after changing inputs or settings." : "Check Settings & CU after changing inputs or settings.").font(.caption).foregroundStyle(.secondary) }
      }
    }.formStyle(.grouped)
  }
  var moodboard: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack { Text("MOOD BOARD").font(.caption.bold()); Spacer(); Button { store.chooseImageInputs(canvas: false) } label: { Image(systemName: "plus") } }
      assetsMenu(canvas: false)
      Text(isNative ? "\(draft?.activeImageInputs.count ?? 0) / 10 active images including canvas" : "Ordered references · up to 8 active").font(.caption2).foregroundStyle(.secondary)
      if let issue = draft?.imageInputIssue { Text(issue).font(.caption).foregroundStyle(.orange) }
      if isNative { Text("Order defines <image1> through <image10>. Review numbered prompts after changing inputs.").font(.caption2) }
      ScrollView {
        LazyVStack(spacing: 14) {
          ForEach(Array((draft?.moodboard ?? []).enumerated()), id: \.element.id) { index, item in
            VStack(spacing: 6) {
              CachedImageThumbnail(path: item.path, maximumPixelSize: 512).frame(height: 115).background(.black.opacity(0.15)).clipped()
              HStack {
                Toggle(referenceLabel(item), isOn: Binding(get: { item.enabled && (!isNative || item.strength > 0) }, set: { enabled in
                  guard let index = store.imageDraft?.moodboard.firstIndex(where: { $0.id == item.id }) else { return }
                  store.imageDraft?.moodboard[index].enabled = enabled
                  if isNative && enabled && item.strength == 0 { store.imageDraft?.moodboard[index].strength = 1 }
                  store.imageEstimate = nil
                }))
                Spacer()
                Button { store.imageDraft?.moodboard.removeAll { $0.id == item.id }; store.imageEstimate = nil } label: { Image(systemName: "xmark") }
              }
              Text(URL(fileURLWithPath: item.path).lastPathComponent).font(.caption2).lineLimit(1)
              Button("Inspect…") { inspection = ImagePreviewSelection(path: item.path, title: referenceLabel(item)) }
              Button("Replace…") { store.replaceImageReference(item.id) }
              HStack {
                if !isNative { Text("Strength \(Int(item.strength * 100))%") }
                Spacer()
                Button { moveReference(index, -1) } label: { Image(systemName: "arrow.up") }.disabled(index == 0)
                Button { moveReference(index, 1) } label: { Image(systemName: "arrow.down") }.disabled(index == (draft?.moodboard.count ?? 0) - 1)
              }
              if !isNative { Slider(value: referenceBinding(item.id, \.strength, item.strength), in: 0...1) }
            }.font(.caption).padding(9).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8)).opacity(item.enabled ? 1 : 0.5)
              .draggable("mood:" + item.id.uuidString)
              .onDrop(of: [UTType.text.identifier], isTargeted: nil) { providers in
                guard let provider = providers.first else { return false }
                provider.loadObject(ofClass: NSString.self) { value, _ in
                  let text = value as? String ?? ""
                  Task { @MainActor in
                    if text.hasPrefix("mood:"), let id = UUID(uuidString: String(text.dropFirst(5))),
                      let from = store.imageDraft?.moodboard.firstIndex(where: { $0.id == id }),
                      let to = store.imageDraft?.moodboard.firstIndex(where: { $0.id == item.id }) {
                      store.imageDraft?.moodboard.move(fromOffsets: IndexSet(integer: from), toOffset: to > from ? to + 1 : to)
                      store.imageEstimate = nil
                    } else { _ = store.dropImageInputs(providers, canvas: false) }
                  }
                }
                return true
              }
          }
          if draft?.moodboard.isEmpty != false { Text("Drop reference images here").foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 140) }
        }
      }.onDrop(of: [UTType.fileURL.identifier, UTType.text.identifier], isTargeted: nil) { store.dropImageInputs($0, canvas: false) }
      if !isNative { Text("FLUX.2/Klein treats positive reference weights as enabled images. Weight is sent unchanged, but may not scale influence. 0% omits the reference.").font(.caption2).foregroundStyle(.secondary) }
    }.padding(14)
  }
  func referenceBinding<T>(_ id: UUID, _ key: WritableKeyPath<ImageWorkspaceInput, T>, _ fallback: T) -> Binding<T> {
    Binding(get: { draft?.moodboard.first { $0.id == id }?[keyPath: key] ?? fallback }, set: { value in
      if let index = store.imageDraft?.moodboard.firstIndex(where: { $0.id == id }) { store.imageDraft?.moodboard[index][keyPath: key] = value; store.imageEstimate = nil }
    })
  }
  func moveReference(_ index: Int, _ offset: Int) {
    store.imageDraft?.moodboard.swapAt(index, index + offset); store.imageEstimate = nil
  }
  func assetsMenu(canvas: Bool) -> some View {
    Menu("From Assets") {
      ForEach(store.allAssets.filter { $0.kind == .image }) { asset in
        Button(asset.name) { store.loadImageInputs([URL(fileURLWithPath: asset.path)], canvas: canvas) }
      }
    }.disabled(!store.allAssets.contains { $0.kind == .image })
  }
  var loraControls: some View {
    let available = store.drawThingsLoRAs(profileID: draft?.profileID ?? "", modelID: draft?.modelID ?? "")
    let hasCatalog = store.drawThingsCatalogs[draft?.profileID ?? ""] != nil
    return VStack(alignment: .leading, spacing: 8) {
      Button("Add / Groups…") { loraLibraryOpen = true }
      if draft?.loras.isEmpty != false {
        Text("Add compatible LoRAs or apply a saved group from the library.")
          .font(.caption).foregroundStyle(.secondary)
      }
      ForEach(draft?.loras ?? []) { lora in
        VStack(alignment: .leading, spacing: 5) {
          HStack {
            Toggle(available.first { $0.id == lora.id }?.name ?? lora.modelID, isOn: loraEnabled(lora.id)).lineLimit(1).help(lora.modelID)
            Button { store.imageDraft?.loras.removeAll { $0.modelID == lora.id }; store.imageEstimate = nil }
              label: { Image(systemName: "xmark") }.help("Remove LoRA")
          }
          if !available.contains(where: { $0.id == lora.id }) {
            Text(hasCatalog ? "Unavailable for this connection or model. Disable or remove before generation."
              : "Saved LoRA · connection not verified")
              .font(.caption2).foregroundStyle(.orange)
          }
          LoRAStrength(value: Binding(get: { draft?.loras.first { $0.modelID == lora.id }?.weight ?? 1 }, set: { value in
            if let index = store.imageDraft?.loras.firstIndex(where: { $0.modelID == lora.id }) {
              store.imageDraft?.loras[index].weight = value; store.imageEstimate = nil
            }
          }))
        }.font(.caption)
      }
    }
  }
  func loraEnabled(_ modelID: String) -> Binding<Bool> {
    Binding(get: { draft?.loras.first { $0.modelID == modelID }?.isEnabled ?? false }, set: { enabled in
      if let index = store.imageDraft?.loras.firstIndex(where: { $0.modelID == modelID }) {
        store.imageDraft?.loras[index].enabled = enabled
      } else if enabled { store.imageDraft?.loras.append(DrawThingsLoRA(modelID: modelID)) }
      store.imageEstimate = nil
    })
  }
  func number(_ value: Any?) -> String { (value as? NSNumber).map { $0.stringValue } ?? "Unknown" }
  static let samplers = ["DPM++ 2M Karras", "Euler A", "DDIM", "PLMS", "DPM++ SDE Karras", "UniPC", "LCM", "Euler A Substep", "DPM++ SDE Substep", "TCD", "Euler A Trailing", "DPM++ SDE Trailing", "DPM++ 2M AYS", "Euler A AYS", "DPM++ SDE AYS", "DPM++ 2M Trailing", "DDIM Trailing", "UniPC Trailing", "UniPC AYS", "TCD Trailing"]
}

struct WorkspaceImage: View {
  var path: String
  var fill = false
  @State private var image: NSImage?
  var body: some View {
    Group {
      if let image { Image(nsImage: image).resizable().aspectRatio(contentMode: fill ? .fill : .fit) }
      else { Image(systemName: "photo").foregroundStyle(.secondary) }
    }.task(id: path) {
      guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
        let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: 1200] as CFDictionary) else { image = nil; return }
      image = NSImage(cgImage: thumbnail, size: .zero)
    }
  }
}
