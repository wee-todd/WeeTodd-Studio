import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct CharacterDirectorWindow: View {
  @Environment(\.dismiss) private var dismiss
  @ObservedObject var controller: CharacterSheetSessionController
  @EnvironmentObject private var store: StudioStore
  var embedded = false
  @State private var tab = "Character"
  @State private var modelPath = ""
  @State private var modelPaths: [URL] = []
  @State private var showModelSetup = false
  @State private var proposalSelections: [UUID: Set<String>] = [:]
  @State private var headX = 0
  @State private var headY = 0
  @State private var headWidth = 0
  @State private var headHeight = 0

  var body: some View {
    VStack(spacing: 0) {
      toolbar
      Divider()
      if let live = controller.bridge.livePreview {
        HStack {
          Text("Live preview · approximate").font(.caption)
          WorkspaceImage(path: live.previewPath ?? "").id(live.previewRevision ?? 0).frame(width: 240, height: 180)
        }
      } else if let preview = controller.previewPath {
        PreviewableImage(path: preview, title: controller.document.title).frame(height: 180)
      }
      HSplitView {
        VStack(alignment: .leading, spacing: 6) {
          ForEach(["Character", "Sources", "Generate", "Panels", "Prompt"], id: \.self) { item in
            Button { tab = item } label: {
              Label(item, systemImage: icon(item)).frame(maxWidth: .infinity, alignment: .leading)
            }.buttonStyle(.plain).padding(7).background(tab == item ? Color.accentColor.opacity(0.18) : .clear).cornerRadius(6)
          }
          Spacer()
          Text(controller.status).font(.caption).foregroundStyle(.secondary)
          if controller.running { ProgressView(value: controller.bridge.fraction) }
        }.padding(12).frame(minWidth: 170, idealWidth: 190, maxWidth: 220)
        ScrollView { content.padding(18).frame(maxWidth: .infinity, alignment: .topLeading) }.disabled(controller.running)
      }
    }
    .frame(minWidth: embedded ? 900 : 1040, minHeight: 700)

    .task { await controller.refreshCatalog(); scanModels() }
    .sheet(isPresented: $showModelSetup) {
      AssistantModelSetupView(store: store, currentModelPath: modelPath) { modelPath = $0; scanModels() }
    }
    .alert("Character Director", isPresented: Binding(get: { controller.error != nil }, set: { if !$0 { controller.error = nil } })) {
      Button("OK") { controller.error = nil }
    } message: { Text(controller.error ?? "") }
  }

  private var toolbar: some View {
    HStack {
      TextField("Character name", text: Binding(get: { controller.document.title }, set: { value in
        controller.edit { $0.title = value }
      })).textFieldStyle(.roundedBorder).font(.title2).frame(maxWidth: 420)
      Spacer()
      Button { controller.undo() } label: { Image(systemName: "arrow.uturn.backward") }.disabled(!controller.canUndo)
      Button { controller.redo() } label: { Image(systemName: "arrow.uturn.forward") }.disabled(!controller.canRedo)
      if controller.running {
        VStack(alignment: .leading) { Text(controller.status); Text(controller.bridge.message).font(.caption) }
        ProgressView(value: controller.bridge.fraction).frame(width: 90)
        Button("Cancel", role: .destructive) { controller.cancel() }
      }
      if embedded { Button("Done") { dismiss() } }
      if !embedded {
        Menu("Recent") {
          if store.characterDirector.recentDocuments.isEmpty { Text("No saved characters") }
          ForEach(store.characterDirector.recentDocuments) { document in
            Button(document.title) { store.characterDirector.open(documentID: document.id) }
          }
        }
        Button("Import…") { importDocument() }
        Button("Export…") { controller.exportDocument() }
      }
    }.padding(12)
  }

  @ViewBuilder private var content: some View {
    switch tab {
    case "Sources": sources
    case "Generate": generation
    case "Panels": panels
    case "Prompt": prompt
    default: CharacterFieldsEditor(controller: controller)
    }
  }

  private var sources: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Source images").font(.title2)
      Text("Analysis proposes structured edits. Nothing replaces accepted fields until you select and apply it.")
        .foregroundStyle(.secondary)
      Toggle("Use the character image as the style image", isOn: Binding(get: {
        controller.document.styleUsesCharacterImage
      }, set: { value in controller.edit { $0.styleUsesCharacterImage = value } })).toggleStyle(.checkbox)
      ForEach(CharacterSourceRole.allCases) { role in sourceRow(role) }
      if let path = controller.preparedSubjectPreviewPath {
        GroupBox("Prepared character analysis input") {
          HStack(alignment: .top, spacing: 12) {
            PreviewableImage(path: path, title: "Apple Vision prepared subject")
              .frame(width: 180, height: 180)
            Text(controller.preparedSubjectStatus ?? "Bounded character overview ready.")
              .font(.caption).foregroundStyle(.secondary)
          }
        }
      }
      Divider()
      HStack {
        Picker("Local vision model", selection: $modelPath) {
          Text("Choose installed Qwen model").tag("")
          ForEach(modelPaths, id: \.path) { Text(LocalPromptModels.label(for: $0.path)).tag($0.path) }
        }.frame(maxWidth: 470)
        Button("Set up…") { showModelSetup = true }
        Button("Rescan") { scanModels() }
      }
      if !controller.document.originalDescription.isEmpty {
        DisclosureGroup("Authored description to map") {
          Text(controller.document.originalDescription).textSelection(.enabled)
          Button("Map description to fields") { controller.launch { try await controller.analyze(role: "text", modelPath: modelPath) } }.disabled(modelPath.isEmpty)
        }
      }
      if let head = controller.document.headReference {
        HStack {
          PreviewableImage(path: head.rgbaCutoutPath, title: "Background-removed head").frame(width: 150, height: 150)
          PreviewableImage(path: head.whiteMattePath, title: "White-matted model input").frame(width: 150, height: 150)
        }
      }
      if let facePath = controller.document.sources[CharacterSourceRole.face.rawValue] {
        manualHeadCrop(path: facePath)
      }
      proposals
    }
  }

  private func manualHeadCrop(path: String) -> some View {
    GroupBox("Manual reference-head crop") {
      VStack(alignment: .leading, spacing: 8) {
        Text("Use source-pixel coordinates when automatic face selection or masking needs correction. Include hair, ears and enough neck for blending.")
          .font(.caption).foregroundStyle(.secondary)
        HStack {
          PreviewableImage(path: path, title: "Reference face source").frame(width: 180, height: 140)
          VStack(alignment: .leading) {
            HStack {
              TextField("X", value: $headX, format: .number).frame(width: 80)
              TextField("Y", value: $headY, format: .number).frame(width: 80)
              TextField("Width", value: $headWidth, format: .number).frame(width: 100)
              TextField("Height", value: $headHeight, format: .number).frame(width: 100)
            }
            HStack {
              Button("Use full image") {
                if let image = NSImage(contentsOfFile: path) {
                  let pixels = image.representations.max { $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh }
                  headX = 0; headY = 0
                  headWidth = pixels?.pixelsWide ?? Int(image.size.width)
                  headHeight = pixels?.pixelsHigh ?? Int(image.size.height)
                }
              }
              Button("Prepare selected head") {
                let selection = CharacterHeadSelection(crop: PanelPixelRect(x: headX, y: headY,
                  width: headWidth, height: headHeight))
                controller.launch { try await controller.prepareHead(selection: selection) }
              }.buttonStyle(.borderedProminent).disabled(headX < 0 || headY < 0 || headWidth <= 0 || headHeight <= 0)
            }
          }
        }
      }
    }
  }

  private func sourceRow(_ role: CharacterSourceRole) -> some View {
    let effective = role == .style && controller.document.styleUsesCharacterImage ? controller.document.sources["character"] : controller.document.sources[role.rawValue]
    return GroupBox(role.label) {
      HStack(spacing: 12) {
        if let effective { PreviewableImage(path: effective, title: role.label).frame(width: 120, height: 90) }
        else { RoundedRectangle(cornerRadius: 5).fill(.quaternary).frame(width: 120, height: 90).overlay(Text("No image").foregroundStyle(.secondary)) }
        VStack(alignment: .leading) {
          Text(effective ?? "Choose an image").font(.caption).lineLimit(2).textSelection(.enabled)
          HStack {
            Button("Choose…") { chooseSource(role) }.disabled(role == .style && controller.document.styleUsesCharacterImage)
            if role != .face {
              Button("Analyze") { controller.launch { try await controller.analyze(role: role.rawValue, modelPath: modelPath) } }
                .disabled(modelPath.isEmpty || effective == nil)
            } else if controller.document.headReference != nil {
              Label("Cutout ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            }
          }
        }
      }
    }
  }

  private var proposals: some View {
    VStack(alignment: .leading, spacing: 10) {
      ForEach(controller.document.proposals) { batch in
        GroupBox("\(batch.role.capitalized) proposals\(batch.stale ? " · stale" : "")") {
          if batch.stale {
            Text("No values were applied. Analyze this source again to refresh these proposals.")
              .font(.caption).foregroundStyle(.orange)
          }
          ForEach(Array((batch.diagnostics ?? []).enumerated()), id: \.offset) { _, diagnostic in
            Label(diagnostic.message, systemImage: "exclamationmark.triangle")
              .font(.caption).foregroundStyle(.orange)
          }
          ForEach(batch.proposals) { proposal in
            Toggle(isOn: proposalBinding(batch: batch, proposal: proposal)) {
              VStack(alignment: .leading) {
                Text("\(proposal.field): \(proposal.value)")
                if !proposal.evidence.isEmpty { Text(proposal.evidence).font(.caption).foregroundStyle(.secondary) }
                if !proposal.uncertainty.isEmpty { Text(proposal.uncertainty).font(.caption).foregroundStyle(.orange) }
              }
            }.toggleStyle(.checkbox).disabled(batch.stale)
          }
          HStack { Spacer(); Button("Accept selected") {
            let selected = proposalSelections[batch.id] ?? []
            Task { await controller.applyProposals(batchID: batch.id, selectedIDs: selected) }
          }.disabled(batch.stale || !batch.proposals.contains { (proposalSelections[batch.id] ?? []).contains($0.id) }) }
        }
      }
    }
  }

  private var generation: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Initial four-panel sheet").font(.title2)
      HStack {
        Picker("Draw Things Local", selection: Binding(get: { controller.document.draft.profileID }, set: { value in
          controller.edit { $0.draft.profileID = value }
          Task { await controller.refreshCatalog() }
        })) { Text("Choose connection").tag(""); ForEach(controller.localConnections) { Text($0.name).tag($0.id) } }
        Picker("Model", selection: Binding(get: { controller.document.draft.modelID }, set: { value in controller.edit { $0.draft.modelID = value } })) {
          Text("Choose Krea 2 Turbo").tag(""); ForEach(controller.models, id: \.id) { Text($0.name).tag($0.id) }
        }
      }
      Picker("Four-view LoRA", selection: Binding(get: { controller.document.draft.characterSheetLoRAID ?? "" }, set: { value in
        controller.edit { $0.draft.characterSheetLoRAID = value.isEmpty ? nil : value }
      })) { Text("Choose compatible LoRA").tag(""); ForEach(controller.loras(model: controller.document.draft.modelID), id: \.id) { Text($0.name).tag($0.id) } }
      HStack {
        Text("1920 × 1088").foregroundStyle(.secondary)
        Stepper("Steps \(controller.document.draft.steps)", value: Binding(get: { controller.document.draft.steps }, set: { value in controller.edit { $0.draft.steps = value } }), in: 1...100)
        TextField("CFG", value: Binding(get: { controller.document.draft.guidance }, set: { value in controller.edit { $0.draft.guidance = value } }), format: .number).frame(width: 80)
        TextField("Seed", value: Binding(get: { controller.document.draft.seed }, set: { value in controller.edit { $0.draft.seed = value } }), format: .number).frame(width: 120)
      }
      Button("Generate initial sheet") { controller.launch { try await controller.generateInitial() } }.buttonStyle(.borderedProminent)
      Divider()
      Text("Panel refinement").font(.title2)
      Picker("FLUX.2 klein 9B", selection: Binding(get: { controller.document.refinement.modelID }, set: { value in controller.edit { $0.refinement.modelID = value } })) {
        Text("Choose exact 9B edit model").tag(""); ForEach(controller.models, id: \.id) { Text($0.name).tag($0.id) }
      }
      loraPicker("HighResolution9B", id: Binding(get: { controller.document.refinement.detailLoRAID }, set: { value in controller.edit { $0.refinement.detailLoRAID = value } }), strength: Binding(get: { controller.document.refinement.detailStrength }, set: { value in controller.edit { $0.refinement.detailStrength = value } }))
      Picker("Detail prompt", selection: Binding(get: { controller.document.refinement.detailPromptStyle ?? .legacy }, set: { value in controller.edit { $0.refinement.detailPromptStyle = value } })) {
        Text("Trigger + quality descriptors").tag(CharacterRefinementPromptStyle.qualityOnly)
        Text("Legacy descriptive prompt").tag(CharacterRefinementPromptStyle.legacy)
      }
      Toggle("Replace faces (head and hair may change)", isOn: Binding(get: { controller.document.refinement.replaceFaces }, set: { value in controller.edit { $0.refinement.replaceFaces = value } })).toggleStyle(.checkbox)
      if controller.document.refinement.replaceFaces {
        loraPicker("BFS rank-64", id: Binding(get: { controller.document.refinement.headLoRAID }, set: { value in controller.edit { $0.refinement.headLoRAID = value } }), strength: Binding(get: { controller.document.refinement.headStrength }, set: { value in controller.edit { $0.refinement.headStrength = value } }))
        Picker("BFS prompt", selection: Binding(get: { controller.document.refinement.headPromptStyle ?? .legacy }, set: { value in controller.edit { $0.refinement.headPromptStyle = value } })) {
          ForEach(CharacterRefinementPromptStyle.allCases) { Text($0.label).tag($0) }
        }
      }
      if controller.document.refinement.replaceFaces {
        Picker("Recipe", selection: Binding(get: { controller.document.refinement.twoPass }, set: { value in controller.edit { $0.refinement.twoPass = value } })) {
          Text("BFS, then HighResolution9B").tag(true); Text("Combined (comparison)").tag(false)
        }.pickerStyle(.segmented)
        Text("BFS runs at the original panel size. Its result is enlarged 2× with Lanczos, then refined with HighResolution9B only. Both passes use the step count below.").font(.caption).foregroundStyle(.secondary)
      }
      HStack {
        Stepper("Steps per pass \(controller.document.refinement.steps)", value: Binding(get: { controller.document.refinement.steps }, set: { value in controller.edit { $0.refinement.steps = value } }), in: 1...100)
        TextField("CFG", value: Binding(get: { controller.document.refinement.guidance }, set: { value in controller.edit { $0.refinement.guidance = value } }), format: .number).frame(width: 80)
        TextField("Seed", value: Binding(get: { controller.document.refinement.seed }, set: { value in controller.edit { $0.refinement.seed = value } }), format: .number).frame(width: 120)
        Button("Refine four panels") { controller.launch { try await controller.refinePanels() } }.buttonStyle(.borderedProminent)
          .disabled(!controller.document.cropsApproved)
      }
      recoveryStages
      candidateReview
    }
  }

  private var recoveryStages: some View {
    let stages = controller.document.pipeline.stages.filter { [.submitted, .uncertain].contains($0.state) }
    return Group {
      if !stages.isEmpty {
        Divider()
        VStack(alignment: .leading, spacing: 8) {
          Label("Generation status needs review", systemImage: "exclamationmark.triangle.fill")
            .font(.headline).foregroundStyle(.orange)
          Text("These requests may already have reached Draw Things. Inspect the saved job/output before allowing another submission.")
            .font(.caption).foregroundStyle(.secondary)
          ForEach(stages) { stage in
            GroupBox {
              HStack {
                VStack(alignment: .leading) {
                  Text(stage.key).font(.callout.monospaced())
                  Text(stage.state.rawValue.capitalized).font(.caption.bold()).foregroundStyle(.orange)
                  if let message = stage.message { Text(message).font(.caption).foregroundStyle(.secondary) }
                }
                Spacer()
                if let directory = stage.recoveryDirectory {
                  Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: directory)])
                  }
                }
                Button("Allow retry…") { controller.allowRetry(stageKey: stage.key) }
              }
            }
          }
        }
      }
    }
  }

  private var panels: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack { Text("Panel crops").font(.title2); Spacer(); Button("Detect panels") { controller.launch { try await controller.detectPanels() } } }
      if let path = controller.document.initialSheetPath { PanelCropOverlay(controller: controller, path: path) }
      if let detection = controller.document.panels {
        HStack {
          Text("\(detection.candidates.count) of 4 crops").font(.caption).foregroundStyle(.secondary)
          Spacer()
          Button("Add crop") { addManualCrop() }.disabled(detection.candidates.count >= 4)
        }
        ForEach(detection.candidates) { candidate in cropEditor(candidate) }
        ForEach(detection.diagnostics, id: \.self) { Label($0, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
        Button(controller.document.cropsApproved ? "Crops approved" : "Approve crops") { controller.approveCrops() }
          .buttonStyle(.borderedProminent).disabled(detection.candidates.count != 4 || controller.document.cropsApproved)
      } else { Text("Generate or select an initial sheet, then detect its actual panel boundaries.").foregroundStyle(.secondary) }
    }
  }

  private func cropEditor(_ candidate: DetectedCharacterPanel) -> some View {
    HStack {
      Picker("Role", selection: cropRole(candidate.id, candidate.role)) { ForEach(CharacterPanelRole.allCases, id: \.self) { Text(roleLabel($0)).tag($0) } }.frame(width: 180)
      ForEach(["x", "y", "width", "height"], id: \.self) { key in
        TextField(key, value: cropNumber(candidate.id, key: key, rect: candidate.sourcePixelRect), format: .number).frame(width: 90)
      }
      Button(role: .destructive) { removeCrop(candidate.id) } label: { Image(systemName: "trash") }
        .help("Remove this crop")
    }
  }

  private func addManualCrop() {
    controller.edit { document in
      guard var detection = document.panels, detection.candidates.count < 4 else { return }
      guard Self.addManualCrop(to: &detection, revision: document.revision + 1) else { return }
      document.panels = detection
      document.cropsApproved = false
    }
  }

  @discardableResult static func addManualCrop(to detection: inout CharacterPanelDetection,
                                                revision: Int) -> Bool {
    guard detection.candidates.count < 4 else { return false }
    let used = Set(detection.candidates.map(\.role))
    let role = CharacterPanelRole.allCases.first { !used.contains($0) } ?? .front
    detection.candidates.append(DetectedCharacterPanel(role: role,
      sourcePixelRect: PanelPixelRect(x: 0, y: 0, width: 240, height: 400),
      evidence: CharacterPanelEvidence(foregroundBounds: []), detectionRevision: revision))
    detection.status = .needsReview
    detection.diagnostics.append("Manual crop added. Position and size it to the actual panel before approval.")
    return true
  }

  private func removeCrop(_ id: UUID) {
    controller.edit { document in
      document.panels?.candidates.removeAll { $0.id == id }
      document.panels?.status = .needsReview
      document.cropsApproved = false
    }
  }

  private var prompt: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Compiled prompt").font(.title2)
      Text("This prompt is managed from structured fields and cannot be edited directly.").foregroundStyle(.secondary)
      ScrollView { Text(controller.compiled.prompt).font(.body.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding() }
        .background(.quaternary.opacity(0.35)).cornerRadius(8).frame(minHeight: 300)
      ForEach(Array(controller.compiled.diagnostics.enumerated()), id: \.offset) { _, issue in
        Label(issue.message, systemImage: issue.severity == .error ? "xmark.octagon" : "exclamationmark.triangle")
          .foregroundStyle(issue.severity == .error ? .red : .orange)
      }
    }
  }

  private var candidateReview: some View {
    VStack(alignment: .leading) {
      ForEach(controller.document.candidates) { asset in
        HStack {
          PreviewableImage(path: asset.path, title: asset.name).frame(width: 150, height: 100)
          VStack(alignment: .leading) { Text(asset.name); Text(asset.path).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
          Spacer()
          if controller.onUse != nil { Button("Use reviewed candidate") { Task { _ = await controller.onUse?(asset) } } }
        }
      }
    }
  }

  private func loraPicker(_ title: String, id: Binding<String>, strength: Binding<Double>) -> some View {
    HStack { Picker(title, selection: id) { Text("Choose installed LoRA").tag(""); ForEach(controller.loras(model: controller.document.refinement.modelID), id: \.id) { Text($0.name).tag($0.id) } }; TextField("Strength", value: strength, format: .percent).frame(width: 90).help("LoRA strength: 80% is a weight of 0.80. This does not change image strength or guidance.") }
  }

  private func chooseSource(_ role: CharacterSourceRole) {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await controller.importSource(url, role: role) }
  }
  private func importDocument() {
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    controller.launch { try await controller.importDocument(url) }
  }
  private func scanModels() { modelPaths = LocalPromptModels.discover(including: modelPath); if modelPath.isEmpty { modelPath = modelPaths.first?.path ?? "" } }
  private func proposalBinding(batch: CharacterProposalBatch, proposal: CharacterFieldProposal) -> Binding<Bool> { Binding(get: { proposalSelections[batch.id, default: []].contains(proposal.id) }, set: { value in if value { proposalSelections[batch.id, default: []].insert(proposal.id) } else { proposalSelections[batch.id, default: []].remove(proposal.id) } }) }
  private func cropRole(_ id: UUID, _ value: CharacterPanelRole) -> Binding<CharacterPanelRole> { Binding(get: { controller.document.panels?.candidates.first { $0.id == id }?.role ?? value }, set: { role in controller.edit { doc in guard let index = doc.panels?.candidates.firstIndex(where: { $0.id == id }) else { return }; doc.panels?.candidates[index].role = role; doc.cropsApproved = false } }) }
  private func cropNumber(_ id: UUID, key: String, rect: PanelPixelRect) -> Binding<Int> { Binding(get: { let r = controller.document.panels?.candidates.first { $0.id == id }?.sourcePixelRect ?? rect; return key == "x" ? r.x : key == "y" ? r.y : key == "width" ? r.width : r.height }, set: { number in controller.edit { doc in guard let index = doc.panels?.candidates.firstIndex(where: { $0.id == id }) else { return }; switch key { case "x": doc.panels?.candidates[index].sourcePixelRect.x = number; case "y": doc.panels?.candidates[index].sourcePixelRect.y = number; case "width": doc.panels?.candidates[index].sourcePixelRect.width = number; default: doc.panels?.candidates[index].sourcePixelRect.height = number }; doc.cropsApproved = false } }) }
  private func roleLabel(_ role: CharacterPanelRole) -> String { role == .closeUp ? "Facial close-up" : role.rawValue.capitalized }
  private func icon(_ item: String) -> String { ["Character": "person.crop.rectangle", "Sources": "photo.on.rectangle", "Generate": "wand.and.stars", "Panels": "rectangle.split.3x1", "Prompt": "text.quote"][item] ?? "circle" }
}

private struct PanelCropOverlay: View {
  @ObservedObject var controller: CharacterSheetSessionController
  let path: String
  var body: some View {
    if let image = NSImage(contentsOfFile: path), image.size.width > 0, image.size.height > 0 {
      GeometryReader { proxy in
        let sx = proxy.size.width / image.size.width, sy = proxy.size.height / image.size.height
        ZStack(alignment: .topLeading) {
          Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
          ForEach(controller.document.panels?.candidates ?? []) { panel in
            Rectangle().stroke(Color.accentColor, lineWidth: 2).background(Color.accentColor.opacity(0.08))
              .frame(width: CGFloat(panel.sourcePixelRect.width) * sx, height: CGFloat(panel.sourcePixelRect.height) * sy)
              .offset(x: CGFloat(panel.sourcePixelRect.x) * sx, y: CGFloat(panel.sourcePixelRect.y) * sy)
              .overlay(alignment: .topLeading) { Text(panel.role.rawValue).font(.caption.bold()).padding(4).background(.regularMaterial) }
              .gesture(DragGesture().onEnded { value in
                controller.edit { doc in
                  guard let index = doc.panels?.candidates.firstIndex(where: { $0.id == panel.id }) else { return }
                  doc.panels?.candidates[index].sourcePixelRect.x += Int(value.translation.width / sx)
                  doc.panels?.candidates[index].sourcePixelRect.y += Int(value.translation.height / sy)
                  doc.cropsApproved = false
                }
              })
          }
        }
      }.aspectRatio(image.size.width / image.size.height, contentMode: .fit).frame(maxHeight: 460)
    }
  }
}
