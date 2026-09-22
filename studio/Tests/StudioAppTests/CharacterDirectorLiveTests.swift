import Combine
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

/// Opt-in qualification using an isolated app-data directory and installed local models.
/// Ordinary test runs never submit generation or load model weights.
@MainActor final class CharacterDirectorLiveTests: XCTestCase {
  func testConfiguredLocalCharacterWorkflow() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let rootPath = environment["WEETODD_CHARACTER_LIVE_ROOT"],
      let mode = environment["WEETODD_CHARACTER_LIVE_MODE"] else {
      throw XCTSkip("Set the isolated live root and mode to run installed-model qualification.")
    }
    let root = URL(fileURLWithPath: rootPath)
    try CharacterFieldCatalog.shared.catalogJSON().write(to: root.appendingPathComponent("field-catalog.json"), options: .atomic)
    let storage = CharacterSheetDocumentStore(root: root.appendingPathComponent("Character Director"))
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.drawThingsConnections = try JSONDecoder().decode([DrawThingsConnection].self,
      from: Data(contentsOf: root.appendingPathComponent("drawthings-connections.json")))
    let originalProject = store.project
    let title = environment["WEETODD_CHARACTER_LIVE_TITLE"] ?? "Local qualification character"
    let document = storage.documents().first { $0.title == title }
      ?? CharacterSheetDocument(title: title)
    let controller = CharacterSheetSessionController(document: document, store: store, storage: storage)
    if controller.document.definition.entry(at: "identity.species")?.state != .value,
      environment["WEETODD_CHARACTER_USER_REFERENCE"] == "1" {
      controller.setText("identity.species", "human")
      controller.setText("identity.type", "person")
      controller.setText("body.plan", "bipedal human")
      controller.edit { $0.draft.seed = 83291; $0.refinement.seed = 83100 }
    } else if controller.document.definition.entry(at: "identity.species")?.state != .value {
      controller.setText("identity.species", "human")
      controller.setText("identity.type", "adult person")
      controller.setText("body.plan", "bipedal human")
      controller.setText("body.build", "medium build, balanced proportions")
      controller.setText("face.covering", "skin")
      controller.setText("hair.style", "short straight dark brown hair")
      controller.setText("eyes.color", "Brown")
      controller.addRecord("garments")
      let garment = try XCTUnwrap(controller.document.definition.appearance.garments.first)
      controller.setText("garments[\(garment.id.uuidString)].type", "blue utility jacket, gray trousers and brown ankle boots")
      controller.edit {
        $0.originalDescription = "An adult human with a medium build, brown eyes and short straight dark brown hair. Wears a blue utility jacket with two chest pockets, gray straight-leg trousers and brown ankle boots. No jewelry or hat."
        $0.draft.seed = 73291; $0.refinement.seed = 73100
      }
    }
    var messages = [String]()
    var previews = Set<String>()
    let observation = controller.bridge.objectWillChange.sink {
      messages.append(controller.bridge.message)
      if let preview = controller.bridge.livePreview, let path = preview.previewPath { previews.insert(path + ":" + String(preview.previewRevision ?? 0)) }
    }
    defer { observation.cancel(); controller.save() }
    let started = Date()
    controller.launch {
      switch mode {
      case "bootstrap":
        await controller.refreshCatalog()
      case "apply":
        break // Apply only after the work task clears the editing guard.
      case "export":
        let destination = try XCTUnwrap(environment["WEETODD_CHARACTER_EXPORT"])
        try storage.export(controller.document, to: URL(fileURLWithPath: destination))
      case "text", "character", "style":
        let model = try XCTUnwrap(environment["WEETODD_CHARACTER_QWEN_MODEL"])
        if mode != "text", let image = environment["WEETODD_CHARACTER_SOURCE_IMAGE"] {
          controller.document.sources[mode] = image
        }
        try await controller.analyze(role: mode, modelPath: model)
        XCTAssertFalse(controller.document.proposals.last?.proposals.isEmpty ?? true)
        XCTAssertNotNil(controller.document.proposals.last?.metadata)
      case "sheet":
        await controller.refreshCatalog()
        try await controller.generateInitial()
        XCTAssertNotNil(controller.document.initialSheetPath)
        XCTAssertEqual(controller.document.candidates.last?.width, 1920)
        XCTAssertEqual(controller.document.candidates.last?.height, 1088)
      case "detect":
        try await controller.detectPanels()
        XCTAssertEqual(controller.document.panels?.candidates.count, 4)
        XCTAssertEqual(controller.document.panels?.status, .detected)
      case "experiment":
        // Replay an exact saved input/configuration, changing only explicitly named controls.
        let file = try XCTUnwrap(environment["WEETODD_CHARACTER_EXPERIMENT"])
        let experiment = try JSONDecoder().decode(PanelExperiment.self,
          from: Data(contentsOf: URL(fileURLWithPath: file)))
        let stage = try XCTUnwrap(controller.document.pipeline.stages.first { $0.key == experiment.sourceStage })
        var draft = try XCTUnwrap(stage.draft)
        if let weight = experiment.detailWeight {
          let id = try XCTUnwrap(draft.characterPanel?.detailLoRAID)
          XCTAssertFalse(draft.characterPanel?.replacesHead ?? true)
          let index = try XCTUnwrap(draft.loras.firstIndex { $0.modelID == id })
          draft.loras[index].weight = weight
        }
        if let seed = experiment.seed { draft.seed = seed }
        if let style = experiment.promptStyle {
          draft.characterPanel?.version = style == .legacy ? 1 : 2
          draft.characterPanel?.promptStyle = style == .legacy ? nil : style
          draft.prompt = try XCTUnwrap(draft.characterPanel).prompt
        }
        draft.name = experiment.name
        XCTAssertNil(draft.managedCharacterPromptIssue)
        _ = try await controller.generate(draft, stageKey: "experiment:" + experiment.name)
      case "face":
        let image = try XCTUnwrap(environment["WEETODD_CHARACTER_SOURCE_IMAGE"])
        controller.document.sources["face"] = image
        try await controller.prepareHead()
        XCTAssertNotNil(controller.document.headReference)
      case "panel":
        await controller.refreshCatalog()
        controller.approveCrops()
        XCTAssertTrue(controller.document.cropsApproved)
        try await controller.validateRecipeDependencies()
        let role = try XCTUnwrap(CharacterPanelRole(rawValue: environment["WEETODD_CHARACTER_PANEL_ROLE"] ?? "front"))
        let offset = Int(environment["WEETODD_CHARACTER_SEED_OFFSET"] ?? "0") ?? 0
        try await self.qualifyPanel(role, seedOffset: offset, controller: controller, storage: storage)
      case "refine", "resume":
        await controller.refreshCatalog()
        controller.approveCrops()
        XCTAssertTrue(controller.document.cropsApproved, controller.error ?? "Crop review failed")
        controller.document.refinement.replaceFaces = environment["WEETODD_CHARACTER_REPLACE_FACE"] == "1"
        let before = controller.document.pipeline.stages.count
        try await controller.refinePanels()
        XCTAssertEqual(controller.document.candidates.last?.width, 3840)
        XCTAssertEqual(controller.document.candidates.last?.height, 2176)
        let takes = controller.document.candidates.last?.generation?.characterAssembly?.panelTakes ?? []
        let separated = controller.document.refinement.twoPass && controller.document.refinement.replaceFaces
        XCTAssertEqual(takes.count, separated ? 8 : 4)
        if separated {
          for index in stride(from: 0, to: takes.count - takes.count % 2, by: 2) {
            XCTAssertEqual(takes[index].loras.map(\.modelID), [controller.document.refinement.headLoRAID])
            XCTAssertEqual(takes[index + 1].loras.map(\.modelID), [controller.document.refinement.detailLoRAID])
            XCTAssertEqual(takes[index].steps, controller.document.refinement.steps)
            XCTAssertEqual(takes[index + 1].steps, controller.document.refinement.steps)
            let crop = try XCTUnwrap(controller.document.panels?.candidates.first { $0.role == takes[index].context.role })
            let rect = crop.sourcePixelRect
            XCTAssertEqual([takes[index].width, takes[index].height],
              [((rect.width + 63) / 64) * 64, ((rect.height + 63) / 64) * 64])
            XCTAssertEqual([takes[index + 1].width, takes[index + 1].height],
              [((rect.width * 2 + 63) / 64) * 64, ((rect.height * 2 + 63) / 64) * 64])
          }
        }
        if mode == "resume" { XCTAssertEqual(controller.document.pipeline.stages.count, before) }
      default: throw StudioError.invalid("Unknown live qualification mode")
      }
    }
    await controller.workTask?.value
    if mode == "apply" {
      let file = try XCTUnwrap(environment["WEETODD_CHARACTER_REVIEWED_IDS"])
      let ids = try JSONDecoder().decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: file)))
      if !ids.isEmpty {
        let batch = try XCTUnwrap(controller.document.proposals.last)
        await controller.applyProposals(batchID: batch.id, selectedIDs: Set(ids))
      }
      if let edits = environment["WEETODD_CHARACTER_REVIEWED_FIELDS"] {
        let fields = try JSONDecoder().decode([String: String].self,
          from: Data(contentsOf: URL(fileURLWithPath: edits)))
        for key in fields.keys.sorted() { controller.setText(key, fields[key]!) }
      }
      XCTAssertTrue(controller.compiled.canGenerate)
    }
    let report: [String: Any] = ["mode": mode, "elapsedSeconds": Date().timeIntervalSince(started),
      "status": controller.status, "error": controller.error ?? "", "previewCount": previews.count,
      "messages": Array(Set(messages)).sorted(), "documentID": controller.id.uuidString]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: root.appendingPathComponent("live-\(mode)-report.json"), options: .atomic)
    XCTAssertNil(controller.error)
    if ["sheet", "refine", "experiment", "panel"].contains(mode), controller.error == nil {
      XCTAssertGreaterThan(previews.count, 0, "Live generation must deliver preview updates.")
    }
    XCTAssertEqual(store.project, originalProject)
    XCTAssertNil(store.imageDraft)
  }

  /// Run one reviewed pair before qualifying the rest. Canonical keys let the full
  /// pipeline prove it can reuse the exact same head/detail requests afterward.
  private func qualifyPanel(_ role: CharacterPanelRole, seedOffset: Int,
    controller: CharacterSheetSessionController, storage: CharacterSheetDocumentStore) async throws {
    let document = controller.document
    let detection = try XCTUnwrap(document.panels)
    let index = try XCTUnwrap(detection.candidates.firstIndex { $0.role == role })
    let panel = detection.candidates[index], rect = panel.sourcePixelRect
    let source = try XCTUnwrap(document.initialSheetPath)
    let head = try XCTUnwrap(document.headReference)
    var settings = document.refinement
    XCTAssertTrue(settings.twoPass && settings.replaceFaces)
    let directory = storage.directory(id: document.id).appendingPathComponent("Panels/\(panel.id.uuidString)")
    let prepared = try await controller.bridge.invoke("character-panel-prepare", runtime: controller.store.runtime,
      payload: ["source": source, "rect": ["x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height], "scale": 1], output: directory)
    let input = try XCTUnwrap(prepared["padded_path"] as? String)
    let dimensions = try XCTUnwrap(prepared["padded_dimensions"] as? [Int])
    var draft = try CharacterPanelRecipe.makeDraft(role: role, definition: document.definition, settings: settings,
      profileID: document.draft.profileID, panelPath: input, headPath: head.whiteMattePath,
      width: dimensions[0], height: dimensions[1], documentID: document.id, seed: settings.seed + index + seedOffset)
    draft = try CharacterPanelRecipe.headOnlyDraft(from: draft)
    let identity = try CharacterArtifactHash.value(["source": CharacterArtifactHash.file(source),
      "rect": CharacterArtifactHash.value(rect), "head": CharacterArtifactHash.file(head.whiteMattePath),
      "mode": "native-head-then-2x-detail-v2", "input": prepared["padded_sha256"] as? String ?? ""])
    let suffix = seedOffset == 0 ? "" : ":seed-\(seedOffset)"
    let key = "panel-\(panel.id.uuidString)"
    let swapped = try await controller.generate(draft, stageKey: key + ":" + identity + suffix)
    let detail = try await controller.bridge.invoke("character-panel-prepare", runtime: controller.store.runtime,
      payload: ["source": swapped.path, "rect": ["x": 0, "y": 0, "width": rect.width, "height": rect.height], "scale": 2],
      output: directory.appendingPathComponent(seedOffset == 0 ? "Detail Input" : "Detail Input Seed \(seedOffset)"))
    let detailPath = try XCTUnwrap(detail["padded_path"] as? String)
    let detailDimensions = try XCTUnwrap(detail["padded_dimensions"] as? [Int])
    settings.replaceFaces = false
    draft = try CharacterPanelRecipe.makeDraft(role: role, definition: document.definition, settings: settings,
      profileID: document.draft.profileID, panelPath: detailPath, headPath: nil,
      width: detailDimensions[0], height: detailDimensions[1], documentID: document.id,
      seed: settings.seed + index + 1000 + seedOffset, preservesInputHead: true)
    _ = try await controller.generate(draft, stageKey: key + ":detail:" + identity + suffix)
  }
}

private struct PanelExperiment: Decodable {
  var name: String
  var sourceStage: String
  var detailWeight: Double?
  var seed: Int?
  var promptStyle: CharacterRefinementPromptStyle?
}
