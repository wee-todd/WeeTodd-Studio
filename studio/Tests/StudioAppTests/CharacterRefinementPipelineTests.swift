import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

@MainActor final class CharacterRefinementPipelineTests: XCTestCase {
  func testSeparatedRefinementRunsNativeBFSThenCroppedTwoTimesLanczosThenDetailOnly() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("character-refinement-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let initial = try artifact(root.appendingPathComponent("initial.png"), contents: "initial-sheet")
    let head = try artifact(root.appendingPathComponent("head.png"), contents: "head-source")
    let rgba = try artifact(root.appendingPathComponent("head-rgba.png"), contents: "head-rgba")
    let matte = try artifact(root.appendingPathComponent("head-white.png"), contents: "head-white")
    let initialHash = try CharacterArtifactHash.file(initial.path)
    let headHash = try CharacterArtifactHash.file(head.path)
    let rgbaHash = try CharacterArtifactHash.file(rgba.path)
    let matteHash = try CharacterArtifactHash.file(matte.path)

    let rects = [
      PanelPixelRect(x: 0, y: 0, width: 301, height: 501),
      PanelPixelRect(x: 301, y: 0, width: 407, height: 503),
      PanelPixelRect(x: 708, y: 0, width: 509, height: 507),
      PanelPixelRect(x: 1217, y: 0, width: 603, height: 511),
    ]
    let panels = zip(CharacterPanelRole.allCases, rects).map { role, rect in
      DetectedCharacterPanel(role: role, sourcePixelRect: rect,
        evidence: .init(foregroundBounds: [rect]), detectionRevision: 1)
    }
    let detection = CharacterPanelDetection(sourceSHA256: initialHash, sourceOrientation: 1,
      detectorVersion: "test", candidates: panels, status: .detected, diagnostics: [])

    struct PrepareCall { let source: String; let rect: [String: Int]; let scale: Int }
    struct GenerationCall { let width: Int; let height: Int; let loras: [String]; let weights: [Double]; let inputs: [String]; let outputPath: String }
    var prepareCalls: [PrepareCall] = []
    var generationCalls: [GenerationCall] = []
    var estimates = 0
    var generationSerial = 0
    var reassemblies = 0
    var commands: [String] = []
    let invocation: Bridge.Invocation = { command, _, payload, output in
      commands.append(command)
      switch command {
      case "character-panel-prepare":
        let source = try XCTUnwrap(payload["source"] as? String)
        let rect = try XCTUnwrap(payload["rect"] as? [String: Int])
        let scale = try XCTUnwrap(payload["scale"] as? Int)
        prepareCalls.append(.init(source: source, rect: rect, scale: scale))
        let width = Self.ceil64(rect["width"]! * scale), height = Self.ceil64(rect["height"]! * scale)
        let directory = try XCTUnwrap(output)
        let path = try self.artifact(directory.appendingPathComponent("prepared.png"),
          contents: "prepared:\(try CharacterArtifactHash.file(source)):\(rect["width"]!):\(rect["height"]!):\(scale)")
        return ["padded_path": path.path, "padded_dimensions": [width, height],
          "padded_sha256": "prepared-\(rect["width"]!)-\(rect["height"]!)-\(scale)"]
      case "dt-estimate":
        estimates += 1
        return ["eligibility": "allowed"]
      case "dt-generate-image":
        let request = try XCTUnwrap(payload["drawThingsRequest"] as? [String: Any])
        let configuration = try XCTUnwrap(request["configuration"] as? [String: Any])
        let width = try XCTUnwrap(configuration["width"] as? Int)
        let height = try XCTUnwrap(configuration["height"] as? Int)
        let loras = (request["loras"] as? [[String: Any]] ?? []).compactMap { $0["modelID"] as? String }
        let weights = (request["loras"] as? [[String: Any]] ?? []).compactMap { $0["weight"] as? Double }
        let inputs = (request["inputs"] as? [[String: Any]] ?? []).compactMap { $0["path"] as? String }
        generationSerial += 1
        let directory = try XCTUnwrap(output)
        let path = try self.artifact(directory.appendingPathComponent("generated.png"), contents: "generation-\(generationSerial)")
        generationCalls.append(.init(width: width, height: height, loras: loras, weights: weights, inputs: inputs,
          outputPath: path.path))
        return ["asset": ["path": path.path, "width": width, "height": height],
          "fingerprint": "generation-\(generationSerial)", "normalizedRequest": request]
      case "character-reassemble":
        reassemblies += 1
        let directory = try XCTUnwrap(output)
        let path = try self.artifact(directory.appendingPathComponent("assembled.png"), contents: "assembly-\(reassemblies)")
        return ["path": path.path]
      default:
        XCTFail("Unexpected bridge command \(command)")
        return [:]
      }
    }

    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false,
      invocation: invocation)
    let connection = DrawThingsConnection(id: "local", host: "127.0.0.1")
    store.drawThingsConnections = [connection]
    let storage = CharacterSheetDocumentStore(root: root.appendingPathComponent("documents"))
    var document = CharacterSheetDocument(title: "Pipeline test")
    document.definition.appearance.setText("identity.species", "human")
    document.definition.appearance.setText("identity.type", "person")
    document.draft.profileID = connection.id
    document.draft.modelID = "krea-model"
    document.draft.characterSheetLoRAID = "sheet-lora"
    document.initialSheetPath = initial.path
    document.panels = detection
    document.cropsApproved = true
    document.sources["face"] = head.path
    document.refinement.modelID = "klein-9b-model"
    document.refinement.detailLoRAID = "high-resolution-9b"
    document.refinement.headLoRAID = "bfs-head-rank64"
    document.refinement.replaceFaces = true
    document.refinement.twoPass = true
    document.headReference = CharacterHeadReference(originalAssetID: UUID(), headCropAssetID: UUID(),
      maskAssetID: UUID(), rgbaCutoutAssetID: UUID(), whiteMatteAssetID: UUID(), sourcePath: head.path,
      rgbaCutoutPath: rgba.path, whiteMattePath: matte.path, sourceSHA256: headHash,
      preprocessingVersion: "test", artifactSHA256: ["rgbaCutout": rgbaHash, "whiteMatte": matteHash])
    let controller = CharacterSheetSessionController(document: document, store: store, storage: storage)
    controller.catalog = [
      "models": [
        ["id": "krea-model", "name": "Krea 2 Turbo"],
        ["id": "klein-9b-model", "name": "FLUX.2 klein 9B KV"],
      ],
      "capabilities": [
        "krea-model": ["operations": ["image": [:]]],
        "klein-9b-model": ["operations": ["image": [:]]],
      ],
      "loras": [
        ["id": "sheet-lora", "name": "Krea2 Character Design 4 View V1", "compatibleModelIDs": ["krea-model"]],
        ["id": "high-resolution-9b", "name": "HighResolution9B", "compatibleModelIDs": ["klein-9b-model"]],
        ["id": "bfs-head-rank64", "name": "BFS head v1 FLUX klein 9B step 3750 rank 64", "compatibleModelIDs": ["klein-9b-model"]],
      ],
    ]

    try await controller.refinePanels()

    XCTAssertEqual(prepareCalls.count, 8)
    XCTAssertEqual(generationCalls.count, 8)
    XCTAssertEqual(estimates, 8)
    XCTAssertEqual(reassemblies, 1)
    XCTAssertEqual(commands, Array(repeating: ["character-panel-prepare", "dt-estimate", "dt-generate-image",
      "character-panel-prepare", "dt-estimate", "dt-generate-image"], count: 4).flatMap { $0 }
      + ["character-reassemble"])
    for index in rects.indices {
      let nativePrepare = prepareCalls[index * 2]
      let detailPrepare = prepareCalls[index * 2 + 1]
      let bfs = generationCalls[index * 2]
      let detail = generationCalls[index * 2 + 1]
      XCTAssertEqual(nativePrepare.source, initial.path)
      XCTAssertEqual(nativePrepare.scale, 1)
      XCTAssertEqual(nativePrepare.rect, ["x": rects[index].x, "y": rects[index].y,
        "width": rects[index].width, "height": rects[index].height])
      XCTAssertEqual(bfs.width, Self.ceil64(rects[index].width))
      XCTAssertEqual(bfs.height, Self.ceil64(rects[index].height))
      XCTAssertEqual(bfs.loras, ["bfs-head-rank64"])
      XCTAssertEqual(bfs.inputs.count, 2)
      XCTAssertEqual(detailPrepare.source, bfs.outputPath)
      XCTAssertEqual(detailPrepare.rect, ["x": 0, "y": 0, "width": rects[index].width, "height": rects[index].height])
      XCTAssertEqual(detailPrepare.scale, 2)
      XCTAssertEqual(detail.width, Self.ceil64(rects[index].width * 2))
      XCTAssertEqual(detail.height, Self.ceil64(rects[index].height * 2))
      XCTAssertEqual(detail.loras, ["high-resolution-9b"])
      XCTAssertEqual(detail.inputs.count, 1)
    }

    let submissions = generationCalls.count
    let commandCount = commands.count
    try await controller.refinePanels()
    XCTAssertEqual(generationCalls.count, submissions, "Completed BFS and detail stages must be reused.")
    XCTAssertEqual(estimates, submissions, "Reused stages must not repeat Draw Things estimates or submissions.")
    XCTAssertEqual(reassemblies, 2)
    XCTAssertEqual(Array(commands.dropFirst(commandCount)),
      Array(repeating: "character-panel-prepare", count: 8) + ["character-reassemble"])

    controller.document.refinement.detailStrength = 0.8
    try await controller.refinePanels()
    XCTAssertEqual(generationCalls.count, submissions + 4, "Changing detail weight must reuse every native BFS result.")
    for call in generationCalls.dropFirst(submissions) {
      XCTAssertEqual(call.loras, ["high-resolution-9b"])
      XCTAssertEqual(call.weights, [0.8])
    }
    let afterWeight = generationCalls.count
    controller.document.refinement.headPromptStyle = .qualityOnly
    try await controller.refinePanels()
    XCTAssertEqual(generationCalls.count, afterWeight + 8, "A changed BFS output must invalidate its downstream detail input.")
  }

  private static func ceil64(_ value: Int) -> Int { ((value + 63) / 64) * 64 }

  @discardableResult private func artifact(_ url: URL, contents: String) throws -> URL {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(contents.utf8).write(to: url, options: .atomic)
    return url
  }
}
