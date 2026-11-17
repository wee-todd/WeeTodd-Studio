import CryptoKit
import CoreGraphics
import Foundation
import ImageIO
import ImageLayout
import XCTest
@testable import StudioCore

final class MOSSLayoutAnalysisTests: XCTestCase {
  func testDenseProposalAcceptsOwnedDetailsPastTheOldRegionCap() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let id = UUID(), sha = String(repeating: "a", count: 64)
    var result = try fixture(folder, id: id, sourceSHA: sha)
    var layout = try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONSerialization.data(withJSONObject: result["layout"]!))
    let subject = layout.regions[0].id
    for i in 1...40 {
      var part = ImageLayoutRegion(type: .object, caption: "Owned detail \(i)", bounds: .init(x: 0.3, y: 0.3, width: 0.1, height: 0.1))
      part.parentID = subject; layout.regions.append(part)
    }
    layout.analysis?.recipe = "moss-image-layout-dense-v4"
    result["recipe"] = "moss-image-layout-dense-v4"
    result["layout"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(layout))
    result["effectivePrompt"] = try MOSSLayoutCompiler.compile(layout).effectivePrompt
    let proposal = try MOSSLayoutAnalysisProposal(result: result, output: folder, requestID: id, sourceSHA256: sha, modelDirectory: "/models/qwen")
    XCTAssertEqual(proposal.layout.regions.count, 41)
    XCTAssertEqual(proposal.layout.regions[1].parentID, subject)
    XCTAssertNil(proposal.layout.sourceReconstruction)
  }
  func testNewAnalysisKeepsSourceAsGuideAndDoesNotEnablePixelConditioning() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let id = UUID(), sha = String(repeating: "a", count: 64)
    let result = try fixture(folder, id: id, sourceSHA: sha)
    let proposal = try MOSSLayoutAnalysisProposal(result: result, output: folder, requestID: id, sourceSHA256: sha, modelDirectory: "/models/qwen")
    XCTAssertNil(proposal.layout.sourceReconstruction)
    var captured = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    captured.selectProvider(.nativeMOSS); captured.mossImage?.transformerDirectory = "/models/SFT"; captured.mossImage?.sharedDirectory = "/models/Shared"
    let applied = try proposal.applying(to: captured, captured: captured)
    let request = try XCTUnwrap(JSONSerialization.jsonObject(with: NativeMOSSImagePreparation.request(draft: applied, requestID: UUID(), seed: 42)) as? [String: Any])
    XCTAssertNil(request["sourceImage"])
    XCTAssertEqual((request["configuration"] as? [String:Any])?["regionalLayout"] as? Bool, true)
    XCTAssertNotNil(applied.mossLayout?.guideImagePath)
  }
  func testProposalBindsAnalyzerFamilyAndTheSeparateMOSSCountingTokenizer() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let id = UUID(), sha = String(repeating: "a", count: 64), counting = "/models/moss-tokenizer"
    var result = try fixture(folder, id: id, sourceSHA: sha)
    result["modelFamily"] = ImageLayoutVisionModel.gemma4_12b.rawValue
    result["mossTokenizerDirectory"] = counting; result["mossTokenizerIdentity"] = String(repeating: "c", count: 64)
    var normalized = try XCTUnwrap(result["normalizedRequest"] as? [String: Any])
    normalized["mossTokenizerDirectory"] = counting; result["normalizedRequest"] = normalized
    func proposal(_ value: [String: Any]) throws -> MOSSLayoutAnalysisProposal {
      try MOSSLayoutAnalysisProposal(result: value, output: folder, requestID: id, sourceSHA256: sha,
        modelDirectory: "/models/qwen", mossTokenizerDirectory: counting, expectedModelFamily: .gemma4_12b)
    }
    XCTAssertNoThrow(try proposal(result))
    for change in [("modelFamily", ImageLayoutVisionModel.qwen35_4b.rawValue),
      ("mossTokenizerDirectory", "/other/tokenizer"), ("mossTokenizerIdentity", "")] {
      var bad = result; bad[change.0] = change.1; XCTAssertThrowsError(try proposal(bad))
    }
  }
  func testGemmaPreparationRequiresMOSSCountingTokenizerAndProcessorAssets() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("gemma"), counting = root.appendingPathComponent("moss-tokenizer")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: counting, withIntermediateDirectories: true)
    try Data(#"{"model_type":"gemma4_unified","text_config":{"hidden_size":3840,"num_hidden_layers":48},"vision_config":{"mm_embed_dim":3840,"model_patch_size":48,"patch_size":16},"quantization":{"bits":8,"group_size":64,"mode":"affine"}}"#.utf8).write(to: model.appendingPathComponent("config.json"))
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "model.safetensors"] {
      try Data("{}".utf8).write(to: model.appendingPathComponent(name))
    }
    let source = root.appendingPathComponent("image.png"); try Data([1,2,3]).write(to: source)
    for name in ["tokenizer.json", "tokenizer_config.json"] { try Data("{}".utf8).write(to: counting.appendingPathComponent(name)) }
    XCTAssertTrue(MOSSLayoutAnalysisPreparation.compatibleModel(model))
    XCTAssertThrowsError(try MOSSLayoutAnalysisPreparation.prepare(source: source, modelDirectory: model.path))
    let prepared = try MOSSLayoutAnalysisPreparation.prepare(source: source, modelDirectory: model.path,
      mossTokenizerDirectory: counting.path)
    let request = try XCTUnwrap(JSONSerialization.jsonObject(with: prepared.bytes) as? [String: Any])
    XCTAssertEqual(request["mossTokenizerDirectory"] as? String, counting.path)
    XCTAssertEqual(prepared.mossTokenizerDirectory, counting.path)
  }
  func testAnalyzerDiscoveryUsesSharedComponentsWhenLibraryRootIsBlankOrStale() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("WeeTodd/assistant-qualification/Qwen3.5-4B-MLX-4bit")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data(#"{"model_type":"qwen3_5","text_config":{"hidden_size":2560,"num_hidden_layers":32},"vision_config":{"hidden_size":1024,"out_hidden_size":2560,"patch_size":16},"quantization":{"bits":4,"group_size":64,"mode":"affine"}}"#.utf8).write(to: model.appendingPathComponent("config.json"))
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "model.safetensors"] {
      try Data("{}".utf8).write(to: model.appendingPathComponent(name))
    }
    let shared = root.appendingPathComponent("WeeTodd/MOSS-Image/Shared").path
    XCTAssertEqual(MOSSLayoutAnalysisPreparation.discoverModel(libraryDirectory: "", sharedDirectory: shared), model.path)
    XCTAssertEqual(MOSSLayoutAnalysisPreparation.discoverModel(libraryDirectory: "/missing/library", sharedDirectory: shared), model.path)
  }
  private func fixture(_ directory: URL, id: UUID, sourceSHA: String) throws -> [String: Any] {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let context = try XCTUnwrap(CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    let image = try XCTUnwrap(context.makeImage()), file = directory.appendingPathComponent("source.png")
    let writer = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(writer, image, nil); XCTAssertTrue(CGImageDestinationFinalize(writer))
    let size = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: 1, sourceHeight: 1)
    var layout = ImageLayoutDraft(width: size.width, height: size.height, caption: "Extracted scene", regions: [
      .init(type: .object, caption: "Central red cup", bounds: .init(x: 0.2, y: 0.2, width: 0.6, height: 0.6))])
    layout.guideImagePath = file.path; layout.guideOpacity = 0.4
    layout.analysis = .init(requestID: id.uuidString, sourceSHA256: sourceSHA, sourceWidth: 1, sourceHeight: 1,
      sourceName: "original.jpg", modelIdentity: String(repeating: "b", count: 64), recipe: "moss-image-layout-vision-ocr-v1", warnings: [])
    return ["schema": "weetodd-moss-layout-result-v1", "requestID": id.uuidString, "sourceSHA256": sourceSHA,
      "sourceWidth": 1, "sourceHeight": 1, "snapshotWidth": 1, "snapshotHeight": 1,
      "snapshotSHA256": SHA256.hash(data: try Data(contentsOf: file)).map { String(format: "%02x", $0) }.joined(),
      "modelIdentity": String(repeating: "b", count: 64), "modelDirectory": "/models/qwen",
      "recipe": "moss-image-layout-vision-ocr-v1", "layout": try JSONSerialization.jsonObject(with: JSONEncoder().encode(layout)),
      "effectivePrompt": try MOSSLayoutCompiler.compile(layout).effectivePrompt, "promptTokens": 100, "warnings": [],
      "releasedStages": ["analysis"], "normalizedRequest": ["requestID": id.uuidString, "sourceSHA256": sourceSHA,
        "modelDirectory": "/models/qwen", "targetPixels": 2_000_000]]
  }
  func testProposalAppliesAllLayoutFieldsWithoutChangingRecipeAndRejectsNewerDraft() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let id = UUID(), sha = String(repeating: "a", count: 64)
    let result = try fixture(folder, id: id, sourceSHA: sha)
    let proposal = try MOSSLayoutAnalysisProposal(result: result, output: folder, requestID: id, sourceSHA256: sha, modelDirectory: "/models/qwen")
    var captured = DrawThingsImageDraft(destination: .init(scope: .project, projectID: UUID()))
    captured.selectProvider(.nativeMOSS); captured.prompt = "Old scene"; captured.steps = 25; captured.guidance = 2.5
    let changed = try proposal.applying(to: captured, captured: captured)
    XCTAssertEqual(changed.prompt, "Extracted scene")
    XCTAssertEqual(changed.width, proposal.layout.width); XCTAssertEqual(changed.height, proposal.layout.height)
    XCTAssertEqual(changed.mossLayout, proposal.layout); XCTAssertEqual(changed.steps, 25); XCTAssertEqual(changed.guidance, 2.5)
    XCTAssertNil(changed.canvas)
    var newer = captured; newer.prompt = "Keep newer user edit"
    XCTAssertThrowsError(try proposal.applying(to: newer, captured: captured))
    XCTAssertEqual(try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(changed)), changed)
  }
  func testWrongIdentityUntrustedGuideAndIncompleteCleanupAreRejected() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let id = UUID(), sha = String(repeating: "a", count: 64)
    let valid = try fixture(folder, id: id, sourceSHA: sha)
    for mutation in [("requestID", UUID().uuidString as Any), ("sourceSHA256", String(repeating: "c", count: 64)),
      ("effectivePrompt", "different prompt"), ("releasedStages", []), ("modelDirectory", "/other/model")] {
      var bad = valid; bad[mutation.0] = mutation.1
      XCTAssertThrowsError(try MOSSLayoutAnalysisProposal(result: bad, output: folder, requestID: id, sourceSHA256: sha, modelDirectory: "/models/qwen"))
    }
    var bad = valid
    var layout = try XCTUnwrap(bad["layout"] as? [String: Any]); layout["guideImagePath"] = "/outside/source.png"; bad["layout"] = layout
    XCTAssertThrowsError(try MOSSLayoutAnalysisProposal(result: bad, output: folder, requestID: id, sourceSHA256: sha, modelDirectory: "/models/qwen"))
  }
}
