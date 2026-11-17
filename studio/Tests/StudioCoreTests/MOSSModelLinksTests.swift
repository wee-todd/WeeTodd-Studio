import Foundation
import XCTest
@testable import StudioCore

final class MOSSModelLinksTests: XCTestCase {
  private func library(_ root: URL, transformer: Bool = true) throws {
    let family = root.appendingPathComponent("WeeTodd/MOSS-Image")
    var files = ["Shared/text_encoder/config.json", "Shared/vae/config.json", "Shared/tokenizer/tokenizer.json", "Shared/tokenizer/tokenizer_config.json"]
    if transformer { files += ["SFT/transformer/config.json", "SFT/transformer/model.safetensors", "Base/transformer/config.json", "Base/transformer/model.safetensors"] }
    for name in files {
      let file = family.appendingPathComponent(name)
      try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data("{}".utf8).write(to: file)
    }
  }
  private func analyzer(_ root: URL, gemma: Bool) throws -> String {
    let model = root.appendingPathComponent(gemma ? "ImgGen/mlx-vlm/gemma-4-12B-it-8bit" : "WeeTodd/assistant-qualification/Qwen3.5-4B-MLX-4bit")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    let config = gemma
      ? #"{"model_type":"gemma4_unified","text_config":{"hidden_size":3840,"num_hidden_layers":48},"vision_config":{"mm_embed_dim":3840,"model_patch_size":48,"patch_size":16},"quantization":{"bits":8,"group_size":64,"mode":"affine"}}"#
      : #"{"model_type":"qwen3_5","text_config":{"hidden_size":2560,"num_hidden_layers":32},"vision_config":{"hidden_size":1024,"out_hidden_size":2560,"patch_size":16},"quantization":{"bits":4,"group_size":64,"mode":"affine"}}"#
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "preprocessor_config.json", "model.safetensors"] {
      try Data("{}".utf8).write(to: model.appendingPathComponent(name))
    }
    try Data(config.utf8).write(to: model.appendingPathComponent("config.json")); return model.path
  }
  private func temporary() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }

  func testFreshDiscoveryFindsMOSSAndPrefersGemmaWithoutWritingFiles() throws {
    let root = temporary(); defer { try? FileManager.default.removeItem(at: root) }
    try library(root); _ = try analyzer(root, gemma: false); let gemma = try analyzer(root, gemma: true)
    let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(.init()), remembered: .init(), libraries: [root])
    XCTAssertEqual(links.libraryDirectory, root.path + "/WeeTodd/MOSS-Image")
    XCTAssertEqual(links.sharedDirectory, root.path + "/WeeTodd/MOSS-Image/Shared")
    XCTAssertEqual(links.transformerDirectories["sft:bf16"], root.path + "/WeeTodd/MOSS-Image/SFT/transformer")
    XCTAssertEqual(links.analyzerDirectory, gemma)
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("moss-model-links.json").path))
  }

  func testExplicitAndRememberedOfflineSelectionsTakePriority() throws {
    let root = temporary(); defer { try? FileManager.default.removeItem(at: root) }
    try library(root); _ = try analyzer(root, gemma: true)
    var saved = MOSSModelLinks(); saved.sharedDirectory = "/offline/saved/Shared"
    saved.transformerDirectories["sft:bf16"] = "/offline/saved/SFT"; saved.analyzerDirectory = "/offline/selected-qwen"
    var settings = MOSSImageSettings(); settings.sharedDirectory = "/manual/Shared"
    let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(settings), remembered: saved, libraries: [root])
    XCTAssertEqual(links.sharedDirectory, "/manual/Shared")
    XCTAssertEqual(links.transformerDirectories["sft:bf16"], "/offline/saved/SFT")
    XCTAssertEqual(links.analyzerDirectory, "/offline/selected-qwen")
    let file = root.appendingPathComponent("preferences/links.json"); try links.write(to: file)
    XCTAssertEqual(try MOSSModelLinks.read(from: file), links)
  }

  func testVariantAndPrecisionNeverReuseAnotherTransformersPath() throws {
    let root = temporary(); defer { try? FileManager.default.removeItem(at: root) }; try library(root)
    var saved = MOSSModelLinks(); saved.transformerDirectories["sft:bf16"] = "/manual/sft-bf16"
    var settings = MOSSImageSettings(); settings.quantizationBits = 8
    let quantized = MOSSModelDiscovery.discover(query: MOSSModelQuery(settings), remembered: saved, libraries: [root])
    XCTAssertEqual(quantized.applying(to: MOSSModelQuery(settings)).transformerDirectory, "")
    settings.quantizationBits = nil; settings.variant = "base"
    let base = MOSSModelDiscovery.discover(query: MOSSModelQuery(settings), remembered: saved, libraries: [root])
    XCTAssertEqual(base.applying(to: MOSSModelQuery(settings)).transformerDirectory, root.path + "/WeeTodd/MOSS-Image/Base/transformer")
    XCTAssertEqual(settings.variant, "base")
  }

  func testSharedOnlyInstallationSupportsAnalysisWithoutPretendingTransformerIsInstalled() throws {
    let root = temporary(); defer { try? FileManager.default.removeItem(at: root) }
    try library(root, transformer: false); let gemma = try analyzer(root, gemma: true)
    let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(.init()), remembered: .init(), libraries: [root])
    XCTAssertEqual(links.sharedDirectory, root.path + "/WeeTodd/MOSS-Image/Shared")
    XCTAssertEqual(links.analyzerDirectory, gemma); XCTAssertTrue(links.transformerDirectories.isEmpty)
    try FileManager.default.removeItem(at: root.appendingPathComponent("WeeTodd/MOSS-Image/Shared/tokenizer/tokenizer_config.json"))
    let incomplete = MOSSModelDiscovery.discover(query: MOSSModelQuery(.init()), remembered: .init(), libraries: [root])
    XCTAssertEqual(incomplete.sharedDirectory, "")
  }

  func testEmptyLibraryAndInvalidPreferencesDoNotInventPaths() throws {
    let root = temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let links = MOSSModelDiscovery.discover(query: MOSSModelQuery(.init()), remembered: .init(), libraries: [root])
    XCTAssertEqual(links, MOSSModelLinks())
    let file = root.appendingPathComponent("links.json"); var invalid = MOSSModelLinks(); invalid.sharedDirectory = "relative/Shared"
    try invalid.write(to: file); XCTAssertThrowsError(try MOSSModelLinks.read(from: file))
    invalid.sharedDirectory = ""; invalid.format = "future-unknown-version"; try invalid.write(to: file)
    XCTAssertThrowsError(try MOSSModelLinks.read(from: file))
  }
}
