import XCTest

@testable import StudioCore

final class ModelSetupTests: XCTestCase {
  let presetJSON =
    #"{"id":"ltx25-t2v","name":"LTX 2.5 Text to Video","engine":"ltx25","task":"t2v","description":"Create a shot","components":[{"key":"checkpoint","label":"Transformer","kind":"file"},{"key":"text_encoder","label":"Text encoder","kind":"directory"}]}"#

  func testCatalogContractDecodesAndRequiresEveryComponent() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var selection = ModelSetupSelection()
    XCTAssertEqual(
      selection.missingComponents(for: preset).map(\.key), ["checkpoint", "text_encoder"])
    selection.components = ["checkpoint": "/models/checkpoint", "text_encoder": "  "]
    XCTAssertEqual(selection.missingComponents(for: preset).map(\.key), ["text_encoder"])
    selection.components["text_encoder"] = "/models/encoder"
    XCTAssertTrue(selection.missingComponents(for: preset).isEmpty)
  }

  func testScanAutoSelectsOnlyUniqueCandidatesAndKeepsExplicitChoice() {
    var selection = ModelSetupSelection()
    selection.applyScan(["checkpoint": ["/a", "/b"], "encoder": ["/encoder", "/encoder"]])
    XCTAssertNil(selection.components["checkpoint"])
    XCTAssertEqual(selection.components["encoder"], "/encoder")
    selection.components["checkpoint"] = "/manual"
    selection.applyScan(["checkpoint": ["/other"], "encoder": []])
    XCTAssertEqual(selection.components["checkpoint"], "/manual")
    XCTAssertEqual(selection.components["encoder"], "/encoder")
  }

  func testDownloadsOnlyAppearForCompatibleEngineAndLegacyCatalogStillDecodes() throws {
    let json =
      #"{"id":"encoder","name":"Encoder","description":"Prepare encoder","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"licenseNotice":"Review source terms"}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3"))
    XCTAssertFalse(download.supports(engine: "ltx25"))
    XCTAssertEqual(download.licenseNotice, "Review source terms")
    let legacy = json.replacingOccurrences(
      of: #","engines":["h3"],"licenseNotice":"Review source terms""#, with: "")
    let legacyDownload = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacy.utf8))
    XCTAssertTrue(legacyDownload.supports(engine: "ltx25"))
  }

  func testRecipeSelectionChecksEngineAndCurrentMediaRoles() throws {
    var preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var clip = Clip(engine: .ltx25)
    XCTAssertTrue(preset.supports(clip))
    clip.engine = .h3
    XCTAssertFalse(preset.supports(clip))
    preset.engine = "h3"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertFalse(preset.supports(clip))
    preset.task = "fflf"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(clip))
  }

  func testH3DownloadsFilterByTaskAndSharedComponentsRemainVisible() throws {
    let json =
      #"{"id":"ref","name":"Reference transformer","description":"Prepared pages","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["ref2va"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "ref2va"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(download.supports(engine: "h3", task: "fflf"))
    XCTAssertFalse(download.supports(engine: "ltx25", task: "ref2va"))
    let sharedJSON = json.replacingOccurrences(of: #","tasks":["ref2va"]"#, with: "")
    let shared = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(sharedJSON.utf8))
    XCTAssertTrue(shared.supports(engine: "h3", task: "fflf"))
  }

  func testComponentDownloadButtonsOnlySelectPackagesProvidingThatComponent() throws {
    let json =
      #"{"id":"support","name":"Support files","description":"Audio and task files","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["t2v","fflf"],"components":["checkpoint","audio_vae","tokenizer","processor"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "fflf", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "ref2va", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v", component: "transformer"))
    let legacyJSON = json.replacingOccurrences(
      of: #","components":["checkpoint","audio_vae","tokenizer","processor"]"#, with: "")
    let legacy = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacyJSON.utf8))
    XCTAssertTrue(legacy.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(legacy.supports(engine: "h3", task: "t2v", component: "audio_vae"))
  }

  func testLTX25BaseRecipeSupportsExistingImageAndAudioTaskFallbacks() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    for role in [MediaRole.first, .audioDriver] {
      var clip = Clip(engine: .ltx25)
      clip.attachments = [Attachment(assetID: UUID(), role: role)]
      XCTAssertTrue(preset.supports(clip))
    }
    var referenceClip = Clip(engine: .ltx25)
    referenceClip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(referenceClip))
  }

  func testMemoryModesUseBackendIdentifiers() {
    XCTAssertEqual(
      ModelSetupMemoryMode.allCases.map(\.rawValue), ["automatic", "lower_memory", "custom"])
  }

  func testNativeSetupCatalogAndRecipeWithoutPython() throws {
    let presets = NativeModelSetup.catalog()
    XCTAssertEqual(Set(presets.map(\.id)), Set(["swift-h3-text", "swift-h3-image",
      "swift-h3-reference", "swift-ltx25-text", "swift-ltx25-image"]))
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let preset = try XCTUnwrap(presets.first { $0.id == "swift-ltx25-text" })
    var selected: [String: String] = [:]
    for component in preset.components {
      let url = root.appendingPathComponent(component.key)
      if component.kind == "directory" {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      } else {
        try Data("test".utf8).write(to: url)
      }
      selected[component.key] = url.path
    }
    let recipe = try NativeModelSetup.recipe(preset: preset, selected: selected,
      memoryMode: .lowerMemory)
    XCTAssertEqual(recipe["engine"] as? String, "ltx25")
    XCTAssertEqual((recipe["config"] as? [String: Any])?["stage2_steps"] as? Int, 3)
    let staged = try NativeModelSetup.stage(recipe, directory: root.appendingPathComponent("profiles").path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: staged))
    selected["audio_vae_path"] = root.appendingPathComponent("missing").path
    XCTAssertThrowsError(try NativeModelSetup.recipe(preset: preset, selected: selected,
      memoryMode: .automatic))
  }

  func testNativeScanUsesHeadersAndManifestsWithoutReadingWeights() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func tensor(_ name: String, metadata: [String: String], names: [String]) throws -> URL {
      let file = root.appendingPathComponent(name)
      var header: [String: Any] = ["__metadata__": metadata]
      for key in names { header[key] = ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]] }
      let bytes = try JSONSerialization.data(withJSONObject: header)
      var prefix = UInt64(bytes.count).littleEndian
      var output = withUnsafeBytes(of: &prefix) { Data($0) }
      output.append(bytes)
      output.append(contentsOf: [0, 0, 0, 0])
      try output.write(to: file)
      return file
    }
    let transformer = try tensor("arbitrary.safetensors", metadata: [
      "model_version": "2.5.0", "config": "{\"transformer\":{\"num_layers\":48}}"
    ], names: ["model.diffusion_model.patchify_proj.weight"])
    _ = try tensor("misleading-ltx-transformer.safetensors", metadata: [:], names: ["wrong.weight"])
    let upscaler = try tensor("upscale.safetensors", metadata: [
      "config": "{\"_class_name\":\"LatentUpsampler\",\"in_channels\":128,\"dims\":3,\"spatial_upsample\":true,\"temporal_upsample\":false}"
    ], names: ["initial_conv.weight"])
    let scan = try NativeModelSetup.scan(presetID: "swift-ltx25-text", roots: [root.path])
    XCTAssertEqual(scan.candidates["transformer_path"], [transformer.path])
    XCTAssertEqual(scan.candidates["spatial_upscaler_path"], [upscaler.path])
    XCTAssertTrue(scan.candidates["audio_vae_path", default: []].isEmpty)
    XCTAssertTrue(scan.warnings.contains { $0.contains("audio VAE") })
  }

  func testNativeScanDistinguishesH3TaskPartitions() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for (name, partition, tasks) in [("FL2VA", "fl2va", ["t2va", "fl2va"]),
      ("Ref2VA", "ref2va", ["ref2va"])] {
      let folder = root.appendingPathComponent(name)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      let document: [String: Any] = ["_minimax_h3": ["partition": partition, "tasks": tasks]]
      try JSONSerialization.data(withJSONObject: document).write(to: folder.appendingPathComponent("model_index.json"))
    }
    let image = try NativeModelSetup.scan(presetID: "swift-h3-image", roots: [root.path])
    let reference = try NativeModelSetup.scan(presetID: "swift-h3-reference", roots: [root.path])
    XCTAssertEqual(image.candidates["checkpoint"], [root.appendingPathComponent("FL2VA").path])
    XCTAssertEqual(reference.candidates["checkpoint"], [root.appendingPathComponent("Ref2VA").path])
  }

  func testInstalledNativeScanFindsCompatibleStacksWhenRequested() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_NATIVE_SCAN_ROOT"] else {
      throw XCTSkip("Opt-in installed-model discovery")
    }
    for presetID in ["swift-h3-image", "swift-h3-reference", "swift-ltx25-text"] {
      let result = try NativeModelSetup.scan(presetID: presetID, roots: [root])
      let missing = result.candidates.filter { $0.value.isEmpty }.map(\.key).sorted()
      XCTAssertTrue(missing.isEmpty, "\(presetID) missing \(missing); \(result.warnings)")
    }
  }
}
