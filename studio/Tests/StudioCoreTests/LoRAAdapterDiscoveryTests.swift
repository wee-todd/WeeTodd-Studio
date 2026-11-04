import XCTest
@testable import StudioCore

final class LoRAAdapterDiscoveryTests: XCTestCase {
  func testAdapterSetupReusesLinkedComponentsAndPinsTheChosenAdapter() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let helper = NativeModelSetupControlTests()
    var components: [String: String] = ["transformer_path": try helper.devPack(root, name: "distilled",
      source: "ltx-2.5-22b-distilled-transformer-bf16.safetensors").path]
    for (key, header) in try NativeLTXConfigResolutionTests().supportHeaders() {
      components[key] = try helper.write(root, name: key, header: header).path
    }
    let adapter = try helper.write(root, name: "chosen-ingredients", header: helper.factors(rank: 128, downscale: "1", version: "2.5"))
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25", "config": [:], "components": components]
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("shared.json"))
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-ingredients" })
    let selected = try NativeModelSetup.linkedSelection(preset: preset, profilesDirectory: root.path,
      adapterPath: adapter.path, componentKey: "ingredients_lora_path")
    XCTAssertEqual(selected["ingredients_lora_path"], adapter.path)
    for key in ["transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path"] {
      XCTAssertEqual(selected[key], components[key])
    }
    XCTAssertEqual(Set(selected.keys), Set(preset.components.map(\.key)))
  }
  func testExcludedChildIsNotScannedThroughAnEnabledParent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let child = root.appendingPathComponent("excluded")
    try FileManager.default.createDirectory(at: child, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data().write(to: root.appendingPathComponent("visible.safetensors"))
    try Data().write(to: child.appendingPathComponent("hidden.safetensors"))
    var excluded = LoRAFolder(path: child.path); excluded.enabled = false
    let result = try NativeLoRAFolderScan.scan([LoRAFolder(path: root.path), excluded])
    XCTAssertEqual((result["entries"] as? [[String: Any]])?.count, 1)
  }
  func testQualifiedSpecializedAndRefinementAdaptersHaveLibraryEntriesWithoutBecomingStyleLoRAs() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let helper = NativeModelSetupControlTests()
    _ = try helper.write(root, name: "refinement", header: helper.distilledHelperHeader())
    _ = try helper.write(root, name: "motion", header: helper.factors(downscale: "2", version: "2.3.0"))
    _ = try helper.write(root, name: "ingredients", header: helper.factors(rank: 128, downscale: "1", version: "2.5"))
    let result = try NativeLoRAFolderScan.scan([LoRAFolder(path: root.path)])
    let entries = try XCTUnwrap(result["entries"] as? [[String: Any]])
    XCTAssertEqual(Set(entries.compactMap { $0["adapterKind"] as? String }), ["distilled_refinement", "motion_track", "ingredients"])
    XCTAssertTrue(entries.allSatisfy { $0["status"] as? String == "specialized" })
    XCTAssertTrue(entries.allSatisfy { $0["loraModel"] as? String != nil })
    XCTAssertThrowsError(try NativeLoRAInspection.inspect(root.appendingPathComponent("ingredients.safetensors"), modelHint: .ltx25))
  }
}
