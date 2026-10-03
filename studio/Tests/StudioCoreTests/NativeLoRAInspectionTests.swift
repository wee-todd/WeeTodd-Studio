import Foundation
import XCTest
@testable import StudioCore

final class NativeLoRAInspectionTests: XCTestCase {
  private var root: URL!
  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }
  override func tearDownWithError() throws { try FileManager.default.removeItem(at: root) }
  private func write(_ name: String = "adapter", metadata: [String: String] = [:],
    transform: ([String: Any]) -> [String: Any] = { $0 }, payloadBytes: UInt64 = 8) throws -> URL {
    let file = root.appendingPathComponent(name + ".safetensors")
    var header: [String: Any] = ["__metadata__": metadata,
      "blocks.0.attn.qkv_proj.lora_A.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
      "blocks.0.attn.qkv_proj.lora_B.weight": ["dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]]]
    header = transform(header)
    let bytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(bytes); try data.write(to: file)
    let handle = try FileHandle(forWritingTo: file); defer { try? handle.close() }
    try handle.truncate(atOffset: UInt64(data.count) + payloadBytes)
    return file
  }
  func testDeclaredModelWinsHintAndMetadataPoorAdapterNeedsChoice() throws {
    let ltx = try write(metadata: ["model_version": "2.3"])
    XCTAssertEqual(try NativeLoRAInspection.inspect(ltx, modelHint: .h3)["loraModel"] as? String, "ltx23")
    let plain = try write("H3-TURBO-by-name")
    XCTAssertNil(try NativeLoRAInspection.inspect(plain)["loraModel"])
    XCTAssertNil(try NativeLoRAInspection.inspect(plain, modelHint: .h3)["loraProfile"])
    XCTAssertEqual(try NativeLoRAInspection.inspect(plain, modelHint: .ltx25)["loraModel"] as? String, "ltx25")
  }
  func testH3TurboCountsLayoutAndAdalnDeclarations() throws {
    let adapter = try write(metadata: ["base_model": "MiniMax H3", "adapter_profile": "turbo",
      "schedule_points": "5", "steps": "4", "qkv_fusion": "block diagonal B"])
    let inspected = try NativeLoRAInspection.inspect(adapter)
    XCTAssertEqual(inspected["loraProfile"] as? String, "turbo")
    XCTAssertEqual(inspected["loraLayout"] as? String, "contiguous_qkv")
    XCTAssertEqual(inspected["loraRequiresAdalnGrid"] as? Bool, false)
    let adaln = try write("adaln", metadata: ["base_model": "MiniMax H3"], transform: { header in
      Dictionary(uniqueKeysWithValues: header.map { ($0.key.replacingOccurrences(of: "attn.qkv_proj", with: "adaln_proj.linear"), $0.value) })
    })
    XCTAssertEqual(try NativeLoRAInspection.inspect(adaln)["loraRequiresAdalnGrid"] as? Bool, true)
  }
  func testExplicitTurboImportRejectsOtherwiseUndeclaredNonFourStepMetadata() throws {
    let adapter = try write(metadata: ["base_model": "MiniMax H3", "inference_steps": "12"])
    XCTAssertNil(try NativeLoRAInspection.inspect(adapter)["loraProfile"])
    XCTAssertThrowsError(try NativeLoRAInspection.inspect(adapter, modelHint: .h3, selectedH3Profile: "turbo"))
    let valid = try write("four-evaluations", metadata: ["base_model": "MiniMax H3", "schedule_points": "5"])
    XCTAssertEqual(try NativeLoRAInspection.inspect(valid, modelHint: .h3, selectedH3Profile: "turbo")["loraProfile"] as? String, "turbo")
    let declaredStandard = try write("standard", metadata: ["base_model": "MiniMax H3", "profile": "standard", "inference_steps": "12"])
    XCTAssertEqual(try NativeLoRAInspection.inspect(declaredStandard, selectedH3Profile: "turbo")["loraProfile"] as? String, "standard")
  }
  func testSharedH3SamplingAdmissionDistinguishesUnknownStandardAndTurbo() throws {
    let unknown = try write(metadata: ["base_model": "MiniMax H3"])
    XCTAssertNil(try NativeLoRAInspection.validateH3Sampling(path: unknown.path, selectedProfile: nil, schedulePoints: 12)["loraProfile"])
    XCTAssertThrowsError(try NativeLoRAInspection.validateH3Sampling(path: unknown.path, selectedProfile: "turbo", schedulePoints: 12))
    let standard = try write("standard", metadata: ["base_model": "MiniMax H3", "profile": "standard", "inference_steps": "12"])
    XCTAssertEqual(try NativeLoRAInspection.validateH3Sampling(path: standard.path, selectedProfile: "standard", schedulePoints: 13)["h3DeclaredEvaluations"] as? Int, 12)
    XCTAssertThrowsError(try NativeLoRAInspection.validateH3Sampling(path: standard.path, selectedProfile: "turbo", schedulePoints: 5))
    let turbo = try write("turbo", metadata: ["base_model": "MiniMax H3", "profile": "turbo", "schedule_points": "5"])
    XCTAssertThrowsError(try NativeLoRAInspection.validateH3Sampling(path: turbo.path, selectedProfile: nil, schedulePoints: 12))
    XCTAssertEqual(try NativeLoRAInspection.validateH3Sampling(path: turbo.path, selectedProfile: nil, schedulePoints: 5)["h3DeclaredEvaluations"] as? Int, 4)
  }

  func testRejectsConflictingFamiliesScalingAndSpecializedContracts() throws {
    for metadata in [["base_model": "MiniMax H3", "model_version": "2.5"],
      ["model_version": "2.4"], ["base_model": "Flux.1-dev"],
      ["base_model": "ltx-video", "adapter_profile": "standard"],
      ["lora_rank": "2", "ss_network_dim": "3"], ["lora_alpha": "nan"],
      ["model_version": "2.5", "reference_downscale_factor": "2"],
      ["model_version": "2.5", "adapter_profile": "pixel-detail"]] {
      let adapter = try write(metadata: metadata)
      XCTAssertThrowsError(try NativeLoRAInspection.inspect(adapter, modelHint: .h3), "\(metadata)")
    }
    for fields in [["steps": "5"], ["steps": "4", "schedule_points": "6"],
      ["profile": "standard", "steps": "4"], ["qkv_layout": "contiguous interleaved"],
      ["qkv_layout": "unknown"], ["steps": "true"]] {
      let metadata = fields.merging(["base_model": "MiniMax H3", "adapter_profile": "turbo"]) { old, _ in old }
      XCTAssertThrowsError(try NativeLoRAInspection.inspect(write(metadata: metadata)), "\(fields)")
    }
  }
  func testRejectsIncompletePairsForeignTensorsBadOffsetsAndBooleanShape() throws {
    let a = "blocks.0.attn.qkv_proj.lora_A.weight", b = "blocks.0.attn.qkv_proj.lora_B.weight"
    let cases: [([String: Any]) -> [String: Any]] = [
      { var h = $0; h.removeValue(forKey: b); return h },
      { var h = $0; h["unrecognized.weight"] = h.removeValue(forKey: b); return h },
      { var h = $0; h[b] = ["dtype": "BF16", "shape": [1, 2], "data_offsets": [4, 8]]; return h },
      { var h = $0; h[a] = ["dtype": "BF16", "shape": [true, 2], "data_offsets": [0, 4]]; return h },
      { var h = $0; h[a] = ["dtype": "BF16", "shape": [1, 2], "data_offsets": [1, 5]]; return h },
      { var h = $0; h["blocks.0.attn.qkv_proj.lora_down.weight"] = h.removeValue(forKey: a); return h },
      { var h = $0; h["orphan.alpha"] = h.removeValue(forKey: b); return h },
      { var h = $0; h["__metadata__"] = ["steps": 4]; return h }]
    for change in cases { XCTAssertThrowsError(try NativeLoRAInspection.inspect(write(transform: change), modelHint: .h3)) }
    let malformed = root.appendingPathComponent("bad.safetensors")
    try Data([255,255,255,255,255,255,255,255]).write(to: malformed)
    XCTAssertThrowsError(try NativeLoRAInspection.inspect(malformed))
  }
  func testHeaderOnlySparseAdapterDoesNotReadOrCopyPayload() throws {
    let size: UInt64 = 512 * 1024 * 1024
    let adapter = try write(metadata: ["model_version": "2.5"], transform: { header in
      var h = header
      h["blocks.0.attn.qkv_proj.lora_B.weight"] = ["dtype": "BF16", "shape": [size / 2, 1], "data_offsets": [4, size + 4]]
      return h
    }, payloadBytes: size + 4)
    let files = try FileManager.default.contentsOfDirectory(atPath: root.path)
    XCTAssertEqual(try NativeLoRAInspection.inspect(adapter)["loraModel"] as? String, "ltx25")
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), files)
    XCTAssertGreaterThan((try FileManager.default.attributesOfItem(atPath: adapter.path)[.size] as? NSNumber)?.uint64Value ?? 0, size)
    // Deliberately no full Data(contentsOf:), hashing, MLX import or payload read.
    // Target compatibility remains the shared worker's responsibility.
  }
  func testFolderScanDeduplicatesLinksAndClassifiesWithoutTraversingSymlinkDirectories() throws {
    let ltx = try write("ltx", metadata: ["model_version": "2.3"])
    _ = try write("needs-choice")
    _ = try write("specialized", metadata: ["model_version": "2.5", "reference_downscale_factor": "2"])
    try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("linked.safetensors"), withDestinationURL: ltx)
    try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("loop"), withDestinationURL: root)
    let nested = root.appendingPathComponent("nested")
    try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: ltx, to: nested.appendingPathComponent("nested.safetensors"))
    try Data().write(to: root.appendingPathComponent("remote.ckpt"))
    var shallow = LoRAFolder(path: root.path); shallow.recursive = false
    let result = try NativeLoRAFolderScan.scan([shallow, LoRAFolder(path: root.path)])
    let entries = try XCTUnwrap(result["entries"] as? [[String: Any]])
    XCTAssertEqual(entries.count, 4)
    XCTAssertEqual(Set(entries.compactMap { $0["path"] as? String }).count, 4)
    XCTAssertEqual(entries.first(where: { $0["name"] as? String == "needs-choice" })?["status"] as? String, "needsModel")
    XCTAssertEqual(entries.first(where: { $0["name"] as? String == "specialized" })?["status"] as? String, "specialized")
    XCTAssertTrue((result["warnings"] as? [String] ?? []).contains(where: { $0.contains(".ckpt") }))
    let limited = try NativeLoRAFolderScan.scan([LoRAFolder(path: root.path)], maximumFiles: 1)
    XCTAssertEqual((limited["entries"] as? [[String: Any]])?.count, 1)
    XCTAssertTrue((limited["warnings"] as? [String] ?? []).contains(where: { $0.contains("Scan limit") }))
    var disabled = shallow; disabled.enabled = false
    XCTAssertTrue((try NativeLoRAFolderScan.scan([disabled])["entries"] as? [[String: Any]] ?? []).isEmpty)
    XCTAssertFalse((try NativeLoRAFolderScan.scan([LoRAFolder(path: root.appendingPathComponent("missing").path)])["warnings"] as? [String] ?? []).isEmpty)
  }
}
