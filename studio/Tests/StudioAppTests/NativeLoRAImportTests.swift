import CryptoKit
import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class NativeLoRAImportTests: XCTestCase {
  private func adapter(_ root: URL, name: String, metadata: [String: String]) throws -> URL {
    let url = root.appendingPathComponent(name + ".safetensors")
    let header: [String: Any] = ["__metadata__": metadata,
      "blocks.0.attn.qkv_proj.lora_A.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
      "blocks.0.attn.qkv_proj.lora_B.weight": ["dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]]]
    let bytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var size = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &size) { Data($0) }; data.append(bytes); data.append(Data(repeating: 0, count: 8))
    try data.write(to: url); return url
  }
  @MainActor func testNativeH3ImportPreservesDeclaredTurboLayoutWithoutPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = try adapter(root, name: "not-a-turbo-filename", metadata: ["base_model": "MiniMax H3",
      "adapter_profile": "turbo", "schedule_points": "5", "qkv_layout": "contiguous_qkv"])
    let original = try Data(contentsOf: url)
    var bridgeCalls = 0
    try FileManager.default.createDirectory(at: root.appendingPathComponent("app"), withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false,
      invocation: { _, _, _, _ in bridgeCalls += 1; throw StudioError.invalid("Unexpected Python bridge invocation.") })
    store.runtime.pythonPath = "/unavailable/python"
    await store.importURLs([url], scope: .global, loraModel: .h3, loraProfile: "standard", loraLayout: "auto")
    XCTAssertNil(store.error)
    XCTAssertEqual(bridgeCalls, 0)
    XCTAssertEqual(store.globalAssets.count, 1)
    XCTAssertEqual(store.globalAssets.first?.loraModel, .h3)
    XCTAssertEqual(store.globalAssets.first?.loraProfile, "turbo")
    XCTAssertEqual(store.globalAssets.first?.loraLayout, "contiguous_qkv")
    XCTAssertEqual(store.globalAssets.first?.path, url.path)
    XCTAssertEqual(try Data(contentsOf: url), original)
  }
  @MainActor func testNativeFolderScanNeedsNoPythonAndDoesNotMutateLibrary() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    _ = try adapter(root, name: "ltx", metadata: ["model_version": "2.3"])
    _ = try adapter(root, name: "unclassified", metadata: [:])
    var bridgeCalls = 0
    try FileManager.default.createDirectory(at: root.appendingPathComponent("app"), withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false,
      invocation: { _, _, _, _ in bridgeCalls += 1; throw StudioError.invalid("Unexpected Python bridge invocation.") })
    store.runtime.pythonPath = "/unavailable/python"
    store.runtime.loraFolders = [LoRAFolder(path: root.path)]
    await store.refreshLoRAFolders()
    XCTAssertEqual(bridgeCalls, 0)
    XCTAssertTrue(store.loraFolderWarnings.isEmpty, store.loraFolderWarnings.joined(separator: "\n"))
    XCTAssertEqual(store.folderLoRAEntries.count, 2)
    XCTAssertEqual(store.folderLoRAEntries.first(where: { $0.name == "ltx" })?.loraModel, .ltx23)
    XCTAssertEqual(store.folderLoRAEntries.first(where: { $0.name == "unclassified" })?.status, "needsModel")
    XCTAssertTrue(store.globalAssets.isEmpty)
    XCTAssertTrue(store.project.assets.isEmpty)
  }
  @MainActor func testExplicitTurboCountMismatchCreatesNoLibraryAsset() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = try adapter(root, name: "not-declared-turbo", metadata: ["base_model": "MiniMax H3", "inference_steps": "12"])
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime.pythonPath = "/unavailable/python"
    await store.importURLs([url], scope: .global, loraModel: .h3, loraProfile: "turbo")
    XCTAssertTrue(store.globalAssets.isEmpty)
    XCTAssertTrue(store.error?.contains("4 evaluations") == true, store.error ?? "No rejection")
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("global-assets.json").path))
  }

  @MainActor func testNativeSelectionAndExplicitLegacyDispatchRemainDistinct() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = try adapter(root, name: "legacy", metadata: ["base_model": "MiniMax H3"])
    var calls = 0
    try FileManager.default.createDirectory(at: root.appendingPathComponent("app"), withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false,
      invocation: { command, _, payload, _ in
        calls += 1; XCTAssertEqual(command, "inspect"); XCTAssertEqual(payload["path"] as? String, url.path)
        return ["kind": "lora", "loraModel": "h3"]
      })
    store.runtime.pythonPath = "/usr/bin/python3"
    XCTAssertTrue(NativeLoRAImport.usesNative(runtime: store.runtime, modelHint: .h3))
    store.runtime.nativeH3Enabled = false; store.runtime.nativeLTX25Enabled = false
    XCTAssertFalse(NativeLoRAImport.usesNative(runtime: store.runtime, modelHint: .h3))
    await store.importURLs([url], scope: .global, loraModel: .h3)
    XCTAssertEqual(calls, 1); XCTAssertNil(store.error)
    XCTAssertEqual(store.globalAssets.first?.path, url.path)
  }

  @MainActor func testInstalledNativeLoRALibraryWithoutPython() async throws {
    guard let manifestPath = ProcessInfo.processInfo.environment["WEETODD_NATIVE_LORA_UI_FIXTURE"] else {
      throw XCTSkip("Opt-in header-only installed LoRA import/scan qualification.")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
    let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let adapters = try XCTUnwrap(manifest["adapters"] as? [[String: Any]])
    guard (1...3).contains(adapters.count) else { throw StudioError.invalid("Pin one to three installed LoRA fixtures.") }
    let output = URL(fileURLWithPath: try XCTUnwrap(manifest["output"] as? String))
    let app = output.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
    var bridgeCalls = 0
    let store = StudioStore(dataDirectory: app, restoreSession: false,
      invocation: { _, _, _, _ in bridgeCalls += 1; throw StudioError.invalid("Unexpected Python invocation.") })
    store.runtime.pythonPath = "/unavailable/python"
    func headerIdentity(_ source: URL) throws -> String {
      let stream = try FileHandle(forReadingFrom: source); defer { try? stream.close() }
      let prefix = try XCTUnwrap(try stream.read(upToCount: 8)); XCTAssertEqual(prefix.count, 8)
      let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
      guard length <= 4 * 1024 * 1024 else { throw StudioError.invalid("Oversized installed LoRA fixture header.") }
      let bytes = try XCTUnwrap(try stream.read(upToCount: Int(length))); XCTAssertEqual(bytes.count, Int(length))
      return SHA256.hash(data: prefix + bytes).map { String(format: "%02x", $0) }.joined()
    }
    var evidence: [[String: Any]] = []
    for item in adapters {
      let source = URL(fileURLWithPath: try XCTUnwrap(item["path"] as? String))
      let model = try XCTUnwrap((item["model"] as? String).flatMap(LoRAModel.init(rawValue:)))
      let digest = try XCTUnwrap(item["headerSHA256"] as? String)
      XCTAssertEqual(try headerIdentity(source), digest)
      let size = try FileManager.default.attributesOfItem(atPath: source.path)[.size] as? NSNumber
      XCTAssertEqual(size?.uint64Value, (item["fileSize"] as? NSNumber)?.uint64Value)
      await store.importURLs([source], scope: .global, loraModel: model,
        loraProfile: item["profile"] as? String, loraLayout: item["layout"] as? String)
      XCTAssertNil(store.error)
      let linked = try XCTUnwrap(store.globalAssets.last)
      XCTAssertEqual(linked.path, source.path); XCTAssertEqual(linked.loraModel, model)
      if let profile = item["profile"] as? String { XCTAssertEqual(linked.loraProfile, profile) }
      if let layout = item["layout"] as? String { XCTAssertEqual(linked.loraLayout, layout) }
      XCTAssertEqual(try headerIdentity(source), digest)
      XCTAssertEqual((try FileManager.default.attributesOfItem(atPath: source.path)[.size] as? NSNumber)?.uint64Value, size?.uint64Value)
      store.runtime.loraFolders = [LoRAFolder(path: source.deletingLastPathComponent().path)]
      await store.refreshLoRAFolders()
      XCTAssertTrue(store.folderLoRAEntries.contains(where: { $0.path == source.path && $0.loraModel == model && $0.status == "ready" }))
      evidence.append(["path": linked.path, "model": model.rawValue, "headerSHA256": digest,
        "fileSize": size ?? 0, "profile": linked.loraProfile ?? "undeclared", "layout": linked.loraLayout ?? "undeclared"])
    }
    XCTAssertEqual(bridgeCalls, 0)
    let saved = try JSONDecoder().decode([MediaAsset].self, from: Data(contentsOf: app.appendingPathComponent("global-assets.json")))
    XCTAssertEqual(saved, store.globalAssets)
    let files = FileManager.default.enumerator(at: app, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
    XCTAssertFalse(files.contains(where: { $0.pathExtension.lowercased() == "safetensors" }))
    guard (testRun?.failureCount ?? 1) == 0 else {
      throw StudioError.invalid("Installed LoRA assertions failed; no passed receipt written.")
    }
    try JSONSerialization.data(withJSONObject: ["status": "passed", "inferenceExecuted": false,
      "pythonBridgeInvocations": bridgeCalls, "weightCopies": 0, "inspectionScope": "bounded headers only; no tensor payload read or whole-file hash",
      "adapters": evidence, "savedLibrary": app.appendingPathComponent("global-assets.json").path], options: [.prettyPrinted, .sortedKeys])
      .write(to: output.appendingPathComponent("qualification.json"))
  }

}
