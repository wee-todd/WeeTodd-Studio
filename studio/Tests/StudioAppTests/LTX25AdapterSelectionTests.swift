import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class LTX25AdapterSelectionTests: XCTestCase {
  @MainActor func testResetRetainsAdapterAfterItsSavedConfigWasDeleted() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false); store.addClip()
    let binding = LTX25AdapterBinding(kind: .msr, path: "/offline/models/msr.safetensors")
    store.applyLTX25Config(.adapter(binding))
    let saved = await store.saveLTX25Config(name: "References")
    XCTAssertTrue(saved)
    store.deleteLTX25Config(try XCTUnwrap(store.savedLTX25Configs.first?.id))
    store.resetLTX25Config()
    XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25AdapterBinding, binding)
    XCTAssertEqual(store.selectedClip?.inferredTask, "ref2va")
  }

  @MainActor func testInstalledAdapterSelectionAndResolutionWhenExplicitlyRequested() async throws {
    guard let profiles = ProcessInfo.processInfo.environment["WEETODD_LORA_DISCOVERY_CHECK_PROFILES"] else {
      throw XCTSkip("Opt-in installed adapter and component header check")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime.profilesDirectory = profiles; store.addClip()
    store.editClip { $0.engine = .ltx25; $0.prompt = "A continuous shot with synchronized sound." }
    await store.refreshLoRAFolders()
    let entries = store.folderLoRAEntries.filter { $0.adapterKind != nil }
    XCTAssertFalse(entries.isEmpty)
    for entry in entries {
      let applied = await store.selectLTX25Adapter(entry)
      XCTAssertTrue(applied, entry.name + ": " + (store.error ?? ""))
      var request = try store.payload()
      request["runtime"] = ["profilesDirectory": profiles, "ffmpegPath": "/usr/bin/true"]
      let description = try NativeLTXPreparation.describe(request: request)
      XCTAssertTrue((description["sourcePaths"] as? [String] ?? []).contains(entry.path), entry.name)
      XCTAssertEqual(store.generationInputReferenceFamily(for: store.selectedClip!),
        (description["generation"] as? [String: Any])?["referenceFamily"] as? String)
    }
    print("Installed specialized adapter selection and component resolution:", entries.count)
  }

  @MainActor func testSelectingAdapterRevealsItsInputChoicesWithoutModelSetupOrLibraryMutation() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    // Header reads classify the actual complete adapter signature; no tensor payload is loaded.
    var header: [String: Any] = ["__metadata__": ["reference_downscale_factor": "2", "model_version": "2.3.0"]]
    for block in 0..<48 {
      for tail in ["attn1.to_k", "attn1.to_out.0", "attn1.to_q", "attn1.to_v", "attn2.to_k",
        "attn2.to_out.0", "attn2.to_q", "attn2.to_v", "ff.net.0.proj", "ff.net.2"] {
        let stem = "diffusion_model.transformer_blocks.\(block).\(tail)"
        header[stem + ".lora_A.weight"] = ["dtype": "BF16", "shape": [32, tail == "ff.net.2" ? 16384 : 4096]]
        header[stem + ".lora_B.weight"] = ["dtype": "BF16", "shape": [tail == "ff.net.0.proj" ? 16384 : 4096, 32]]
      }
    }
    let bytes = try JSONSerialization.data(withJSONObject: header); var size = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &size) { Data($0) }; data.append(bytes)
    let file = root.appendingPathComponent("motion.safetensors"); try data.write(to: file)
    let report = try NativeLoRAFolderScan.scan([LoRAFolder(path: root.path)])
    let entries = try JSONDecoder().decode([LoRAFolderEntry].self,
      from: JSONSerialization.data(withJSONObject: report["entries"]!))
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    store.addClip(); store.editClip { $0.prompt = "Keep this prompt."; $0.seed = 321 }
    let before = try XCTUnwrap(store.selectedClip), assets = store.allAssets
    let applied = await store.selectLTX25Adapter(try XCTUnwrap(entries.first))
    XCTAssertTrue(applied, store.error ?? "")
    let clip = try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(clip.inferredTask, "control")
    XCTAssertEqual(store.generationInputReferenceFamily(for: clip), "motion_track")
    XCTAssertEqual(store.generationInputSupportedTasks(for: clip), ["control"])
    XCTAssertEqual(GenerationControlPurpose.available(engine: clip.engine,
      referenceFamily: store.generationInputReferenceFamily(for: clip),
      supportedTasks: store.generationInputSupportedTasks(for: clip)), [.motionTracks])
    XCTAssertFalse(store.supportsEndpoint(.first, for: clip))
    XCTAssertEqual(clip.prompt, before.prompt); XCTAssertEqual(clip.seed, before.seed)
    XCTAssertEqual(store.allAssets, assets); XCTAssertFalse(store.showRuntime)
    XCTAssertFalse(FileManager.default.fileExists(atPath: store.runtime.profilesDirectory))
    store.undo(); XCTAssertEqual(store.selectedClip, before)
  }
}
