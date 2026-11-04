import AppKit
import Foundation
import StudioCore
import SwiftUI
import XCTest
@testable import WeeToddStudio

final class LoRAAutomaticDiscoveryTests: XCTestCase {
  @MainActor func testInstalledLibraryWhenExplicitlyRequested() async throws {
    guard let profiles = ProcessInfo.processInfo.environment["WEETODD_LORA_DISCOVERY_CHECK_PROFILES"] else {
      throw XCTSkip("Opt-in header-only installed-library check")
    }
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    store.runtime.profilesDirectory = profiles; store.addClip()
    store.project.clips[0].engine = .ltx25
    let before = store.project
    await store.refreshLoRAFolders()
    XCTAssertEqual(store.project, before)
    XCTAssertFalse(store.folderLoRAEntries.isEmpty, store.loraFolderWarnings.joined(separator: "\n"))
    XCTAssertTrue(store.folderLoRAEntries.contains { $0.adapterKind == .msr })
    XCTAssertTrue(store.folderLoRAEntries.contains { $0.adapterKind == .dfr })
    XCTAssertTrue(store.folderLoRAEntries.contains { $0.adapterKind == .distilledRefinement })
    let counts = Dictionary(grouping: store.folderLoRAEntries, by: { $0.adapterKind?.rawValue ?? $0.status }).mapValues(\.count)
    print("Installed-library header check:", counts, "folders:", store.automaticallyDiscoveredLoRAFolders.count)
    if let destination = ProcessInfo.processInfo.environment["WEETODD_LORA_LIBRARY_SNAPSHOT"] {
      _ = NSApplication.shared
      let host = NSHostingView(rootView: LoRALibrary().environmentObject(store).environment(\.colorScheme, .dark))
      let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 920, height: 660),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua)
      window.contentView = host; defer { window.close() }
      host.frame = window.contentView!.bounds; host.layoutSubtreeIfNeeded()
      RunLoop.main.run(until: Date().addingTimeInterval(0.1)); host.layoutSubtreeIfNeeded()
      let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
      host.cacheDisplay(in: host.bounds, to: bitmap)
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: destination))
      XCTAssertFalse(window.isVisible)
      XCTAssertEqual(store.project, before)
    }
  }
  @MainActor func testDataDirectoryOwnsDefaultModelSetupsAndDiscoveryDoesNotInvalidateTakes() throws {
    let root = try directory(), store = StudioStore(dataDirectory: root, restoreSession: false)
    XCTAssertEqual(store.runtime.profilesDirectory, root.appendingPathComponent("Profiles").path)
    store.addClip(); let clip = try XCTUnwrap(store.selectedClip), before = store.generationRequestKey(for: clip)
    store.runtime.automaticallyDiscoverLoRAs = false
    XCTAssertEqual(store.generationRequestKey(for: clip), before)
  }
  private func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    return root
  }
  private func adapter(_ root: URL, name: String, metadata: [String: String] = [:]) throws -> URL {
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let header: [String: Any] = ["__metadata__": metadata,
      "blocks.0.attn.qkv_proj.lora_A.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
      "blocks.0.attn.qkv_proj.lora_B.weight": ["dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]]]
    let bytes = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(bytes.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(bytes); data.append(Data(repeating: 0, count: 8))
    let file = root.appendingPathComponent(name + ".safetensors")
    try data.write(to: file); return file
  }
  @MainActor func testLinkedModelSetupsDiscoverLibrariesAndUseOnlyPerFileModelHints() async throws {
    let root = try directory(), library = root.appendingPathComponent("external/models/loras")
    let linked = try adapter(library, name: "linked-metadata-poor")
    let neighbor = try adapter(library, name: "declared-23", metadata: ["model_version": "2.3"])
    let unknown = try adapter(library, name: "unlinked-metadata-poor")
    _ = try adapter(library, name: "h3", metadata: ["base_model": "MiniMax H3"])
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx25",
      "config": [:], "components": ["loras": [[linked.path, 1.0]]]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profiles.appendingPathComponent("linked.json"))
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    store.runtime.profilesDirectory = profiles.path; store.addClip()
    let before = store.project
    await store.refreshLoRAFolders()
    XCTAssertEqual(Set(store.compatibleLoRAs(for: .ltx25).map(\.path)), [linked.path, neighbor.path])
    XCTAssertEqual(store.folderLoRAEntries.first { $0.path == unknown.path }?.status, "needsModel")
    XCTAssertEqual(store.project, before)
    XCTAssertTrue(store.globalAssets.isEmpty)
    XCTAssertFalse(FileManager.default.fileExists(atPath: store.defaultLoRAFolder.appendingPathComponent(linked.lastPathComponent).path))
  }
  @MainActor func testImportedLibraryLinksDiscoverTheirNeighborsAndPreserveExplicitExclusions() async throws {
    let root = try directory(), library = root.appendingPathComponent("external/loras")
    let linked = try adapter(library, name: "known", metadata: ["model_version": "2.5"])
    let neighbor = try adapter(library, name: "neighbor", metadata: ["model_version": "2.5"])
    let store = StudioStore(dataDirectory: root.appendingPathComponent("app"), restoreSession: false)
    var asset = MediaAsset(name: "Known", kind: .lora, path: linked.path, scope: .global)
    asset.loraModel = .ltx25; store.globalAssets = [asset]
    await store.refreshLoRAFolders()
    XCTAssertTrue(store.folderLoRAEntries.contains { $0.path == neighbor.path })
    var excluded = LoRAFolder(path: library.path); excluded.enabled = false
    store.runtime.loraFolders = [excluded]
    await store.refreshLoRAFolders()
    XCTAssertTrue(store.folderLoRAEntries.isEmpty)
    store.runtime.loraFolders = []
    await store.refreshLoRAFolders()
    XCTAssertTrue(store.folderLoRAEntries.isEmpty)
    XCTAssertEqual(store.globalAssets, [asset])
  }
  func testLegacyLTX23DictionaryAdapterBindingsDiscoverLibraries() throws {
    let root = try directory(), library = root.appendingPathComponent("external/loras")
    let linked = try adapter(library, name: "legacy-control")
    let profiles = root.appendingPathComponent("Profiles")
    try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
    // The legacy setup generator moves control_lora_path into this dictionary array.
    let recipe: [String: Any] = ["format": "weetodd-headless-v2", "engine": "ltx23",
      "task": "control", "config": [:], "components": ["ic_loras": [
        ["path": linked.path, "family": "union", "strength": 1.0]]]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: profiles.appendingPathComponent("legacy.json"))
    let result = try LoRAFolderDiscovery.discover(folders: [], profilesDirectory: profiles.path,
      assets: [], groups: [])
    XCTAssertEqual(result.automaticFolders.map(\.path), [library.path])
    XCTAssertEqual(result.modelHints[linked.path], .ltx23)
  }
  func testPerFileHintsStayWithinLegacyBridgeLimit() throws {
    let root = try directory(), source = try adapter(root, name: "source")
    var assets: [MediaAsset] = []
    for index in 0...1000 {
      let file = root.appendingPathComponent("linked-\(index).safetensors")
      try FileManager.default.linkItem(at: source, to: file)
      var asset = MediaAsset(name: "Linked", kind: .lora, path: file.path, scope: .global)
      asset.loraModel = .ltx25; assets.append(asset)
    }
    let result = try LoRAFolderDiscovery.discover(folders: [], profilesDirectory: "", assets: assets, groups: [])
    XCTAssertEqual(result.modelHints.count, 1000)
    XCTAssertTrue(result.warnings.contains { $0.contains("hints are limited") })
  }
}
