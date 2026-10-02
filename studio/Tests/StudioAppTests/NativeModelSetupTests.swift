import XCTest
import StudioCore
@testable import WeeToddStudio

final class NativeModelSetupTests: XCTestCase {
  @MainActor func testNativeFolderScanUpdatesSetupWithoutPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let checkpoint = root.appendingPathComponent("FL2VA")
    try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: ["_minimax_h3": [
      "partition": "fl2va", "tasks": ["t2va", "fl2va"]
    ]]).write(to: checkpoint.appendingPathComponent("model_index.json"))
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python",
      profilesDirectory: root.appendingPathComponent("profiles").path)
    let state = ModelSetupState()
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-h3-image" })
    state.begin(preset)
    state.roots = [root.path]
    await state.scan(store: store)
    XCTAssertTrue(state.error.isEmpty, state.error)
    XCTAssertEqual(state.selection.components["checkpoint"], checkpoint.path)
    XCTAssertFalse(state.scanning)
  }

  @MainActor func testInstalledComponentsPassSwiftSetupPreflightWithoutPython() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let source = environment["WEETODD_NATIVE_SETUP_SOURCE"],
      let worker = environment["WEETODD_NATIVE_SETUP_WORKER"],
      let ffmpeg = environment["WEETODD_NATIVE_SETUP_FFMPEG"] else {
      throw XCTSkip("Opt-in installed-component setup preflight")
    }
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: source))) as! [String: Any]
    let engine = try XCTUnwrap(original["engine"] as? String)
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first {
      $0.id == (engine == "h3" ? "swift-h3-text" : "swift-ltx25-text")
    })
    let sourceComponents = try XCTUnwrap(original["components"] as? [String: Any])
    let selected = try Dictionary(uniqueKeysWithValues: preset.components.map { field in
      (field.key, try XCTUnwrap(sourceComponents[field.key] as? String))
    })
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python",
      profilesDirectory: root.appendingPathComponent("profiles").path)
    store.runtime.ffmpegPath = ffmpeg
    if engine == "h3" {
      store.runtime.h3WorkerPath = worker; store.runtime.nativeH3Enabled = true
    } else {
      store.runtime.ltx25WorkerPath = worker; store.runtime.nativeLTX25Enabled = true
    }
    let state = ModelSetupState()
    await state.loadCatalog(runtime: store.runtime)
    state.begin(preset)
    state.selection.components = selected
    await state.createRecipe(store: store)
    XCTAssertTrue(state.error.isEmpty, state.error)
    XCTAssertTrue(store.profiles.contains { $0.id == state.resultPath })
  }

  @MainActor func testLTXSetupCreatesDiscoverableProfileWithoutPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = root.appendingPathComponent("worker")
    try """
    #!/bin/sh
    test "$1" = preflight || exit 5
    echo '{"status":"success","result":{"nativeRuntime":"swift-mlx"}}'
    """.write(to: worker, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python",
      profilesDirectory: root.appendingPathComponent("profiles").path)
    store.runtime.ltx25WorkerPath = worker.path
    store.runtime.nativeLTX25Enabled = true
    let state = ModelSetupState()
    await state.loadCatalog(runtime: store.runtime)
    XCTAssertTrue(state.catalogError.isEmpty)
    let preset = try XCTUnwrap(state.presets.first { $0.id == "swift-ltx25-text" })
    state.begin(preset)
    for component in preset.components {
      let url = root.appendingPathComponent(component.key)
      if component.kind == "directory" {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      } else {
        try Data("test".utf8).write(to: url)
      }
      state.selection.components[component.key] = url.path
    }
    await state.createRecipe(store: store)
    XCTAssertTrue(state.error.isEmpty, state.error)
    XCTAssertTrue(FileManager.default.fileExists(atPath: state.resultPath))
    XCTAssertTrue(store.profiles.contains { $0.id == state.resultPath })
  }
}
