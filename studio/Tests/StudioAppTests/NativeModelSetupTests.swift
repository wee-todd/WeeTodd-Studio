import XCTest
import CryptoKit
import StudioCore
@testable import WeeToddStudio

final class NativeModelSetupTests: XCTestCase {
  @MainActor func testInstalledNativeDownloadPackageWithPythonUnavailable() async throws {
    guard let app = ProcessInfo.processInfo.environment["WEETODD_NATIVE_DOWNLOAD_APP"] else {
      throw XCTSkip("Opt-in packaged native model download")
    }
    let bundle = URL(fileURLWithPath: app)
    let catalog = bundle.appendingPathComponent("Contents/Resources/RendererSource/src/wee_todd_mlx/model_download_catalog.json")
    let worker = bundle.appendingPathComponent("Contents/MacOS/WeeToddH3MLXWorker")
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: root.appendingPathComponent("profiles").path)
    store.runtime.nativeH3Enabled = true; store.runtime.h3WorkerPath = worker.path
    let state = ModelSetupState(); state.nativeDownloadCatalogURL = catalog
    state.readNativeDownloadToken = { nil }
    await state.loadCatalog(runtime: store.runtime)
    XCTAssertTrue(state.catalogError.isEmpty, state.catalogError)
    let preset = try XCTUnwrap(state.presets.first { $0.id == "swift-h3-text" })
    state.begin(preset); state.selectedDownloadID = "h3-dt-tokenizer"
    state.downloadDestination = root.appendingPathComponent("library").path
    await state.download(store: store)
    XCTAssertTrue(state.error.isEmpty, state.error)
    let package = try XCTUnwrap(NativeModelDownloads.catalog(at: catalog).first { $0.descriptor.id == "h3-dt-tokenizer" })
    let installed = root.appendingPathComponent("library/h3-dt-tokenizer")
    for file in package.files {
      XCTAssertEqual((try FileManager.default.attributesOfItem(atPath: installed.appendingPathComponent(file.target).path)[.size] as? NSNumber)?.int64Value, file.size)
    }
    XCTAssertFalse(store.bridge.busy)
    XCTAssertTrue(state.downloadMessage.contains("Swift"))
  }
  @MainActor func testNativeDownloadCatalogAndExistingFileInstallWithoutPython() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let bytes = Data("installed fixture".utf8), source = root.appendingPathComponent("existing")
    try bytes.write(to: source)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let descriptor: [String: Any] = ["id": "fixture", "name": "Fixture", "description": "Test",
      "downloadBytes": bytes.count, "requiredDiskBytes": bytes.count,
      "sourceURL": "https://huggingface.co/fixture/model", "licenseURL": "https://huggingface.co/fixture/model/LICENSE",
      "outputKind": "directory", "engines": ["ltx25"]]
    let file: [String: Any] = ["repo": "fixture/model", "revision": String(repeating: "a", count: 40),
      "filename": "model", "target": "model", "size": bytes.count, "sha256": digest]
    let catalog = root.appendingPathComponent("catalog.json")
    try JSONSerialization.data(withJSONObject: [["descriptor": descriptor, "kind": "ltx25", "files": [file]]]).write(to: catalog)
    let worker = root.appendingPathComponent("worker"); try Data("#!/bin/sh\nexit 0\n".utf8).write(to: worker)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: root.appendingPathComponent("profiles").path)
    store.runtime.ltx25WorkerPath = worker.path; store.runtime.nativeLTX25Enabled = true
    let state = ModelSetupState(); state.nativeDownloadCatalogURL = catalog
    state.readNativeDownloadToken = { nil }
    await state.loadCatalog(runtime: store.runtime)
    XCTAssertTrue(state.catalogError.isEmpty, state.catalogError)
    XCTAssertEqual(state.downloads.map(\.id), ["fixture"])
    state.begin(try XCTUnwrap(state.presets.first { $0.id == "swift-ltx25-text" }))
    state.roots = [source.path]; state.selectedDownloadID = "fixture"
    state.downloadDestination = root.appendingPathComponent("library").path
    await state.download(store: store)
    XCTAssertTrue(state.error.isEmpty, state.error); XCTAssertFalse(state.downloading)
    let installed = root.appendingPathComponent("library/fixture/model")
    XCTAssertEqual(try Data(contentsOf: installed), bytes)
    XCTAssertFalse(store.bridge.busy); XCTAssertTrue(state.downloadMessage.contains("Swift"))
  }
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
    let sourceConfig = original["config"] as? [String: Any] ?? [:]
    let rounds = sourceConfig["dfr_temporal_rounds"] as? Int ?? 0
    let id = engine == "h3" ? "swift-h3-text" : sourceConfig["dfr_enabled"] as? Bool == true
      ? (rounds == 0 ? "swift-ltx25-dfr-spatial" : "swift-ltx25-dfr-temporal-\(rounds)")
      : "swift-ltx25-text"
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == id })
    let sourceComponents = try XCTUnwrap(original["components"] as? [String: Any])
    let selected = try Dictionary(uniqueKeysWithValues: preset.components.map { field in
      (field.key, try XCTUnwrap((sourceComponents[field.key] ?? sourceConfig[field.key]) as? String))
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
