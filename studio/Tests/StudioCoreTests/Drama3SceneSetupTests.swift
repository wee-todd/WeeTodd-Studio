import CryptoKit
import Drama3Contracts
import XCTest
@testable import StudioCore

final class Drama3SceneSetupTests: XCTestCase {
  private let bytes = Data("pinned-scene-source".utf8)
  private func fixture() throws -> (root: URL, model: URL, catalog: URL) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let model = root.appendingPathComponent("installed-model")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let file: [String: Any] = ["repo": "fixture/model", "revision": String(repeating: "a", count: 40),
      "filename": "original.safetensors", "target": ".original-head-source.safetensors",
      "size": bytes.count, "sha256": SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()]
    let descriptor: [String: Any] = ["id": "drama3-4bit", "name": "Native", "description": "Fixture",
      "downloadBytes": bytes.count, "requiredDiskBytes": bytes.count,
      "sourceURL": "https://huggingface.co/fixture/model", "licenseURL": "https://huggingface.co/fixture/model/LICENSE",
      "outputKind": "directory", "engines": ["drama3"]]
    let catalog = root.appendingPathComponent("catalog.json")
    try JSONSerialization.data(withJSONObject: [["descriptor": descriptor, "kind": "drama3-native", "files": [file]]]).write(to: catalog)
    return (root, model, catalog)
  }

  func testInstalledModelReusesVerifiedHiddenSharedShardWithoutDownload() async throws {
    let fixture = try fixture()
    let shared = fixture.root.appendingPathComponent("ai-models/.retained")
    try FileManager.default.createDirectory(at: shared, withIntermediateDirectories: true)
    let original = shared.appendingPathComponent("original.safetensors")
    try bytes.write(to: original)
    let source = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
      model: fixture.model, existingRoots: [shared.deletingLastPathComponent().path], progress: { _, _ in },
      transfer: { _, _, _, _ in XCTFail("A verified existing original shard must be reused."); throw CancellationError() })
    XCTAssertEqual(source.deletingLastPathComponent().path, fixture.model.path)
    XCTAssertEqual(try Data(contentsOf: source), bytes)
    XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: source.path)[.systemFileNumber] as? NSNumber,
                   try FileManager.default.attributesOfItem(atPath: original.path)[.systemFileNumber] as? NSNumber)
    // A failed extraction can retry the retained verified source without another acquisition.
    let retry = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
      model: fixture.model, existingRoots: [], progress: { _, _ in },
      transfer: { _, _, _, _ in XCTFail("Retained source must survive extraction failure."); throw CancellationError() })
    XCTAssertEqual(retry, source)
  }

  func testCancelledSourceTransferRetainsPartialAndRetryUsesSameModelFolder() async throws {
    let fixture = try fixture(), payload = bytes
    do {
      _ = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
        model: fixture.model, existingRoots: [], progress: { _, _ in }, transfer: { _, partial, _, _ in
          try payload.prefix(5).write(to: partial); throw CancellationError()
        })
      XCTFail("Cancellation must stop setup.")
    } catch is CancellationError {}
    let source = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
      model: fixture.model, existingRoots: [], progress: { _, _ in }, transfer: { _, partial, _, _ in
        XCTAssertEqual(try Data(contentsOf: partial), payload.prefix(5))
        try payload.write(to: partial)
      })
    XCTAssertEqual(try Data(contentsOf: source), payload)
    XCTAssertEqual(source.deletingLastPathComponent().path, fixture.model.path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.model.appendingPathComponent(".drama3-scene-controls.lock").path))
  }

  func testCorruptSupplementNeverBecomesTheInstalledSource() async throws {
    let fixture = try fixture(), payload = bytes
    do {
      _ = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
        model: fixture.model, existingRoots: [], progress: { _, _ in }, transfer: { _, partial, _, _ in
          try Data(repeating: 0, count: payload.count).write(to: partial)
        })
      XCTFail("Reject an incorrect checksum.")
    } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.model.appendingPathComponent(".original-head-source.safetensors").path))
  }
  func testStaleLockFileAllowsRetryAndActiveOwnerExcludesAnotherSetup() async throws {
    let fixture = try fixture(), payload = bytes
    let lock = fixture.model.appendingPathComponent(".drama3-scene-controls.lock")
    try Data("stale process".utf8).write(to: lock)
    do {
      let owner = try Drama3SceneSetupLock(model: fixture.model)
      defer { withExtendedLifetime(owner) {} }
      do {
        _ = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
          model: fixture.model, existingRoots: [], progress: { _, _ in },
          transfer: { _, _, _, _ in XCTFail("Active ownership must exclude transfers."); throw CancellationError() })
        XCTFail("Reject concurrent setup.")
      } catch { XCTAssertTrue(error.localizedDescription.contains("Another scene-control setup")) }
    }
    let source = try await NativeModelDownloads.prepareDramaSceneSource(catalog: fixture.catalog,
      model: fixture.model, existingRoots: [], progress: { _, _ in },
      transfer: { _, partial, _, _ in try payload.write(to: partial) })
    XCTAssertEqual(try Data(contentsOf: source), payload)
    XCTAssertTrue(FileManager.default.fileExists(atPath: lock.path), "Keep the stable inode to avoid unlink/reopen races.")
  }

  func testSceneSetupRejectsSymlinkedLockFiles() throws {
    let fixture = try fixture()
    let outside = fixture.root.appendingPathComponent("outside")
    try Data().write(to: outside)
    try FileManager.default.createSymbolicLink(at: fixture.model.appendingPathComponent(".drama3-scene-controls.lock"), withDestinationURL: outside)
    XCTAssertThrowsError(try Drama3SceneSetupLock(model: fixture.model))
  }

}
