import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class DrawThingsRequestSafetyTests: XCTestCase {
  @MainActor private func fixture(invocation: Bridge.Invocation? = nil) throws -> (StudioStore, URL) {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
    let store = StudioStore(dataDirectory: folder, restoreSession: false, invocation: invocation)
    let connection = DrawThingsConnection(id: "test")
    store.drawThingsConnections = [connection]
    store.addClip()
    store.editClip {
      $0.engine = .drawThings
      $0.prompt = "Test prompt"
      $0.drawThings = DrawThingsSelection(profileID: connection.id, modelID: "h3", modelFamily: "minimaxh3")
    }
    return (store, folder)
  }

  @MainActor func testLateDrawThingsRenderPreservesTrimAndClosedDocument() async throws {
    for reopen in [false, true] {
      var continuation: CheckedContinuation<[String: Any], Error>?
      let entered = expectation(description: "render suspended \(reopen)")
      let (store, folder) = try fixture { command, _, _, _ in
        if command == "dt-prepare-clip" { return ["eligibility": "allowed"] }
        if command == "dt-generate-clip" {
          return try await withCheckedThrowingContinuation {
            continuation = $0; entered.fulfill()
          }
        }
        return [:]
      }
      store.editClip { $0.sourcePath = "/tmp/original.mov" }
      let copy = folder.appendingPathComponent("copy.weetodd")
      try ProjectStorage.write(store.project, to: copy)
      await store.prepareDrawThingsClip()
      XCTAssertTrue(store.canGenerateSelected)
      let task = Task { await store.renderDrawThingsClip() }
      await fulfillment(of: [entered], timeout: 2)
      if reopen { store.load(copy) } else { store.editClip { $0.sourceIn = 2 } }
      let video = folder.appendingPathComponent("completed.mov")
      try Data("video".utf8).write(to: video)
      continuation?.resume(returning: ["video": video.path, "manifestPath": "manifest.json"])
      await task.value
      XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/original.mov")
      if reopen {
        XCTAssertTrue(store.selectedClip?.versions.isEmpty == true)
        XCTAssertTrue(store.error?.contains(video.path) == true)
      } else {
        XCTAssertEqual(store.selectedClip?.sourceIn, 2)
        XCTAssertEqual(store.selectedClip?.versions.last?.path, video.path)
      }
    }
  }

  @MainActor func testPreparedDrawThingsCannotBeUsedInReopenedDocument() throws {
    let (store, folder) = try fixture()
    let clip = try XCTUnwrap(store.selectedClip)
    let prepared = PreparedDrawThingsClip(projectID: store.project.id,
      documentSessionID: store.documentSessionID, clipID: clip.id,
      signature: store.signature(for: clip), connection: store.drawThingsConnections[0])
    store.preparedDrawThingsClip = prepared
    XCTAssertTrue(store.canGenerateSelected)
    let file = folder.appendingPathComponent("copy.weetodd")
    try ProjectStorage.write(store.project, to: file)
    store.load(file)
    // Even a late preflight callback holding the original snapshot cannot authorize this copy.
    store.preparedDrawThingsClip = prepared
    XCTAssertFalse(store.canGenerateSelected)
  }

  @MainActor func testMissingOrChangedAttachmentInvalidatesPreparation() async throws {
    let (store, folder) = try fixture()
    let file = folder.appendingPathComponent("reference.png")
    try Data("first".utf8).write(to: file)
    let asset = MediaAsset(name: "Reference", kind: .image, path: file.path, scope: .project)
    store.change { $0.assets.append(asset) }
    store.editClip { $0.attachments = [Attachment(assetID: asset.id, role: .first)] }
    try await store.attachmentDigests.resolve([file.path])
    let clip = try XCTUnwrap(store.selectedClip)
    let signature = store.signature(for: clip)
    store.preparedDrawThingsClip = PreparedDrawThingsClip(projectID: store.project.id,
      documentSessionID: store.documentSessionID, clipID: clip.id, signature: signature,
      connection: store.drawThingsConnections[0])
    XCTAssertTrue(store.canGenerateSelected)
    try Data("changed".utf8).write(to: file, options: .atomic)
    XCTAssertFalse(store.canGenerateSelected)
    try await store.attachmentDigests.resolve([file.path])
    XCTAssertNotEqual(store.signature(for: clip), signature)
    try FileManager.default.removeItem(at: file)
    XCTAssertFalse(store.canGenerateSelected)
  }
}
