import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeImageGenerationTests: XCTestCase {
  @MainActor func testPreflightMessagesRemainVisibleForBothProviders() {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.imageEstimate = ["eligibility": "blocked", "issues": ["Choose a valid model manifest."]]
    XCTAssertEqual(store.imagePreflightIssues, ["Choose a valid model manifest."])
    store.imageEstimate = ["eligibility": "blocked", "issues": [["message": "Connect Draw Things."]]]
    XCTAssertEqual(store.imagePreflightIssues, ["Connect Draw Things."])
  }
  @MainActor func testNativePreflightAndGenerationRequireNoRemoteConnection() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var commands: [String] = []
    let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, _, payload, _ in
      commands.append(command)
      XCTAssertNil(payload["connection"])
      let request = try XCTUnwrap(payload["nativeImageRequest"] as? [String: Any])
      if command == "image-preflight" { return ["eligibility": "allowed"] }
      return ["asset": ["path": root.appendingPathComponent("image.png").path, "width": 512, "height": 512],
        "fingerprint": "test", "normalizedRequest": request]
    })
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMLX); draft.nativeImage?.manifestPath = "/model/manifest.json"
    store.imageDraft = draft
    await store.prepareImageGeneration()
    XCTAssertEqual(store.imageEstimate?["eligibility"] as? String, "allowed")
    await store.generateImageAsset()
    XCTAssertEqual(commands, ["image-preflight", "image-generate"])
    XCTAssertEqual(store.project.assets.last?.generation?.provider, "nativeMLX")
  }
}
