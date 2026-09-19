import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeEditorialDurationTests: XCTestCase {
  @MainActor func renderedStore(seconds: Double, preserve: Bool? = true, automatic: Bool = false) async throws -> StudioStore {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, _, _ in
      if command == "render" {
        return ["video": "/tmp/coverage-take.mp4", "metadata": ["generation": ["duration_mode": automatic ? "automatic" : "manual"]]]
      }
      if command == "inspect" { return ["duration": seconds, "fps": 24.0] }
      return [:]
    })
    store.addClip(.ltx25)
    store.editClip { $0.duration = 4.8; $0.sourcePath = "/tmp/previous-take.mp4" }
    store.preparedRecipe = "/tmp/job/prepared/recipe.json"
    store.preparedFingerprint = store.signature(for: store.selectedClip!)
    var report: [String: Any] = ["nativeFPS": 24.0]
    if let preserve { report["preserveEditorialDuration"] = preserve }
    store.preparedReport = String(decoding: try JSONSerialization.data(withJSONObject: report), as: UTF8.self)
    await store.renderPrepared()
    return store
  }

  @MainActor func testCoveredManualRenderPreservesEditorialTrimAndFullTake() async throws {
    let store = try await renderedStore(seconds: 121.0 / 24)
    let clip = try XCTUnwrap(store.selectedClip)
    XCTAssertNil(store.error)
    XCTAssertEqual(clip.duration, 4.8)
    XCTAssertEqual(clip.sourcePath, "/tmp/coverage-take.mp4")
    XCTAssertEqual(clip.versions.last?.usableDuration, 121.0 / 24)
    XCTAssertEqual(store.project.assets.last?.duration, 121.0 / 24)
    XCTAssertEqual(clip.renderedSignature, store.signature(for: clip))
  }

  @MainActor func testShortRenderIsSavedWithoutPromotingOrShorteningTimeline() async throws {
    let store = try await renderedStore(seconds: 113.0 / 24)
    let clip = try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(clip.duration, 4.8)
    XCTAssertEqual(clip.sourcePath, "/tmp/previous-take.mp4")
    XCTAssertEqual(clip.versions.last?.path, "/tmp/coverage-take.mp4")
    XCTAssertEqual(clip.versions.last?.usableDuration, 113.0 / 24)
    XCTAssertEqual(store.project.assets.last?.path, "/tmp/coverage-take.mp4")
    XCTAssertTrue(store.error?.contains("shorter") == true)
  }

  @MainActor func testFractionalFrameRoundingDoesNotMoveCut() async throws {
    let store = try await renderedStore(seconds: 115.0 / 24)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.duration, 4.8)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/coverage-take.mp4")
  }

  @MainActor func testLegacyPreparedManualTakeStillCannotShortenTimeline() async throws {
    let store = try await renderedStore(seconds: 113.0 / 24, preserve: nil)
    XCTAssertEqual(store.selectedClip?.duration, 4.8)
    XCTAssertTrue(store.error?.contains("shorter") == true)
  }

  @MainActor func testAutomaticDurationRetainsItsExistingAdoptionPath() async throws {
    let store = try await renderedStore(seconds: 113.0 / 24, preserve: false, automatic: true)
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.duration, 113.0 / 24)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/coverage-take.mp4")
  }
}
