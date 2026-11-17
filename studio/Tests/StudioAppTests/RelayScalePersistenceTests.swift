import Foundation
import InferenceContracts
import StudioCore
import XCTest
@testable import WeeToddStudio

final class RelayScalePersistenceTests: XCTestCase {
  @MainActor func testWholeMovieScalePersistsAndDefaultsToTwoForOldPreferences() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    XCTAssertEqual(store.relayVSRScale, .x2)
    store.relayVSRScale = .x4; store.videoUpscaleTool = .relayVSR; store.saveVideoUpscalePreferences()
    let reopened = StudioStore(dataDirectory: folder, restoreSession: false)
    XCTAssertEqual(reopened.relayVSRScale, .x4)
    let file = folder.appendingPathComponent("video-upscale.json")
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
    object.removeValue(forKey: "relayScale")
    try JSONSerialization.data(withJSONObject: object).write(to: file)
    let legacy = StudioStore(dataDirectory: folder, restoreSession: false)
    XCTAssertEqual(legacy.relayVSRScale, .x2)
  }
}
