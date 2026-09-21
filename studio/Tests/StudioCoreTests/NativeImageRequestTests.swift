import Foundation
import XCTest
@testable import StudioCore

final class NativeImageRequestTests: XCTestCase {
  func testTenReferencesCountCanvasAndKeepDisabledCards() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
    draft.selectProvider(.nativeMLX)
    draft.moodboard = (0..<10).map { ImageWorkspaceInput(path: "/fixture/\($0).png") }
    XCTAssertEqual(draft.activeImageInputs.count, 10)
    XCTAssertNil(draft.imageInputIssue)
    draft.canvas = ImageWorkspaceInput(path: "/fixture/canvas.png")
    XCTAssertNotNil(draft.imageInputIssue)
    draft.moodboard[0].enabled = false
    XCTAssertNil(draft.imageInputIssue)
    XCTAssertEqual(draft.activeImageInputs.first?.role, "canvas")
    XCTAssertEqual(draft.activeImageInputs.last?.index, 10)
    XCTAssertEqual(draft.moodboard.count, 10)
  }

  func testProviderSwitchRestoresSettingsAndDoesNotChangeInputs() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
    draft.modelID = "remote-model"; draft.steps = 7; draft.sampler = 17
    draft.moodboard = [ImageWorkspaceInput(path: "/fixture.png")]
    let inputs = draft.moodboard
    draft.selectProvider(.nativeMLX)
    XCTAssertEqual(draft.modelID, "Qwen/Qwen-Image-2.1")
    XCTAssertEqual(draft.steps, 40)
    XCTAssertEqual(draft.width, 1024)
    XCTAssertThrowsError(try draft.request(id: "wrong-serializer"))
    draft.steps = 23
    draft.selectProvider(.drawThings)
    XCTAssertEqual(draft.modelID, "remote-model")
    XCTAssertEqual(draft.steps, 7)
    XCTAssertEqual(draft.sampler, 17)
    XCTAssertEqual(draft.moodboard, inputs)
    draft.selectProvider(.nativeMLX)
    XCTAssertEqual(draft.steps, 23)
  }

  func testLegacyDraftDefaultsToDrawThingsWithoutMigration() throws {
    let draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(draft)) as? [String: Any])
    object.removeValue(forKey: "provider")
    object.removeValue(forKey: "nativeImage")
    let restored = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONSerialization.data(withJSONObject: object))
    XCTAssertEqual(restored.executionProvider, .drawThings)
    XCTAssertEqual(restored.steps, 4)
  }

  func testNativeRequestUsesOrderedIDsAndOmitsInactiveMissingFiles() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try Data("fixture".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
    draft.selectProvider(.nativeMLX)
    draft.nativeImage?.manifestPath = "/model/manifest.json"
    draft.canvas = ImageWorkspaceInput(path: url.path)
    draft.moodboard = [ImageWorkspaceInput(path: "/missing.png"), ImageWorkspaceInput(path: url.path)]
    draft.moodboard[0].strength = 0
    let request = try draft.nativeRequest(id: "native")
    let inputs = try XCTUnwrap(request["inputs"] as? [[String: Any]])
    XCTAssertEqual(inputs.compactMap { $0["imageIndex"] as? Int }, [1, 2])
    XCTAssertEqual(inputs[1]["id"] as? String, draft.moodboard[1].id.uuidString)
    XCTAssertEqual(request["schema"] as? String, "weetodd-native-image-request-v1")
    XCTAssertNil(request["billingPolicy"])
    XCTAssertEqual((request["configuration"] as? [String: Any])?["livePreview"] as? Bool, true)
  }
}
