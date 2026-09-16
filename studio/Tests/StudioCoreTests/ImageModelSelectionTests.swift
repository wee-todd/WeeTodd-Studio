import XCTest
@testable import StudioCore

final class ImageModelSelectionTests: XCTestCase {
  func testRestoreAndSameConnectionPreserveModelAndLoRAs() throws {
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    draft.profileID = "local"; draft.modelID = "klein"
    draft.loras = [DrawThingsLoRA(modelID: "detail", weight: 0.7)]
    let saved = try JSONEncoder().encode(draft)
    var restored = try JSONDecoder().decode(DrawThingsImageDraft.self, from: saved)
    restored.selectConnection("local")
    XCTAssertEqual(restored, draft)
    restored.selectConnection("cloud")
    XCTAssertEqual(restored.modelID, "")
    XCTAssertEqual(restored.loras, draft.loras)
  }
  func testExplicitModelChangesKeepIncompatibleLoRAsForReviewAndPreserveInputs() {
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    draft.loras = [DrawThingsLoRA(modelID: "detail", weight: 0.7)]
    draft.moodboard = [ImageWorkspaceInput(path: "/reference.png")]
    draft.selectModel("klein", compatibleLoRAIDs: nil)
    XCTAssertEqual(draft.loras.count, 1)
    draft.selectModel("krea", compatibleLoRAIDs: [])
    XCTAssertEqual(draft.loras, [DrawThingsLoRA(modelID: "detail", weight: 0.7)])
    draft.selectModel("", compatibleLoRAIDs: nil)
    XCTAssertEqual(draft.modelID, "")
    XCTAssertEqual(draft.loras, [DrawThingsLoRA(modelID: "detail", weight: 0.7)])
    XCTAssertEqual(draft.moodboard.count, 1)
  }
}
