import XCTest
@testable import StudioCore

final class ReferenceSheetTests: XCTestCase {
  func testTemplatesKeepObjectDefinitionAndDoNotGrantApproval() throws {
    for kind in PlanningSubjectKind.allCases {
      let context = ReferenceSheetContext(subjectKey: "workflow:object", name: "Sample", kind: kind,
        description: "Exact approved description.", linkedDefinitions: "prop_123: Silver key.")
      var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
      draft.modelID = "custom-user-model"; draft.steps = 8; context.apply(to: &draft)
      XCTAssertTrue(draft.prompt.contains("Exact approved description."))
      XCTAssertTrue(draft.prompt.contains("prop_123"))
      XCTAssertEqual(draft.modelID, "custom-user-model"); XCTAssertEqual(draft.steps, 8)
      XCTAssertEqual(draft.width % 64, 0); XCTAssertEqual(draft.height % 64, 0)
      XCTAssertEqual(try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft)), draft)
    }
  }
  func testExplicitReferenceBindingsSurviveWithoutAnAgentReview() throws {
    let data = Data(#"{"id":"cat","kind":"character","name":"Cat","description":"Black cat","aliases":[],"evidence":[],"suggestions":[],"referenceAssets":["generated:one"]}"#.utf8)
    let subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: data)
    XCTAssertEqual(subject.referenceAssetKeys, ["generated:one"])
    XCTAssertNil(subject.descriptionReview)
    let decoded = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: JSONEncoder().encode(subject))
    XCTAssertEqual(decoded.referenceAssetKeys, subject.referenceAssetKeys)
  }

  func testReferenceDraftDoesNotReplaceOrdinaryWorkspace() {
    let destination = ImageAssetDestination(scope: .project, projectID: UUID())
    var normal = DrawThingsImageDraft(destination: destination); normal.prompt = "Keep my work"
    var sheet = normal
    ReferenceSheetContext(subjectKey: "one", name: "Corgi", kind: .character, description: "Short legs").apply(to: &sheet)
    var library = ImageWorkspaceLibrary(); library.record(normal, preview: nil); library.record(sheet, preview: nil)
    XCTAssertEqual(library.sessions.count, 2)
    XCTAssertEqual(library.sessions[destination.storageKey]?.draft.prompt, "Keep my work")
    XCTAssertTrue(sheet.prompt.contains("quadruped stays on four legs"))
  }
}
