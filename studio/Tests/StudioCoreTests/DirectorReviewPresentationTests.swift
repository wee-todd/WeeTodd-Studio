import Foundation
import XCTest
@testable import StudioCore

final class DirectorReviewPresentationTests: XCTestCase {
  private func definition(version: String = "1.1.0", detailed: Bool = false) -> [String: JSONValue] {
    let ids = ["creative_brief", "subjects", "classify", "inventory", "design", "subjects_coverage", "story", "clips", "prompt_preview"]
    let required = detailed ? Set(["creative_brief", "classify", "inventory", "design", "subjects_coverage", "story", "clips", "prompt_preview"]) : Set(["creative_brief", "subjects_coverage", "clips"])
    return ["id": .string("weetodd.guided-movie-planning"), "version": .string(version), "steps": .array(ids.map {
      .object(["id": .string($0), "name": .string($0), "operation": .string("keep.me@1"), "requiresApproval": .boolean(required.contains($0))])
    })]
  }
  private func run(_ states: [String: (String, Bool)], awaiting: String? = nil) -> WorkflowRunSummary {
    WorkflowRunSummary(status: "paused", totalSeconds: 0, steps: states.mapValues {
      .init(name: "Stage", status: $0.0, approved: $0.1)
    }, outputs: [:], awaitingStep: awaiting)
  }
  func testReviewModeChangesOnlyNewSupportedDefinitionsAndPreservesOtherStepFields() throws {
    let original = definition()
    let changed = try DirectorReviewMode.detailed.applying(to: original, hasStarted: false)
    let model = DirectorReviewPresentation(definition: changed, result: nil)
    XCTAssertEqual(model.requiredStepIDs, ["creative_brief", "classify", "inventory", "design", "subjects_coverage", "story", "clips", "prompt_preview"])
    guard case .array(let steps) = changed["steps"], case .object(let first) = steps[0] else { return XCTFail() }
    XCTAssertEqual(first["operation"], .string("keep.me@1"))
    XCTAssertThrowsError(try DirectorReviewMode.focused.applying(to: changed, hasStarted: true))
    XCTAssertThrowsError(try DirectorReviewMode.focused.applying(to: definition(version: "1.0.0", detailed: true), hasStarted: false))
    XCTAssertEqual(DirectorReviewPresentation(definition: original, result: nil).requiredStepIDs, ["creative_brief", "subjects_coverage", "clips"])
  }
  func testSavedDetailedJobStillStopsAtEarlierUnapprovedClassification() {
    let result = run(["creative_brief": ("completed", true), "classify": ("completed", false), "subjects_coverage": ("completed", true)])
    let model = DirectorReviewPresentation(definition: definition(version: "1.0.0", detailed: true), result: result)
    XCTAssertEqual(model.nextAction, .review("classify"))
    XCTAssertEqual(model.phase(for: "classify"), .subjects)
  }
  func testApprovalContinuesPastStaleAwaitingPointerAndIncompleteWorkNeverImports() {
    let result = run(["creative_brief": ("completed", true)], awaiting: "creative_brief")
    let model = DirectorReviewPresentation(definition: definition(), result: result)
    XCTAssertEqual(model.nextAction, .continuePlanning)
    XCTAssertFalse(model.canImport)
  }
  func testPhaseSelectionPrefersActualAwaitingGateThenFinalCompletedSubjectOutput() {
    var result = run(["creative_brief": ("completed", true), "classify": ("completed", false), "subjects_coverage": ("completed", false)], awaiting: "classify")
    XCTAssertEqual(DirectorReviewPresentation(definition: definition(detailed: true), result: result).stepID(for: .subjects), "classify")
    result.awaitingStep = nil
    XCTAssertEqual(DirectorReviewPresentation(definition: definition(), result: result).stepID(for: .subjects), "subjects_coverage")
  }
  func testBatchApprovalNeedsSavedCompleteContentButNotSeparateItemApprovals() {
    var step = WorkflowRunSummary.Step(name: "Subjects", status: "completed", approved: false, items: ["one": .init(status: "completed", approved: false)])
    XCTAssertTrue(DirectorReviewPresentation.canApprove(step, hasUnsavedDraft: false))
    XCTAssertFalse(DirectorReviewPresentation.canApprove(step, hasUnsavedDraft: true))
    step.items?["one"]?.status = "stale"
    XCTAssertFalse(DirectorReviewPresentation.canApprove(step, hasUnsavedDraft: false))
  }
  func testFocusedImportNeedsGeneratedPreviewAndAllActualGatesButNoExtraPreviewApproval() {
    let ids = ["creative_brief", "subjects", "classify", "inventory", "design", "subjects_coverage", "story", "clips", "prompt_preview"]
    var result = run(Dictionary(uniqueKeysWithValues: ids.map { ($0, ("completed", ["creative_brief", "subjects_coverage", "clips"].contains($0))) }))
    XCTAssertFalse(DirectorReviewPresentation(definition: definition(), result: result).canImport)
    result.steps["prompt_preview"]?.outputs = ["h3_prompts": .object([:])]
    XCTAssertTrue(DirectorReviewPresentation(definition: definition(), result: result).canImport)
    XCTAssertEqual(DirectorReviewPresentation(definition: definition(), result: result).nextAction, .addToProject)
    XCTAssertFalse(DirectorReviewPresentation(definition: definition(detailed: true), result: result).canImport)
  }
  func testEveryGuidedExecutionStepNavigatesWithinItsCreativePhase() {
    let model = DirectorReviewPresentation(definition: definition(), result: nil)
    let expected: [(String, DirectorReviewPhase)] = [
      ("describe", .brief), ("creative_brief", .brief), ("resolve", .brief),
      ("subjects", .subjects), ("classify", .subjects), ("links", .subjects), ("inventory", .subjects),
      ("design", .subjects), ("subjects_coverage", .subjects), ("story", .shots),
      ("clips", .shots), ("endpoints", .shots), ("check", .shots), ("prompt_preview", .shots)]
    for (id, phase) in expected { XCTAssertEqual(model.phase(for: id), phase, id) }
  }
  func testBriefNavigationKeepsEditableBriefAfterResolveCompletes() {
    var definition = definition()
    definition["steps"] = .array([.object(["id": .string("creative_brief"), "requiresApproval": .boolean(true)]), .object(["id": .string("resolve")])])
    let result = run(["creative_brief": ("completed", true), "resolve": ("completed", false)])
    XCTAssertEqual(DirectorReviewPresentation(definition: definition, result: result).stepID(for: .brief), "creative_brief")
  }
  func testPreparationRequiresBriefAndEachDeclaredModelBinding() {
    let missing = DirectorPreparationState(requiredModels: ["assistant"], bindings: ["other": "/models/other"], brief: "A story")
    XCTAssertEqual(missing.missingModel, "assistant")
    XCTAssertFalse(missing.canPrepare)
    XCTAssertEqual(DirectorPreparationState(requiredModels: ["assistant"], bindings: ["assistant": "  \n"], brief: "A story").missingModel, "assistant")
    let empty = DirectorPreparationState(requiredModels: ["assistant"], bindings: ["assistant": "/models/qwen"], brief: " \n")
    XCTAssertNil(empty.missingModel); XCTAssertFalse(empty.canPrepare)
    XCTAssertTrue(DirectorPreparationState(requiredModels: ["assistant"], bindings: ["assistant": "/models/qwen"], brief: "A story").canPrepare)
  }
  func testRepairDiffReportsChangedFieldsWithoutHidingUnexpectedChanges() {
    let before = WorkflowClipDraft(id: "clip-1", startFrame: 0, frameCount: 120, action: "Walk", startState: "Door", endState: "Desk", location: "Office", characters: ["Ada"], continuity: "cut")
    var after = before; after.action = "Run"; after.location = "Hall"
    let changes = DirectorShotRepairScope.action.changes(from: before, to: after)
    XCTAssertEqual(changes.map(\.field), ["Action", "Location"])
    XCTAssertEqual(changes.map(\.withinScope), [true, false])
    XCTAssertEqual(changes[0].before, "Walk"); XCTAssertEqual(changes[0].after, "Run")
  }
}
