import Foundation
import XCTest
@testable import StudioCore

final class WorkflowJobTests: XCTestCase {
  func testFailedRunOpensFailedStepInsteadOfEarlierCompletedOutput() throws {
    let data = Data(#"{"status":"failed","totalSeconds":1,"outputs":{},"steps":{"timing":{"name":"Music timing","status":"completed","outputs":{"music_timing":{}}},"inventory":{"name":"Inventory","status":"completed","outputs":{"subjects":[]}},"subjects":{"name":"Subjects","status":"failed","error":"Invalid selection","outputs":null}}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.preferredReviewStepID, "subjects")
  }
  func testGuidedImportRequiresEveryUpstreamApproval() throws {
    let data = Data(#"{"status":"paused","totalSeconds":1,"outputs":{},"steps":{"brief":{"name":"Brief","status":"completed","approved":false},"preview":{"name":"Preview","status":"completed","approved":true,"outputs":{"h3_prompts":{}}}}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertFalse(run.canImportGuidedPlan(requiredSteps: ["brief", "preview"], allSteps: ["brief", "preview"]))
    XCTAssertFalse(run.canImportGuidedPlan(requiredSteps: [], allSteps: ["preview"]))
    XCTAssertTrue(run.canImportGuidedPlan(requiredSteps: ["preview"], allSteps: ["preview"]))
    XCTAssertFalse(run.canImportGuidedPlan(requiredSteps: ["preview"], allSteps: ["preview", "pending_check"]))
    XCTAssertFalse(run.canImportGuidedPlan(requiredSteps: ["preview"], allSteps: []))
  }
  func testStaleCoverageDoesNotOverrideCurrentInventoryReview() throws {
    let data = Data(#"{"status":"paused","totalSeconds":1,"outputs":{},"steps":{"subjects_coverage":{"name":"Coverage","status":"stale","outputs":{"subjects":[]}},"inventory":{"name":"Inventory","status":"completed","outputs":{"subjects":[]}},"subjects":{"name":"Extracted","status":"completed","outputs":{"subjects":[]}}}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.preferredSubjectStepID, "inventory")
  }
  func testCompletedGuidedRunOpensFinalPreviewInsteadOfEarlyClassification() throws {
    let data = Data(#"{"status":"completed","totalSeconds":1,"outputs":{},"steps":{"classify":{"name":"Classify","status":"completed","outputs":{"subjects":[]}},"subjects_coverage":{"name":"Coverage","status":"completed","outputs":{"subjects":[]}},"prompt_preview":{"name":"Preview","status":"completed","outputs":{"h3_prompts":{}}}}}"#.utf8)
    var run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.preferredReviewStepID, "prompt_preview")
    run.awaitingStep = "subjects_coverage"
    XCTAssertEqual(run.preferredReviewStepID, "subjects_coverage")
  }
  func testNumericFieldsRejectInvalidValuesWithoutSilentlyChangingThem() throws {
    XCTAssertEqual(try WorkflowJob.input("24", type: "integer"), .integer(24))
    XCTAssertEqual(try WorkflowJob.input("10.1", type: "number"), .number(10.1))
    XCTAssertThrowsError(try WorkflowJob.input("24.5", type: "integer"))
    XCTAssertThrowsError(try WorkflowJob.input("NaN", type: "number"))
    XCTAssertThrowsError(try WorkflowJob.input("", type: "number"))
  }
  func testJobRoundTripKeepsBindingsAndResumeDirectory() throws {
    let job = WorkflowJob(definition: ["id": .string("custom.test")],
      inputs: ["images": .array([.string("asset:frame")])],
      models: ["assistant": "/models/qwen.ckpt"], assets: ["asset:frame": "/media/frame.png"],
      runDirectory: "/runs/test")
    let decoded = try JSONDecoder().decode(WorkflowJob.self, from: JSONEncoder().encode(job))
    XCTAssertEqual(decoded.inputs, job.inputs)
    XCTAssertEqual(decoded.models, job.models)
    XCTAssertEqual(decoded.assets, job.assets)
    XCTAssertEqual(decoded.runDirectory, job.runDirectory)
    XCTAssertNil(decoded.maxSteps)
  }
  func testRunSummaryShowsReviewEvenWhenExecutionCompletes() throws {
    let data = Data(#"{"status":"completed","totalSeconds":12.2,"steps":{"check":{"name":"Check","status":"completed","outputs":{"review":{"status":"needs_attention","items":[{"severity":"warning","message":"Check identity"}]}}}},"outputs":{"prompt":"A fox.","review":{"status":"needs_attention","items":[]}}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.prompt, "A fox.")
    XCTAssertTrue(run.needsAttention)
    XCTAssertEqual(run.steps["check"]?.status, "completed")
  }
  func testReviewCatalogRoundTripDoesNotReplaceFrozenExecutionCatalog() throws {
    let frozen: JSONValue = .array([.object(["id": .string("original")])])
    var job = WorkflowJob(definition: [:], inputs: ["library": frozen], models: [:], assets: [:], runDirectory: "/runs/test")
    var subject = PlanningSubject(name: "Fresh jacket", kind: .clothing)
    subject.details = "Copper leather."
    job.coverageLibraryCandidates = [LibraryObjectCandidate(subject: subject, packageID: UUID(), version: 2, scope: "global")]
    let restored = try JSONDecoder().decode(WorkflowJob.self, from: JSONEncoder().encode(job))
    XCTAssertEqual(restored.inputs["library"], frozen)
    XCTAssertEqual(restored.coverageLibraryCandidates?.first?.name, "Fresh jacket")
    XCTAssertEqual(restored.coverageLibraryCandidates?.first?.version, 2)
  }
}

final class WorkflowReviewTests: XCTestCase {
  func testApprovalIsSeparateFromStructureValidityAndOldRecordsStillDecode() throws {
    let data = Data(#"{"status":"awaiting_approval","revision":"abc","awaitingStep":"clips","totalSeconds":12,"steps":{"clips":{"name":"Clips","status":"completed","approved":false,"items":{"clip-1":{"status":"completed","approved":true}}}},"outputs":{}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.revision, "abc")
    XCTAssertEqual(run.awaitingStep, "clips")
    XCTAssertEqual(run.steps["clips"]?.items?["clip-1"]?.approved, true)
    XCTAssertFalse(run.steps["clips"]?.approved ?? true)
  }
  func testEditingOneBeatPreservesTimingAndOtherClips() throws {
    let raw = #"{"fps":24,"totalFrames":240,"characters":[{"id":"fox","description":"Red fox"}],"clips":[{"id":"clip-1","startFrame":0,"frameCount":120,"action":"Walk","startState":"Outside","endState":"Door","location":"Forest","characters":["fox"],"continuity":"cut"},{"id":"clip-2","startFrame":120,"frameCount":120,"action":"Rest","startState":"Door","endState":"Inside","location":"Forest","characters":["fox"],"continuity":"continue"}]}"#
    var plan = try JSONDecoder().decode(WorkflowClipPlan.self, from: Data(raw.utf8))
    let original = plan.clips[1]
    var edited = plan.clips[0]; edited.action = "Approach the door carefully"
    try plan.replace(edited)
    XCTAssertEqual(plan.clips[1], original)
    XCTAssertEqual(plan.clips[0].frameCount, 120)
    XCTAssertEqual(plan.totalFrames, 240)
    edited.frameCount = 100
    XCTAssertThrowsError(try plan.replace(edited))
  }
}

final class WorkflowRepairResultTests: XCTestCase {
  func testFailedShotDirectionsExcludeApprovedAndCompletedItems() throws {
    let data = Data(#"{"name":"Clips","status":"failed","items":{"clip-2":{"status":"failed"},"clip-1":{"status":"completed","approved":true},"clip-10":{"status":"failed"},"clip-3":{"status":"failed","approved":true}}}"#.utf8)
    var step = try JSONDecoder().decode(WorkflowRunSummary.Step.self, from: data)
    XCTAssertEqual(step.failedShotIDs(operation: "music.plan_beats@1"), ["clip-2", "clip-10"])
    XCTAssertEqual(step.failedShotIDs(operation: "movie.plan_creative_beats@1"), ["clip-2", "clip-10"])
    XCTAssertTrue(step.failedShotIDs(operation: "project.design_subjects@1").isEmpty)
    step.approved = true
    XCTAssertTrue(step.failedShotIDs(operation: "music.plan_beats@1").isEmpty)
    step.approved = false; step.status = "running"
    XCTAssertTrue(step.failedShotIDs(operation: "music.plan_beats@1").isEmpty)
  }
  func testFailedCancelledAndBlockedRepairsDoNotReportSuccess() throws {
    for status in ["failed", "cancelled", "awaiting_approval"] {
      let data = Data("{\"status\":\"\(status)\",\"awaitingStep\":\"story\",\"totalSeconds\":1,\"steps\":{\"clips\":{\"name\":\"Clips\",\"status\":\"completed\",\"items\":{\"clip-2\":{\"status\":\"completed\"}}}},\"outputs\":{}}".utf8)
      let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
      XCTAssertFalse(run.repairCompleted(stepID: "clips", itemID: "clip-2"))
    }
    let data = Data(#"{"status":"awaiting_approval","awaitingStep":"clips","totalSeconds":1,"steps":{"clips":{"name":"Clips","status":"completed","items":{"clip-2":{"status":"completed"}}}},"outputs":{}}"#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertTrue(run.repairCompleted(stepID: "clips", itemID: "clip-2"))
  }
}
