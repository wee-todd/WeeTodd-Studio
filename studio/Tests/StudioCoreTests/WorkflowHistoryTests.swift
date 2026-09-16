import XCTest
@testable import StudioCore

final class WorkflowHistoryTests: XCTestCase {
  func testOlderCheckpointCountsNestedSavedCallsWithoutInventingHistory() throws {
    let data = Data(#"{"status":"awaiting_approval","totalSeconds":221.9,"definition":{"steps":[{"id":"review","name":"Review objects","operation":"project.review_object_coverage@1"}]},"steps":{"review":{"status":"completed","seconds":90,"calls":[],"items":{"dog":{"calls":[{"text":"one"},{"text":"two"}]}}}}}"#.utf8)
    let snapshot = try JSONDecoder().decode(WorkflowHistorySnapshot.self, from: data)
    XCTAssertTrue(snapshot.executionHistory.isEmpty)
    XCTAssertEqual(snapshot.steps["review"]?.savedResponses, 2)
    XCTAssertEqual(snapshot.steps["review"]?.seconds, 90)
    XCTAssertEqual(snapshot.definition.steps.first?.id, "review")
    XCTAssertTrue(WorkflowOperationDescription.text("project.review_object_coverage@1").contains("missing"))
  }

  func testRecordedCallsRetriesAndReuseRemainDistinct() throws {
    let data = Data(#"{"status":"completed","totalSeconds":20,"definition":{"steps":[]},"steps":{},"historyOmitted":3,"executionHistory":[{"id":"a","stepID":"plan","name":"Plan","operation":"text.plan_edits@1","reason":"Workflow step","startedAt":100,"status":"completed","seconds":20,"modelCalls":2,"reusedCalls":1,"retries":1,"events":[{"id":"c","kind":"model","message":"Plan","status":"returned","seconds":10,"purpose":"Plan edits","requestKey":"abc"}]}]}"#.utf8)
    let snapshot = try JSONDecoder().decode(WorkflowHistorySnapshot.self, from: data)
    let entry = try XCTUnwrap(snapshot.executionHistory.first)
    XCTAssertEqual(entry.modelCalls, 2)
    XCTAssertEqual(entry.reusedCalls, 1)
    XCTAssertEqual(entry.retries, 1)
    XCTAssertEqual(entry.elapsed(at: Date(timeIntervalSince1970: 150)), 20)
    XCTAssertEqual(snapshot.historyOmitted, 3)
  }

  func testLiveEntryUsesWallTimeOnlyWhileRunning() throws {
    let data = Data(#"{"id":"a","stepID":"plan","name":"Plan","operation":"text.plan_edits@1","reason":"Workflow step","startedAt":100,"status":"running","seconds":2,"modelCalls":1,"reusedCalls":0,"retries":0,"events":[]}"#.utf8)
    let entry = try JSONDecoder().decode(WorkflowHistorySnapshot.Entry.self, from: data)
    XCTAssertEqual(entry.elapsed(at: Date(timeIntervalSince1970: 105)), 5)
  }
}
