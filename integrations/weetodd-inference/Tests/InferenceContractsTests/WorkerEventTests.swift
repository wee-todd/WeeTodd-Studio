import Foundation
import XCTest
@testable import InferenceContracts

final class WorkerEventTests: XCTestCase {
  func testProgressPreviewAndResidencyFollowOneJobLifecycle() throws {
    let job = UUID()
    var stream = WorkerEventValidator(jobID: job)
    let events: [WorkerEvent.Payload] = [.accepted,
      .stageStarted(name: "transformer"), .progress(completed: 1, total: 8),
      .preview(relativePath: "previews/step-1.png", width: 768, height: 512),
      .stageReleased(name: "transformer"), .stageStarted(name: "decode"),
      .stageReleased(name: "decode"), .completed(artifacts: ["result.mp4"])]
    for (index, payload) in events.enumerated() {
      let event = WorkerEvent(jobID: job, sequence: UInt64(index), payload: payload)
      let decoded = try WorkerEvent.decode(JSONEncoder().encode(event))
      try stream.accept(decoded)
    }
    XCTAssertTrue(stream.isTerminal)
    XCTAssertNil(stream.residentStage)
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 8, payload: .stageStarted(name: "late"))))
  }

  func testWrongJobDuplicateSequenceAndMissingReleaseCannotUpdateProgress() throws {
    let job = UUID()
    var stream = WorkerEventValidator(jobID: job)
    try stream.accept(WorkerEvent(jobID: job, sequence: 0, payload: .accepted))
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: UUID(), sequence: 1, payload: .stageStarted(name: "text"))))
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 0, payload: .stageStarted(name: "text"))))
    try stream.accept(WorkerEvent(jobID: job, sequence: 1, payload: .stageStarted(name: "text")))
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 2, payload: .stageStarted(name: "transformer"))))
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 2, payload: .completed(artifacts: ["result.mp4"]))))
    try stream.accept(WorkerEvent(jobID: job, sequence: 2, payload: .stageReleased(name: "text")))
    XCTAssertNil(stream.residentStage)
  }

  func testProgressCannotRegressOrExceedTotal() throws {
    let job = UUID()
    var stream = WorkerEventValidator(jobID: job)
    try stream.accept(WorkerEvent(jobID: job, sequence: 0, payload: .accepted))
    try stream.accept(WorkerEvent(jobID: job, sequence: 1, payload: .stageStarted(name: "sample")))
    try stream.accept(WorkerEvent(jobID: job, sequence: 2, payload: .progress(completed: 3, total: 8)))
    for payload: WorkerEvent.Payload in [.progress(completed: 2, total: 8), .progress(completed: 9, total: 8),
      .progress(completed: 4, total: 9), .progress(completed: 0, total: 0)] {
      XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 3, payload: payload)))
    }
    try stream.accept(WorkerEvent(jobID: job, sequence: 3, payload: .progress(completed: 4, total: 8)))
  }

  func testRejectsLargeMessagesUnknownVersionsAndUnsafePreviewPaths() throws {
    XCTAssertThrowsError(try WorkerEvent.decode(Data(repeating: 32, count: 65537)))
    let job = UUID()
    let event = WorkerEvent(jobID: job, sequence: 0, payload: .accepted)
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(event)) as! [String: Any]
    json["version"] = 2
    XCTAssertThrowsError(try WorkerEvent.decode(JSONSerialization.data(withJSONObject: json)))
    var stream = WorkerEventValidator(jobID: job)
    try stream.accept(event)
    try stream.accept(WorkerEvent(jobID: job, sequence: 1, payload: .stageStarted(name: "sample")))
    for path in ["../outside.png", "/tmp/preview.png", "previews/../outside.png", "", "https://example.com/image"] {
      XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 2,
        payload: .preview(relativePath: path, width: 512, height: 512))))
    }
    XCTAssertThrowsError(try stream.accept(WorkerEvent(jobID: job, sequence: 2,
      payload: .preview(relativePath: "preview.png", width: 100000, height: 100000))))
  }

  func testFailureReportsResidentStageUntilWorkerIsReaped() throws {
    let job = UUID()
    var stream = WorkerEventValidator(jobID: job)
    try stream.accept(WorkerEvent(jobID: job, sequence: 0, payload: .accepted))
    try stream.accept(WorkerEvent(jobID: job, sequence: 1, payload: .stageStarted(name: "sample")))
    try stream.accept(WorkerEvent(jobID: job, sequence: 2, payload: .failed(message: "Allocation failed")))
    XCTAssertTrue(stream.isTerminal)
    XCTAssertEqual(stream.residentStage, "sample")
  }
}
