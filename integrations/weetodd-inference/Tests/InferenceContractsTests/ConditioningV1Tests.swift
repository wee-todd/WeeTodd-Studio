import XCTest
@testable import InferenceContracts

final class ConditioningV1Tests: XCTestCase {
  func testAcceptsVersionedTaskAndAudioPolicyWithUniqueInputs() throws {
    let inputs = try ConditioningV1.inputs(
      ["version": 1, "task": "fflf", "audio_policy": "generated",
       "inputs": [["id": "first", "kind": "image"], ["id": "last", "kind": "image"]]],
      task: "fflf", audioPolicy: "generated", count: 1...2)
    XCTAssertEqual(inputs.count, 2)
  }

  func testRejectsTaskPolicyAndIdentityMismatch() throws {
    let base: [String: Any] = ["version": 1, "task": "a2v", "audio_policy": "source",
      "inputs": [["id": "voice", "kind": "audio"]]]
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["task": "i2v"]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...1))
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["audio_policy": "generated"]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...1))
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["audio_policy": true]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...1))
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["version": 1.0]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...1))
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["inputs": [
      ["id": "voice"], ["id": "voice"]]]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...2))
    XCTAssertThrowsError(try ConditioningV1.inputs(base.merging(["unsupported": true]) { _, new in new },
      task: "a2v", audioPolicy: "source", count: 1...1))
  }
}
