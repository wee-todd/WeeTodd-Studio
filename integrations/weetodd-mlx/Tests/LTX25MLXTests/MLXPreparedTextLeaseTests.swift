import Foundation
import MLX
import XCTest
@testable import LTX25MLX

final class MLXPreparedTextLeaseTests: XCTestCase {
  func testPolicyDecoderRejectsUnknownMissingAndUnsafeFields() throws {
    let original: [String:Any] = ["head_checkpoint_path":"/tmp/head.safetensors","minimum_seconds":0.25,"maximum_seconds":30]
    let decoded = try JSONDecoder().decode(MLXAutomaticDurationPolicy.self, from: JSONSerialization.data(withJSONObject:original))
    XCTAssertEqual(try decoded.maximumFrames(fps:24),713)
    var unknown = original; unknown["ignore_bounds"] = true
    XCTAssertThrowsError(try JSONDecoder().decode(MLXAutomaticDurationPolicy.self, from: JSONSerialization.data(withJSONObject:unknown)))
    var missing = original; missing.removeValue(forKey:"maximum_seconds")
    XCTAssertThrowsError(try JSONDecoder().decode(MLXAutomaticDurationPolicy.self, from: JSONSerialization.data(withJSONObject:missing)))
    for (key,value) in [("head_checkpoint_path","/tmp/head\0.safetensors" as Any),
      ("head_checkpoint_path","/"+String(repeating:"a",count:4096) as Any),
      ("head_checkpoint_path","relative.safetensors" as Any), ("minimum_seconds",true as Any),
      ("maximum_seconds",30.01 as Any), ("minimum_seconds",0.24 as Any)] {
      var invalid = original; invalid[key] = value
      XCTAssertThrowsError(try JSONDecoder().decode(MLXAutomaticDurationPolicy.self, from:JSONSerialization.data(withJSONObject:invalid)))
    }
    XCTAssertThrowsError(try MLXTextPreparationBinding(originalRecipeSHA256:String(repeating:"a",count:64),
      prompt:"room",negativePrompt:nil,gemmaRoot:"/tmp/gemma\0",connectorCheckpoint:"/tmp/connector"))
    var pinned = original; pinned["head_header_sha256"] = String(repeating:"a",count:64)
    XCTAssertEqual(try JSONDecoder().decode(MLXAutomaticDurationPolicy.self,from:JSONSerialization.data(withJSONObject:pinned)).headHeaderSHA256,String(repeating:"a",count:64))
    pinned["head_header_sha256"] = "not-a-header-hash"
    XCTAssertThrowsError(try JSONDecoder().decode(MLXAutomaticDurationPolicy.self,from:JSONSerialization.data(withJSONObject:pinned)))
  }
  private static func binding(negative: String? = nil, prompt: String = "A quiet room.") throws -> MLXTextPreparationBinding {
    try MLXTextPreparationBinding(originalRecipeSHA256: String(repeating: "a", count: 64),
      prompt: prompt, negativePrompt: negative, gemmaRoot: "/tmp/gemma", connectorCheckpoint: "/tmp/connector.safetensors")
  }
  private static func output(_ token: Int) -> MLXTextEncoder.Output {
    MLXTextEncoder.Output(video: MLXArray([Float(token)], [1,1]),
      audio: MLXArray([Float(token+1)], [1,1]), tokenIDs: [token])
  }
  func testPositiveIsEncodedOnceAndResolutionPrecedesNegativeAndConsumption() throws {
    try Device.withDefaultDevice(.cpu) {
      let binding = try Self.binding(negative: "Blurred."), policy = MLXAutomaticDurationPolicy(headCheckpointPath: "/tmp/head.safetensors")
      var events: [String] = []
      let lease = try MLXAutomaticTextPreparation.prepare(binding: binding, policy: policy, fps: 24,
        encode: { prompt, negative in
          events.append(negative ? "negative" : "positive")
          XCTAssertEqual(prompt, negative ? "Blurred." : "A quiet room.")
          return Self.output(negative ? 8 : 7)
        }, predict: { value in
          XCTAssertEqual(value.tokenIDs, [7]); events.append("predict"); return 3
        }, admitResolvedFrames: { frames in XCTAssertEqual(frames,65); events.append("admit") })
      XCTAssertEqual(events,["positive","predict","admit","negative"])
      let value = try lease.consume(expectedBinding: binding, resolvedFrames: 65, fps: 24)
      XCTAssertEqual(value.positive.tokenIDs,[7]); XCTAssertEqual(value.negative?.tokenIDs,[8])
      XCTAssertEqual(lease.resolution.effectiveDurationSeconds,65.0/24)
      XCTAssertTrue(lease.isReleased)
      XCTAssertThrowsError(try lease.consume(expectedBinding: binding, resolvedFrames: 65, fps: 24))
    }
  }
  func testManualUnconditionalAbsenceDoesNotEncodeEmptyNegativeAndAdmissionFailureStopsIt() throws {
    try Device.withDefaultDevice(.cpu) {
      let policy = MLXAutomaticDurationPolicy(headCheckpointPath: "/tmp/head.safetensors")
      var encodes = 0
      let lease = try MLXAutomaticTextPreparation.prepare(binding: Self.binding(), policy: policy, fps: 24,
        encode: { _, negative in XCTAssertFalse(negative); encodes += 1; return Self.output(2) },
        predict: { _ in 3 }, admitResolvedFrames: { _ in })
      XCTAssertEqual(encodes,1); lease.release(); XCTAssertTrue(lease.isReleased)
      enum Refusal: Error { case tooLarge }
      encodes = 0
      XCTAssertThrowsError(try MLXAutomaticTextPreparation.prepare(binding: Self.binding(negative: ""), policy: policy, fps: 24,
        encode: { _, _ in encodes += 1; return Self.output(2) }, predict: { _ in 3 },
        admitResolvedFrames: { _ in throw Refusal.tooLarge })) { XCTAssertTrue($0 is Refusal) }
      XCTAssertEqual(encodes,1)
    }
  }
  func testBindingAndGeometryMismatchReleaseInsteadOfReusingDifferentPrompt() throws {
    try Device.withDefaultDevice(.cpu) {
      let binding = try Self.binding(), policy = MLXAutomaticDurationPolicy(headCheckpointPath: "/tmp/head.safetensors")
      func make() throws -> MLXPreparedTextLease {
        try MLXAutomaticTextPreparation.prepare(binding: binding, policy: policy, fps: 24,
          encode: { _, _ in Self.output(4) }, predict: { _ in 3 }, admitResolvedFrames: { _ in })
      }
      let wrongPrompt = try make()
      XCTAssertThrowsError(try wrongPrompt.consume(expectedBinding: Self.binding(prompt: "A different shot."), resolvedFrames: 65, fps: 24))
      XCTAssertTrue(wrongPrompt.isReleased)
      let wrongFrames = try make()
      XCTAssertThrowsError(try wrongFrames.consume(expectedBinding: binding, resolvedFrames: 73, fps: 24))
      XCTAssertTrue(wrongFrames.isReleased)
      let wrongFPS = try make()
      XCTAssertThrowsError(try wrongFPS.consume(expectedBinding: binding, resolvedFrames: 65, fps: 30))
      XCTAssertTrue(wrongFPS.isReleased)
    }
  }
  func testInvalidGridRejectsBeforeEncodingAndCancelledConsumeDropsContexts() async throws {
    try Device.withDefaultDevice(.cpu) {
      var encodes = 0
      XCTAssertThrowsError(try MLXAutomaticTextPreparation.prepare(binding: Self.binding(),
        policy: MLXAutomaticDurationPolicy(headCheckpointPath: "/tmp/head.safetensors", minimumSeconds:2.4, maximumSeconds:2.5), fps: 24,
        encode: { _, _ in encodes += 1; return Self.output(2) }, predict: { _ in 3 }, admitResolvedFrames: { _ in }))
      XCTAssertEqual(encodes,0)
    }
    let task = Task<Void, Error>.detached { @Sendable in
      try Device.withDefaultDevice(.cpu) {
        let binding = try MLXPreparedTextLeaseTests.binding()
        let lease = try MLXAutomaticTextPreparation.prepare(binding: binding,
          policy: MLXAutomaticDurationPolicy(headCheckpointPath: "/tmp/head.safetensors"), fps: 24,
          encode: { _, _ in MLXPreparedTextLeaseTests.output(2) }, predict: { _ in 3 }, admitResolvedFrames: { _ in })
        withUnsafeCurrentTask { $0?.cancel() }
        do { _ = try lease.consume(expectedBinding: binding, resolvedFrames: 65, fps: 24); XCTFail("Cancelled consume returned contexts") }
        catch is CancellationError { XCTAssertTrue(lease.isReleased); throw CancellationError() }
      }
    }
    do { try await task.value; XCTFail("Cancellation was lost") }
    catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
  }
}
