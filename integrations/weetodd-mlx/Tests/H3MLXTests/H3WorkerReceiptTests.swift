import Foundation
import Darwin
import XCTest
@testable import H3MLX

final class H3WorkerReceiptTests: XCTestCase {
  func testContinuationDecodeAndInterleavedPreviewsUseOnePublishedFrameClock() {
    var last = 0.83
    // Raw 90 frames contain 22 context frames and publish 68 frames. Preview
    // callbacks can arrive before a lagging chunk-level decode callback.
    let events = [0, 22, 23, 22, 34, 30, 51, 45, 90]
    let expectedCounts = [0, 0, 1, 0, 12, 8, 29, 23, 68]
    let expectedHighWaterCounts = [0, 0, 1, 1, 12, 12, 29, 29, 68]
    for (index, rawCompleted) in events.enumerated() {
      let progress = H3WorkerReceipt.progress(stage: "video_decode",
        completed: rawCompleted, total: 90, evaluations: 4,
        publishedFrames: 68, overlapFrames: 22, previousFraction: last)
      XCTAssertEqual(progress.completed, expectedCounts[index])
      XCTAssertEqual(progress.total, 68)
      XCTAssertEqual(progress.fraction,
        0.84 + 0.13 * Double(expectedHighWaterCounts[index]) / 68, accuracy: 1e-12)
      XCTAssertGreaterThanOrEqual(progress.fraction, last)
      last = progress.fraction
    }
    XCTAssertEqual(last, 0.97, accuracy: 1e-12)
  }

  func testOrdinaryDecodePreservesFrameClockAndFiniteStageProgress() {
    var last = 0.0
    let stages: [(String, Int, Int, Double)] = [
      ("control_video_encode", 1, 2, 0.015), ("text", 25, 50, 0.04),
      ("transformer_prepare", 50, 50, 0.1), ("sampling_block_1", 25, 50, 0.19125),
      ("sampling", 4, 4, 0.83), ("video_decode", 34, 90, 0.84 + 0.13 * 34 / 90),
      ("video_decode", 90, 90, 0.97), ("audio_decode", 1, 2, 0.98)]
    for (stage, completed, total, expected) in stages {
      let progress = H3WorkerReceipt.progress(stage: stage, completed: completed,
        total: total, evaluations: 4, publishedFrames: 90, overlapFrames: 0,
        previousFraction: last)
      XCTAssertEqual(progress.completed, completed)
      XCTAssertEqual(progress.total, total)
      XCTAssertTrue(progress.fraction.isFinite)
      XCTAssertEqual(progress.fraction, expected, accuracy: 1e-12)
      last = progress.fraction
    }
    for previous in [Double.nan, .infinity, -.infinity, -1, 2] {
      for stage in ["text", "video_decode", "audio_decode", "sampling", "unknown"] {
        let progress = H3WorkerReceipt.progress(stage: stage, completed: Int.min,
          total: 0, evaluations: 0, publishedFrames: 0, overlapFrames: 22,
          previousFraction: previous)
        XCTAssertTrue(progress.fraction.isFinite)
        XCTAssertGreaterThanOrEqual(progress.fraction, 0)
        XCTAssertLessThanOrEqual(progress.fraction, 0.995)
      }
    }
  }

  func testInstalledWorkerPolicyIsRenderOnlyAndReportsActualPriorityBeforeInference() throws {
    let env = ProcessInfo.processInfo.environment
    guard let worker = env["WEETODD_H3_EXECUTION_POLICY_WORKER"],
      let request = env["WEETODD_H3_EXECUTION_POLICY_REQUEST"] else {
      throw XCTSkip("Set a packaged worker and valid header-only request for installed execution-policy qualification.")
    }
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-h3-policy-\(UUID().uuidString)",isDirectory:true)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:directory) }
    func run(_ operation: String, output: URL) throws -> (Int32,[[String:Any]]) {
      // Rebind only the private output identity. Preserve the supplied recipe,
      // digest, job, engine and FFmpeg fields and never modify the source request.
      let supplied = try Data(contentsOf:URL(fileURLWithPath:request))
      guard supplied.count <= 65536,
        var envelope = try JSONSerialization.jsonObject(with:supplied) as? [String:Any] else {
        throw H3CheckpointError.invalid("Invalid installed execution-policy envelope.")
      }
      envelope["outputDirectory"] = output.path
      let privateRequest = directory.appendingPathComponent(operation + "-request.json")
      try JSONSerialization.data(withJSONObject:envelope,options:[.sortedKeys])
        .write(to:privateRequest,options:.withoutOverwriting)
      let log = directory.appendingPathComponent(operation + ".jsonl")
      XCTAssertTrue(FileManager.default.createFile(atPath:log.path,contents:Data()))
      let handle = try FileHandle(forWritingTo:log)
      defer { try? handle.close() }
      let process = Process()
      process.executableURL = URL(fileURLWithPath:worker)
      process.arguments = [operation,"--request",privateRequest.path,"--output",output.path]
      var environment = env
      environment["WEETODD_PYTHON"] = "/nonexistent"
      process.environment = environment
      process.standardOutput = handle; process.standardError = handle
      let finished = DispatchSemaphore(value:0)
      process.terminationHandler = { _ in finished.signal() }
      try process.run()
      guard finished.wait(timeout:.now()+60) == .success else {
        process.terminate()
        if finished.wait(timeout:.now()+5) != .success {
          _ = Darwin.kill(process.processIdentifier,SIGKILL)
          _ = finished.wait(timeout:.now()+5)
        }
        throw H3CheckpointError.invalid("Installed execution-policy worker exceeded the bounded CPU-only deadline.")
      }
      let lines = String(decoding:try Data(contentsOf:log),as:UTF8.self).split(separator:"\n")
      let events = lines.compactMap { line -> [String:Any]? in
        try? JSONSerialization.jsonObject(with:Data(line.utf8)) as? [String:Any]
      }
      return (process.terminationStatus,events)
    }
    let preflight = try run("preflight",output:directory.appendingPathComponent("unused-preflight-output"))
    XCTAssertEqual(preflight.0,0,"The supplied fixture must pass actual header-only preflight.")
    XCTAssertTrue(preflight.1.contains { $0["status"] as? String == "success" })
    XCTAssertFalse(preflight.1.contains { $0["stage"] as? String == "execution_policy" })
    // This new private directory already exists, so rendering must reject it
    // after policy acquisition and before the inference lease/weighted work.
    let render = try run("render",output:directory)
    XCTAssertEqual(render.0,1)
    let events = render.1.filter { $0["stage"] as? String == "execution_policy" }
    XCTAssertEqual(events.count,1)
    let policy = try XCTUnwrap(events.first)
    XCTAssertEqual(policy["event"] as? String,"progress")
    XCTAssertEqual(policy["executionPolicy"] as? String,"user_initiated_allowing_idle_system_sleep")
    XCTAssertEqual(policy["taskPriority"] as? Int,Int(TaskPriority.userInitiated.rawValue))
    XCTAssertEqual(policy["idleSystemSleepAllowed"] as? Bool,true)
    XCTAssertEqual(policy["latencyCritical"] as? Bool,false)
    XCTAssertTrue(render.1.contains { $0["status"] as? String == "error"
      && ($0["error"] as? String)?.contains("destination must be new") == true })
    XCTAssertFalse(render.1.contains { $0["status"] as? String == "success" })
    XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("render.mp4").path))
  }

  func testBackendReceiptSeparatesMPPOutputsFromFirstUseAndKnownFallback() throws {
    let report = H3BackendReport(preparedWindowSize: 1, eligibleProjectionCalls: 12,
      mppProjectionCalls: 8, knownFallbackProjectionCalls: 2,
      firstUseReferenceProjectionCalls: 2, verifiedMPPSignatures: 1,
      rejectedMPPSignatures: 1, vsaOriginalRowsIndexedCalls: 50)
    let metadata = report.metadata
    XCTAssertEqual(metadata["projectionBackend"] as? String, "mlx_with_verified_mpp")
    XCTAssertEqual(report.eligibleProjectionCalls, report.mppProjectionCalls
      + report.knownFallbackProjectionCalls + report.firstUseReferenceProjectionCalls)
    let result = H3WorkerReceipt.renderResult(video: URL(fileURLWithPath: "/tmp/take/render.mp4"),
      metadata: ["transformerExecution": metadata], jobID: UUID())
    let saved = try XCTUnwrap((result["metadata"] as? [String: Any])?["transformerExecution"] as? [String: Any])
    XCTAssertEqual(saved["mppEligibleProjectionCalls"] as? Int, 12)
    XCTAssertEqual(saved["mppProjectionCalls"] as? Int, 8)
    XCTAssertEqual(saved["mppKnownFallbackProjectionCalls"] as? Int, 2)
    XCTAssertEqual(saved["mppFirstUseReferenceProjectionCalls"] as? Int, 2)
    XCTAssertEqual(saved["vsaOriginalRowsIndexedCalls"] as? Int, 50)
    XCTAssertEqual(saved["vsaGroupedSparseCalls"] as? Int, 0)
    XCTAssertEqual(result["productionQualified"] as? Bool, false)
    let policy = try XCTUnwrap(saved["mppVerificationPolicy"] as? String)
    XCTAssertTrue(policy.contains("first-use"))
    XCTAssertTrue(policy.contains("not a universal proof"))
    XCTAssertTrue(policy.contains("full-media qualification is separate"))
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: result))
  }

  func testFirstUseVerificationDoesNotClaimAnMPPOutput() {
    let report = H3BackendReport(preparedWindowSize: 2, eligibleProjectionCalls: 1,
      firstUseReferenceProjectionCalls: 1, verifiedMPPSignatures: 1,
      vsaGroupedSparseCalls: 3, vsaDenseCalls: 1)
    XCTAssertEqual(report.metadata["projectionBackend"] as? String, "mlx")
    XCTAssertEqual(report.metadata["preparedWindowSize"] as? Int, 2)
    XCTAssertEqual(report.metadata["vsaGroupedSparseCalls"] as? Int, 3)
    XCTAssertEqual(report.metadata["vsaDenseCalls"] as? Int, 1)
  }

  func testBackendSnapshotIsScalarAndUnpreparedDefaultIsExplicit() throws {
    var active = H3BackendReport(preparedWindowSize: 1, mppProjectionCalls: 4,
      vsaOriginalRowsIndexedCalls: 2)
    let captured = active
    active = H3BackendReport()
    XCTAssertEqual(captured.mppProjectionCalls, 4)
    XCTAssertEqual(captured.vsaOriginalRowsIndexedCalls, 2)
    XCTAssertEqual(active.metadata["preparedWindowPolicy"] as? String, "unprepared")
    XCTAssertEqual(active.metadata["preparedWindowSize"] as? Int, 0)
    XCTAssertEqual(active.metadata["projectionBackend"] as? String, "mlx")
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: active.metadata))
  }

  func testContinuationReceiptCarriesCollectableNativeContextAndExactUsableWindow() {
    let result=H3WorkerReceipt.renderResult(video:URL(fileURLWithPath:"/tmp/take/render.mp4"),
      metadata:["frames":90,"fps":24,"continuationManifest":"/tmp/take/continuation/manifest.json",
        "continuationManifestSHA256":String(repeating:"a",count:64),
        "continuationPayloadSHA256":String(repeating:"b",count:64)],jobID:UUID())
    let artifact=result["continuation_artifact"] as? [String:String]
    XCTAssertEqual(artifact?["payload_filename"],"latents.f32")
    XCTAssertEqual(artifact?["manifest_sha256"],String(repeating:"a",count:64))
    XCTAssertEqual(result["usable_source_in"] as? Double,0)
    XCTAssertEqual(result["usable_duration"] as? Double,3.75)
    XCTAssertEqual(result["use_complete_duration"] as? Bool,true)
  }
  func testMotionReceiptPublishesRecoveredSourceInsteadOfExpandedSamplingClock() {
    let result=H3WorkerReceipt.renderResult(video:URL(fileURLWithPath:"/tmp/motion/render.mp4"),
      metadata:["task":"motion_fidelity","frames":60,"sampledFrames":124,"fps":24],jobID:UUID())
    XCTAssertEqual(result["usable_duration"] as? Double,2.5)
    XCTAssertEqual(result["usable_source_in"] as? Double,0)
    XCTAssertEqual(result["use_complete_duration"] as? Bool,true)
    XCTAssertNil(result["continuation_artifact"])
  }
  func testRenderCompletionPreservesRequestIdentity() {
    let jobID = UUID(uuidString: "00000000-0000-0000-0000-000000000123")!
    let video = URL(fileURLWithPath: "/tmp/take/render.mp4")
    let result = H3WorkerReceipt.renderResult(video: video,
      metadata: ["frames": 124], jobID: jobID)
    XCTAssertEqual(result["jobID"] as? String, jobID.uuidString)
    XCTAssertEqual(result["video"] as? String, video.path)
    XCTAssertEqual(result["nativeRuntime"] as? String, "swift-mlx")
    XCTAssertEqual(result["productionQualified"] as? Bool, false)
  }
}
