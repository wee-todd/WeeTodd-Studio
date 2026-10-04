import Foundation
import XCTest
@testable import H3MLX

final class H3WorkerReceiptTests: XCTestCase {
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
