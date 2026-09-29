import Foundation
import XCTest
@testable import H3MLX

final class H3WorkerReceiptTests: XCTestCase {
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
