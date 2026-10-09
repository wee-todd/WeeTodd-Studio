import Foundation
import XCTest
@testable import H3MLX

final class H3VideoPrecisionPolicyTests: XCTestCase {
  func testDefaultAndExplicitProductAdmissionNeverChangesMemoryMode() throws {
    for mode: H3VideoDecodeMemoryMode? in [nil,.normal,.lowMemoryBF16] {
      XCTAssertNoThrow(try H3VideoDecodePrecision.float32.validate(memoryMode:mode))
    }
    XCTAssertNoThrow(try H3VideoDecodePrecision.float16.validate(memoryMode:.lowMemoryBF16))
    for mode: H3VideoDecodeMemoryMode? in [nil,.normal] {
      XCTAssertThrowsError(try H3VideoDecodePrecision.float16.validate(memoryMode:mode))
      XCTAssertThrowsError(try H3AVOutputDecoder.decode(videoRows:[],audioRows:[],
        geometry:H3Geometry(width:32,height:32,durationSeconds:2.5),
        videoVAE:URL(fileURLWithPath:"/nonexistent/video"),audioVAE:URL(fileURLWithPath:"/nonexistent/audio"),
        videoDecodeMemoryMode:mode,videoDecodePrecision:.float16,onFrame:{ _,_ in },onAudio:{ _,_ in })) {
        XCTAssertTrue(String(describing:$0).contains("low_memory_bf16"))
      }
    }
  }
  func testMotionNoopReportsRequestedPrecisionWithoutClaimingDecoderExecution() {
    let noop = H3T2VARunner.Result(videoFrames:0,audioSamplesPerChannel:1,audioSampleRate:32_000)
    XCTAssertNil(noop.videoDecodePrecision)
    let report = H3VideoDecodePrecision.executionDiagnostics(requested:.float16,applied:noop.videoDecodePrecision)
    XCTAssertEqual(report["requestedComputePrecision"] as? String,"float16")
    XCTAssertEqual(report["computePrecision"] as? String,"not_applied")
    XCTAssertEqual(report["precisionApplied"] as? Bool,false)
    let applied = H3VideoDecodePrecision.executionDiagnostics(requested:.float16,applied:.float16)
    XCTAssertEqual(applied["computePrecision"] as? String,"float16")
    XCTAssertEqual(applied["precisionApplied"] as? Bool,true)
  }
}
