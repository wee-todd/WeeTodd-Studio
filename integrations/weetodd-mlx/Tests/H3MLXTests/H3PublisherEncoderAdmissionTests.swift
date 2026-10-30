import XCTest
@testable import H3MLX

final class H3PublisherEncoderAdmissionTests:XCTestCase {
  func testMeasuredFullFrameSourceRejects32And64GiBBeforeWeights() throws {
    let bytes=try H3PublisherEncoderAdmission.requiredPeakBytes(width:1280,height:736,paddedFrames:124)
    XCTAssertEqual(bytes,100_545_855_488)
    for limit:UInt64 in [32,64] {
      XCTAssertThrowsError(try H3PublisherEncoderAdmission.validate(width:1280,height:736,paddedFrames:124,maximumStageBytes:limit<<30,physicalMemory:256<<30))
    }
    XCTAssertNoThrow(try H3PublisherEncoderAdmission.validate(width:1280,height:736,paddedFrames:124,maximumStageBytes:96<<30,physicalMemory:256<<30))
    XCTAssertThrowsError(try H3PublisherEncoderAdmission.validate(width:1280,height:736,paddedFrames:124,maximumStageBytes:96<<30,physicalMemory:64<<30))
  }
  func testFixed17FrameEncoderWindowMakesBudgetIndependentOfClipCount() throws {
    XCTAssertEqual(try H3PublisherEncoderAdmission.requiredPeakBytes(width:1280,height:736,paddedFrames:22),
      try H3PublisherEncoderAdmission.requiredPeakBytes(width:1280,height:736,paddedFrames:124))
    XCTAssertLessThan(try H3PublisherEncoderAdmission.requiredPeakBytes(width:32,height:32,paddedFrames:22),3<<30)
    XCTAssertThrowsError(try H3PublisherEncoderAdmission.requiredPeakBytes(width:1280,height:720,paddedFrames:124))
    XCTAssertThrowsError(try H3PublisherEncoderAdmission.requiredPeakBytes(width:1280,height:736,paddedFrames:120))
  }
}
