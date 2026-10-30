import XCTest
@testable import H3MLX

final class H3RegenerateAdmissionTests:XCTestCase {
  func testExact2K4KAndPortraitCountsAndCenteredCrops() throws {
    let budget:UInt64=192*1024*1024*1024
    let two=try H3RegenerateAdmission(width:2560,height:1440,visibleFrames:121,paddedFrames:124,fps:24,
      textRows:256,maximumStageBytes:budget,availableTemporaryBytes:budget)
    XCTAssertEqual(two.videoRows,133200)
    XCTAssertEqual(two.workingWidth,2560); XCTAssertEqual(two.cropY,0)
    let four=try H3RegenerateAdmission(width:3840,height:2160,visibleFrames:121,paddedFrames:124,fps:24,
      textRows:256,maximumStageBytes:budget,availableTemporaryBytes:budget)
    XCTAssertEqual(four.videoRows,301920);XCTAssertEqual(four.workingHeight,2176);XCTAssertEqual(four.cropY,8)
    let portrait=try H3RegenerateAdmission(width:1080,height:1920,visibleFrames:121,paddedFrames:124,fps:24,
      textRows:256,maximumStageBytes:budget,availableTemporaryBytes:budget)
    XCTAssertEqual(portrait.workingWidth,1088);XCTAssertEqual(portrait.cropX,4)
    XCTAssertEqual(portrait.packedRows,portrait.videoRows+portrait.audioRows+256)
    XCTAssertThrowsError(try two.validateExecution(qualificationProbe:false))
    XCTAssertNoThrow(try two.validateExecution(qualificationProbe:true))
  }
  func testCadenceOverflowBudgetsAndOrdinaryCapsRemainStrict() throws {
    for fps in [23.976,25,30,Double.nan] {
      XCTAssertThrowsError(try H3RegenerateAdmission(width:2560,height:1440,visibleFrames:121,paddedFrames:124,fps:fps,
        textRows:256,maximumStageBytes:192*1024*1024*1024,availableTemporaryBytes:192*1024*1024*1024))
    }
    let invalid:[(Int,Int,UInt64,UInt64)] = [
      (Int.max,1440,192*1024*1024*1024,192*1024*1024*1024),(2560,1440,1,192*1024*1024*1024),
      (2560,1440,192*1024*1024*1024,1)]
    for (width,height,stage,disk) in invalid {
      XCTAssertThrowsError(try H3RegenerateAdmission(width:width,height:height,visibleFrames:121,paddedFrames:124,fps:24,
        textRows:256,maximumStageBytes:stage,availableTemporaryBytes:disk))
    }
    XCTAssertEqual(H3CanvasAdmission.ordinary.maximumPackedRows,40000)
    XCTAssertEqual(H3CanvasAdmission.spatialRefinement.maximumPackedRows,64000)
    XCTAssertThrowsError(try H3CanvasAdmission.ordinary.validatePackedRows(40001))
    XCTAssertThrowsError(try H3CanvasAdmission.spatialRefinement.validatePackedRows(64001))
  }
}
