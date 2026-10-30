import XCTest
@testable import H3MLX

final class H3RegenerateWorkerPreparationTests:XCTestCase {
  func testSilenceRowsUseActualEncoderSamplePaddingWithoutChangingOrdinaryClock() throws {
    for frames in [5,22,39,56,124,362] {
      let samples=Int((Double(frames)/24*32000).rounded(.toNearestOrEven))
      XCTAssertEqual(try H3RegenerateAdmission.silenceSampleCount(paddedFrames:frames),samples)
      let geometry=try H3Geometry(regenerateWidth:32,height:32,paddedFrames:frames)
      XCTAssertEqual(geometry.audioLatentFrames,(samples+799)/800)
      let admission=try H3RegenerateAdmission(width:32,height:32,visibleFrames:frames,paddedFrames:frames,fps:24,textRows:1,
        maximumStageBytes:32*1024*1024*1024,availableTemporaryBytes:32*1024*1024*1024)
      XCTAssertEqual(admission.audioRows,2*geometry.audioLatentFrames)
    }
    XCTAssertThrowsError(try H3RegenerateAdmission.silenceSampleCount(paddedFrames:0))
    XCTAssertEqual(try H3Geometry(width:32,height:32,durationSeconds:4.4).audioLatentFrames,178)
  }
  func testReleasedDecoderRegenerateTilePlanDoesNotExpandOrdinaryAdmission() throws {
    let full=try H3Geometry(regenerateWidth:2560,height:1440,paddedFrames:124)
    XCTAssertEqual(try H3VideoVAEDecoder.preflightRegenerate(geometry:full),104)
    XCTAssertThrowsError(try H3VideoVAEDecoder.preflightSpatial(geometry:full))
    let fourK=try H3Geometry(regenerateWidth:3840,height:2176,paddedFrames:124)
    XCTAssertEqual(try H3VideoVAEDecoder.preflightRegenerate(geometry:fourK),220)
    XCTAssertThrowsError(try H3VideoVAEDecoder.preflightRegenerate(geometry:H3Geometry(regenerateWidth:32,height:32,paddedFrames:5)))
  }
}
