import XCTest
@testable import LTX25MLX

final class MLXMovieUpscalePlanTests: XCTestCase {
  func testEnhancePreservesFullHDCanvasAndThreeEvaluationDenoiseStrength() throws {
    let mode=try XCTUnwrap(MLXMovieUpscalePlan.Mode(rawValue:"enhance"))
    let plan=try MLXMovieUpscalePlan(mode:mode,width:1920,height:1088,frames:120,
      fps:24,sizePolicy:.strict,refinementStrength:0.5)
    XCTAssertEqual(plan.size.outputWidth,1920);XCTAssertEqual(plan.size.outputHeight,1088)
    XCTAssertEqual(plan.frames,120);XCTAssertEqual(plan.paddedFrames,121)
    XCTAssertEqual(plan.fps,24);XCTAssertEqual(plan.seconds,5)
    XCTAssertEqual(plan.sigmas.count,4)
    XCTAssertEqual(plan.sigmas[0],0.5,accuracy:1e-15)
    XCTAssertEqual(plan.sigmas[1],0.3986254295532646,accuracy:1e-15)
    XCTAssertEqual(plan.sigmas[2],0.23195876288659795,accuracy:1e-15)
    XCTAssertEqual(plan.sigmas[3],0)
    XCTAssertEqual(try plan.chunks().map(\.frames),[120])
    for policy in [MLXMovieUpscalePlan.SizePolicy.fitNearest,.centerCrop,.strict] {
      XCTAssertThrowsError(try MLXMovieUpscalePlan(mode:mode,width:1920,height:1080,frames:24,
        fps:24,sizePolicy:policy,refinementStrength:0.5))
    }
  }
  func testArbitraryVisibleFramesPadCausalTailWithoutMovingLastAnchorOrAddingSlots() throws {
    let plan = try MLXMovieUpscalePlan(mode: .pixelSpatial, width: 672, height: 384,
      frames: 98, fps: 24, sizePolicy: .strict)
    XCTAssertEqual(plan.paddedFrames, 105)
    XCTAssertEqual(plan.visibleLastFrame, 97)
    XCTAssertEqual(plan.size.outputWidth, 1344); XCTAssertEqual(plan.size.outputHeight, 768)
    XCTAssertEqual(plan.seconds, 98.0 / 24)
    XCTAssertEqual(try MLXMovieUpscalePlan.paddedFrameCount(1), 1)
    XCTAssertEqual(try MLXMovieUpscalePlan.paddedFrameCount(2), 9)
    XCTAssertThrowsError(try MLXMovieUpscalePlan.paddedFrameCount(Int.max))
  }
  func testExactThreeRefinementSigmasAndLatentOnlyBypass() throws {
    for mode in [MLXMovieUpscalePlan.Mode.refine, .pixelSpatial] {
      let p = try MLXMovieUpscalePlan(mode: mode, width: 64, height: 32, frames: 9,
        fps: 24, sizePolicy: .strict, refinementStrength: 0.35)
      XCTAssertEqual(p.sigmas.count, 4)
      for (actual, expected) in zip(p.sigmas, [0.35, 0.725 * 0.35 / 0.909375, 0.421875 * 0.35 / 0.909375, 0]) {
        XCTAssertEqual(actual, expected, accuracy: 1e-15)
      }
    }
    XCTAssertTrue(try MLXMovieUpscalePlan(mode: .latentOnly, width: 64, height: 32,
      frames: 9, fps: 24, sizePolicy: .strict).sigmas.isEmpty)
  }
  func testGridPolicyPreservesAspectOrRecordsExactCenteredCrop() throws {
    let fit = try MLXMovieUpscalePlan.prepareSize(width: 1280, height: 720, policy: .fitNearest)
    XCTAssertEqual(fit.width, 1312); XCTAssertEqual(fit.height, 736); XCTAssertTrue(fit.resized)
    XCTAssertLessThan(abs((Double(fit.width) / Double(fit.height)) / (1280.0 / 720) - 1), 0.005)
    let crop = try MLXMovieUpscalePlan.prepareSize(width: 673, height: 385, policy: .centerCrop)
    XCTAssertEqual(crop.width, 672); XCTAssertEqual(crop.height, 384)
    XCTAssertEqual(crop.cropLeft, 0); XCTAssertEqual(crop.cropRight, 1)
    XCTAssertEqual(crop.cropTop, 0); XCTAssertEqual(crop.cropBottom, 1); XCTAssertFalse(crop.resized)
    XCTAssertThrowsError(try MLXMovieUpscalePlan.prepareSize(width: 673, height: 385, policy: .strict))
  }
  func testChunkPlanExactCoverageQualityFloorAndHardCutSelection() throws {
    let plan = try MLXMovieUpscalePlan(mode: .pixelSpatial, width: 672, height: 384,
      frames: 98, fps: 24, sizePolicy: .strict)
    let chunks = try plan.chunks(frameMegapixelBudget: 50.6, cutFrames: [49])
    XCTAssertEqual(chunks.map(\.startFrame), [0, 49]); XCTAssertEqual(chunks.map(\.endFrame), [49, 98])
    XCTAssertEqual(chunks.map(\.paddedFrames), [49, 49]); XCTAssertEqual(chunks[0].reason, "scene cut")
    XCTAssertEqual(chunks[1].visibleLastFrame, 48)
    XCTAssertThrowsError(try plan.chunks(frameMegapixelBudget: 20))
    let short = try MLXMovieUpscalePlan(mode: .refine, width: 64, height: 32,
      frames: 2, fps: 24, sizePolicy: .strict)
    XCTAssertEqual(try short.chunks().count, 1)
    XCTAssertEqual(try MLXMovieUpscalePlan.sceneCuts(adjacentLuminanceDifferences: [0.01, 0.012, 0.8, 0.009, 0.01]), [3])
    XCTAssertThrowsError(try MLXMovieUpscalePlan.sceneCuts(adjacentLuminanceDifferences: [.nan]))
  }
}
