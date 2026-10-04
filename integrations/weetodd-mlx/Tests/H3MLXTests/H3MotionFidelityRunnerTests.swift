import XCTest
import MLX
@testable import H3MLX

final class H3MotionFidelityRunnerTests: XCTestCase {
  func testActualAnalysisReductionPreservesTemporalPhaseAndNormalization() throws {
    try Device.withDefaultDevice(.cpu) {
      var pixels = Array(repeating: Float(2), count: 22 * 2 * 2 * 24)
      for index in (10 * 96)..<(11 * 96) { pixels[index] = 22 }
      let jerk = try H3MotionFidelityRunner.temporalJerk(
        MLXArray(pixels, [1,22,2,2,24]), mean: Array(repeating: 2, count: 24),
        standardDeviation: Array(repeating: 2, count: 24))
      XCTAssertEqual(jerk, [0,0,0,0,0,0,0,10,30,30,10,0,0,0,0,0,0,0,0])
    }
  }
  func testExpandedPixelIndicesRecoverOriginalBytesAndPadOnlyLastFrame() throws {
    let source: [UInt8] = [1,2,3,4,5,6,7,8,9]
    let expanded = try H3MotionFidelityRunner.expandedPixels(source, width: 1, height: 1,
      indices: [0,0,1,2,2,2])
    XCTAssertEqual(expanded, [1,2,3,1,2,3,4,5,6,7,8,9,7,8,9,7,8,9])
    XCTAssertThrowsError(try H3MotionFidelityRunner.expandedPixels(source, width: 1, height: 1, indices: [3]))
  }
  func testOnlyKnownCodecEndPaddingCanBeDiscarded() throws {
    try Device.withDefaultDevice(.cpu) {
      let latent = MLXArray(Array(repeating: Float(1), count: 2 * 122 * 32), [2,122,32])
      let bounded = try H3MotionFidelityRunner.cropAudioPadding(latent, samples: 97333, expectedFrames: 121)
      XCTAssertEqual(bounded.shape, [2,121,32]); XCTAssertEqual(bounded.sum().item(Float.self), Float(2 * 121 * 32))
      XCTAssertThrowsError(try H3MotionFidelityRunner.cropAudioPadding(latent, samples: 96000, expectedFrames: 120))
      XCTAssertThrowsError(try H3MotionFidelityRunner.cropAudioPadding(latent, samples: 96800, expectedFrames: 121))
    }
  }
}
