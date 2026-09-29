import XCTest
@testable import H3MLX

final class H3WorkerProgressTests: XCTestCase {
  func testFL2VAStageBoundariesAreRecorded() {
    XCTAssertTrue(H3WorkerStageBoundary.tracks(task: "fl2va"))
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "keyframe_video_weights_released",
      completed: 1, total: 1), "keyframeVideoEncode")
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "video_weights_released",
      completed: 1, total: 1), "videoDecode")
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "audio_weights_released",
      completed: 1, total: 1), "audioDecode")
    XCTAssertFalse(H3WorkerStageBoundary.tracks(task: "t2va"))
  }

  func testRef2VAStageBoundariesSeparatePreparationFromSampling() {
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "text_weights_released",
      completed: 1, total: 1), "qwen")
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "transformer_prepare",
      completed: 49, total: 50), nil)
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "transformer_prepare",
      completed: 50, total: 50), "transformerPreparation")
    XCTAssertEqual(H3WorkerStageBoundary.name(stage: "transformer_weights_released",
      completed: 1, total: 1), "sampling")
  }

  func testTransformerPreparationAdvancesImmediatelyAfterReferenceEncoding() throws {
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "transformer_prepare",
      completed: 1, total: 50, evaluations: 4)), 0.0804, accuracy: 0.000001)
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "transformer_prepare",
      completed: 50, total: 50, evaluations: 4)), 0.1, accuracy: 0.000001)
  }

  func testSamplingBlocksAdvanceAcrossEvaluations() throws {
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "sampling_block_1",
      completed: 25, total: 50, evaluations: 2)), 0.2825, accuracy: 0.000001)
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "sampling",
      completed: 1, total: 2, evaluations: 2)), 0.465, accuracy: 0.000001)
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "sampling_block_2",
      completed: 25, total: 50, evaluations: 2)), 0.6475, accuracy: 0.000001)
    XCTAssertEqual(try XCTUnwrap(H3WorkerProgress.fraction(stage: "sampling",
      completed: 2, total: 2, evaluations: 2)), 0.83, accuracy: 0.000001)
  }

  func testMalformedSamplingCountsCannotAdvanceProgress() {
    XCTAssertNil(H3WorkerProgress.fraction(stage: "sampling_block_3",
      completed: 1, total: 50, evaluations: 2))
    XCTAssertNil(H3WorkerProgress.fraction(stage: "sampling_block_1",
      completed: 51, total: 50, evaluations: 2))
  }
}
