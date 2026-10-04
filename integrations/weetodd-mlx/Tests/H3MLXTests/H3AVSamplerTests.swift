import MLX
import XCTest
@testable import H3MLX

final class H3AVSamplerTests: XCTestCase {
  private final class UnitVelocity: H3VelocityPredictor {
    let layout: H3PackedLayout
    var calls = 0
    init(layout: H3PackedLayout) { self.layout = layout }
    func predict(videoLatents: MLXArray, audioLatents: MLXArray,
      timestepIndices: [Int32], progress: (Int, Int) -> Void) throws
      -> H3FinalLayer.Output {
      calls += 1
      progress(1, 2)
      progress(2, 2)
      return H3FinalLayer.Output(
        video: MLXArray.ones(videoLatents.shape),
        audio: MLXArray.ones(audioLatents.shape))
    }
  }

  func testJointSchedulesKeepReferenceRowsFixed() throws {
    try Device.withDefaultDevice(.cpu) {
      for method in [H3SamplingMethod.euler, .resMultistep] {
      let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
      let layout = try H3PackedLayout(geometry: geometry,
        textTags: [1, 1], anchors: [.first])
      let video = try H3Schedule(requestedSteps: 3, shift: 12)
      let audio = try H3Schedule(requestedSteps: 3, shift: 3)
      let rows = try H3RowSchedule(layout: layout, video: video, audio: audio)
      let videoCount = layout.conditionVideoRows + geometry.videoRows
      let condition = MLXArray([Float](repeating: 5,
        count: layout.conditionVideoRows * 96),
        [1, layout.conditionVideoRows, 96])
      let targets = MLXArray.zeros([1, geometry.videoRows, 96])
      let initialVideo = concatenated([condition, targets], axis: 1)
      let initialAudio = MLXArray.zeros([1, geometry.audioRows, 32])
      let predictor = UnitVelocity(layout: layout)
      var progress: [(Int, Int)] = []
      var blocks: [(Int, Int, Int)] = []
      let result = try H3AVSampler.run(predictor: predictor,
        videoSchedule: video, audioSchedule: audio,
        rowSchedule: rows, videoLatents: initialVideo,
        audioLatents: initialAudio, samplingMethod: method) { done, total in
        progress.append((done, total))
      } blockProgress: { step, done, total in
        blocks.append((step, done, total))
      }
      XCTAssertEqual(predictor.calls, 2)
      XCTAssertEqual(progress.map(\.0), [1, 2])
      XCTAssertEqual(blocks.map(\.0), [1, 1, 2, 2])
      XCTAssertEqual(blocks.map(\.1), [1, 2, 1, 2])
      XCTAssertEqual(result.video.shape, [1, videoCount, 96])
      XCTAssertEqual(result.audio.shape, [1, geometry.audioRows, 32])
      XCTAssertEqual(result.video[0, 0, 0].item(Float.self), 5)
      XCTAssertEqual(result.video[0, layout.conditionVideoRows, 0].item(Float.self), 1,
        accuracy: 0.000001)
      XCTAssertEqual(result.audio[0, 0, 0].item(Float.self), 1,
        accuracy: 0.000001)
    }
    }
  }
}
