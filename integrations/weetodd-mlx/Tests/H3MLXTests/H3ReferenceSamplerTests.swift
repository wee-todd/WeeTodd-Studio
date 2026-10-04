import MLX
import XCTest
@testable import H3MLX

final class H3ReferenceSamplerTests: XCTestCase {
  private final class UnitVelocity: H3ReferenceVelocityPredictor {
    let layout: H3ReferenceLayout
    var calls = 0
    init(layout: H3ReferenceLayout) { self.layout = layout }
    func predict(videoLatents: MLXArray, audioLatents: MLXArray,
      timestepIndices: [Int32], progress: (Int, Int) -> Void) throws
      -> H3FinalLayer.Output {
      calls += 1
      return H3FinalLayer.Output(video: MLXArray.ones(videoLatents.shape),
        audio: MLXArray.ones(audioLatents.shape))
    }
  }

  func testSamplingUpdatesOnlyTargetsAndKeepsReferenceAudioAndVideoFixed() throws {
    try Device.withDefaultDevice(.cpu) {
      for method in [H3SamplingMethod.euler, .resMultistep] {
      let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
      let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1], references: [
        .image(latentHeight: 2, latentWidth: 2), .audio(latents: 2),
      ])
      let videoSchedule = try H3Schedule(requestedSteps: 3, shift: 12)
      let audioSchedule = try H3Schedule(requestedSteps: 3, shift: 3)
      let rows = try H3ReferenceRowSchedule(layout: layout,
        video: videoSchedule, audio: audioSchedule)
      let referenceVideo = MLXArray([Float](repeating: 7, count: 96), [1, 1, 96])
      let referenceAudio = MLXArray([Float](repeating: 9, count: 4 * 32), [1, 4, 32])
      let initialVideo = concatenated([referenceVideo,
        MLXArray.zeros([1, geometry.videoRows, 96])], axis: 1)
      let initialAudio = concatenated([referenceAudio,
        MLXArray.zeros([1, geometry.audioRows, 32])], axis: 1)
      let predictor = UnitVelocity(layout: layout)
      let output = try H3ReferenceSampler.run(predictor: predictor,
        videoSchedule: videoSchedule, audioSchedule: audioSchedule,
        rowSchedule: rows, videoLatents: initialVideo, audioLatents: initialAudio, samplingMethod: method)
      XCTAssertEqual(predictor.calls, 2)
      XCTAssertEqual(output.video[0, 0, 0].item(Float.self), 7)
      XCTAssertEqual(output.audio[0, 0, 0].item(Float.self), 9)
      XCTAssertEqual(output.video[0, 1, 0].item(Float.self), 1, accuracy: 0.000001)
      XCTAssertEqual(output.audio[0, 4, 0].item(Float.self), 1, accuracy: 0.000001)
    }
    }
  }
}
