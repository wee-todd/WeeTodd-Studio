import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3FlexSamplingTests: XCTestCase {
  func testReleasedKnotsProduceSixDistinctVideoAndAudioEvaluations() throws {
    let method = try XCTUnwrap(H3SamplingMethod(rawValue: "pika_flex"))
    let video = try H3Schedule(requestedSteps: 7, shift: 12, samplingMethod: method)
    let audio = try H3Schedule(requestedSteps: 7, shift: 3, samplingMethod: method)
    let expectedVideo: [Float] = [1, 0.972972973, 0.923076923, 0.8, 0.571428571, 0.333333333, 0]
    let expectedAudio: [Float] = [1, 0.9, 0.75, 0.5, 0.25, 0.111111111, 0]
    XCTAssertEqual(video.timesteps.count, 6)
    for i in 0..<7 {
      XCTAssertEqual(video.sigmas[i], expectedVideo[i], accuracy: 0.0000002)
      XCTAssertEqual(audio.sigmas[i], expectedAudio[i], accuracy: 0.0000002)
    }
    XCTAssertThrowsError(try H3Schedule(requestedSteps: 6, shift: 12, samplingMethod: method))
    XCTAssertThrowsError(try H3Schedule(requestedSteps: 8, shift: 12, samplingMethod: method))
  }

  func testFlexUsesCleanPredictionAndFreshNoiseInsteadOfEulerHistory() throws {
    try Device.withDefaultDevice(.cpu) {
      let method = try XCTUnwrap(H3SamplingMethod(rawValue: "pika_flex"))
      let schedule = try H3Schedule(sigmas: [0.5, 0.25, 0])
      var stepper = H3SamplingStepper(schedule: schedule, method: method)
      // x0 = 2 + .5*3 = 3.5; next = .75*3.5 + .25*(-2) = 2.125.
      let value = try stepper.advance(sample: MLXArray([Float(2)]),
        velocity: MLXArray([Float(3)]), index: 0, freshNoise: MLXArray([Float(-2)]))
      XCTAssertEqual(value.item(Float.self), 2.125, accuracy: 0.000001)
      let final = try stepper.advance(sample: value, velocity: MLXArray([Float(4)]), index: 1)
      XCTAssertEqual(final.item(Float.self), 3.125, accuracy: 0.000001)
    }
  }

  func testTerminalPredictionDrawsNoNoiseAndInvalidNoiseDoesNotAdvanceState() throws {
    try Device.withDefaultDevice(.cpu) {
      let method = try XCTUnwrap(H3SamplingMethod(rawValue: "pika_flex"))
      let schedule = try H3Schedule(sigmas: [0.5, 0.25, 0])
      var stepper = H3SamplingStepper(schedule: schedule, method: method)
      XCTAssertThrowsError(try stepper.advance(sample: MLXArray([Float(2)]),
        velocity: MLXArray([Float(3)]), index: 0, freshNoise: MLXArray([Float(1), 2])))
      let value = try stepper.advance(sample: MLXArray([Float(2)]),
        velocity: MLXArray([Float(3)]), index: 0, freshNoise: MLXArray([Float(-2)]))
      MLXRandom.seed(123)
      let expectedNext = MLXRandom.normal([3]).asArray(Float.self)
      MLXRandom.seed(123)
      _ = try stepper.advance(sample: value, velocity: MLXArray([Float(4)]), index: 1)
      XCTAssertEqual(MLXRandom.normal([3]).asArray(Float.self).map(\.bitPattern), expectedNext.map(\.bitPattern))
    }
  }
}
