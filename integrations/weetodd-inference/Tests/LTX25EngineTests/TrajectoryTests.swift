import XCTest

@testable import LTX25Engine

final class TrajectoryTests: XCTestCase {
  func testDeterministicTrajectoryUsesEverySigmaAndBothVelocityStreams() throws {
    let sampler = try EulerTrajectory()
    let plan = try SamplingSchedule(sigmas: [1, 0.5, 0.25, 0], eta: 0)
    var visited: [Float] = []
    var completed: [Int] = []
    let result = try sampler.evaluate(
      video: [2, 4], audio: [-1], schedule: plan,
      predict: { state, sigma in
        visited.append(sigma)
        return AVLatents(video: [1, 2], audio: [-2])
      }, progress: { completed.append($0.completedSteps) })
    XCTAssertEqual(visited, [1, 0.5, 0.25])
    XCTAssertEqual(completed, [1, 2, 3])
    XCTAssertEqual(result.video, [1, 2])
    XCTAssertEqual(result.audio, [1])
  }

  func testAncestralNoiseIsOrderedAndTerminalDoesNotRequestNoise() throws {
    let sampler = try EulerTrajectory()
    let plan = try SamplingSchedule(sigmas: [1, 0.5, 0], eta: 1)
    var calls: [String] = []
    let result = try sampler.evaluate(
      video: [3], audio: [-3], schedule: plan,
      noise: { step, modality, count in
        calls.append("\(step)-\(modality.rawValue)-\(count)")
        return modality == .video ? [1] : [-1]
      }, predict: { state, _ in AVLatents(video: [0], audio: [0]) })
    XCTAssertEqual(calls, ["0-video-1", "0-audio-1"])
    XCTAssertEqual(result.video[0], 2.47140452, accuracy: 1e-6)
    XCTAssertEqual(result.audio[0], -2.47140452, accuracy: 1e-6)
  }

  func testRejectsWholeScheduleAndMissingNoiseBeforePrediction() throws {
    for sigmas in [[1.0], [1, 0.5], [1, 0, 0], [1, 0.6, 0.7, 0], [1, .nan, 0]] {
      XCTAssertThrowsError(try SamplingSchedule(sigmas: sigmas))
    }
    let sampler = try EulerTrajectory()
    let plan = try SamplingSchedule(sigmas: [1, 0.5, 0])
    XCTAssertThrowsError(
      try sampler.evaluate(
        video: [1], audio: [1], schedule: plan,
        predict: { _, _ in
          XCTFail("Missing noise reached inference")
          return AVLatents(video: [0], audio: [0])
        }))
  }

  func testPreviewFailureStopsTrajectoryAndReentrancyFailsSafely() throws {
    enum Failure: Error { case stop }
    let sampler = try EulerTrajectory()
    let plan = try SamplingSchedule(sigmas: [1, 0.5, 0], eta: 0)
    var visited = 0
    XCTAssertThrowsError(
      try sampler.evaluate(
        video: [2], audio: [2], schedule: plan,
        predict: { _, _ in
          visited += 1
          XCTAssertThrowsError(
            try sampler.evaluate(
              video: [2], audio: [2], schedule: plan,
              predict: { _, _ in
                XCTFail("Reentrant prediction executed")
                return AVLatents(video: [0], audio: [0])
              }))
          return AVLatents(video: [1], audio: [1])
        },
        preview: { state, event in
          XCTAssertEqual(event.completedSteps, 1)
          XCTAssertEqual(state.video, [1.5])
          throw Failure.stop
        }))
    XCTAssertEqual(visited, 1)
    let result = try sampler.evaluate(
      video: [2], audio: [2], schedule: plan, predict: { _, _ in AVLatents(video: [1], audio: [1]) }
    )
    XCTAssertEqual(result.video, [1])
  }

  func testRejectsTimestepThatUnderflowsTheModelBoundary() throws {
    XCTAssertThrowsError(try SamplingSchedule(sigmas: [1, 1e-50, 0], eta: 0))
  }

  func testCallbackFailureStopsBeforeNextEvaluationAndAllowsRetry() throws {
    enum Failure: Error { case stop }
    let sampler = try EulerTrajectory()
    let plan = try SamplingSchedule(sigmas: [1, 0.5, 0], eta: 0)
    var evaluations = 0
    XCTAssertThrowsError(
      try sampler.evaluate(
        video: [1], audio: [1], schedule: plan,
        predict: { _, _ in
          evaluations += 1
          return AVLatents(video: [0], audio: [0])
        }, progress: { _ in throw Failure.stop }))
    XCTAssertEqual(evaluations, 1)
    let output = try sampler.evaluate(
      video: [1], audio: [1], schedule: plan,
      predict: { _, _ in
        AVLatents(video: [0], audio: [0])
      })
    XCTAssertEqual(output.video, [1])
  }
}
