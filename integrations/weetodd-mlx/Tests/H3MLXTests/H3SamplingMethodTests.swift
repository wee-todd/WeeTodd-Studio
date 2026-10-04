import MLX
import XCTest
@testable import H3MLX

final class H3SamplingMethodTests: XCTestCase {
  // Independent owned-Python CPU fixtures: scheduler.py's sequential float32
  // update, not an official MiniMax or whole-checkpoint numerical claim.
  func testNonconstantVelocitySequenceMatchesOwnedPythonBothModalityShifts() throws {
    try Device.withDefaultDevice(.cpu) {
      let velocities: [[Float]] = [[0.3, -0.1, 0.2], [-0.2, 0.5, -0.4],
        [0.7, -0.3, 0.1], [-0.1, 0.2, 0.9]]
      let fixtures: [(Float, [[Float]])] = [
        (12, [[0.4081081152, -0.2027027011, 1.5054054260],
          [0.3739513457, -0.1487416029, 1.4564338923],
          [0.6097599268, -0.3172954917, 1.5489921570],
          [0.5297599435, -0.1572954804, 2.2689921856]]),
        (3, [[0.4300000072, -0.2100000083, 1.5199999809],
          [0.3331371844, -0.0547646433, 1.3797646761],
          [0.7487342358, -0.3353988826, 1.5213823318],
          [0.6987342238, -0.2353988886, 1.9713823795]])]
      for (shift, expected) in fixtures {
        let schedule = try H3Schedule(requestedSteps: 5, shift: shift)
        var stepper = H3SamplingStepper(schedule: schedule, method: .resMultistep)
        var sample = MLXArray([Float(0.4), -0.2, 1.5])
        for index in 0..<4 {
          sample = try stepper.advance(sample: sample,
            velocity: MLXArray(velocities[index]), index: index)
          let actual = sample.asArray(Float.self)
          for column in 0..<3 {
            XCTAssertEqual(actual[column], expected[index][column], accuracy: 0.000002)
          }
        }
      }
    }
  }

  func testEulerDelegatesUnchangedAndNewRunHasIndependentHistory() throws {
    try Device.withDefaultDevice(.cpu) {
      let schedule = try H3Schedule(requestedSteps: 5, shift: 12)
      var stepper = H3SamplingStepper(schedule: schedule, method: .euler)
      var sample = MLXArray([Float(0.4), -0.2, 1.5])
      for index in 0..<4 {
        let velocity = MLXArray([Float(index) * 0.3, -0.1, 0.2])
        let expected = try schedule.advance(sample: sample, velocity: velocity, index: index)
        sample = try stepper.advance(sample: sample, velocity: velocity, index: index)
        XCTAssertEqual(sample.asArray(Float.self).map(\.bitPattern),
          expected.asArray(Float.self).map(\.bitPattern))
      }
      var first = H3SamplingStepper(schedule: schedule, method: .resMultistep)
      var fresh = H3SamplingStepper(schedule: schedule, method: .resMultistep)
      let initial = MLXArray([Float(0.4), -0.2, 1.5])
      let velocity = MLXArray([Float(0.3), -0.1, 0.2])
      let a = try first.advance(sample: initial, velocity: velocity, index: 0)
      _ = try first.advance(sample: a, velocity: velocity, index: 1)
      XCTAssertEqual(a.asArray(Float.self).map(\.bitPattern),
        try fresh.advance(sample: initial, velocity: velocity, index: 0)
          .asArray(Float.self).map(\.bitPattern))
    }
  }

  func testBadIndicesAndShapesDoNotAdvanceHistoryAndOneEvaluationUsesX0() throws {
    try Device.withDefaultDevice(.cpu) {
      let schedule = try H3Schedule(requestedSteps: 2, shift: 3)
      var stepper = H3SamplingStepper(schedule: schedule, method: .resMultistep)
      let initial = MLXArray([Float(0.4), -0.2, 1.5])
      let velocity = MLXArray([Float(0.3), -0.1, 0.2])
      XCTAssertThrowsError(try stepper.advance(sample: initial, velocity: velocity, index: 1))
      XCTAssertThrowsError(try stepper.advance(sample: initial,
        velocity: MLXArray([Float(1)]), index: 0))
      let actual = try stepper.advance(sample: initial, velocity: velocity, index: 0)
      XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),
        (initial + velocity).asArray(Float.self).map(\.bitPattern))
      XCTAssertThrowsError(try stepper.advance(sample: initial, velocity: velocity, index: 0))
    }
  }
  func testCancelledTaskRejectsBeforeAdvancingHistory() async throws {
    let task = Task {
      try Device.withDefaultDevice(.cpu) {
        let schedule = try H3Schedule(requestedSteps: 5, shift: 12)
        var stepper = H3SamplingStepper(schedule: schedule, method: .resMultistep)
        let sample = MLXArray([Float(1), 2, 3])
        withUnsafeCurrentTask { $0?.cancel() }
        let value = try stepper.advance(sample: sample, velocity: sample, index: 0)
        eval(value)
      }
    }
    do {
      _ = try await task.value
      XCTFail("Cancelled sampler advanced a target")
    } catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
  }

}
