import MLX
import XCTest
@testable import H3MLX

final class H3ScheduleTests: XCTestCase {
  func testFourEvaluationsHaveIndependentVideoAndAudioShifts() throws {
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let expectedVideo: [Float] = [1, 0.9729729891, 0.9230769277, 0.8000000119, 0]
    let expectedAudio: [Float] = [1, 0.8999999762, 0.75, 0.5, 0]
    XCTAssertEqual(video.sigmas, expectedVideo)
    XCTAssertEqual(audio.sigmas, expectedAudio)
    XCTAssertEqual(video.timesteps[1], 1 - expectedVideo[1])
    XCTAssertEqual(audio.timesteps[1], 1 - expectedAudio[1])
  }

  func testDataWardVelocityAndTerminalDenoise() throws {
    let schedule = try H3Schedule(requestedSteps: 5, shift: 12)
    let sample = MLXArray([Float(2)], [1, 1])
    let velocity = MLXArray([Float(1)], [1, 1])
    let first = try schedule.advance(sample: sample, velocity: velocity, index: 0)
      .asArray(Float.self)[0]
    XCTAssertEqual(first, 2.027027, accuracy: 0.00001)
    let final = try schedule.advance(sample: sample, velocity: velocity, index: 3)
      .asArray(Float.self)[0]
    XCTAssertEqual(final, 2.8, accuracy: 0.00001)
  }

  func testFiftyOnePointGridMatchesLongerFloat32Reference() throws {
    let video = try H3Schedule(requestedSteps: 51, shift: 12)
    let audio = try H3Schedule(requestedSteps: 51, shift: 3)
    XCTAssertEqual(video.sigmas.count, 51)
    XCTAssertEqual(audio.sigmas.count, 51)
    XCTAssertEqual(Array(video.sigmas.prefix(5)),
      [1, 0.9983021617, 0.9965397716, 0.9947089553, 0.99280577898] as [Float])
    XCTAssertEqual(Array(audio.sigmas.prefix(5)),
      [1, 0.9932432771, 0.9863013029, 0.9791666269, 0.9718309045] as [Float])
  }

  func testScheduleRejectsUnsupportedStepCountsAndIndices() throws {
    XCTAssertThrowsError(try H3Schedule(requestedSteps: 1, shift: 12))
    XCTAssertThrowsError(try H3Schedule(requestedSteps: 5, shift: 0))
    let schedule = try H3Schedule(requestedSteps: 5, shift: 12)
    XCTAssertThrowsError(try schedule.advance(sample: MLXArray([Float(1)], [1, 1]),
      velocity: MLXArray([Float(1)], [1, 1]), index: 4))
  }
}
