import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3JointRefinementTests: XCTestCase {
  func testSuffixUsesCeilingOfEvaluationsAndRetainsExactExistingSigmaValues() throws {
    let full = try H3Schedule(requestedSteps: 5, shift: 12)
    let controls = try H3JointRefinement(strength: 0.35)
    let schedules = try controls.schedules(requestedSteps: 5)
    XCTAssertEqual(schedules.video.sigmas, Array(full.sigmas.suffix(3)))
    XCTAssertEqual(schedules.video.timesteps.count, 2)
    let unchanged = try H3JointRefinement(strength: 1).schedules(requestedSteps: 5)
    XCTAssertEqual(unchanged.video.sigmas, full.sigmas)
    XCTAssertEqual(unchanged.audio.sigmas, try H3Schedule(requestedSteps: 5, shift: 3).sigmas)
  }
  func testExplicitNoiseFractionUsesOneBaseClockAcrossDifferentModalityShifts() throws {
    let schedules = try H3JointRefinement(strength: 0.5, startVideoSigma: 0.25,
      evaluations: 4).schedules(requestedSteps: 20)
    // Independent NumPy float32 fixture from the owned refinement_sigmas formula.
    let video: [UInt32] = [0x3e800001, 0x3e4bab23, 0x3e109091, 0x3d9a90e8, 0]
    let audio: [UInt32] = [0x3d9d89d9, 0x3d6f6069, 0x3d21af29, 0x3ca3d70a, 0]
    XCTAssertEqual(schedules.video.sigmas.map(\.bitPattern), video)
    XCTAssertEqual(schedules.audio.sigmas.map(\.bitPattern), audio)
    XCTAssertEqual(schedules.video.timesteps.count, 4)
    XCTAssertNotEqual(schedules.video.sigmas[0], schedules.audio.sigmas[0])
  }
  func testExplicitAndSuffixAdmissionRejectsMalformedControlsBeforeModelUse() throws {
    for value in [0.0, -0.01, 1.01, Double.infinity, Double.nan] {
      XCTAssertThrowsError(try H3JointRefinement(strength: value))
      XCTAssertThrowsError(try H3JointRefinement(strength: 0.5, startVideoSigma: value))
    }
    XCTAssertThrowsError(try H3JointRefinement(strength: 0.5, evaluations: 3))
    XCTAssertThrowsError(try H3JointRefinement(strength: 0.5, startVideoSigma: 0.5, evaluations: 65))
    XCTAssertThrowsError(try H3Schedule(sigmas: [0.5, 0.5, 0]))
    XCTAssertThrowsError(try H3Schedule(sigmas: [0.5, 0.25]))
  }
  func testInitializedTargetMathRetainsFloat32MultiplyOrderAndRejectsNonfiniteRows() throws {
    try Device.withDefaultDevice(.cpu) {
      let schedule = try H3Schedule(sigmas: [0.25, 0])
      let source = MLXArray([Float](repeating: 2, count: 32), [1, 1, 32])
      let noise = MLXArray([Float](repeating: 10, count: 32), [1, 1, 32])
      let result = try H3JointRefinement.initialize(source: source, noise: noise, schedule: schedule)
      XCTAssertEqual(result.asArray(Float.self), [Float](repeating: 4, count: 32))
      let bad = MLXArray([Float](repeating: .nan, count: 32), [1, 1, 32])
      XCTAssertThrowsError(try H3JointRefinement.initialize(source: bad, noise: noise, schedule: schedule))
    }
  }
}
