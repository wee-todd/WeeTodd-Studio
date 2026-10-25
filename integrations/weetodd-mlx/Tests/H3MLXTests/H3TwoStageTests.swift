import Foundation
import XCTest
@testable import H3MLX

final class H3TwoStageTests: XCTestCase {
  func testAllTaskRunnersExposeLatentOnlySampling() {
    let _: (H3T2VARequest) throws -> H3SampledLatents = {try H3T2VARunner.sample($0)}
    let _: (H3FL2VARequest) throws -> H3SampledLatents = {try H3FL2VARunner.sample($0)}
    let _: (H3Ref2VAStillRequest) throws -> H3SampledLatents = {try H3Ref2VAStillRunner.sample($0)}
  }
  func testCustomFinalCanvasIsExactAndOnlyDraftRounds() throws {
    let plan = try H3TwoStagePlan(width: 1536, height: 864, durationSeconds: 3)
    XCTAssertEqual(plan.target.width, 1536); XCTAssertEqual(plan.target.height, 864)
    XCTAssertEqual(plan.base.width, 768); XCTAssertEqual(plan.base.height, 448)
    let portrait = try H3TwoStagePlan(width: 864, height: 1536, durationSeconds: 3)
    XCTAssertEqual(portrait.base.width, 448); XCTAssertEqual(portrait.base.height, 768)
    let square = try H3TwoStagePlan(width: 1024, height: 1024, durationSeconds: 3, draftScale: 1.5)
    XCTAssertEqual(square.base.width, 704); XCTAssertEqual(square.base.height, 704)
    XCTAssertEqual(square.target.width, 1024)
  }
  func testInvalidFinalAndOverBudgetDurationRejectBeforeSampling() throws {
    for canvas in [(1920,1080), (1952,1088), (1920,1120), (32,64)] {
      XCTAssertThrowsError(try H3TwoStagePlan(width: canvas.0, height: canvas.1, durationSeconds: 3))
    }
    XCTAssertThrowsError(try H3TwoStagePlan(width: 1920, height: 1088, durationSeconds: 5))
    XCTAssertNoThrow(try H3TwoStagePlan(width: 1920, height: 1088, durationSeconds: 3))
    XCTAssertThrowsError(try H3TwoStagePlan(width: 1344, height: 768, durationSeconds: 3, draftScale: .nan))
    XCTAssertThrowsError(try H3TwoStagePlan(width: 1344, height: 768, durationSeconds: 3, evaluations: 0))
  }
  func testRefinementBudgetIsIndependentAndUsesModalityShifts() throws {
    let plan = try H3TwoStagePlan(width: 1344, height: 768, durationSeconds: 3)
    let short = try plan.refinement.schedules(requestedSteps: 5)
    let long = try plan.refinement.schedules(requestedSteps: 21)
    XCTAssertEqual(short.video.sigmas, long.video.sigmas)
    XCTAssertEqual(short.audio.sigmas, long.audio.sigmas)
    XCTAssertEqual(short.video.timesteps.count, 3)
    XCTAssertEqual(short.video.sigmas[0], Float(12 * 0.35 / (1 + 11 * 0.35)), accuracy: 1e-6)
    XCTAssertEqual(short.audio.sigmas[0], Float(3 * 0.35 / (1 + 2 * 0.35)), accuracy: 1e-6)
    XCTAssertTrue(plan.refinement.preserveAudio)
  }
}
