import XCTest
@testable import H3MLX

final class H3MotionFidelityPlanTests: XCTestCase {
  func testUniformRecoveryIsExactAcrossAllSourcePhasesAndHoldCounts() throws {
    XCTAssertEqual((0..<18).map(H3MotionFidelityPlan.latentIndex),
      [0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5])
    for count in 60...86 { for hold in 2...4 {
      let plan = try H3MotionFidelityPlan(sourceFrames: count,
        settings: H3MotionFidelitySettings(mode: .uniform, maxHold: hold))
      XCTAssertFalse(plan.adaptiveAnalysisPerformed)
      XCTAssertEqual(plan.paddedFrames % 17, 5)
      XCTAssertEqual(plan.recovery.map { plan.expansionIndices[$0] }, Array(0..<count))
      XCTAssertEqual(plan.expansionIndices.last, count - 1)
    } }
  }
  func testOwnedAdaptiveBurstGoldenAndQuietNoop() throws {
    let settings = try H3MotionFidelitySettings(maxHold: 4)
    let quiet = try H3MotionFidelityPlan(sourceFrames: 73, settings: settings,
      temporalJerk: Array(repeating: 0, count: 19))
    XCTAssertTrue(quiet.adaptiveAnalysisPerformed)
    XCTAssertTrue(quiet.noop); XCTAssertEqual(quiet.expandedFrames, 73)
    let burst = try H3MotionFidelityPlan(sourceFrames: 73, settings: settings,
      temporalJerk: [0,0,0,0,0,0,0,10,30,30,10,0,0,0,0,0,0,0,0])
    XCTAssertEqual(burst.holds, Array(repeating: 1, count: 32) +
      [2,3,4,4,4,4,4,3,2] + Array(repeating: 1, count: 32))
    XCTAssertEqual(burst.expandedFrames, 94); XCTAssertEqual(burst.paddedFrames, 107)
    XCTAssertEqual(burst.recovery.map { burst.expansionIndices[$0] }, Array(0..<73))
    XCTAssertEqual(burst.scores[30], Double(Float(0.33333)))
    XCTAssertEqual(burst.scores[34], 1)
    XCTAssertFalse(burst.noop)
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(burst)) as! [String: Any]
    XCTAssertEqual(encoded["noop"] as? Bool, false)
    XCTAssertEqual(encoded["adaptiveAnalysisPerformed"] as? Bool, true)
    XCTAssertEqual(encoded["expandedSeconds"] as? Double, Double(107) / 24)
    XCTAssertTrue(zip(burst.holds, burst.holds.dropFirst()).allSatisfy { abs($0.0 - $0.1) <= 1 })
    XCTAssertTrue(burst.audioFilter.contains("end_sample=42667"))
    XCTAssertTrue(burst.audioFilter.contains("atempo=0.5,atempo=0.666666666667"))
    XCTAssertTrue(burst.audioFilter.contains("atempo=0.5,atempo=0.5"))
    XCTAssertTrue(burst.audioFilter.hasSuffix("atrim=end_sample=142667[out]"))
  }
  func testBudgetAndNonfiniteAnalysisRejectBeforeEncoding() throws {
    XCTAssertThrowsError(try H3MotionFidelityPlan(sourceFrames: 100,
      settings: H3MotionFidelitySettings(mode: .uniform, maxHold: 4)))
    XCTAssertThrowsError(try H3MotionFidelityPlan(sourceFrames: 73,
      settings: H3MotionFidelitySettings(), temporalJerk: [0]))
    XCTAssertThrowsError(try H3MotionFidelityPlan(sourceFrames: 73,
      settings: H3MotionFidelitySettings(), temporalJerk: Array(repeating: .nan, count: 19)))
    XCTAssertThrowsError(try H3MotionFidelitySettings(evaluations: 65))
    XCTAssertThrowsError(try H3MotionFidelitySettings(strength: 0))
  }
  func testAudioBoundaryUsesCumulativeClockRatherThanRoundedFrameLength() throws {
    XCTAssertEqual((0...4).map(H3MotionFidelityPlan.audioSampleBoundary), [0,1333,2667,4000,5333])
    let plan = try H3MotionFidelityPlan(sourceFrames: 60,
      settings: H3MotionFidelitySettings(mode: .uniform, maxHold: 3))
    XCTAssertEqual(plan.expandedFrames, 180); XCTAssertEqual(plan.paddedFrames, 192)
    XCTAssertTrue(plan.audioFilter.contains("atrim=end_sample=240000[a0]"))
    XCTAssertTrue(plan.audioFilter.hasSuffix("atrim=end_sample=256000[out]"))
  }
}
