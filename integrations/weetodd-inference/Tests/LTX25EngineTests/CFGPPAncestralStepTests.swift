import XCTest
@testable import LTX25Engine

final class CFGPPAncestralStepTests: XCTestCase {
  func testAuthoredTimestepRoundsFeaturesAfterRawFloat32Trigonometry() throws {
    // Independent scalar Float32 libm fixture, checked against primary model
    // input ordering; this does not claim Torch GPU trigonometric bit parity.
    let rows:[(Float,[Float])]=[
      (1,[0.5625,0.796875,0.99609375,0.828125,0.107421875]),
      (0.99375,[0.53515625,1,0.99609375,0.84375,0.1064453125]),
      (0.98125,[0.4765625,0.201171875,0.99609375,0.87890625,0.10546875]),
      (0.725,[-0.7578125,-0.80859375,0.99609375,0.6484375,0.07763671875]),
      (0.421875,[0.62109375,0.216796875,1,0.78515625,0.04541015625]),
      (0,[1,1,1,0,0])]
    for (sigma,expected) in rows {
      let actual=try DenoiserMath.authoredTimestep(sigma)
      XCTAssertEqual(actual.count,256)
      XCTAssertEqual([0,31,127,128,255].map { actual[$0] },expected)
      XCTAssertTrue(actual.allSatisfy { $0 == DenoiserMath.bfloat16($0) })
    }
    XCTAssertNotEqual(try DenoiserMath.authoredTimestep(0.725),try DenoiserMath.timestep(0.725))
    XCTAssertThrowsError(try DenoiserMath.authoredTimestep(.nan))
  }
  func testRectifiedFlowUsesRawUnconditionalPredictionEvenAtCFGOne() throws {
    let step = try CFGPPAncestralStep(sigma: 0.725, nextSigma: 0.421875)
    let next = step.sampleScale * 0.7 + step.predictionScale * 0.2
      + step.unconditionalScale * -0.1 + step.noiseScale * 0.3
    XCTAssertEqual(next, 0.35441775816288656, accuracy: 1e-7)
    XCTAssertLessThan(step.unconditionalScale, 0)
    let ordinary = try EulerStep(sigma: 0.725, nextSigma: 0.421875, eta: 0)
    XCTAssertGreaterThan(abs(next - (ordinary.sampleScale * 0.7 + ordinary.predictionScale * 0.2)), 0.1)
  }

  func testMaximumSigmaHasFiniteLimitAndTerminalReturnsPositiveClean() throws {
    let first = try CFGPPAncestralStep(sigma: 1, nextSigma: 0.99375)
    XCTAssertEqual(first.sampleScale, 0)
    XCTAssertEqual(first.unconditionalScale, 0)
    XCTAssertEqual(first.predictionScale, 0.00625, accuracy: 1e-7)
    XCTAssertEqual(first.noiseScale, 0.99375, accuracy: 1e-7)
    let last = try CFGPPAncestralStep(sigma: 0.421875, nextSigma: 0)
    XCTAssertTrue(last.terminal)
    XCTAssertEqual(last.sampleScale, 0)
    XCTAssertEqual(last.unconditionalScale, 0)
    XCTAssertEqual(last.predictionScale, 1)
    XCTAssertEqual(last.noiseScale, 0)
  }

  func testInvalidSigmasFailBeforeModelWork() {
    for pair in [(Double.nan, 0.5), (1.01, 0.5), (0, 0), (0.5, 0.5), (0.5, -0.1)] {
      XCTAssertThrowsError(try CFGPPAncestralStep(sigma: pair.0, nextSigma: pair.1))
    }
  }
}
