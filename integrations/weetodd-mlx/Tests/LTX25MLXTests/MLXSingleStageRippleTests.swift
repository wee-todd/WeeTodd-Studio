import XCTest
import LTX25Engine
import AdapterRuntime
@testable import LTX25MLX

final class MLXSingleStageRippleTests: XCTestCase {
  func testRippleUsesEightDeterministicStepsAndAdmitsAppendedGuide() throws {
    let geometry = try AVGeometry(width: 64, height: 64, frames: 9, fps: 24)
    let plan = try MLXSingleStageRipple.plan(geometry: geometry, strength: 1,
      maximumActivationBytes: 2 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.schedule.sigmas,
      [1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0])
    XCTAssertEqual(plan.schedule.eta, 0)
    XCTAssertEqual(plan.layout.videoTokens, geometry.videoTokens * 2)
    XCTAssertEqual(plan.configuration.videoTokens, geometry.videoTokens * 2)
    XCTAssertFalse(plan.schedule.steps.contains(where: \.ancestral))
  }

  func testRippleRejectsGuideThatExceedsTransformerAdmission() throws {
    let geometry = try AVGeometry(width: 1024, height: 1024, frames: 513, fps: 24)
    XCTAssertThrowsError(try MLXSingleStageRipple.plan(geometry: geometry, strength: 1,
      maximumActivationBytes: 2 * 1024 * 1024 * 1024))
  }

  func testRippleSamplerAdmitsSeparateImageAnchorsInTokenPlan() throws {
    let geometry = try AVGeometry(width: 64, height: 64, frames: 17, fps: 24)
    let anchors = [RippleImageAnchor(frame: 8, strength: 1)]
    let plan = try MLXSingleStageRipple.plan(geometry: geometry, strength: 1,
      anchors: anchors, maximumActivationBytes: 2 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.layout.videoTokens, geometry.videoTokens * 2 + 4)
    XCTAssertEqual(plan.configuration.videoTokens, plan.layout.videoTokens)
    XCTAssertEqual(plan.schedule.sigmas.count, 9)
  }

  func testFullRateFiveSecondRippleTokenBudgetIsAdmittedBeforeWeights() throws {
    let geometry = try AVGeometry(width: 1376, height: 768, frames: 129, fps: 24)
    let anchors = (1...8).map { RippleImageAnchor(frame: $0 * 16, strength: 1) }
    let plan = try MLXSingleStageRipple.plan(geometry: geometry, strength: 1,
      anchors: anchors, maximumActivationBytes: 32 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.layout.videoTokens, 43_344)
    XCTAssertEqual(plan.configuration.videoTokens, 43_344)
    XCTAssertThrowsError(try MLXSingleStageRipple.plan(geometry: geometry, strength: 1,
      anchors: anchors, maximumActivationBytes: 1024 * 1024 * 1024))
  }

  func testInstalledTransformerAndRippleHeadersWhenRequested() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let transformer = environment["WEETODD_LTX25_TRANSFORMER"],
      let adapter = environment["WEETODD_RIPPLE_ADAPTER"] else {
      throw XCTSkip("Installed Ripple transformer qualification is opt-in.")
    }
    let geometry = try AVGeometry(width: 64, height: 64, frames: 9, fps: 24)
    _ = try MLXSingleStageRipple(geometry: geometry, referenceStrength: 1,
      transformerRoot: URL(fileURLWithPath: transformer),
      adapters: [LoRAAdapter(path: adapter, strength: 1.35)])
  }
}
