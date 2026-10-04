import Foundation
import XCTest
@testable import H3MLX

final class H3SignedLoRAStrengthTests: XCTestCase {
  func testSignedPublicBoundsAndDeferredDescriptorPreserveEffectiveStrength() throws {
    for strength: Float in [-10, -1, 0, 1, 10] {
      let adapter = try H3LoRAAdapter(url: URL(fileURLWithPath: "/adapter.safetensors"),
        strength: strength, profile: .standard, qkvLayout: .contiguousQKV,
        startAfterEvaluations: 3)
      XCTAssertEqual(adapter.strength, strength)
      try adapter.validate(requestedSteps: 20, samplingMethod: .resMultistep)
      XCTAssertFalse(adapter.isActive(evaluation: 2)); XCTAssertTrue(adapter.isActive(evaluation: 3))
    }
    for strength: Float in [-10.01, 10.01, .infinity, -.infinity, .nan] {
      XCTAssertThrowsError(try H3LoRAAdapter(url: URL(fileURLWithPath: "/adapter.safetensors"), strength: strength))
    }
  }
}
