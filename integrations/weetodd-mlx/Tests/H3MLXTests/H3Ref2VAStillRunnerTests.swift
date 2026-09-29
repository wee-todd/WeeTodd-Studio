import Foundation
import XCTest
@testable import H3MLX

final class H3Ref2VAStillRunnerTests: XCTestCase {
  private let missing = URL(fileURLWithPath: "/missing")

  func testRequestRejectsMissingReferencesBeforeModelIO() throws {
    XCTAssertThrowsError(try makeRequest(references: []))
  }

  func testRequestRejectsMalformedReferenceBytesBeforeModelIO() throws {
    let bad = H3StillReference(rgb8: Data(count: 7), width: 64, height: 64)
    XCTAssertThrowsError(try makeRequest(references: [bad]))
  }

  func testRequestRejectsExcessReferencesBeforeModelIO() throws {
    let valid = H3StillReference(rgb8: Data(count: 64 * 64 * 3),
      width: 64, height: 64)
    XCTAssertThrowsError(try makeRequest(references: Array(repeating: valid, count: 10)))
  }

  func testRequestRejectsReferenceRowsBeyondBudgetBeforeModelIO() throws {
    let valid = H3StillReference(rgb8: Data(count: 256 * 256 * 3),
      width: 256, height: 256)
    XCTAssertThrowsError(try makeRequest(references: Array(repeating: valid, count: 9),
      width: 768, height: 768, duration: 15))
  }

  private func makeRequest(references: [H3StillReference], width: Int = 64,
    height: Int = 64, duration: Double = 2.5) throws -> H3Ref2VAStillRequest {
    try H3Ref2VAStillRequest(prompt: "A person walks", references: references,
      width: width, height: height, durationSeconds: duration, seed: 1,
      requestedSteps: 5, transformer: missing, qwenPages: missing,
      qwenVision: missing, tokenizer: missing, videoVAE: missing,
      audioVAE: missing)
  }
}
