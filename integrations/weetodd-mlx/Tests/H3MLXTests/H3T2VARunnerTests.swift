import Foundation
import XCTest
@testable import H3MLX

final class H3T2VARunnerTests: XCTestCase {
  func testOversizedPackedGeometryFailsBeforeModelPathsAreInspected() throws {
    let missing = URL(fileURLWithPath: "/missing")
    XCTAssertThrowsError(try H3T2VARequest(prompt: "Young Beowulf trains",
      width: 768, height: 768, durationSeconds: 15, seed: 1,
      requestedSteps: 5, transformer: missing, qwenPages: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing)) { error in
        XCTAssertTrue(String(describing: error).contains("packed rows"))
      }
  }

  func testTextOnlyVerticalSliceRejectsInvalidGeometryAndEmptyPromptBeforeModelIO() throws {
    let missing = URL(fileURLWithPath: "/missing")
    XCTAssertThrowsError(try H3T2VARequest(prompt: "",
      width: 32, height: 32, durationSeconds: 2.5, seed: 1,
      requestedSteps: 5, transformer: missing, qwenPages: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing))
    XCTAssertThrowsError(try H3T2VARequest(prompt: "A person walks",
      width: 33, height: 32, durationSeconds: 2.5, seed: 1,
      requestedSteps: 5, transformer: missing, qwenPages: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing))
    XCTAssertThrowsError(try H3T2VARequest(prompt: "A person walks",
      width: 32, height: 32, durationSeconds: 2.5, seed: 1,
      requestedSteps: 201, transformer: missing, qwenPages: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing))
  }
}
