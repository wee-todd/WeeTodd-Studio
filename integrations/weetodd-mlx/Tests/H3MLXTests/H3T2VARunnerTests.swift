import Foundation
import XCTest
@testable import H3MLX

final class H3T2VARunnerTests: XCTestCase {
  func testFastH3RequiresReleasedFourEvaluationEulerScheduleBeforeModelIO() throws {
    let missing = URL(fileURLWithPath:"/missing")
    for variant in [H3FastVariant.denseV1,.vsaV1] {
      for (points,method) in [(5,H3SamplingMethod.euler),(9,.euler),(5,.resMultistep)] {
        let create = { try H3T2VARequest(prompt:"Beowulf trains",width:672,height:384,
          durationSeconds:5,seed:42,requestedSteps:points,transformer:missing,qwenPages:missing,
          tokenizer:missing,videoVAE:missing,audioVAE:missing,fastVariant:variant,samplingMethod:method) }
        if points == 5 && method == .euler { XCTAssertEqual(try create().fastVariant,variant) }
        else { XCTAssertThrowsError(try create()) }
      }
    }
  }

  func testOversizedPackedGeometryFailsBeforeModelPathsAreInspected() throws {
    let missing = URL(fileURLWithPath: "/missing")
    XCTAssertThrowsError(try H3T2VARequest(prompt: "Young Beowulf trains",
      width: 768, height: 768, durationSeconds: 15, seed: 1,
      requestedSteps: 5, transformer: missing, qwenPages: missing,
      tokenizer: missing, videoVAE: missing, audioVAE: missing)) { error in
        XCTAssertTrue(String(describing: error).contains("packed rows"))
      }
  }

  func testFastH3DoesNotSilentlyIgnoreAnAdapterWhenEffectiveStackIsEmpty() throws {
    let missing = URL(fileURLWithPath:"/missing")
    XCTAssertThrowsError(try H3T2VARequest(prompt:"Beowulf trains",width:672,height:384,
      durationSeconds:5,seed:42,requestedSteps:5,transformer:missing,qwenPages:missing,
      tokenizer:missing,videoVAE:missing,audioVAE:missing,turboLoRA:missing,
      loRAAdapters:[],fastVariant:.denseV1))
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
