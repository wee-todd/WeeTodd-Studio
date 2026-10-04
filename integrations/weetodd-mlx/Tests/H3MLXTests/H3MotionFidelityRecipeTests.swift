import Foundation
import XCTest
@testable import H3MLX

final class H3MotionFidelityRecipeTests: XCTestCase {
  private func recipe() -> [String: Any] {
    ["engine":"h3", "prompt":"Preserve the original movement and character.",
     "components":["task":"t2va"], "config":["steps":20],
     "conditioning":["task":"t2v","inputs":[]],
     "motion_fidelity":["version":1,"source_video":"/source.mp4", "source_sha256":String(repeating:"a",count:64),
       "source_in":0.125,"duration_seconds":2.5,"ffprobe":"/ffprobe","mode":"adaptive",
       "strength":0.5,"max_hold":2,"sensitivity":0.5,"seed":42,"max_frames":345]]
  }
  private func prepared(_ root: [String: Any]) throws -> H3MotionFidelityRecipe.Prepared? {
    try H3MotionFidelityRecipe.prepare(data: JSONSerialization.data(withJSONObject: root))
  }
  func testExplicitMotionControlsAreRemovedBeforeOrdinaryCompilation() throws {
    let input = recipe(); let result = try XCTUnwrap(prepared(input))
    XCTAssertEqual(result.sourceIn, 0.125); XCTAssertEqual(result.durationSeconds, 2.5)
    XCTAssertEqual(result.settings.seed, 42); XCTAssertEqual(result.settings.maxHold, 2)
    var expected = input; expected.removeValue(forKey:"motion_fidelity")
    XCTAssertEqual(try JSONSerialization.jsonObject(with:result.ordinaryRecipe) as? NSDictionary,
      expected as NSDictionary)
    XCTAssertNil(try prepared(expected))
  }
  func testUnsupportedCombinationsAndMistypedControlsFailBeforeSourceLoading() throws {
    for key in ["joint_latents","continuation","refinement"] {
      var input = recipe(); input[key] = ["version":1]
      XCTAssertThrowsError(try prepared(input))
    }
    for (key, value) in [("version",true), ("seed",true), ("max_hold",2.5), ("duration_seconds",2.501), ("source_in",0.01), ("unknown",0)] as [(String,Any)] {
      var input = recipe(), controls = input["motion_fidelity"] as! [String:Any]
      controls[key] = value; input["motion_fidelity"] = controls
      XCTAssertThrowsError(try prepared(input))
    }
    var turbo = recipe(); turbo["config"] = ["steps":5]
    XCTAssertThrowsError(try prepared(turbo))
    var ref = recipe(); ref["components"] = ["task":"ref2va"]
    XCTAssertThrowsError(try prepared(ref))
  }
}
