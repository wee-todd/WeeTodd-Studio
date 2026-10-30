import Foundation
import XCTest
@testable import InferenceContracts

final class NativeFlashVSRRecipeTests: XCTestCase {
  private func recipe(compression: Int = 4, frames: Int = 17) throws -> NativeFlashVSRRecipe {
    try .init(sourceID: String(repeating: "a", count: 64), inputRGBPath: "/jobs/input.rgb",
      inputSHA256: String(repeating: "b", count: 64), outputRGBPath: "/jobs/output.rgb",
      scratchDirectory: "/jobs/scratch", modelRoot: "/models/FlashVSR", width: 256, height: 128,
      frames: frames, fpsNumerator: 30000, fpsDenominator: 1001, temporalCompression: compression,
      variant: .turbo, decoder: .tiny, seed: 1)
  }
  func testExpandedClockAndDecoderLookaheadNeverChangeVisibleCadence() throws {
    let ordinary = try recipe()
    XCTAssertEqual(ordinary.expandedFrames, 17)
    XCTAssertEqual(ordinary.groupCount, 2)
    XCTAssertEqual(ordinary.projectionExpandedIndices(group: 0), Array(0..<17))
    XCTAssertEqual(ordinary.projectionExpandedIndices(group: 1), Array(repeating: 16, count: 16))
    let dense = try recipe(compression: 1)
    XCTAssertEqual(dense.expandedFrames, 65)
    XCTAssertEqual(dense.sourceIndex(expandedIndex: 1), 1)
    XCTAssertEqual(dense.sourceIndex(expandedIndex: 4), 1)
    XCTAssertEqual(dense.sourceIndex(expandedIndex: 5), 2)
    XCTAssertEqual(dense.fpsNumerator, 30000)
    XCTAssertEqual(dense.fpsDenominator, 1001)
    XCTAssertEqual(dense.groupCount, 5)
    XCTAssertEqual(try recipe(frames: 1).groupCount, 1)
    XCTAssertEqual(try recipe(compression: 2, frames: 14).expandedFrames, 29)
  }
  func testStrictRoundTripAndScheduleBackendFieldsRejectBeforeWeights() throws {
    let original = try recipe()
    let data = try JSONEncoder().encode(original)
    XCTAssertEqual(try NativeFlashVSRRecipe.prepare(data: data), original)
    for (key, value) in [("version", 2 as Any), ("temporalCompression", 3), ("pythonPath", "/bin/python"), ("frames", 0)] {
      var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      object[key] = value
      XCTAssertThrowsError(try NativeFlashVSRRecipe.prepare(data: JSONSerialization.data(withJSONObject: object)))
    }
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    object["sampling"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(NativeCapabilityDescriptor.h3Regenerate.sampling))
    XCTAssertThrowsError(try NativeFlashVSRRecipe.prepare(data: JSONSerialization.data(withJSONObject: object)))
  }
}
