import XCTest
import TensorIO
import InferenceTestSupport
@testable import AdapterRuntime

final class LoRAPlanTests: XCTestCase {
  let standard: [(String, [Int], String)] = [
    ("layer.lora_A.weight", [2, 3], "BF16"), ("layer.lora_B.weight", [4, 2], "BF16")]

  func testPerTargetAlphaOverridesGlobalScaleAndRetainsSourceNames() throws {
    try withTensorFile(metadata: ["lora_alpha": "8", "lora_rank": "2"],
      tensors: standard + [("layer.alpha", [], "F32")], scalars: ["layer.alpha": 1]) { url in
      let plan = try LoRAPlan(file: SafeTensorFile(url: url), strength: 0.8,
        targetShapes: ["layer": [4, 3]])
      XCTAssertEqual(plan.pairs.count, 1)
      XCTAssertEqual(plan.pairs[0].scale, 0.4, accuracy: 1e-7)
      XCTAssertEqual(plan.pairs[0].downTensor, "layer.lora_A.weight")
      XCTAssertEqual(plan.pairs[0].upTensor, "layer.lora_B.weight")
    }
  }

  func testZeroStrengthDoesNotExcuseUnsupportedTargetsOrUnknownTensors() throws {
    try withTensorFile(tensors: standard) { url in
      XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 0,
        targetShapes: ["other": [4, 3]]))
    }
    try withTensorFile(tensors: standard + [("unexpected.bias", [4], "F32")]) { url in
      XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 1,
        targetShapes: ["layer": [4, 3]]))
    }
  }

  func testRejectsOrphansMixedSchemasAndIncompatibleDestinationShapes() throws {
    for tensors in [Array(standard.prefix(1)),
      [standard[0], ("layer.lora_up.weight", [4, 2], "BF16")],
      [standard[0], ("layer.lora_B.weight", [4, 3], "BF16")],
      standard + [("other.alpha", [], "F32")]] {
      try withTensorFile(tensors: tensors) { url in
        XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 1,
          targetShapes: ["layer": [4, 3]]))
      }
    }
    try withTensorFile(tensors: standard) { url in
      XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 1,
        targetShapes: ["layer": [3, 4]]))
    }
  }

  func testRejectsConflictingMetadataAndNonfiniteScaling() throws {
    for metadata in [["lora_alpha": "8", "ss_network_alpha": "4"], ["lora_alpha": "nan"],
                     ["lora_rank": "0"], ["lora_alpha": "-1"]] {
      try withTensorFile(metadata: metadata, tensors: standard) { url in
        XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 1,
          targetShapes: ["layer": [4, 3]]))
      }
    }
    try withTensorFile(tensors: standard + [("layer.alpha", [], "F32")],
      scalars: ["layer.alpha": .infinity]) { url in
      XCTAssertThrowsError(try LoRAPlan(file: SafeTensorFile(url: url), strength: 1,
        targetShapes: ["layer": [4, 3]]))
    }
  }

  func testDownUpConventionUsesGlobalAlphaOverActualRankAndSignedStrength() throws {
    try withTensorFile(metadata: ["ss_network_alpha": "4"], tensors: [
      ("layer.lora_down.weight", [2, 3], "F16"), ("layer.lora_up.weight", [4, 2], "F16")]) { url in
      let plan = try LoRAPlan(file: SafeTensorFile(url: url), strength: -0.5,
        targetShapes: ["layer": [4, 3]])
      XCTAssertEqual(plan.pairs[0].scale, -1)
    }
  }
}
