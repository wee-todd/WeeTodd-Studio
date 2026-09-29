import Foundation
import XCTest
import InferenceTestSupport
@testable import LTX25NNC

final class BlockWeightSourceTests: XCTestCase {
  func testMapsOfficialDenseNamesAndRejectsIncompatibleHeader() throws {
    let name = "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"
    try withTensorFile(tensors: [(name, [2, 4], "BF16")]) { url in
      let source = try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["ff.proj_in.weight": [2, 4]])
      XCTAssertEqual(source.decodedWeightBytes, 32)
      XCTAssertEqual(try source.read("ff.proj_in.weight", shape: [2, 4]), [Float](repeating: 0, count: 8))
      XCTAssertThrowsError(try source.read("ff.proj_in.weight", shape: [4, 2]))
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["ff.proj_in.weight": [4, 2]]))
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 1, expectedShapes: ["ff.proj_in.weight": [2, 4]]))
    }
  }

  func testValidatesPackedCompanionsAndRejectsUnconsumedSelectedBlockWeights() throws {
    let stem = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
    let tensors = [(stem + ".weight", [2, 16], "U32"), (stem + ".scales", [2, 1], "BF16"),
      (stem + ".biases", [2, 1], "BF16")]
    try withTensorFile(tensors: tensors) { url in
      let source = try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["attn1.to_q.weight": [2, 64]])
      XCTAssertEqual(source.decodedWeightBytes, 512)
      XCTAssertEqual(try source.read("attn1.to_q.weight", shape: [2, 64]), [Float](repeating: 0, count: 128))
    }
    try withTensorFile(tensors: Array(tensors.dropLast())) { url in
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["attn1.to_q.weight": [2, 64]]))
    }
    try withTensorFile(tensors: tensors + [(stem + ".unsupported", [2], "BF16")]) { url in
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["attn1.to_q.weight": [2, 64]]))
    }
  }

  func testRejectsAliasedDuplicateKeysAndMemoryBudget() throws {
    let prefix = "model.diffusion_model.transformer_blocks.0."
    try withTensorFile(tensors: [(prefix + "attn1.to_out.0.bias", [2], "F32"),
      (prefix + "attn1.to_out.bias", [2], "F32")]) { url in
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 0, expectedShapes: ["attn1.to_out.bias": [2]]))
    }
    try withTensorFile(tensors: [(prefix + "attn1.to_out.bias", [2], "F32")]) { url in
      XCTAssertThrowsError(try BlockWeightSource(url: url, blockIndex: 0,
        expectedShapes: ["attn1.to_out.bias": [2]], maximumDecodedWeightBytes: 4))
    }
  }
}
