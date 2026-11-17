import Foundation
import XCTest
@testable import ImageLayout

final class VisionModelTests: XCTestCase {
  func testModelAdmissionKeepsQwenAndAddsOnlyTheQualifiedGemmaFamily() throws {
    let qwen: [String: Any] = ["model_type": "qwen3_5", "text_config": ["hidden_size": 2560, "num_hidden_layers": 32],
      "vision_config": ["hidden_size": 1024, "out_hidden_size": 2560, "patch_size": 16],
      "quantization": ["bits": 4, "group_size": 64, "mode": "affine"]]
    XCTAssertEqual(try ImageLayoutVisionModel.identify(qwen), .qwen35_4b)
    var gemma: [String: Any] = ["model_type": "gemma4_unified", "text_config": ["hidden_size": 3840, "num_hidden_layers": 48],
      "vision_config": ["mm_embed_dim": 3840, "model_patch_size": 48, "patch_size": 16],
      "quantization": ["bits": 8, "group_size": 64, "mode": "affine"]]
    XCTAssertEqual(try ImageLayoutVisionModel.identify(gemma), .gemma4_12b)
    gemma["quantization"] = ["bits": 4, "group_size": 64, "mode": "affine"]
    XCTAssertThrowsError(try ImageLayoutVisionModel.identify(gemma))
    gemma["quantization"] = ["bits": 8, "group_size": 64, "mode": "affine"]
    gemma.removeValue(forKey: "vision_config")
    XCTAssertThrowsError(try ImageLayoutVisionModel.identify(gemma))
  }
}
