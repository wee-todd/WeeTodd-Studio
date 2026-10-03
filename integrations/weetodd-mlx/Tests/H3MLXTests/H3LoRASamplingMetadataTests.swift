import Foundation
import XCTest
@testable import H3MLX

/// Header/alpha admission only: no MLX arrays, weights or model evaluation.
final class H3LoRASamplingMetadataTests: XCTestCase {
  private func withAdapter(_ metadata: [String: String],
    _ body: (URL) throws -> Void) throws {
    let target = "diffusion_model.blocks.0.mlp.fc2"
    let downCount = 14336 * 2, upCount = 5376 * 2
    var payload = Data(repeating: 0, count: downCount + upCount)
    var alpha = Float(1).bitPattern.littleEndian
    payload.append(withUnsafeBytes(of: &alpha) { Data($0) })
    let base = ["target_format": "ComfyUI generic LoRA",
      "qkv_fusion": "block diagonal B"]
    let header: [String: Any] = [
      "__metadata__": base.merging(metadata) { _, new in new },
      target + ".lora_A.weight": ["dtype": "BF16", "shape": [1, 14336],
        "data_offsets": [0, downCount]],
      target + ".lora_B.weight": ["dtype": "BF16", "shape": [5376, 1],
        "data_offsets": [downCount, downCount + upCount]],
      target + ".alpha": ["dtype": "F32", "shape": [],
        "data_offsets": [downCount + upCount, payload.count]],
    ]
    let json = try JSONSerialization.data(withJSONObject: header)
    var size = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of: &size) { Data($0) }
    bytes.append(json); bytes.append(payload)
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".safetensors")
    try bytes.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    try body(url)
  }

  func testDeclaredTurboNeedsFourEvaluationsAndFiveSchedulePoints() throws {
    try withAdapter(["adapter_profile": "turbo", "inference_steps": "4"]) { url in
      XCTAssertEqual(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5).targetCount, 1)
      for steps in [4, 6, 12] {
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: steps))
      }
    }
    for metadata in [["distillation_profile": "turbo", "steps": "12"]] {
      try withAdapter(metadata) { url in
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
      }
    }
  }

  func testTurboWithoutDeclaredCountDefaultsToFourEvaluations() throws {
    for metadata in [["profile": "turbo"], ["adapter_role": "turbo"]] {
      try withAdapter(metadata) { url in
        XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 4))
        XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1))
      }
    }
  }

  func testCountAliasesNormalizeEvaluationsAndSchedulePoints() throws {
    try withAdapter(["profile": "Turbo", "inference_steps": "4",
      "num_inference_steps": "4", "steps": "4", "transformer_evaluations": "4",
      "schedule_points": "5"]) { url in
      XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
    }
    try withAdapter(["adapter_role": "turbo", "schedule_points": "5"]) { url in
      XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
    }
  }

  func testLowDeclaredEvaluationCountInfersTurboWithoutFilenameGuessing() throws {
    for count in 1...8 {
      try withAdapter(["inference_steps": String(count)]) { url in
        if count == 4 {
          XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
          XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 12))
        } else {
          XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
        }
      }
    }
  }

  func testConflictingProfilesAndCountsFailClosed() throws {
    for metadata in [["profile": "standard", "adapter_role": "turbo", "steps": "4"],
      ["profile": "standard", "steps": "4"],
      ["profile": "turbo", "adapter_profile": "quality", "steps": "4"],
      ["inference_steps": "4", "schedule_points": "6"]] {
      try withAdapter(metadata) { url in
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
      }
    }
  }

  func testMalformedExplicitMetadataFailsClosed() throws {
    for value in ["", "0", "-4", "4.0", "+4", "04", "four", String(repeating: "9", count: 30)] {
      try withAdapter(["inference_steps": value]) { url in
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
      }
    }
    for metadata in [["profile": "unknown"], ["profile": "true"], ["profile": "false"], ["profile": ""],
      ["adapter_role": "unknown"], ["schedule_points": "1"]] {
      try withAdapter(metadata) { url in
        XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 5))
      }
    }
  }

  func testGenericStandardRetainsArbitrarySupportedSchedules() throws {
    for metadata in [[:], ["profile": "standard"], ["profile": "quality"],
      ["profile": "base"], ["adapter_role": "style"], ["inference_steps": "12"]] {
      try withAdapter(metadata) { url in
        for steps in [5, 12, 50] {
          XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1, requestedSteps: steps))
        }
      }
    }
  }

  func testOptionalRequestedStepsPreservesProjectionCallersButChecksMetadata() throws {
    try withAdapter(["profile": "turbo", "steps": "4"]) { url in
      XCTAssertNoThrow(try H3LoRAFile(url: url, strength: 1))
    }
    try withAdapter(["profile": "turbo", "steps": "8"]) { url in
      XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1))
    }
  }
}
