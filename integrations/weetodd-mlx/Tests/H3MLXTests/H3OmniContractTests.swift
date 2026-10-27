import CryptoKit
import Foundation
import XCTest
@testable import H3MLX

// Contract data, not implementation: the independently pinned release's sidecar fields.
func omniContractFixture() -> [String: Any] {
  ["schema": "fasth3-inference-contract-v1", "schema_version": "fasth3-inference-contract-v1",
   "model_type": "ref2va", "transformer_component": "transformer_ref",
   "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
   "conditioning": "fixed_ordered_references_target_only_flow", "pdd_steps": 32,
   "pdd_step_indices": [0,4,8,12,16,20,24,28,32], "num_inference_steps": 8,
   "transformer_forwards": 8, "grid_max_t": 0.999, "video_scheduler_shift": 12.0,
   "audio_scheduler_shift": 3.0, "guidance_scale": 1.0,
   "attention_backend": "VIDEO_SPARSE_ATTN_H3", "vsa_tile_size": 128,
   "vsa_sparsity": 0.9, "vsa_ref_policy": "p2_multi_region", "vsa_ref_keep_rate": 0.1]
}
func omniContractData(_ fixture: [String: Any] = omniContractFixture()) throws -> Data {
  try JSONSerialization.data(withJSONObject: fixture, options: [.sortedKeys])
}

final class H3OmniContractTests: XCTestCase {
  func testOfficialContractAdmitsOnlyTrainedExecution() throws {
    let contract = try H3OmniContract(data: omniContractData())
    let plan = try H3PDDPlan(contract: contract)
    XCTAssertEqual(plan.evaluationCount, 8)
    XCTAssertEqual(contract.pddSteps, 32)
    XCTAssertEqual(contract.vsaTileSize, 128)
    XCTAssertEqual(contract.vsaRefPolicy, "p2_multi_region")
    XCTAssertEqual(contract.vsaRefKeepRate, 0.1)
  }

  func testEveryAlteredOrMissingOfficialFieldIsRejected() throws {
    let original = omniContractFixture()
    for key in original.keys {
      var missing = original; missing.removeValue(forKey: key)
      XCTAssertThrowsError(try H3OmniContract(data: omniContractData(missing)), key)
      var altered = original
      if original[key] is String { altered[key] = "unsupported" }
      else if original[key] is [Int] { altered[key] = [0,4,8,12,16,20,24,27,32] }
      else { altered[key] = 0.125 }
      XCTAssertThrowsError(try H3OmniContract(data: omniContractData(altered)), key)
    }
    var unknown = original; unknown["dmd_denoising_steps"] = [1000, 750, 500, 250]
    XCTAssertThrowsError(try H3OmniContract(data: omniContractData(unknown)))
  }

  func testBooleansStringsNonfiniteAndMalformedJSONAreRejected() throws {
    for key in ["pdd_steps", "transformer_forwards", "num_inference_steps", "vsa_tile_size",
                "guidance_scale", "grid_max_t", "vsa_sparsity", "vsa_ref_keep_rate",
                "video_scheduler_shift", "audio_scheduler_shift"] {
      for value: Any in [true, false, "1", NSNull()] {
        var fixture = omniContractFixture(); fixture[key] = value
        XCTAssertThrowsError(try H3OmniContract(data: omniContractData(fixture)), key)
      }
    }
    var booleanIndex = omniContractFixture(); booleanIndex["pdd_step_indices"] = [false,4,8,12,16,20,24,28,32] as [Any]
    XCTAssertThrowsError(try H3OmniContract(data: omniContractData(booleanIndex)))
    for json in ["[]", "null", "{}", "{\"grid_max_t\":NaN}", "{\"grid_max_t\":1e999}"] {
      XCTAssertThrowsError(try H3OmniContract(data: Data(json.utf8)))
    }
    let complete = String(decoding: try omniContractData(), as: UTF8.self)
    for token in ["NaN", "Infinity", "1e999"] {
      let nonfinite = complete.replacingOccurrences(of: "\"grid_max_t\":0.999", with: "\"grid_max_t\":" + token)
      XCTAssertNotEqual(nonfinite, complete)
      XCTAssertThrowsError(try H3OmniContract(data: Data(nonfinite.utf8)))
    }
    // Otherwise-valid JSON must fail the bound, rather than merely fail JSON
    // parsing. Trailing JSON whitespace is legal.
    let oversized = try omniContractData() + Data(repeating: 32, count: 65_537)
    XCTAssertThrowsError(try H3OmniContract(data: oversized))
  }

  func testSidecarDigestBindsExactBytesAndRejectsUnboundedOrNonfileInputs() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = folder.appendingPathComponent("contract.json")
    let data = try omniContractData(); try data.write(to: url)
    let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    XCTAssertNoThrow(try H3OmniContract.load(url: url, expectedSHA256: sha))
    for invalid in [sha.uppercased(), String(repeating: "0", count: 64), "bad"] {
      XCTAssertThrowsError(try H3OmniContract.load(url: url, expectedSHA256: invalid))
    }
    try (data + Data("\n".utf8)).write(to: url)
    XCTAssertThrowsError(try H3OmniContract.load(url: url, expectedSHA256: sha))
    let oversized = data + Data(repeating: 32, count: 65_537)
    try oversized.write(to: url)
    let oversizedSHA = SHA256.hash(data: oversized).map { String(format: "%02x", $0) }.joined()
    XCTAssertThrowsError(try H3OmniContract.load(url: url, expectedSHA256: oversizedSHA))
    XCTAssertThrowsError(try H3OmniContract.load(url: folder, expectedSHA256: sha))
    XCTAssertThrowsError(try H3OmniContract.load(url: URL(string: "https://example.com/contract.json")!, expectedSHA256: sha))
  }
}
