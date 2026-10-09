import Foundation
import XCTest
@_spi(H3SolDiagnostic) @testable import H3MLX

final class H3AttentionPolicyRecipeTests: XCTestCase {
  private func recipe(_ policy: Any? = nil) -> [String: Any] {
    var config: [String: Any] = ["width": 64, "height": 64,
      "duration_seconds": 2.5, "steps": 5, "seed": 123,
      "sampling_method": "euler", "drop_adaln": true]
    if let policy { config["attention_policy"] = policy }
    return ["format": "weetodd-headless-v2", "engine": "h3", "prompt": "A reference subject moves.",
      "components": ["task": "ref2va", "transformer": "/tmp/transformer.safetensors",
        "text_encoder": "/tmp/qwen-pages", "tokenizer": "/tmp/tokenizer",
        "video_vae": "/tmp/video.safetensors", "audio_vae": "/tmp/audio.safetensors",
        "loras": [["/tmp/turbo.safetensors", 1.0]]],
      "config": config, "conditioning": ["version": 1, "task": "ref2va",
        "audio_policy": "generated", "inputs": [["id": "subject", "kind": "image",
          "role": "reference", "path": "/tmp/reference.png",
          "sha256": String(repeating: "a", count: 64)]]]]
  }

  private func compile(_ root: [String: Any], resolutions: inout Int) throws -> H3Ref2VAStillRequest {
    try H3StudioRecipe.compileMediaReferences(data: JSONSerialization.data(withJSONObject: root)) { _, _, _ in
      resolutions += 1
      return .image(H3StillReference(rgb8: Data(count: 64 * 64 * 3), width: 64, height: 64))
    }
  }

  func testMissingAndExplicitDensePreserveExistingRequest() throws {
    var resolutions = 0
    let missing = try compile(recipe(), resolutions: &resolutions)
    let explicit = try compile(recipe("dense"), resolutions: &resolutions)
    XCTAssertEqual(missing.attentionPolicy, .dense)
    XCTAssertEqual(explicit.attentionPolicy, .dense)
    XCTAssertEqual(missing.seed, explicit.seed)
    XCTAssertEqual(missing.requestedSteps, explicit.requestedSteps)
    XCTAssertEqual(missing.transformerBackend, explicit.transformerBackend)
    XCTAssertEqual(resolutions, 2)
  }

  func testExplicitSolMapsToFixedVersionOnePolicy() throws {
    var resolutions = 0
    let request = try compile(recipe("sol_experimental"), resolutions: &resolutions)
    XCTAssertEqual(request.attentionPolicy, .solExperimental)
    XCTAssertEqual(request.transformerBackend, .mlx)
    XCTAssertEqual(request.requestedSteps - 1, 4)
    XCTAssertEqual(resolutions, 1)
    let report = H3SolReport(tau: 0.5, generatedVideoStart: 100, generatedVideoEnd: 200,
      completedEvaluations: 4, completedSolConsumers: 90, completedDenseBlocks: 110,
      selectedExactBlockPairs: 1, coarseBlockPairs: 2, maximumPreparedLogicalBytes: 3).metadata
    XCTAssertEqual(report["policyVersion"] as? Int, 1)
    XCTAssertEqual(report["tau"] as? Float, 0.5)
    XCTAssertEqual(report["blockSize"] as? Int, 64)
    XCTAssertEqual(report["queryBlockSize"] as? Int, 64)
    XCTAssertEqual(report["denseWarmupEvaluations"] as? Int, 2)
    XCTAssertEqual(report["excludedBlocks"] as? [Int], [0, 1, 33, 34, 35])
  }

  func testUnknownPolicyAndUnsupportedTasksFailBeforeMedia() throws {
    for invalid in ["unknown", "SOL_EXPERIMENTAL", 1, true, NSNull()] as [Any] {
      var resolutions = 0
      XCTAssertThrowsError(try compile(recipe(invalid), resolutions: &resolutions))
      XCTAssertEqual(resolutions, 0)
    }
    for key in ["fasth3", "vdn", "motion_fidelity", "joint_refinement", "joint_latents", "refinement", "continuation"] {
      var root = recipe("sol_experimental")
      root[key] = [String: Any]()
      var resolutions = 0
      XCTAssertThrowsError(try compile(root, resolutions: &resolutions))
      XCTAssertEqual(resolutions, 0, key)
    }
    for change in [["steps": 9], ["sampling_method": "res_multistep"],
      ["drop_adaln": false], ["drop_adaln": 1], ["transformer_backend": "nnc_experimental"]] as [[String: Any]] {
      var root = recipe("sol_experimental")
      root["config"] = (root["config"] as! [String: Any]).merging(change) { _, new in new }
      var resolutions = 0
      XCTAssertThrowsError(try compile(root, resolutions: &resolutions))
      XCTAssertEqual(resolutions, 0)
    }
    for task in ["t2va"] {
      var root = recipe("sol_experimental")
      root["components"] = (root["components"] as! [String: Any]).merging(["task": task]) { _, new in new }
      XCTAssertThrowsError(try H3AttentionPolicy.admitRecipe(root, canvasAdmission: .ordinary))
    }
    XCTAssertThrowsError(try H3AttentionPolicy.admitRecipe(recipe("sol_experimental"), canvasAdmission: .spatialRefinement))
    var fun = recipe("sol_experimental")
    fun["components"] = (fun["components"] as! [String: Any]).merging(["fun_controlnet": "/tmp/fun"]) { _, new in new }
    XCTAssertThrowsError(try H3AttentionPolicy.admitRecipe(fun, canvasAdmission: .ordinary))
  }

  func testTurboAndPackedRowAdmission() throws {
    for stack in [[], [["/tmp/turbo.safetensors", 0.5]],
      [["/tmp/a.safetensors", 1.0], ["/tmp/b.safetensors", 1.0]]] as [[[Any]]] {
      var root = recipe("sol_experimental")
      root["components"] = (root["components"] as! [String: Any]).merging(["loras": stack]) { _, new in new }
      var resolutions = 0
      XCTAssertThrowsError(try compile(root, resolutions: &resolutions))
      XCTAssertEqual(resolutions, 0)
    }
    let url = URL(fileURLWithPath: "/tmp/turbo.safetensors")
    let good = try H3LoRAAdapter(url: url, strength: 1, profile: .turbo, qkvLayout: .contiguousQKV)
    try H3SolTaskPolicy.validateSettings(steps: 5, samplingMethod: .euler, adapters: [good], packedRows: 40_000)
    XCTAssertThrowsError(try H3SolTaskPolicy.validateSettings(steps: 5, samplingMethod: .euler, adapters: [good], packedRows: 40_001))
    let invalid = [try H3LoRAAdapter(url: url, strength: 1, profile: .standard),
      try H3LoRAAdapter(url: url, strength: 1, qkvLayout: .nativeInterleaved),
      try H3LoRAAdapter(url: url, strength: 1, startAfterEvaluations: 1)]
    for adapter in invalid {
      XCTAssertThrowsError(try H3SolTaskPolicy.validateSettings(steps: 5, samplingMethod: .euler, adapters: [adapter], packedRows: 100))
    }
  }

  func testDiagnosticEnvironmentCanOnlyAgreeWithSavedSelection() throws {
    let flag = "WEETODD_H3_EXPERIMENTAL_SOL", tau = "WEETODD_H3_EXPERIMENTAL_SOL_TAU"
    try H3AttentionPolicy.validateDiagnosticEnvironment([:], saved: .dense)
    try H3AttentionPolicy.validateDiagnosticEnvironment([:], saved: .solExperimental)
    try H3AttentionPolicy.validateDiagnosticEnvironment([flag: "0"], saved: .dense)
    try H3AttentionPolicy.validateDiagnosticEnvironment([flag: "1", tau: "0.5"], saved: .solExperimental)
    for env in [[flag: "1"], [flag: "unknown"], [tau: "0.5"]] {
      XCTAssertThrowsError(try H3AttentionPolicy.validateDiagnosticEnvironment(env, saved: .dense))
    }
    for env in [[flag: "0"], [flag: "1", tau: "nan"], [flag: "1", tau: "0.75"]] {
      XCTAssertThrowsError(try H3AttentionPolicy.validateDiagnosticEnvironment(env, saved: .solExperimental))
    }
    let data = try JSONSerialization.data(withJSONObject: recipe("sol_experimental"))
    XCTAssertEqual(try H3SolDiagnostic.validateSavedRecipe(data: data, environment: [:]), .solExperimental)
    XCTAssertThrowsError(try H3SolDiagnostic.validateSavedRecipe(data: data, environment: [flag: "0"]))
  }
}
