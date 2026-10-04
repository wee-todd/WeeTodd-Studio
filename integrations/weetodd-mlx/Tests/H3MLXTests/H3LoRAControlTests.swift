import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3LoRAControlTests: XCTestCase {
  private let target = "diffusion_model.blocks.0.mlp.fc2"
  private func adapter(rank: Int = 1, profile: String? = nil) throws -> URL {
    let aBytes = rank * 14336 * 2, bBytes = rank * 5376 * 2
    var metadata = ["target_format": "ComfyUI generic LoRA", "qkv_fusion": "block diagonal B"]
    if let profile { metadata["adapter_profile"] = profile }
    let header: [String: Any] = ["__metadata__": metadata,
      target + ".lora_A.weight": ["dtype": "BF16", "shape": [rank, 14336], "data_offsets": [0, aBytes]],
      target + ".lora_B.weight": ["dtype": "BF16", "shape": [5376, rank], "data_offsets": [aBytes, aBytes + bBytes]],
      target + ".alpha": ["dtype": "F32", "shape": [], "data_offsets": [aBytes + bBytes, aBytes + bBytes + 4]]]
    let json = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(json.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(json)
    let start = data.count
    // Only one nonzero pair; alpha/rank compensates for the declared rank.
    var payload = Data(count: aBytes + bBytes + 4)
    payload[0] = 0x80; payload[1] = 0x3f
    payload[aBytes] = 0x80; payload[aBytes + 1] = 0x3f
    var alpha = Float(rank).bitPattern.littleEndian
    withUnsafeBytes(of: &alpha) { payload.replaceSubrange((aBytes + bBytes)..<(aBytes + bBytes + 4), with: $0) }
    data.append(payload)
    XCTAssertEqual(data.count, start + aBytes + bBytes + 4)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    try data.write(to: url); return url
  }
  private func recipe(_ stack: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "h3",
      "prompt": "A robot waves", "components": ["task": "t2va", "transformer": "/tmp/base",
        "text_encoder": "/tmp/qwen", "tokenizer": "/tmp/tokenizer", "video_vae": "/tmp/video", "audio_vae": "/tmp/audio"],
      "config": ["width": 64, "height": 32, "duration_seconds": 2.5, "steps": 5, "seed": 3],
      "conditioning": ["version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated"], "loras": stack])
  }
  private func descriptor(_ changes: [String: Any] = [:]) -> [String: Any] {
    ["path": "/tmp/a.safetensors", "strength": 0.5, "profile": "standard",
      "qkv_layout": "native_interleaved", "start_after_evaluations": 2]
      .merging(changes) { _, new in new }
  }

  func testDescriptorPropagationAndInvalidControlsFailBeforeFiles() throws {
    let request = try H3StudioRecipe.compile(data: recipe(["version": 1, "adapters": [descriptor()]]))
    let entry = try XCTUnwrap(request.loRAAdapters.first)
    XCTAssertEqual(entry.profile, .standard); XCTAssertEqual(entry.qkvLayout, .nativeInterleaved)
    XCTAssertEqual(entry.startAfterEvaluations, 2); XCTAssertEqual(entry.strength, 0.5)
    for bad: [String: Any] in [["profile": true], ["qkv_layout": "guessed"],
      ["start_after_evaluations": true], ["start_after_evaluations": 1.5],
      ["start_after_evaluations": 4], ["adaln_input_grid": "base"], ["unknown": 0],
      ["profile": "turbo"], ["strength": true]] {
      XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(["version": 1, "adapters": [descriptor(bad)]])))
    }
    for version: Any in [true, 1.5, 2] {
      XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(["version": version, "adapters": [descriptor()]])))
    }
    let entries = (0..<8).map { descriptor(["path": "/tmp/\($0).safetensors"]) }
    XCTAssertEqual(try H3StudioRecipe.compile(data: recipe(["version": 1, "adapters": entries])).loRAAdapters.count, 8)
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(["version": 1, "adapters": entries + [descriptor()]])))
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: recipe(["version": 1, "adapters": [descriptor(), descriptor()]])))
    var both = try XCTUnwrap(JSONSerialization.jsonObject(with: recipe(["version": 1, "adapters": [descriptor()]])) as? [String: Any])
    var components = try XCTUnwrap(both["components"] as? [String: Any])
    components["loras"] = [["/tmp/pair.safetensors", 1.0]]; both["components"] = components
    XCTAssertThrowsError(try H3StudioRecipe.compile(data: JSONSerialization.data(withJSONObject: both)))
  }

  func testDeferredSequentialEightAdapterMathAndPreparationPolicy() throws {
    let urls = try (0..<8).map { _ in try adapter() }
    defer { for url in urls { try? FileManager.default.removeItem(at: url) } }
    let controls = try urls.enumerated().map {
      try H3LoRAAdapter(url: $0.element, strength: Float($0.offset + 1) / 8,
        profile: .standard, qkvLayout: .nativeInterleaved, startAfterEvaluations: $0.offset % 3)
    }
    let stack = try H3LoRAStack(adapters: controls)
    try Device.withDefaultDevice(.cpu) {
      var source = [Float](repeating: 0, count: 14336); source[0] = 2
      let input = MLXArray(source, [1, 1, 14336]), base = MLXArray.ones([1, 1, 5376])
      for evaluation: Int? in [nil, 0, 1, 2, 3] {
        stack.evaluation = evaluation
        let actual = try stack.apply(base: base, input: input, target: target).asArray(Float.self)
        let sum = controls.filter { evaluation == nil || $0.startAfterEvaluations <= evaluation! }
          .reduce(Float(1)) { $0 + 2 * $1.strength }
        XCTAssertEqual(actual[0], sum); XCTAssertTrue(actual.dropFirst().allSatisfy { $0 == 1 })
      }
    }
    XCTAssertThrowsError(try H3LoRAStack(adapters: controls + [controls[0]]))
    XCTAssertThrowsError(try H3LoRAStack(adapters: [controls[0], controls[0]]))
  }

  func testExplicitTurboAndRankBoundsHaveHeaderAdmission() throws {
    let standard = try adapter(rank: 512), oversized = try adapter(rank: 513), turbo = try adapter(profile: "turbo")
    defer { for url in [standard, oversized, turbo] { try? FileManager.default.removeItem(at: url) } }
    XCTAssertNoThrow(try H3LoRAFile(url: standard, strength: 1, requestedSteps: 20, profile: .standard, startAfterEvaluations: 2))
    XCTAssertThrowsError(try H3LoRAFile(url: oversized, strength: 1))
    XCTAssertThrowsError(try H3LoRAFile(url: turbo, strength: 1, requestedSteps: 5, startAfterEvaluations: 1))
    XCTAssertThrowsError(try H3LoRAFile(url: turbo, strength: 1, requestedSteps: 5, samplingMethod: .resMultistep))
    XCTAssertThrowsError(try H3LoRAFile(url: standard, strength: 1, requestedSteps: 20, profile: .turbo))
    XCTAssertNoThrow(try H3LoRAFile(url: standard, strength: 1, requestedSteps: 5, profile: .turbo))
  }

  func testNativeAndContiguousQKVAreDifferentExplicitPermutations() throws {
    Device.withDefaultDevice(.cpu) {
      let input = MLXArray([Float(2)], [1, 1, 1]), base = MLXArray.zeros([1, 1, 12])
      let a = MLXArray([Float(1)], [1, 1]), b = MLXArray((1...12).map(Float.init), [12, 1])
      let native = H3LoRAProjection.apply(base: base, input: input, a: a, b: b,
        alpha: 1, strength: 1, reorderQKV: false).asArray(Float.self)
      let contiguous = H3LoRAProjection.apply(base: base, input: input, a: a, b: b,
        alpha: 1, strength: 1, reorderQKV: true, qkvHeads: 2, qkvHeadSize: 2).asArray(Float.self)
      XCTAssertEqual(native, (1...12).map { Float(2 * $0) })
      XCTAssertEqual(contiguous, [2, 4, 10, 12, 18, 20, 6, 8, 14, 16, 22, 24])
      XCTAssertNotEqual(native, contiguous)
    }
  }
  func testSignedAdapterActuallyAppliesNegativeContributionAfterDeferredBoundary() throws {
    let url = try adapter(); defer { try? FileManager.default.removeItem(at: url) }
    let control = try H3LoRAAdapter(url: url, strength: -2, profile: .standard,
      qkvLayout: .nativeInterleaved, startAfterEvaluations: 2)
    let stack = try H3LoRAStack(adapters: [control])
    try Device.withDefaultDevice(.cpu) {
      var values = [Float](repeating: 0, count: 14336); values[0] = 3
      let input = MLXArray(values, [1,1,14336]), base = MLXArray.ones([1,1,5376])
      stack.evaluation = 1
      XCTAssertEqual(try stack.apply(base: base, input: input, target: target).asArray(Float.self),
        [Float](repeating: 1, count: 5376))
      stack.evaluation = 2
      let actual = try stack.apply(base: base, input: input, target: target).asArray(Float.self)
      XCTAssertEqual(actual[0], -5); XCTAssertTrue(actual.dropFirst().allSatisfy { $0 == 1 })
    }
  }
  func testNegativePackedQKVContributionPreservesExplicitPermutationAndSign() {
    Device.withDefaultDevice(.cpu) {
      let input = MLXArray([Float(2)], [1,1,1]), base = MLXArray.ones([1,1,12])
      let a = MLXArray([Float(1)], [1,1]), b = MLXArray((1...12).map(Float.init), [12,1])
      let actual = H3LoRAProjection.apply(base: base, input: input, a: a, b: b,
        alpha: 1, strength: -0.5, reorderQKV: true, qkvHeads: 2, qkvHeadSize: 2).asArray(Float.self)
      XCTAssertEqual(actual, [0,-1,-4,-5,-8,-9,-2,-3,-6,-7,-10,-11])
    }
  }

}
