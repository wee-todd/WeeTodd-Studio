import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3LoRAControlTests: XCTestCase {
  private let target = "diffusion_model.blocks.0.mlp.fc2"
  private func adapter(rank: Int = 1, profile: String? = nil, target requestedTarget: String? = nil) throws -> URL {
    let target = requestedTarget ?? self.target
    let inputWidth = target.hasSuffix("attn.qkv_proj") ? 5376 : 14336
    let outputWidth = target.hasSuffix("attn.qkv_proj") ? 21504 : 5376
    let aBytes = rank * inputWidth * 2, bBytes = rank * outputWidth * 2
    var metadata = ["target_format": "ComfyUI generic LoRA", "qkv_fusion": "block diagonal B"]
    if let profile { metadata["adapter_profile"] = profile }
    let header: [String: Any] = ["__metadata__": metadata,
      target + ".lora_A.weight": ["dtype": "BF16", "shape": [rank, inputWidth], "data_offsets": [0, aBytes]],
      target + ".lora_B.weight": ["dtype": "BF16", "shape": [outputWidth, rank], "data_offsets": [aBytes, aBytes + bBytes]],
      target + ".alpha": ["dtype": "F32", "shape": [], "data_offsets": [aBytes + bBytes, aBytes + bBytes + 4]]]
    let json = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(json.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(json)
    let start = data.count
    // Only one nonzero pair; alpha/rank compensates for the declared rank.
    var payload = Data(count: aBytes + bBytes + 4)
    payload[0] = 0x80; payload[1] = 0x3f
    payload[aBytes] = 0x80; payload[aBytes + 1] = 0x3f
    if target.hasSuffix("attn.qkv_proj") {
      for row in 0..<outputWidth {
        // Distinct BF16 values across head/Q/K/V groups exercise the permutation.
        let bits = UInt16(0x3f00 + (row / 128) % 127)
        let offset = aBytes + row * rank * 2
        payload[offset] = UInt8(truncatingIfNeeded: bits)
        payload[offset + 1] = UInt8(truncatingIfNeeded: bits >> 8)
      }
    }
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
        let queued = try stack.applyQueued(base: base, input: input, target: target).asArray(Float.self)
        XCTAssertEqual(queued, actual, "Queued adapters must preserve order, strength and activation boundaries.")
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

  func testPreparedBlockPairsReuseAcrossRowChunksAndPreserveOrderedBF16Math() throws {
    let urls = try (0..<8).map { _ in try adapter() }
    defer { for url in urls { try? FileManager.default.removeItem(at: url) } }
    let controls = try urls.enumerated().map {
      try H3LoRAAdapter(url: $0.element, strength: Float($0.offset - 3) / 7,
        profile: .standard, qkvLayout: .nativeInterleaved, startAfterEvaluations: $0.offset % 3)
    }
    let stack = try H3LoRAStack(adapters: controls)
    try Device.withDefaultDevice(.cpu) {
      for evaluation: Int? in [nil, 0, 1, 2] {
        stack.evaluation = evaluation
        let scope = try XCTUnwrap(stack.prepareBlock(index: 0))
        defer { scope.close() }
        XCTAssertEqual(scope.storageBytes, 0, "Binding must not eagerly load any adapter pairs.")
        for rows in [3, 2, 1, 3, 2] {
          var values = [Float](repeating: 0, count: rows * 14336)
          for row in 0..<rows { values[row * 14336] = Float(row + 1) / 3 }
          let input = MLXArray(values, [1, rows, 14336]).asType(.bfloat16)
          let base = MLXArray.full([1, rows, 5376], values: MLXArray(Float(0.7)), dtype: .bfloat16)
          let reference = try stack.applyQueued(base: base, input: input, target: target).asArray(Float.self)
          let actual = try scope.applyQueued(base: base, input: input, target: target).asArray(Float.self)
          XCTAssertEqual(actual, reference)
        }
        let activeCount = controls.filter { $0.isActive(evaluation: evaluation) }.count
        XCTAssertEqual(scope.acquiredPairCount, activeCount)
        XCTAssertEqual(scope.residentPairCount, activeCount)
        XCTAssertEqual(scope.storageBytes, activeCount * (14336 + 5376) * 2)
        scope.retire(target: target)
        XCTAssertEqual(scope.storageBytes, 0)
        XCTAssertEqual(scope.residentPairCount, 0)
        let input = MLXArray.zeros([1, 1, 14336], dtype: .bfloat16)
        let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
        XCTAssertThrowsError(try scope.applyQueued(base: base, input: input, target: target))
        scope.close(); scope.close()
        XCTAssertTrue(scope.isClosed)
        XCTAssertThrowsError(try scope.applyQueued(base: base, input: input, target: target))
      }
    }
  }

  func testPreparedScopeCapturesActivationAndRejectsForeignBlocksAndChangedWidths() throws {
    let url = try adapter()
    defer { try? FileManager.default.removeItem(at: url) }
    let control = try H3LoRAAdapter(url: url, strength: 1, profile: .standard,
      qkvLayout: .nativeInterleaved, startAfterEvaluations: 2)
    let stack = try H3LoRAStack(adapters: [control])
    try Device.withDefaultDevice(.cpu) {
      stack.evaluation = 1
      let inactive = try XCTUnwrap(stack.prepareBlock(index: 0)); defer { inactive.close() }
      stack.evaluation = 2
      let active = try XCTUnwrap(stack.prepareBlock(index: 0)); defer { active.close() }
      stack.evaluation = 0
      var values = [Float](repeating: 0, count: 14336); values[0] = 2
      let input = MLXArray(values, [1, 1, 14336]).asType(.bfloat16)
      let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
      XCTAssertEqual(try inactive.applyQueued(base: base, input: input, target: target).asArray(Float.self),
        base.asArray(Float.self))
      XCTAssertEqual(inactive.acquiredPairCount, 0)
      XCTAssertEqual(try active.applyQueued(base: base, input: input, target: target)[0,0,0].item(Float.self), 2)
      XCTAssertThrowsError(try active.applyQueued(base: base, input: input,
        target: "diffusion_model.blocks.1.mlp.fc2"))
      XCTAssertThrowsError(try active.applyQueued(base: base,
        input: MLXArray.zeros([1, 1, 14335], dtype: .bfloat16), target: target))
      XCTAssertEqual(active.acquiredPairCount, 1)
    }
  }

  func testPreparedRawQKVPairsKeepBothExplicitLayoutsExact() throws {
    let qkvTarget = "diffusion_model.blocks.0.attn.qkv_proj"
    let url = try adapter(target: qkvTarget)
    defer { try? FileManager.default.removeItem(at: url) }
    try Device.withDefaultDevice(.cpu) {
      for layout: H3LoRAQKVLayout in [.nativeInterleaved, .contiguousQKV] {
        let file = try H3LoRAFile(url: url, strength: -0.5, qkvLayout: layout)
        let scope = try XCTUnwrap(file.prepareBlock(index: 0)); defer { scope.close() }
        var values = [Float](repeating: 0, count: 5376); values[0] = 2
        let input = MLXArray(values, [1, 1, 5376]).asType(.bfloat16)
        let base = MLXArray.ones([1, 1, 21504], dtype: .bfloat16)
        for reorder in [true, false, true] {
          XCTAssertEqual(try scope.applyQueued(base: base, input: input,
            target: qkvTarget, reorderQKV: reorder).asArray(Float.self),
            try file.applyQueued(base: base, input: input,
              target: qkvTarget, reorderQKV: reorder).asArray(Float.self))
        }
        XCTAssertEqual(scope.acquiredPairCount, 1)
      }
    }
  }

  func testInstalledPreparedWindowRetiresAdapterScopeWithItsOwner() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_PREPARED_BLOCK_TESTS"] == "1",
      let checkpoint = env["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed prepared-window adapter ownership qualification.")
    }
    let url = try adapter()
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try H3LoRAFile(url: url, strength: 1)
    let window = try H3PreparedBlockWindow(checkpointURL: URL(fileURLWithPath: checkpoint),
      blockCount: 1, windowSize: 1)
    defer { window.close() }
    let owner = try window.weights(for: 0)
    let baseBytes = owner.storageBytes
    let scope = try XCTUnwrap(try owner.prepareLoRA(file) as? H3LoRABlock)
    XCTAssertTrue(scope === (try owner.prepareLoRA(file) as? H3LoRABlock))
    try Device.withDefaultDevice(.cpu) {
      let input = MLXArray.ones([1, 1, 14336], dtype: .bfloat16)
      let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
      _ = try scope.applyQueued(base: base, input: input, target: target).asArray(Float.self)
    }
    XCTAssertEqual(owner.storageBytes, baseBytes + scope.storageBytes)
    XCTAssertGreaterThan(scope.storageBytes, 0)
    try window.finish(index: 0)
    XCTAssertTrue(scope.isClosed)
    XCTAssertTrue(owner.isClosed)
    XCTAssertEqual(scope.storageBytes, 0)
    XCTAssertEqual(owner.storageBytes, 0)
    XCTAssertEqual(window.storageBytes, 0)
    XCTAssertThrowsError(try owner.prepareLoRA(file))
  }

  func testPreparedCachedPairStillRejectsSourceMutation() throws {
    let url = try adapter()
    defer { try? FileManager.default.removeItem(at: url) }
    try Device.withDefaultDevice(.cpu) {
      let file = try H3LoRAFile(url: url, strength: 1)
      let scope = try XCTUnwrap(file.prepareBlock(index: 0)); defer { scope.close() }
      let input = MLXArray.ones([1, 1, 14336], dtype: .bfloat16)
      let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
      _ = try scope.applyQueued(base: base, input: input, target: target).asArray(Float.self)
      let handle = try FileHandle(forWritingTo: url)
      try handle.seekToEnd(); try handle.write(contentsOf: Data([0])); try handle.close()
      XCTAssertThrowsError(try scope.applyQueued(base: base, input: input, target: target))
      scope.close()
      XCTAssertEqual(scope.storageBytes, 0)
    }
  }

  func testPreparedCancellationBeforeAcquisitionAndErrorUnwindReleaseScope() async throws {
    let url = try adapter()
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try H3LoRAFile(url: url, strength: 1)
    let targetName = target
    let task = Task {
      let taskFile = try H3LoRAFile(url:url,strength:1)
      let cancelledScope = try XCTUnwrap(taskFile.prepareBlock(index:0))
      withUnsafeCurrentTask { $0?.cancel() }
      let wasCancelled = try Device.withDefaultDevice(.cpu) {
        defer { cancelledScope.close() }
        let input = MLXArray.ones([1, 1, 14336], dtype: .bfloat16)
        let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
        do {
          _ = try cancelledScope.applyQueued(base:base,input:input,target:targetName)
          return false
        } catch is CancellationError { return true }
      }
      return (wasCancelled,cancelledScope.isClosed,cancelledScope.acquiredPairCount,
        cancelledScope.storageBytes)
    }
    let cancelled = try await task.value
    XCTAssertTrue(cancelled.0)
    XCTAssertTrue(cancelled.1)
    XCTAssertEqual(cancelled.2, 0)
    XCTAssertEqual(cancelled.3, 0)

    let scope = try XCTUnwrap(file.prepareBlock(index: 0))
    enum Stop: Error { case injected }
    XCTAssertThrowsError(try Device.withDefaultDevice(.cpu) {
      defer { scope.close() }
      let input = MLXArray.ones([1, 1, 14336], dtype: .bfloat16)
      let base = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
      _ = try scope.applyQueued(base: base, input: input, target: target).asArray(Float.self)
      throw Stop.injected
    })
    XCTAssertTrue(scope.isClosed)
    XCTAssertEqual(scope.storageBytes, 0)
    XCTAssertEqual(scope.residentPairCount, 0)
  }

}
