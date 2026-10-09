import Foundation
import MLX
import MLXRandom
import CryptoKit
import TensorIO
import XCTest
@testable import H3MLX

final class H3TransformerBlockTests: XCTestCase {
  func testPreparedAttentionLifetimeAndRaggedMLPChunksPreserveDenseBlock() throws {
    let rows = 7, width = 4
    let x = MLXRandom.normal([1, rows, width], key: MLXRandom.key(963)).asType(.bfloat16)
    let modulation = (MLXRandom.normal([1, 18 * width], key: MLXRandom.key(964)) * Float(0.1)).asType(.bfloat16)
    let indices = MLXArray((0..<rows).map { Int32($0 % 3) })
    let angles = H3RotaryAngles(rows: rows,
      cosine: MLXArray.ones([1, 1, rows, 4], dtype: .bfloat16),
      sine: MLXArray.zeros([1, 1, rows, 4], dtype: .bfloat16))
    let weights = Dictionary(uniqueKeysWithValues: [
      ("attn.qkv_proj", 24, 4), ("attn.out_proj", 4, 8),
      ("mlp.fc1", 12, 4), ("mlp.fc2", 4, 6),
    ].enumerated().map { i, item in
      (item.0, (MLXRandom.normal([item.1, item.2], key: MLXRandom.key(UInt64(965 + i))) * Float(0.1)).asType(.bfloat16))
    })
    func run(chunk: Int?, bounded: Bool) throws -> (MLXArray, [String]) {
      var order: [String] = []
      let output = try H3TransformerBlock.evaluateKernel(input: x, modulation: modulation,
        modulationIndices: indices, angles: angles,
        hiddenWidth: 4, heads: 2, headWidth: 4, feedWidth: 6, rotaryWidth: 4,
        read: { _, shape in MLXArray.ones(shape, dtype: .bfloat16) },
        project: { activation, name, _, _, _ in
          order.append(name)
          return matmul(activation, weights[name]!.T)
        }, feedRowChunk: chunk, drainAttentionInputs: bounded,
        retireQKV: { order.append("retire_qkv") },
        retireAttention: { order.append("retire_attention") })
      return (output, order)
    }
    let expected = try run(chunk: nil, bounded: false).0.view(dtype: .uint16).asArray(UInt16.self)
    for chunk in [1, 2, 3, 8] {
      let (actual, order) = try run(chunk: chunk, bounded: true)
      XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), expected)
      XCTAssertLessThan(try XCTUnwrap(order.firstIndex(of: "attn.qkv_proj")),
        try XCTUnwrap(order.firstIndex(of: "retire_qkv")))
      XCTAssertLessThan(try XCTUnwrap(order.firstIndex(of: "retire_qkv")),
        try XCTUnwrap(order.firstIndex(of: "attn.out_proj")))
      XCTAssertLessThan(try XCTUnwrap(order.firstIndex(of: "attn.out_proj")),
        try XCTUnwrap(order.firstIndex(of: "retire_attention")))
      XCTAssertLessThan(try XCTUnwrap(order.firstIndex(of: "retire_attention")),
        try XCTUnwrap(order.firstIndex(of: "mlp.fc1")))
      XCTAssertEqual(order.filter { $0 == "mlp.fc1" }.count, (rows + chunk - 1) / chunk)
    }
    XCTAssertThrowsError(try run(chunk: 0, bounded: true))
  }

  func testCancellationAtMLPChunkBoundaryStopsBeforeTheNextProjection() async throws {
    let task = Task { () throws -> Bool in
      let input = MLXArray.ones([1, 5, 2], dtype: .bfloat16)
      let mod = MLXArray.ones([1, 36], dtype: .bfloat16) * Float(0.1)
      var completedChunks = 0
      do {
        _ = try H3TransformerBlock.evaluateKernel(input: input, modulation: mod,
          modulationIndices: MLXArray.zeros([5], dtype: .int32),
          angles: H3RotaryAngles(rows: 5,
            cosine: MLXArray.ones([1, 1, 5, 2], dtype: .bfloat16),
            sine: MLXArray.zeros([1, 1, 5, 2], dtype: .bfloat16)),
          hiddenWidth: 2, heads: 1, headWidth: 2, feedWidth: 2, rotaryWidth: 2,
          read: { _, shape in MLXArray.ones(shape, dtype: .bfloat16) },
          project: { x, name, rows, columns, _ in
            if name == "mlp.fc2" { completedChunks += 1; withUnsafeCurrentTask { $0?.cancel() } }
            return matmul(x, MLXArray.ones([rows, columns], dtype: .bfloat16).T)
          }, feedRowChunk: 2)
        return false
      } catch is CancellationError { return completedChunks == 1 }
    }
    let stopped = try await task.value
    XCTAssertTrue(stopped)
  }

  func testInstalledAffineProjectionLoadingComparison() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let outputPath = environment["WEETODD_H3_AFFINE_LOAD_OUTPUT"],
      let checkpointPath = environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed affine weight-loading comparison.")
    }
    guard !FileManager.default.fileExists(atPath: outputPath) else {
      throw H3CheckpointError.invalid("Affine loading probe output already exists.")
    }
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    let layout = try H3CheckpointLayout(url: checkpoint)
    let url = try H3CheckpointSource.fileURL(checkpoint, block: 0)
    let file = try SafeTensorFile(url: url)
    let stem = layout.prefix + "blocks.0.mlp.fc1"
    let input = MLXRandom.normal([1,9909,5376], key: MLXRandom.key(901)).asType(.bfloat16)
    eval(input)
    let previous = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previous }
    var reports: [[String: Any]] = []
    var expected: String?
    for mapped in [true, false] {
      var times: [Double] = []
      var output: MLXArray?
      Memory.clearCache(); Memory.peakMemory = Memory.activeMemory
      for _ in 0..<4 {
        let start = CFAbsoluteTimeGetCurrent()
        let weight: H3QwenQ8Projection
        if mapped { weight = try H3QwenQ8Projection(file: file, name: stem + ".weight") }
        else {
          let arrays = try loadArrays(url: url)
          weight = try H3QwenQ8Projection(packed: XCTUnwrap(arrays[stem + ".weight"]),
            scales: XCTUnwrap(arrays[stem + ".scales"]), biases: XCTUnwrap(arrays[stem + ".biases"]), columns: 5376)
        }
        output = try weight.project(input); eval(output!); Stream.gpu.synchronize()
        times.append(CFAbsoluteTimeGetCurrent() - start)
      }
      let peak = Memory.peakMemory
      let result = try XCTUnwrap(output)
      var digest = SHA256()
      for start in stride(from: 0, to: 9909, by: 256) {
        result[0, start..<min(start+256,9909), 0..<28672].asArray(Float.self)
          .withUnsafeBytes { digest.update(data: Data($0)) }
      }
      let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
      if let expected { XCTAssertEqual(hash, expected) } else { expected = hash }
      reports.append(["loader": mapped ? "scoped-mapping-copy" : "mlx-file-backed",
        "seconds": times, "peakMLXBytes": peak, "outputFloat32SHA256": hash])
      try file.checkUnchanged(at: url)
    }
    try JSONSerialization.data(withJSONObject: reports, options: [.prettyPrinted,.sortedKeys])
      .write(to: URL(fileURLWithPath: outputPath))
  }
  /// One complete trained block, including weight reads and all projections.
  /// Boundary timings deliberately synchronize and are diagnostic only; the
  /// four ordinary calls below use the uninstrumented production execution.
  func testInstalledFastVSABlockFrozenParityAndTiming() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let outputPath = environment["WEETODD_H3_VSA_BLOCK_OUTPUT"],
      let checkpointPath = environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed trained VSA complete-block probe.")
    }
    guard !FileManager.default.fileExists(atPath: outputPath) else {
      throw H3CheckpointError.invalid("VSA block probe output already exists.")
    }
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    XCTAssertEqual(try H3CheckpointLayout(url: checkpoint).fastVariant, .vsaV1)
    let tiles = try H3FastTiles(prefixSegments: [171, 414], videoGrid: [37, 12, 21])
    let input = MLXRandom.normal([1, tiles.rows, 5376], key: MLXRandom.key(901)).asType(.bfloat16)
    let modulation = (MLXRandom.normal([1, 96768], key: MLXRandom.key(902)) * 0.05).asType(.bfloat16)
    let indices = MLXArray((0..<tiles.rows).map { Int32($0 < 171 ? 0 : ($0 < 585 ? 1 : 2)) })
    let positions = MLXArray((0..<tiles.rows).flatMap { row -> [Float] in
      if row < 585 { return [Float(row), 0, 0] }
      let video = row - 585
      return [Float(video / 252), Float((video / 21) % 12), Float(video % 21)]
    }, [tiles.rows, 3])
    eval([input, modulation, indices, positions])
    let angles = try H3TransformerBlock.prepareRotaryAngles(checkpointURL: checkpoint, positions: positions)
    let previousLimit = Memory.cacheLimit
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    Memory.peakMemory = Memory.activeMemory
    var seconds: [Double] = []
    var output: MLXArray?
    for _ in 0..<4 {
      let start = CFAbsoluteTimeGetCurrent()
      output = try H3TransformerBlock.evaluate(checkpointURL: checkpoint, index: 0,
        input: input, modulation: modulation, modulationIndices: indices,
        positions: positions, rotaryAngles: angles, fastTiles: tiles, observe: { _, _ in })
      Stream.gpu.synchronize()
      seconds.append(CFAbsoluteTimeGetCurrent() - start)
    }
    let peak = Memory.peakMemory
    XCTAssertLessThanOrEqual(peak, 2_791_728_742,
      "Consumer-owned modulation must retain the fixed 2.6 GiB complete-block bound.")
    if let maximum = environment["WEETODD_H3_VSA_BLOCK_MAX_WARM_SECONDS"].flatMap(Double.init) {
      XCTAssertLessThanOrEqual(seconds.dropFirst().sorted()[1], maximum)
    }
    let values = try XCTUnwrap(output).asArray(Float.self)
    let hash = values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    if let expected = environment["WEETODD_H3_VSA_BLOCK_EXPECTED_SHA256"] { XCTAssertEqual(hash, expected) }
    if let budget = environment["WEETODD_H3_VSA_BLOCK_MAX_MLX_BYTES"].flatMap(Int.init) { XCTAssertLessThanOrEqual(peak, budget) }
    var stages: [[String: Any]] = []
    Memory.peakMemory = Memory.activeMemory
    var start = CFAbsoluteTimeGetCurrent()
    let diagnostic = try H3TransformerBlock.evaluate(checkpointURL: checkpoint, index: 0,
      input: input, modulation: modulation, modulationIndices: indices,
      positions: positions, rotaryAngles: angles, fastTiles: tiles, observe: { name, value in
        eval(value); Stream.gpu.synchronize()
        let end = CFAbsoluteTimeGetCurrent()
        stages.append(["boundary": name, "secondsSincePriorBoundary": end - start,
          "activeMLXBytes": Memory.activeMemory, "peakMLXBytes": Memory.peakMemory])
        Memory.peakMemory = Memory.activeMemory
        start = end
      })
    XCTAssertEqual(diagnostic.asArray(Float.self), values)
    let report: [String: Any] = ["scope": "one installed trained block; no whole-render speed claim",
      "inputShape": input.shape, "inputSeeds": [901, 902], "outputFloat32SHA256": hash,
      "secondsIncludingFirstCompilation": seconds, "warmSeconds": Array(seconds.dropFirst()),
      "peakMLXBytes": peak, "instrumentedBoundaryTimings": stages]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: outputPath))
  }
  func testInstalledRotaryFrequencyKeepsFloat32PhaseUntilTrig() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set an installed H3 checkpoint for rotary phase parity.")
    }
    let positions = MLXArray([Float(320), 0, 0], [1, 3])
    let angles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: URL(fileURLWithPath: path), positions: positions)
    // The released rope.inv_freq[1] is 0.56234133. Python multiplies in FP32,
    // takes cosine, and only then rounds to the attention query's BF16 dtype.
    let expected = MLXArray([cos(Float(320) * Float(0.56234133))])
      .asType(.bfloat16).asType(.float32).item(Float.self)
    let actual = angles.cosine[0, 0, 0, 1].asType(.float32).item(Float.self)
    XCTAssertEqual(actual, expected, accuracy: 0.004)
  }

  func testPreparedRotaryAnglesMatchInlineBlockOutput() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set an installed H3 checkpoint for rotary block parity.")
    }
    let checkpoint = URL(fileURLWithPath: path)
    let rows = 4
    let input = MLXRandom.normal([1, rows, 5376], key: MLXRandom.key(21))
      .asType(.bfloat16)
    let modulation = MLXRandom.normal([1, 96768], key: MLXRandom.key(22))
      .asType(.bfloat16) * 0.05
    let indices = MLXArray.zeros([rows], dtype: .int32)
    let positions = MLXArray([Float](repeating: 0, count: rows * 3),
      [rows, 3])
    let ordinary = try H3TransformerBlock.evaluate(checkpointURL: checkpoint,
      index: 0, input: input, modulation: modulation,
      modulationIndices: indices, positions: positions)
    let angles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: checkpoint, positions: positions)
    let prepared = try H3TransformerBlock.evaluate(checkpointURL: checkpoint,
      index: 0, input: input, modulation: modulation,
      modulationIndices: indices, positions: positions,
      rotaryAngles: angles, observe: { _, _ in })
    let error = max(abs(ordinary.asType(.float32) - prepared.asType(.float32)))
      .item(Float.self)
    XCTAssertEqual(error, 0)
  }

  func testInstalledRotatedBlockComparisonDiagnostic() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_ROTATED_BLOCK_PROBE"] == "1",
      let checkpointPath = environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Enable the installed rotated-block diagnostic explicitly.")
    }
    let rows = 4
    let input = MLXRandom.normal([1, rows, 5376], key: MLXRandom.key(2026))
      .asType(.bfloat16)
    let modulation = MLXRandom.normal([1, 96768], key: MLXRandom.key(2027))
      .asType(.bfloat16) * 0.05
    let indices = MLXArray.zeros([rows], dtype: .int32)
    let positions = MLXArray.zeros([rows, 3], dtype: .float32)
    eval([input, modulation, indices, positions])
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    var decodedStages: [String: [Float]] = [:]
    let decodedStart = Date()
    let decoded = try H3TransformerBlock.evaluate(checkpointURL: checkpoint,
      index: 0, input: input, modulation: modulation,
      modulationIndices: indices, positions: positions,
      projectionMode: .weightDecoded) { stage, value in
        decodedStages[stage] = value.asType(.float32).asArray(Float.self)
      }
    let decodedSeconds = Date().timeIntervalSince(decodedStart)
    let rotatedStart = Date()
    let rotated = try H3TransformerBlock.evaluate(checkpointURL: checkpoint,
      index: 0, input: input, modulation: modulation,
      modulationIndices: indices, positions: positions,
      projectionMode: .activationRotated) { stage, value in
        let actual = value.asType(.float32).asArray(Float.self)
        let expected = decodedStages[stage]!
        let errors = zip(actual, expected).map { abs($0 - $1) }
        print("H3_ROTATED_BLOCK_STAGE stage=\(stage) max=\(errors.max()!) mean=\(errors.reduce(0, +) / Float(errors.count))")
      }
    let rotatedSeconds = Date().timeIntervalSince(rotatedStart)
    let error = abs(rotated.asType(.float32) - decoded.asType(.float32))
    print("H3_ROTATED_BLOCK_PROBE rows=\(rows) max=\(max(error).item(Float.self)) mean=\(mean(error).item(Float.self)) decoded_seconds=\(decodedSeconds) rotated_seconds=\(rotatedSeconds)")
    XCTAssertEqual(rotated.shape, decoded.shape)
  }

  func testInstalledRef2VABlockWithTurboProfile() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_REF2VA_BLOCK_PROBE"] == "1" else {
      throw XCTSkip("Enable the installed Ref2VA block probe explicitly.")
    }
    let environment = ProcessInfo.processInfo.environment
    guard let checkpointPath = environment["WEETODD_H3_TEST_CHECKPOINT"],
      let adapterPath = environment["WEETODD_H3_TEST_TURBO_LORA"],
      FileManager.default.isReadableFile(atPath: checkpointPath),
      FileManager.default.isReadableFile(atPath: adapterPath) else {
      throw XCTSkip("Set installed H3 checkpoint and Turbo LoRA paths.")
    }
    let rows = 13_315
    let input = MLXArray.zeros([1, rows, 5376], dtype: .bfloat16)
    let modulation = MLXArray.zeros([1, 96768], dtype: .bfloat16)
    let indices = MLXArray.zeros([rows], dtype: .int32)
    let positions = MLXArray.zeros([rows, 3], dtype: .float32)
    eval([input, modulation, indices, positions])
    let adapter = try H3LoRAFile(url: URL(fileURLWithPath: adapterPath), strength: 1)
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    _ = try H3CheckpointLayout(url: checkpoint)
    let anglesStarted = Date()
    let angles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: checkpoint, positions: positions)
    let anglesSeconds = Date().timeIntervalSince(anglesStarted)
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = Date()
    var previous = started
    var stages: [(String, Double)] = []
    let output = try H3TransformerBlock.evaluate(
      checkpointURL: checkpoint, index: 0,
      input: input, modulation: modulation, modulationIndices: indices,
      positions: positions, lora: adapter, rotaryAngles: angles) { stage, value in
        eval(value)
        let now = Date()
        stages.append((stage, now.timeIntervalSince(previous)))
        previous = now
      }
    XCTAssertEqual(output.shape, [1, rows, 5376])
    XCTAssertTrue(output[0, 0, 0].item(Float.self).isFinite)
    print("H3_REF2VA_BLOCK_PROBE rows=\(rows) angles_seconds=\(anglesSeconds) block_seconds=\(Date().timeIntervalSince(started)) peak_mlx_bytes=\(Memory.peakMemory)")
    for (stage, seconds) in stages {
      print("H3_REF2VA_BLOCK_STAGE stage=\(stage) seconds=\(seconds)")
    }
  }

  func testInstalledRef2VABlockRowWindowComparison() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_BLOCK_WINDOW_PROBE"] == "1",
      let checkpointPath = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let adapterPath = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_TURBO_LORA"] else {
      throw XCTSkip("Enable the installed H3 block row-window probe explicitly.")
    }
    let rows = 13_315
    let input = MLXArray.zeros([1, rows, 5376], dtype: .bfloat16)
    let modulation = MLXArray.zeros([1, 96768], dtype: .bfloat16)
    let indices = MLXArray.zeros([rows], dtype: .int32)
    let positions = MLXArray.zeros([rows, 3], dtype: .float32)
    eval([input, modulation, indices, positions])
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    let adapter = try H3LoRAFile(url: URL(fileURLWithPath: adapterPath), strength: 1)
    let angles = try H3TransformerBlock.prepareRotaryAngles(
      checkpointURL: checkpoint, positions: positions)
    var reference: [Float]?
    for window in [8192, 16384] {
      Memory.clearCache()
      Memory.peakMemory = Memory.activeMemory
      let started = Date()
      let output = try H3TransformerBlock.evaluate(
        checkpointURL: checkpoint, index: 0,
        input: input, modulation: modulation, modulationIndices: indices,
        positions: positions, lora: adapter, rotaryAngles: angles,
        rowWindow: window) { _, _ in }
      let elapsed = Date().timeIntervalSince(started)
      let sample = output.asType(.float32).asArray(Float.self)
      if let reference {
        XCTAssertEqual(sample, reference)
      } else {
        reference = sample
      }
      print("H3_BLOCK_WINDOW_PROBE window=\(window) seconds=\(elapsed) "
        + "peak_mlx_bytes=\(Memory.peakMemory)")
    }
  }

  func testInstalledBlockAtRepresentativeProductionRowCount() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_PRODUCTION_BLOCK_PROBE"] == "1" else {
      throw XCTSkip("Enable the installed H3 production-row block probe explicitly.")
    }
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      FileManager.default.isReadableFile(atPath: path) else {
      throw XCTSkip("Set WEETODD_H3_TEST_CHECKPOINT to the installed transformer.")
    }
    let geometry = try H3Geometry(width: 768, height: 768, durationSeconds: 5)
    let layout = try H3PackedLayout(geometry: geometry,
      textTags: [Int32](repeating: 0, count: 16), anchors: [])
    let count = layout.tags.count
    XCTAssertEqual(count, 21_742)
    let input = MLXArray.zeros([1, count, 5376], dtype: .bfloat16)
    let modulation = MLXArray.zeros([1, 96768], dtype: .bfloat16)
    let indices = MLXArray.zeros([count], dtype: .int32)
    let positions = MLXArray.zeros([count, 3], dtype: .float32)
    eval([input, modulation, indices, positions])
    let checkpoint = URL(fileURLWithPath: path)
    let validationStarted = Date()
    _ = try H3CheckpointLayout(url: checkpoint)
    let validationSeconds = Date().timeIntervalSince(validationStarted)
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = Date()
    var previous = started
    var stages: [(String, Double)] = []
    let result = try H3TransformerBlock.evaluate(
      checkpointURL: checkpoint, index: 0,
      input: input, modulation: modulation,
      modulationIndices: indices, positions: positions) { stage, value in
        eval(value)
        let now = Date()
        stages.append((stage, now.timeIntervalSince(previous)))
        previous = now
      }
    let seconds = Date().timeIntervalSince(started)
    XCTAssertEqual(result.shape, [1, count, 5376])
    XCTAssertTrue(result[0, 0, 0].item(Float.self).isFinite)
    print("H3_PRODUCTION_BLOCK_PROBE rows=\(count) layout_validation_seconds=\(validationSeconds) block_seconds=\(seconds) peak_mlx_bytes=\(Memory.peakMemory)")
    for (stage, elapsed) in stages {
      print("H3_PRODUCTION_BLOCK_STAGE stage=\(stage) seconds=\(elapsed)")
    }
  }

  func testActivationRotatedBlockZeroStaysCloseToReferenceOutput() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_ROTATED_QUALITY_TEST"] == "1" else {
      throw XCTSkip("Activation-rotated ConvRot is experimental and fails the installed block quality gate.")
    }
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let refinerFixture = ProcessInfo.processInfo.environment["WEETODD_H3_REFINER_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_BLOCK_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint, refiner, AdaLN and block oracle paths.")
    }
    func bf16(_ url: URL, shape: [Int]) throws -> MLXArray {
      try Data(contentsOf: url).withUnsafeBytes {
        MLXArray($0, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let input = try bf16(URL(fileURLWithPath: refinerFixture)
      .appendingPathComponent("final.u16"), shape: [1, 4, 5376])
    let modulation = try bf16(URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16"), shape: [3, 96768])
    let root = URL(fileURLWithPath: fixture)
    let indices = try Data(contentsOf: root.appendingPathComponent("indices.i32"))
      .withUnsafeBytes { MLXArray($0, [4], type: Int32.self) }
    let positions = try Data(contentsOf: root.appendingPathComponent("positions.f32"))
      .withUnsafeBytes { MLXArray($0, [4, 3], type: Float.self) }
    let actual = try H3TransformerBlock.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), index: 0,
      input: input, modulation: modulation, modulationIndices: indices,
      positions: positions, projectionMode: .activationRotated)
    let expected = try bf16(root.appendingPathComponent("output.u16"),
      shape: [1, 4, 5376])
    let error = abs(actual.asType(.float32) - expected.asType(.float32))
    let maximum = max(error).item(Float.self)
    let average = mean(error).item(Float.self)
    print("H3 activation-rotated block 0 max=\(maximum) mean=\(average)")
    XCTAssertLessThan(maximum, 1)
    XCTAssertLessThan(average, 0.05)
  }

  func testInstalledBlockZeroMatchesReferenceAtEveryBoundary() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let refinerFixture = ProcessInfo.processInfo.environment["WEETODD_H3_REFINER_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_BLOCK_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint, refiner, AdaLN and block oracle paths.")
    }
    func bf16(_ url: URL, shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: url)
      return data.withUnsafeBytes {
        MLXArray($0, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
    }
    let input = try bf16(URL(fileURLWithPath: refinerFixture)
      .appendingPathComponent("final.u16"), shape: [1, 4, 5376])
    let modulation = try bf16(URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16"), shape: [3, 96768])
    let root = URL(fileURLWithPath: fixture)
    let indices = try Data(contentsOf: root.appendingPathComponent("indices.i32"))
      .withUnsafeBytes { MLXArray($0, [4], type: Int32.self) }
    let positions = try Data(contentsOf: root.appendingPathComponent("positions.f32"))
      .withUnsafeBytes { MLXArray($0, [4, 3], type: Float.self) }
    _ = try H3CheckpointLayout(url: URL(fileURLWithPath: checkpoint))
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = Date()
    let output = try H3TransformerBlock.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), index: 0,
      input: input, modulation: modulation, modulationIndices: indices,
      positions: positions) { name, value in
      let expected = try bf16(root.appendingPathComponent("\(name).u16"),
        shape: value.shape)
      XCTAssertEqual(max(abs(value.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, name)
    }
    print("H3 installed exact block with boundary checks seconds=\(Date().timeIntervalSince(started)) "
      + "peak_mlx=\(Memory.peakMemory)")
    XCTAssertEqual(output.shape, [1, 4, 5376])
  }
}
