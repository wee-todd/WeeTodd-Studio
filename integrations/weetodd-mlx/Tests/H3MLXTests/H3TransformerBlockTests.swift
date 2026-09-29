import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3TransformerBlockTests: XCTestCase {
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
