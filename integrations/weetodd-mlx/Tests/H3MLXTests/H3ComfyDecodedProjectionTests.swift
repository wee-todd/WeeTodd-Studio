import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3ComfyDecodedProjectionTests: XCTestCase {
  func testInstalledQKVDecodeMatmulAndTurboSplitProfile() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_QKV_SPLIT_PROBE"] == "1",
      let checkpointPath = environment["WEETODD_H3_TEST_CHECKPOINT"],
      let adapterPath = environment["WEETODD_H3_TEST_TURBO_LORA"] else {
      throw XCTSkip("Enable the installed QKV split probe and select its checkpoint and Turbo adapter explicitly.")
    }
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    let layout = try H3CheckpointLayout(url: checkpoint)
    guard layout.curveRank == nil, layout.quantizedProjections == 250 else {
      throw H3CheckpointError.invalid("QKV split profiling requires the full-width Comfy INT8 checkpoint.")
    }
    let file = try SafeTensorFile(url: checkpoint)
    let adapter = try H3LoRAFile(url: URL(fileURLWithPath: adapterPath), strength: 1)
    let rows = 5136, columns = 5376, outputColumns = 21504
    // Match H3TransformerBlock.evaluate, rather than the standalone loader's
    // smaller default. No sampler setting or weighted implementation changes.
    let rowWindow = 16384
    let target = "diffusion_model.blocks.0.attn.qkv_proj"
    let previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousCacheLimit }
    Stream.gpu.synchronize(); Memory.clearCache()
    let before = Memory.activeMemory
    var phases: [[String: Any]] = []
    var maximumError: Float = .infinity, adapterDifference: Float = 0
    try autoreleasepool {
      let input = autoreleasepool { () -> MLXArray in
        let values = (0..<(rows * columns)).map { index -> Float in
          let numerator = (index * 31) % 255 - 127
          return Float(numerator == 0 ? 1 : numerator) / 256
        }
        let value = MLXArray(values, [1, rows, columns]).asType(.bfloat16)
        eval(value); return value
      }
      XCTAssertTrue((input .!= 0).all().item(Bool.self), "Every synthetic activation must be nonzero.")
      XCTAssertTrue(MLX.isFinite(input).all().item(Bool.self))
      func measure(_ name: String, _ operation: () throws -> MLXArray) rethrows -> MLXArray {
        Stream.gpu.synchronize(); Memory.clearCache()
        let residentBefore = Memory.activeMemory
        Memory.peakMemory = residentBefore
        let started = Date()
        let output = try operation()
        eval(output); Stream.gpu.synchronize()
        phases.append(["phase": name, "seconds": Date().timeIntervalSince(started),
          "active_bytes_before": residentBefore, "active_bytes_after": Memory.activeMemory,
          "peak_mlx_bytes": Memory.peakMemory])
        return output
      }
      let weight = try measure("checkpoint_decode") {
        try H3ComfyDecodedProjection.load(file: file, checkpointURL: checkpoint,
          name: layout.prefix + "blocks.0.attn.qkv_proj.weight", rows: outputColumns,
          columns: columns, reorderQKV: true, rowWindow: rowWindow)
      }
      let base = measure("base_bf16_matmul") { matmul(input, weight.T) }
      let split = try measure("turbo_read_and_apply") {
        try adapter.apply(base: base, input: input, target: target, reorderQKV: true)
      }
      // The unsplit expression is exactly the block projector's current call
      // sequence. It checks that the added evaluation barriers preserve output.
      let combined = try measure("unsplit_parity_reference") {
        try adapter.apply(base: matmul(input, weight.T), input: input,
          target: target, reorderQKV: true)
      }
      XCTAssertEqual(split.shape, [1, rows, outputColumns]); XCTAssertEqual(split.dtype, .bfloat16)
      XCTAssertTrue(MLX.isFinite(split).all().item(Bool.self))
      XCTAssertTrue(MLX.isFinite(combined).all().item(Bool.self))
      maximumError = max(abs(split.asType(.float32) - combined.asType(.float32))).item(Float.self)
      adapterDifference = max(abs(split.asType(.float32) - base.asType(.float32))).item(Float.self)
      XCTAssertEqual(maximumError, 0, "Phase boundaries must not change the combined BF16 projection.")
      XCTAssertGreaterThan(adapterDifference, 0, "The admitted Turbo adapter must actually affect the projection.")
      try file.checkUnchanged(at: checkpoint)
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    let after = Memory.activeMemory
    XCTAssertEqual(after, before, "The probe must release its input, decoded weight, adapter factors and outputs.")
    let report: [String: Any] = ["format": "weetodd-h3-qkv-split-probe-v1",
      "checkpoint": checkpointPath, "adapter": adapterPath, "adapter_strength": 1,
      "adapter_target_count": adapter.targetCount, "rows": rows, "columns": columns,
      "output_columns": outputColumns, "decode_row_window": rowWindow,
      "dtype": "BF16", "input_pattern": "nonzero centered integer multiples of 1/256",
      "source_qkv_int8_bytes": outputColumns * columns,
      "active_mlx_bytes_before": before, "active_mlx_bytes_after_release": after,
      "maximum_absolute_parity_error": maximumError, "maximum_adapter_difference": adapterDifference,
      "phases": phases, "scope": "single projection; no sampling or model generation"]
    let json = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    print("H3_QKV_SPLIT_PROBE " + String(decoding: json, as: UTF8.self))
  }

  func testInstalledQKVRowWindowLatencyAndMemoryProbe() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_QKV_WINDOW_PROBE"] == "1",
      let checkpointPath = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Enable the installed QKV row-window probe explicitly.")
    }
    let checkpoint = URL(fileURLWithPath: checkpointPath)
    let file = try SafeTensorFile(url: checkpoint)
    let name = "model.diffusion_model.blocks.0.attn.qkv_proj.weight"
    var reference: [Float]?
    for window in [4096, 8192, 16384] {
      Memory.clearCache()
      Memory.peakMemory = Memory.activeMemory
      let started = Date()
      let weight = try H3ComfyDecodedProjection.load(file: file,
        checkpointURL: checkpoint, name: name, rows: 21504,
        columns: 5376, reorderQKV: true, rowWindow: window)
      let elapsed = Date().timeIntervalSince(started)
      let sample = weight[0..<16, 0..<5376].asType(.float32).asArray(Float.self)
      if let reference {
        XCTAssertEqual(sample, reference)
      } else {
        reference = sample
      }
      print("H3_QKV_WINDOW_PROBE window=\(window) seconds=\(elapsed) "
        + "peak_mlx_bytes=\(Memory.peakMemory)")
    }
  }

  func testInstalledFullQKVLoadReordersHeadRows() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_QKV_FULL_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_CONVROT_ORACLE"] else {
      throw XCTSkip("Enable the installed full QKV load qualification explicitly.")
    }
    let weight = try H3ComfyDecodedProjection.load(
      checkpointURL: URL(fileURLWithPath: checkpoint),
      name: "model.diffusion_model.blocks.0.attn.qkv_proj.weight",
      rows: 21504, columns: 5376, reorderQKV: true)
    XCTAssertEqual(weight.shape, [21504, 5376])
    for (letter, start) in [("q", 0), ("k", 128), ("v", 256)] {
      let data = try Data(contentsOf: URL(fileURLWithPath: fixture)
        .appendingPathComponent("qkv-\(letter).u16"))
      let expected = data.withUnsafeBytes {
        MLXArray($0, [16, 5376], type: UInt16.self).view(dtype: .bfloat16)
      }
      let actual = weight[start..<(start + 16), 0..<5376]
      XCTAssertEqual(max(abs(actual.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, letter)
    }
  }

  func testInstalledConvRotRowsMatchReferenceBeforeQKVReorder() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_CONVROT_ORACLE"] else {
      throw XCTSkip("Set installed Comfy H3 checkpoint and ConvRot oracle paths.")
    }
    for (stem, rows, columns) in [
      ("blocks.0.adaln_proj.linear", 96768, 2688),
      ("blocks.0.attn.qkv_proj", 21504, 5376),
    ] {
      let actual = try H3ComfyDecodedProjection.decodeRows(
        checkpointURL: URL(fileURLWithPath: checkpoint),
        name: "model.diffusion_model." + stem + ".weight",
        rows: rows, columns: columns, range: 0..<16)
      let data = try Data(contentsOf: URL(fileURLWithPath: fixture)
        .appendingPathComponent(stem.replacingOccurrences(of: ".", with: "-") + ".u16"))
      let expected = data.withUnsafeBytes {
        MLXArray($0, [16, columns], type: UInt16.self).view(dtype: .bfloat16)
      }
      XCTAssertEqual(max(abs(actual.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, stem)
    }
  }

}
