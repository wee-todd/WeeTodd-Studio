import Foundation
import MLX
import MLXRandom
import TensorIO
import XCTest
@testable import H3MLX

final class H3ComfyDecodedProjectionTests: XCTestCase {
  func testStreamingProjectionPreservesBF16BiasAcrossRowWindows() throws {
    let rows = 1025, columns = 4
    let marker = try JSONSerialization.data(withJSONObject: [
      "format": "int8_tensorwise", "convrot": false])
    let scaleBytes = [Float](repeating: 1, count: rows)
      .map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
    let biasBits: [UInt16] = [0x3e80, 0xbf00, 0x3f40] // 0.25, -0.5, 0.75
    let biasBytes = (0..<rows).map { biasBits[$0 % 3].littleEndian }
      .withUnsafeBytes { Data($0) }
    var payload = Data((0..<(rows * columns)).map { UInt8($0 % columns + 1) })
    let scaleStart = payload.count; payload.append(scaleBytes)
    let markerStart = payload.count; payload.append(marker)
    let biasStart = payload.count; payload.append(biasBytes)
    let header: [String: Any] = [
      "projection.weight": ["dtype": "I8", "shape": [rows, columns], "data_offsets": [0, scaleStart]],
      "projection.weight_scale": ["dtype": "F32", "shape": [rows, 1], "data_offsets": [scaleStart, markerStart]],
      "projection.comfy_quant": ["dtype": "U8", "shape": [marker.count], "data_offsets": [markerStart, biasStart]],
      "projection.bias": ["dtype": "BF16", "shape": [rows], "data_offsets": [biasStart, payload.count]],
    ]
    let json = try JSONSerialization.data(withJSONObject: header)
    var size = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of: &size) { Data($0) }; bytes.append(json); bytes.append(payload)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    try bytes.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    let input = MLXArray([Float(2), 1, -1, 0.5], [1, columns]).asType(.bfloat16)
    let output = try H3ComfyDecodedProjection.projectStreaming(checkpointURL: url,
      name: "projection.weight", rows: rows, columns: columns, input: input, rowWindow: 1024)
    XCTAssertEqual(output.dtype, .bfloat16); XCTAssertEqual(output.shape, [1, rows])
    let expected: [Float] = [3.25, 2.5, 3.75]
    XCTAssertEqual(output.asArray(Float.self), (0..<rows).map { expected[$0 % 3] })
  }

  /// Metadata and invalid-range admission stop before any MLX tensor is created.
  func testSmallMetadataAdmissionRejectsMalformedInputsBeforeTensorCreation() throws {
    let validMarker: [String: Any] = ["format": "int8_tensorwise", "convrot": false]
    try withSmallMetadata(marker: validMarker, scales: [0.5, 1]) { url in
      XCTAssertThrowsError(try H3ComfyDecodedProjection.decodeRows(checkpointURL: url,
        name: "projection.weight", rows: 2, columns: 4, range: -1..<0)) { error in
        XCTAssertTrue(String(describing: error).contains("decode rows exceed"))
      }
    }
    for badScale: Float in [0, -1, .infinity, .nan] {
      try withSmallMetadata(marker: validMarker, scales: [0.5, badScale]) { url in
        XCTAssertThrowsError(try H3ComfyDecodedProjection.decodeRows(checkpointURL: url,
          name: "projection.weight", rows: 2, columns: 4, range: -1..<0)) { error in
          XCTAssertTrue(String(describing: error).contains("Invalid Comfy H3 row scales"))
        }
      }
    }
    for marker: [String: Any] in [[:], ["format": "other", "convrot": false],
      ["format": "int8_tensorwise", "convrot": false, "unknown": 1],
      ["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 64]] {
      try withSmallMetadata(marker: marker, scales: [0.5, 1]) { url in
        XCTAssertThrowsError(try H3ComfyDecodedProjection.decodeRows(checkpointURL: url,
          name: "projection.weight", rows: 2, columns: 4, range: -1..<0)) { error in
          XCTAssertTrue(String(describing: error).contains("Unsupported Comfy H3"))
        }
      }
    }
  }

  func testSuccessfulWeightDecodePreservesStageAllocationPool() throws {
    let previous = Memory.cacheLimit
    Stream.gpu.synchronize(); Memory.clearCache()
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previous }
    autoreleasepool {
      let scratch = MLXRandom.normal([4096, 1024], key: MLXRandom.key(914))
      eval(scratch)
    }
    Stream.gpu.synchronize()
    let cachedBefore = Memory.cacheMemory
    XCTAssertGreaterThan(cachedBefore, 8 * 1024 * 1024)
    try withSmallMetadata(marker: ["format": "int8_tensorwise", "convrot": false],
      scales: [0.5, 1]) { url in
      let value = try H3ComfyDecodedProjection.load(checkpointURL: url,
        name: "projection.weight", rows: 2, columns: 4)
      XCTAssertEqual(value.asArray(Float.self), Array(repeating: 0, count: 8))
      XCTAssertGreaterThanOrEqual(Memory.cacheMemory, cachedBefore)
      XCTAssertEqual(Memory.cacheLimit, 128 * 1024 * 1024)
    }
  }

  private func withSmallMetadata(marker: [String: Any], scales: [Float],
    _ body: (URL) throws -> Void) throws {
    let markerData = try JSONSerialization.data(withJSONObject: marker)
    let scaleData = scales.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
    var payload = Data(repeating: 0, count: 8)
    payload.append(scaleData); payload.append(markerData)
    let header: [String: Any] = [
      "projection.weight": ["dtype": "I8", "shape": [2, 4], "data_offsets": [0, 8]],
      "projection.weight_scale": ["dtype": "F32", "shape": [2, 1],
        "data_offsets": [8, 8 + scaleData.count]],
      "projection.comfy_quant": ["dtype": "U8", "shape": [markerData.count],
        "data_offsets": [8 + scaleData.count, payload.count]],
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
