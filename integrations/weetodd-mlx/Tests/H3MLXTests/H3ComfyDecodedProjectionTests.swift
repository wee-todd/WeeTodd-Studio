import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3ComfyDecodedProjectionTests: XCTestCase {
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
