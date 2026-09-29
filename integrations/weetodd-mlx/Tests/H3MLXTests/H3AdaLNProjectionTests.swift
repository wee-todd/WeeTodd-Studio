import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3AdaLNProjectionTests: XCTestCase {
  func testExperimentalLargerWindowMatchesInstalledAdaLN() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_WINDOW_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let timeFixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"] else {
      throw XCTSkip("Enable the installed larger-window AdaLN qualification explicitly.")
    }
    let time = try Data(contentsOf: URL(fileURLWithPath: timeFixture)
      .appendingPathComponent("output.f32")).withUnsafeBytes {
        MLXArray($0, [3, 2688], type: Float.self)
      }
    let expected = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16")).withUnsafeBytes {
        MLXArray($0, [3, 96768], type: UInt16.self).view(dtype: .bfloat16)
      }
    if ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_PROFILE"] == "1" {
      let checkpointURL = URL(fileURLWithPath: checkpoint)
      let coldStarted = Date()
      _ = try H3CheckpointLayout(url: checkpointURL)
      let coldSeconds = Date().timeIntervalSince(coldStarted)
      let warmStarted = Date()
      _ = try H3CheckpointLayout(url: checkpointURL)
      print("H3 layout first=\(coldSeconds) cached=\(Date().timeIntervalSince(warmStarted))")
    }
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = Date()
    let actual = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: time, rowWindow: 16384)
    let elapsed = Date().timeIntervalSince(started)
    let error = abs(actual.asType(.float32) - expected.asType(.float32))
    let peak = max(error).item(Float.self)
    let average = mean(error).item(Float.self)
    print("H3 AdaLN 16384-row window seconds=\(elapsed) max=\(peak) "
      + "mean=\(average) peak_mlx=\(Memory.peakMemory)")
    XCTAssertEqual(peak, 0)
  }

  func testInstalledLayoutCacheRejectsReplacedCheckpointPath() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_LAYOUT_CACHE_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Enable the installed H3 layout cache check explicitly.")
    }
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-h3-layout-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let link = directory.appendingPathComponent("checkpoint.safetensors")
    try FileManager.default.createSymbolicLink(at: link,
      withDestinationURL: URL(fileURLWithPath: checkpoint))
    let first = try H3CheckpointLayout(url: link)
    let second = try H3CheckpointLayout(url: link)
    XCTAssertEqual(first.quantizedProjections, second.quantizedProjections)
    let invalid = directory.appendingPathComponent("invalid.safetensors")
    try Data([0]).write(to: invalid)
    try FileManager.default.removeItem(at: link)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: invalid)
    XCTAssertThrowsError(try H3CheckpointLayout(url: link))
  }
  func testInstalledBlockZeroMatchesStreamedReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let timeFixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"] else {
      throw XCTSkip("Set the installed H3 checkpoint, time and AdaLN oracle paths.")
    }
    let timeData = try Data(contentsOf: URL(fileURLWithPath: timeFixture)
      .appendingPathComponent("output.f32"))
    let time = timeData.withUnsafeBytes { MLXArray($0, [3, 2688], type: Float.self) }
    let expectedData = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16"))
    let expected = expectedData.withUnsafeBytes {
      MLXArray($0, [3, 96768], type: UInt16.self).view(dtype: .bfloat16)
    }
    let actual = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: time)
    XCTAssertEqual(actual.shape, [3, 96768])
    XCTAssertEqual(actual.dtype, .bfloat16)
    XCTAssertEqual(max(abs(actual.asType(.float32) - expected.asType(.float32)))
      .item(Float.self), 0)
    let repeatedTime = concatenated(Array(repeating: time, count: 7), axis: 0)[0..<20, 0..<2688]
    let repeatedData = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0-20.u16"))
    let repeatedExpected = repeatedData.withUnsafeBytes {
      MLXArray($0, [20, 96768], type: UInt16.self).view(dtype: .bfloat16)
    }
    let repeated = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: repeatedTime)
    XCTAssertEqual(repeated.shape, [20, 96768])
    // The 16+4 bounded GEMM and one 20-row GEMM select different Metal kernels;
    // the drift is below one BF16 quantum at the tested projection scale.
    XCTAssertLessThan(max(abs(repeated.asType(.float32)
      - repeatedExpected.asType(.float32))).item(Float.self), 0.005)
  }
}
