import MLX
import XCTest
@testable import H3MLX

final class H3QwenTextEncoderTests: XCTestCase {
  func testInstalledSingleImageReferenceMatchesQwenVisualOracle() throws {
    guard let paged = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let compact = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let pixelsRoot = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"],
      let trace = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_MIXED"] else {
      throw XCTSkip("Set installed Qwen and visual oracle paths.")
    }
    let pixelData = try Data(contentsOf: URL(fileURLWithPath: pixelsRoot)
      .appendingPathComponent("pixels.u16"))
    let pixels = pixelData.withUnsafeBytes {
      MLXArray($0, [16, 1536], type: UInt16.self).view(dtype: .bfloat16)
    }
    let result = try H3QwenTextEncoder.encodeReferences(
      prompt: "A quick brown fox leaps.", pixels: pixels,
      references: [.image(grid: .init(temporal: 1, height: 4, width: 4))],
      checkpointRoot: URL(fileURLWithPath: paged),
      visionCheckpointURL: URL(fileURLWithPath: compact),
      tokenizerURL: URL(fileURLWithPath: tokenizer))
    let data = try Data(contentsOf: URL(fileURLWithPath: trace)
      .appendingPathComponent("layer-50.f32"))
    let expected = data.withUnsafeBytes { MLXArray($0, [18, 5120], type: Float.self) }
    XCTAssertEqual(max(abs(result.hidden.asType(.float32) - expected)).item(Float.self), 0)
    XCTAssertEqual(result.tags.filter({ $0 == 0 }).count, 6)
  }

  func testReferenceConditionerRejectsInvalidPresentationBeforeWeights() throws {
    guard let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set installed Qwen tokenizer path.")
    }
    let grid = H3QwenRequest.Grid(temporal: 1, height: 2, width: 2)
    let pixels = MLXArray.zeros([4, 1536]).asType(.bfloat16)
    XCTAssertThrowsError(try H3QwenTextEncoder.encodeReferences(
      prompt: "A sailor speaks.", pixels: pixels,
      references: [.video(blocks: [.init(timestampSeconds: 1, grid: grid),
        .init(timestampSeconds: 0, grid: grid)], hasAudio: true)],
      checkpointRoot: URL(fileURLWithPath: "/missing"),
      visionCheckpointURL: URL(fileURLWithPath: "/missing"),
      tokenizerURL: URL(fileURLWithPath: tokenizer))) { error in
      guard case H3CheckpointError.invalid(let detail) = error else {
        return XCTFail("Expected H3 reference admission error")
      }
      XCTAssertTrue(detail.contains("ordered finite timestamps"))
    }
  }

  func testInstalledMixedImageConditionerMatchesFiftyLayerOracle() throws {
    guard let paged = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let compact = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let pixelsRoot = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_VISION_BLOCK"],
      let trace = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_MIXED"] else {
      throw XCTSkip("Set installed Qwen and mixed-image oracle paths.")
    }
    let pixelData = try Data(contentsOf: URL(fileURLWithPath: pixelsRoot)
      .appendingPathComponent("pixels.u16"))
    let pixels = pixelData.withUnsafeBytes {
      MLXArray($0, [16, 1536], type: UInt16.self).view(dtype: .bfloat16)
    }
    let checkpoints = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50]
    var differences: [(Int, Float)] = []
    let result = try H3QwenTextEncoder.encodeKeyframesWithTrace(
      prompt: "A quick brown fox leaps.", pixels: pixels,
      grids: [.init(temporal: 1, height: 4, width: 4)],
      checkpointRoot: URL(fileURLWithPath: paged),
      visionCheckpointURL: URL(fileURLWithPath: compact),
      tokenizerURL: URL(fileURLWithPath: tokenizer)) { index, hidden in
      guard checkpoints.contains(index) else { return }
      let data = try Data(contentsOf: URL(fileURLWithPath: trace)
        .appendingPathComponent(String(format: "layer-%02d.f32", index)))
      let expected = data.withUnsafeBytes { MLXArray($0, [18, 5120], type: Float.self) }
      differences.append((index,
        max(abs(hidden.asType(.float32) - expected)).item(Float.self)))
    }
    XCTAssertEqual(differences.map(\.1).max(), 0)
    XCTAssertEqual(result.hidden.shape, [18, 5120])
    XCTAssertEqual(result.tags.filter({ $0 == 0 }).count, 6)
  }

  func testInstalledPerLayerTraceIdentifiesConditionerDrift() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let trace = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TRACE"] else {
      throw XCTSkip("Set installed Qwen paths and a reference trace for per-layer qualification.")
    }
    var differences: [Float] = []
    _ = try H3QwenTextEncoder.encodeWithTrace(prompt: "A quick brown fox leaps.",
      checkpointRoot: URL(fileURLWithPath: root), tokenizerURL: URL(fileURLWithPath: tokenizer),
      progress: { _, _ in }, observe: { index, hidden in
        let path = URL(fileURLWithPath: trace)
          .appendingPathComponent(String(format: "layer-%03d.f32", index))
        let bytes = try Data(contentsOf: path)
        let actual = hidden.asType(.float32).asArray(Float.self)
        XCTAssertEqual(bytes.count, actual.count * 4)
        let expected = bytes.withUnsafeBytes { raw in
          (0..<actual.count).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
        }
        differences.append(zip(actual, expected).map { abs($0 - $1) }.max()!)
      })
    XCTAssertLessThan(differences.max()!, 0.05)
  }

  func testInstalledFiftyLayerHiddenStateMatchesH3Conditioner() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set installed Qwen page and tokenizer paths for the optional 50-layer check.")
    }
    let result = try H3QwenTextEncoder.encode(prompt: "A quick brown fox leaps.",
      checkpointRoot: URL(fileURLWithPath: root), tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(result.tokenIDs, [32, 3974, 13876, 38835, 83458, 13])
    XCTAssertEqual(result.tags, [1, 1, 1, 1, 1, 1])
    XCTAssertEqual(result.hidden.shape, [6, 5120])
    let values = result.hidden.asType(.float32).asArray(Float.self)
    let first: [Float] = [-0.018920898, -1.28125, -1.3046875, -3.71875,
      4.1875, -3.09375, 0.146484375, -3.890625]
    let last: [Float] = [-6.0625, 2.875, -0.5859375, -1.2109375,
      -3.28125, -0.859375, 1.59375, -1.4296875]
    for index in 0..<8 {
      XCTAssertEqual(values[index], first[index], accuracy: 0.05, "first \(index)")
      XCTAssertEqual(values[5 * 5120 + index], last[index], accuracy: 0.05, "last \(index)")
    }
  }
}
