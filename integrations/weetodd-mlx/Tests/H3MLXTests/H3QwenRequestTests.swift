import XCTest
@testable import H3MLX

final class H3QwenRequestTests: XCTestCase {
  func testOrderedRef2VAPresentationKeepsAudioLabelsAndVideoPads() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set WEETODD_H3_QWEN_TOKENIZER for installed Qwen request parity.")
    }
    let tokenizer = try H3QwenTokenizer(url: URL(fileURLWithPath: path))
    let image = H3QwenRequest.Grid(temporal: 1, height: 2, width: 2)
    let video = H3QwenRequest.Grid(temporal: 1, height: 2, width: 4)
    let request = try H3QwenRequest.references(prompt: "A sailor speaks.",
      references: [.image(grid: image),
        .video(blocks: [.init(timestampSeconds: 0, grid: video),
          .init(timestampSeconds: 1, grid: video)], hasAudio: true),
        .audio], tokenizer: tokenizer)
    let start = try XCTUnwrap(tokenizer.encode("<|vision_start|>").first)
    let end = try XCTUnwrap(tokenizer.encode("<|vision_end|>").first)
    let imagePad = try XCTUnwrap(tokenizer.encode("<|image_pad|>").first)
    let videoPad = try XCTUnwrap(tokenizer.encode("<|video_pad|>").first)
    XCTAssertEqual(request.visualRanges.count, 3)
    XCTAssertEqual(request.tokenIDs[request.visualRanges[0]], [start, imagePad, end])
    XCTAssertEqual(request.tokenIDs[request.visualRanges[1]],
      [start, videoPad, videoPad, end])
    XCTAssertEqual(request.tokenIDs[request.visualRanges[2]],
      [start, videoPad, videoPad, end])
    XCTAssertEqual(request.tags.filter { $0 == 0 }.count, 11)
    let firstLabel = try tokenizer.encode("<Picture 1>: ")
    XCTAssertEqual(Array(request.tokenIDs.prefix(request.visualRanges[0].lowerBound)),
      firstLabel)
    let betweenRange = request.visualRanges[0].upperBound..<request.visualRanges[1].lowerBound
    let between = Array(request.tokenIDs[betweenRange])
    let middleLabel = try tokenizer.encode("<Audio 1>: ")
      + tokenizer.encode("<Video 1>: ")
      + tokenizer.encode("<0.0 seconds>")
    XCTAssertEqual(between, middleLabel)
    let promptTokens = try tokenizer.encode("A sailor speaks.")
    XCTAssertEqual(Array(request.tokenIDs.suffix(promptTokens.count)), promptTokens)
  }

  func testRef2VAPresentationRejectsUnorderedVideoTimestamps() throws {
    let tokenizer = try H3QwenTokenizer(vocabulary: ["x": 1], merges: [], regex: ".")
    let grid = H3QwenRequest.Grid(temporal: 1, height: 2, width: 2)
    XCTAssertThrowsError(try H3QwenRequest.references(prompt: "x",
      references: [.video(blocks: [.init(timestampSeconds: 1, grid: grid),
        .init(timestampSeconds: 0, grid: grid)], hasAudio: false)], tokenizer: tokenizer))
  }

  func testInstalledTwoKeyframePresentationMatchesPythonTokenAndTagOracle() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set WEETODD_H3_QWEN_TOKENIZER for installed Qwen request parity.")
    }
    let tokenizer = try H3QwenTokenizer(url: URL(fileURLWithPath: path))
    let request = try H3QwenRequest.keyframes(prompt: "A quick brown fox leaps.",
      grids: [.init(temporal: 1, height: 4, width: 4),
        .init(temporal: 1, height: 2, width: 2)], tokenizer: tokenizer)
    XCTAssertEqual(request.tokenIDs, [21604, 3826, 220, 16, 26818, 220,
      151652, 151655, 151655, 151655, 151655, 151653,
      21604, 3826, 220, 17, 26818, 220,
      151652, 151655, 151653,
      32, 3974, 13876, 38835, 83458, 13])
    let expectedTags: [Int32] = [Int32](repeating: 1, count: 6) + [Int32](repeating: 0, count: 6)
      + [Int32](repeating: 1, count: 6) + [Int32](repeating: 0, count: 3)
      + [Int32](repeating: 1, count: 6)
    XCTAssertEqual(request.tags, expectedTags)
    XCTAssertEqual(request.visualRanges, [6..<12, 18..<21])
  }

  func testInvalidVisualGeometryRejectedBeforeInference() throws {
    let tokenizer = try H3QwenTokenizer(vocabulary: ["x": 1], merges: [], regex: ".")
    XCTAssertThrowsError(try H3QwenRequest.keyframes(prompt: "x",
      grids: [.init(temporal: 1, height: 3, width: 4)], tokenizer: tokenizer))
    XCTAssertThrowsError(try H3QwenRequest.keyframes(prompt: "x",
      grids: [.init(temporal: 1, height: 64, width: 64)], tokenizer: tokenizer))
  }
}
