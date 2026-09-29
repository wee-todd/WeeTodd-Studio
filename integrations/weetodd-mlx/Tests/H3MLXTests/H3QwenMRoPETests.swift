import XCTest
@testable import H3MLX

final class H3QwenMRoPETests: XCTestCase {
  func testTwoKeyframePositionsMatchQwenReference() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set installed tokenizer for mixed Qwen position parity.")
    }
    let tokenizer = try H3QwenTokenizer(url: URL(fileURLWithPath: path))
    let grids: [H3QwenRequest.Grid] = [.init(temporal: 1, height: 4, width: 4),
      .init(temporal: 1, height: 2, width: 2)]
    let request = try H3QwenRequest.keyframes(prompt: "A quick brown fox leaps.",
      grids: grids, tokenizer: tokenizer)
    let positions = try H3QwenMRoPE.positions(request: request, grids: grids)
    XCTAssertEqual(positions, [
      [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
      [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    ])
  }
}
