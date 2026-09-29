import XCTest
@testable import H3MLX

final class H3QwenTokenizerTests: XCTestCase {
  func testByteLevelMergeAndNoAutomaticChatTokens() throws {
    let tokenizer = try H3QwenTokenizer(vocabulary: [
      "H": 1, "e": 2, "l": 3, "o": 4, "He": 5, "ll": 6,
      "Hell": 7, "Hello": 8,
    ], merges: ["H e", "l l", "He ll", "Hell o"],
      regex: #"\p{L}+|\s+|."#)
    XCTAssertEqual(try tokenizer.encode("Hello"), [8])
  }

  func testInstalledQwenTokenizerMatchesReferencePrompts() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set WEETODD_H3_QWEN_TOKENIZER for the optional installed vocabulary check.")
    }
    let tokenizer = try H3QwenTokenizer(url: URL(fileURLWithPath: path))
    XCTAssertEqual(try tokenizer.encode("A quick brown fox leaps."),
      [32, 3974, 13876, 38835, 83458, 13])
    XCTAssertEqual(try tokenizer.encode("<Picture 1>: a dragon, red and gold"),
      [21604, 3826, 220, 16, 26818, 264, 25105, 11, 2518, 323, 6623])
    XCTAssertEqual(try tokenizer.encode("Café naïve 🎥 4K — rain"),
      [34, 2577, 963, 94880, 586, 11162, 236, 98, 220, 19, 42, 1959, 11174])
  }
}
