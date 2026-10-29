import Tokenizers
import XCTest

@testable import Drama3MLX

final class TokenizerParityTests: XCTestCase {
  func testReleasedStructuralTokensEncodeAsSingleEntriesWhenQualified() async throws {
    guard let root = ProcessInfo.processInfo.environment["DRAMA3_NUMERICAL_ROOT"] else {
      throw XCTSkip("Set DRAMA3_NUMERICAL_ROOT for tokenizer qualification.")
    }
    let tokenizer = try await AutoTokenizer.from(
      modelFolder: URL(fileURLWithPath: root).appendingPathComponent("balanced"))
    for token in [
      "<|im_start|>", "<|im_end|>", "<|voice|>", "<|speaker:0|>", "<|speaker:1|>",
      "<|audio_separator|>", "<|endofturn|>", "<|fim_suffix|>", "<|fim_prefix|>",
    ] {
      let ids = tokenizer.encode(text: token, addSpecialTokens: false)
      print("Structural token \(token): \(ids)")
      XCTAssertEqual(ids, [try XCTUnwrap(tokenizer.convertTokenToId(token))])
    }
  }
}
