import Foundation
import Tokenizers

public final class MOSSQwenTokenizer {
  let tokenizer: any Tokenizer
  public static func render(_ prompt: String) -> String {
    "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
  }
  public init(directory: URL) async throws { tokenizer = try await AutoTokenizer.from(modelFolder: directory) }
  public func tokens(_ prompt: String) throws -> [Int] {
    let tokens = tokenizer.encode(text: Self.render(prompt), addSpecialTokens: false)
    guard !tokens.isEmpty, tokens.count <= 4096, tokens.allSatisfy({ (0..<248320).contains($0) }) else {
      throw MOSSError("MOSS accepts up to 4096 tokens including its template. Shorten the prompt or layout; nothing was truncated.")
    }
    return tokens
  }
}
