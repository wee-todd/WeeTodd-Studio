import Foundation
import MLXLMCommon
import Tokenizers
import ImageLayout

struct MOSSLayoutTokenizerLoader: TokenizerLoader {
  func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
    let templateURL = directory.appendingPathComponent("chat_template.jinja")
    let template = try String(contentsOf: templateURL, encoding: .utf8)
    return MOSSLayoutTokenizer(base: try await AutoTokenizer.from(modelFolder: directory), template: template)
  }
}

final class MOSSLayoutTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
  static func specialNames(_ family: ImageLayoutVisionModel) -> [String] {
    family == .qwen35_4b ? ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
      : ["<bos>", "<eos>", "<|turn>", "<turn|>", "<|image>", "<|image|>", "<image|>"]
  }
  static func chatStop(family: ImageLayoutVisionModel, tokens: [String: Int], declaredEOS: Int) throws -> Int {
    if family == .qwen35_4b { try validateSpecialTokens(tokens, eos: declaredEOS); return 248046 }
    let expected = ["<bos>": 2, "<eos>": 1, "<|turn>": 105, "<turn|>": 106,
      "<|image>": 255999, "<|image|>": 258880, "<image|>": 258882]
    guard declaredEOS == 1, expected.allSatisfy({ tokens[$0.key] == $0.value }) else {
      throw MOSSError("The analyzer tokenizer must preserve Gemma4's turn and image tokens.")
    }
    return 106
  }
  static func validateSpecialTokens(_ tokens: [String: Int], eos: Int) throws {
    let expected = ["<|im_start|>": 248045, "<|im_end|>": 248046,
      "<|vision_start|>": 248053, "<|vision_end|>": 248054, "<|image_pad|>": 248056]
    guard eos == expected["<|im_end|>"], expected.allSatisfy({ tokens[$0.key] == $0.value }) else {
      throw MOSSError("The analyzer tokenizer must preserve Qwen3.5's chat and vision tokens.")
    }
  }
  let base: any Tokenizers.Tokenizer
  let template: String
  init(base: any Tokenizers.Tokenizer, template: String) { self.base = base; self.template = template }
  func encode(text: String, addSpecialTokens: Bool) -> [Int] { base.encode(text: text, addSpecialTokens: addSpecialTokens) }
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { base.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
  func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
  func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
  var bosToken: String? { base.bosToken }
  var eosToken: String? { base.eosToken }
  var unknownToken: String? { base.unknownToken }
  func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    additionalContext: [String: any Sendable]?) throws -> [Int] {
    try base.applyChatTemplate(messages: messages, chatTemplate: .literal(template), addGenerationPrompt: true,
      truncation: false, maxLength: nil, tools: tools, additionalContext: additionalContext)
  }
}
