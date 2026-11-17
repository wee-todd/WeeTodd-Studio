import Foundation
import ImageLayout

final class MOSSLayoutTokenBudget {
  let tokenizer: MOSSQwenTokenizer
  let directory: URL
  let identity: String
  static func directory(provided: String?, model: URL, family: ImageLayoutVisionModel) throws -> URL {
    if let provided {
      guard provided.hasPrefix("/"), !provided.utf8.contains(0) else { throw MOSSError("Choose a local MOSS tokenizer folder.") }
      return URL(fileURLWithPath: provided).standardizedFileURL.resolvingSymlinksInPath()
    }
    guard family == .qwen35_4b else {
      throw MOSSError("Gemma analysis requires the MOSS Qwen tokenizer folder to validate the final layout's token budget.")
    }
    return model
  }
  static func validateVocabulary(_ tokens: [String: Int]) throws {
    let expected = ["<|im_start|>": 248045, "<|im_end|>": 248046, "<|vision_start|>": 248053,
      "<|vision_end|>": 248054, "<|image_pad|>": 248056]
    guard expected.allSatisfy({ tokens[$0.key] == $0.value }) else {
      throw MOSSError("Count MOSS prompts with its compatible Qwen3.5 tokenizer, not the vision analyzer's vocabulary.")
    }
  }
  init(provided: String?, model: URL, family: ImageLayoutVisionModel) async throws {
    directory = try Self.directory(provided: provided, model: model, family: family)
    identity = try Self.assets(directory)
    tokenizer = try await MOSSQwenTokenizer(directory: directory)
    var special: [String: Int] = [:]
    for name in MOSSLayoutTokenizer.specialNames(.qwen35_4b) { special[name] = tokenizer.tokenizer.convertTokenToId(name) }
    try Self.validateVocabulary(special)
    try checkUnchanged()
  }
  func checkUnchanged() throws {
    guard try Self.assets(directory) == identity else { throw MOSSError("The MOSS prompt tokenizer changed during analysis.") }
  }
  private static func assets(_ directory: URL) throws -> String {
    let assets = try ["tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja"].map { name in
      let file = directory.appendingPathComponent(name)
      if !FileManager.default.fileExists(atPath: file.path), ["config.json", "chat_template.jinja"].contains(name) { return name + ":absent" }
      let data = try Data(contentsOf: file)
      guard data.count <= 64 * 1024 * 1024 else { throw MOSSError("The MOSS tokenizer exceeds its size limit.") }
      return name + ":" + MOSSImagePreflight.digest(data)
    }.joined(separator: "\n")
    return MOSSImagePreflight.digest(Data((directory.path + "\n" + assets).utf8))
  }
}
