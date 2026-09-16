import BinaryResources
import CoreFoundation
import Foundation
import LLM
import NNC
import Tokenizer

public struct LocalTextError: Error {
  public let code: String
  init(_ code: String) { self.code = code }
}

enum LocalTextModel: String {
  case qwen35_4B = "qwen_3.5_4b_i8x.ckpt"
  case qwen35_9B = "qwen_3.5_9b_i5x.ckpt"
  static func identify(_ path: String) throws -> Self {
    guard path.hasPrefix("/"), let model = Self(rawValue: URL(fileURLWithPath: path).lastPathComponent) else {
      throw LocalTextError("text_model_unsupported")
    }
    return model
  }
}

struct LocalTextRequest {
  let modelPath: String
  let model: LocalTextModel
  let systemPrompt: String
  let prompt: String
  let maxTokens: Int
  let images: [LocalPromptImage]
  init(_ value: [String: Any]) throws {
    guard Set(value.keys).isSubset(of: ["requestID", "modelPath", "systemPrompt", "prompt", "maxTokens", "images"]),
      let path = value["modelPath"] as? String,
      let system = value["systemPrompt"] as? String, let prompt = value["prompt"] as? String,
      let limit = value["maxTokens"] as? NSNumber,
      CFGetTypeID(limit) != CFBooleanGetTypeID(), limit.doubleValue == Double(limit.intValue),
      (1...1024).contains(limit.intValue), !system.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      ![system, prompt].contains(where: { text in
        ["<|", "<think>", "</think>"].contains(where: text.contains)
      }) else { throw LocalTextError("text_request_invalid") }
    guard system.utf8.count + prompt.utf8.count <= 24_000 else {
      throw LocalTextError("text_input_bytes_exceeded")
    }
    modelPath = path; model = try LocalTextModel.identify(path)
    systemPrompt = system; self.prompt = prompt; maxTokens = limit.intValue
    guard let records = value["images"] as? [[String: Any]] ?? (value["images"] == nil ? [] : nil), records.count <= 8 else {
      throw LocalTextError("vision_inputs_invalid")
    }
    images = try records.map { image in
      guard Set(image.keys) == ["path", "label"], let imagePath = image["path"] as? String,
        imagePath.hasPrefix("/"), let label = image["label"] as? String, !label.isEmpty,
        label.utf8.count <= 300, !label.contains("<|"), !label.contains("<think>"), !label.contains("</think>") else {
        throw LocalTextError("vision_inputs_invalid")
      }
      return LocalPromptImage(path: imagePath, label: label)
    }
    guard images.isEmpty || model == .qwen35_4B else { throw LocalTextError("vision_model_unsupported") }
  }
}

/// Runs in a request-owned helper process; exiting releases the model and all GPU caches.
/// Installed DT stores are opened read-only. No download, conversion or persistent KV cache.
public enum LocalTextGeneration {
  private static let eos: Set<Int32> = [248_044, 248_046]
  private static let tokenizer = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_5_json, merges: BinaryResources.merges_qwen3_5_txt,
    specialTokens: ["<|endoftext|>": 248_044, "<|im_start|>": 248_045,
      "<|im_end|>": 248_046, "<think>": 248_068, "</think>": 248_069,
      "<|vision_start|>": 248_053, "<|vision_end|>": 248_054, "<|image_pad|>": 248_056],
    unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|im_end|>")

  private static func chatTokens(_ request: LocalTextRequest) -> [Int32] {
    let images = request.images.enumerated().map { index, image in
      "Image \(index + 1) (\(image.label)):\n<|vision_start|><|image_pad|><|vision_end|>\n"
    }.joined()
    let chat = "<|im_start|>system\n" + request.systemPrompt + "<|im_end|>\n"
      + "<|im_start|>user\n" + images + request.prompt + "<|im_end|>\n"
      + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tokenizer.tokenize(text: chat, addSpecialTokens: false).0
  }
  static func promptTokens(_ request: LocalTextRequest) throws -> [Int32] {
    let tokens = chatTokens(request)
    guard !tokens.isEmpty, tokens.count <= 4096 else { throw LocalTextError("text_context_too_long") }
    return tokens
  }

  static func multimodalTokens(_ request: LocalTextRequest, grids: [(t: Int, h: Int, w: Int)]) throws -> (tokenIDs: [Int32], tokenTypeIDs: [Int32]) {
    guard grids.count == request.images.count else { throw LocalTextError("vision_inputs_invalid") }
    let result = Qwen3_5ExpandedMultimodalTokenIDs(chatTokens(request), imageGridThw: grids)
    guard !result.tokenIDs.isEmpty, result.tokenIDs.count <= 4096 else { throw LocalTextError("text_context_too_long") }
    return result
  }

  private struct Prepared {
    let request: LocalTextRequest
    let tokens: [Int32]
    let vision: LocalVisionInput?
    let expanded: (tokenIDs: [Int32], tokenTypeIDs: [Int32])?
    let budget: [String: Any]
  }

  private static func prepare(_ value: [String: Any]) throws -> Prepared {
    let request = try LocalTextRequest(value)
    let tokens = try promptTokens(request)
    let vision = request.images.isEmpty ? nil : try LocalVisionInput.prepare(request.images)
    let expanded = try vision.map { try multimodalTokens(request, grids: $0.grids) }
    let count = expanded?.tokenIDs.count ?? tokens.count
    // Configured policy, not the model's architectural context capacity. Existing
    // runtime allocations reserve input + requested output; expose both explicitly.
    guard count + request.maxTokens <= 5120 else { throw LocalTextError("text_context_too_long") }
    return Prepared(request: request, tokens: tokens, vision: vision, expanded: expanded,
      budget: ["valid": true, "inputBytes": request.systemPrompt.utf8.count + request.prompt.utf8.count,
        "textTokens": tokens.count, "inputTokens": count, "imageTokens": count - tokens.count,
        "imagesUsed": request.images.count, "outputTokenBudget": request.maxTokens,
        "inputByteLimit": 24_000, "inputTokenLimit": 4096, "totalTokenLimit": 5120])
  }

  /// Exact model-free budget check. Validates the format binding, tokenizer and image
  /// grids without opening the checkpoint or loading weights. It is not a health check.
  public static func preflight(_ value: [String: Any]) throws -> [String: Any] {
    try prepare(value).budget
  }

  public static func run(_ value: [String: Any], progress: ([String: Any]) -> Void) throws -> [String: Any] {
    let preparationStarted = Date()
    let prepared = try prepare(value)
    let request = prepared.request
    let preparationMilliseconds = Date().timeIntervalSince(preparationStarted) * 1000
    let resolved = URL(fileURLWithPath: request.modelPath).resolvingSymlinksInPath()
    guard (try? resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
      let file = FileHandle(forReadingAtPath: request.modelPath) else {
      throw LocalTextError("text_model_unavailable")
    }
    let header = try file.read(upToCount: 16)
    try file.close()
    guard header == Data("SQLite format 3\0".utf8) else { throw LocalTextError("text_model_invalid_store") }
    let tokens = prepared.tokens
    progress(["stage": "loading", "tokens": 0])
    let start = Date()
    let generator = Qwen3_5TextGeneration<Float16>(filePath: request.modelPath,
      configuration: request.model == .qwen35_4B ? .qwen3_5_4B : .qwen3_5_9B,
      tieEmbedding: request.model == .qwen35_4B)
    var timing: [String: Any] = [:]
    let generated: [Int32]
    do {
      if let vision = prepared.vision, let expanded = prepared.expanded {
        progress(["stage": "vision", "images": request.images.count])
        timing["inputTokensWithImages"] = expanded.tokenIDs.count
        generated = try generator.generateMultimodal(graph: DynamicGraph(),
          promptTokenIds: expanded.tokenIDs, tokenTypeIds: expanded.tokenTypeIDs,
          imagePatches: vision.patches, imageGridThw: vision.grids, maxTokens: request.maxTokens,
          partialHandler: { ids in
            if ids.count == 1 || ids.count % 8 == 0 { progress(["stage": "writing", "tokens": ids.count]) }
            return true
          })
      } else {
      generated = try generator.generate(graph: DynamicGraph(), promptTokenIds: tokens,
        maxTokens: request.maxTokens, prefillChunkSize: 512,
        partialHandler: { ids in
          if ids.count == 1 || ids.count % 8 == 0 { progress(["stage": "writing", "tokens": ids.count]) }
          return true
        }, timingHandler: { value in
          timing = ["loadAndCompileMilliseconds": value.loadAndCompileMilliseconds,
            "prefillMilliseconds": value.prefillMilliseconds,
            "decodeCompileMilliseconds": value.decodeCompileMilliseconds,
            "decodeLoopMilliseconds": value.decodeLoopMilliseconds,
            "tokensPerSecond": value.decodeLoopTokensPerSecond]
        })
      }
    } catch let error as LocalTextError { throw error }
      catch { throw LocalTextError("text_generation_failed") }
    let content = generated.prefix(while: { !eos.contains($0) })
    let text = tokenizer.decode(Array(content)).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { throw LocalTextError("text_result_empty") }
    timing["requestPreparationMilliseconds"] = preparationMilliseconds
    return ["text": text, "model": request.model.rawValue, "inputTokens": timing["inputTokensWithImages"] ?? tokens.count,
      "imagesUsed": request.images.count,
      "outputTokens": content.count, "truncated": !generated.contains(where: eos.contains) && generated.count >= request.maxTokens,
      "totalSeconds": Date().timeIntervalSince(start), "timing": timing,
      "preflight": prepared.budget]
  }
}
