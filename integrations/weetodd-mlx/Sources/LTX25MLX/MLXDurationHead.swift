import CoreFoundation
import CryptoKit
import Foundation
import MLX
import LTX25Engine
import TensorIO

/// The small trained prompt-duration head uses positive post-connector streams.
/// This object retains a validated reader, never resident weight arrays or Gemma states.
public final class MLXDurationHead {
  public let checkpoint: URL
  public let tensorBytes: UInt64
  public let headerSHA256: String
  private let file: SafeTensorFile
  private let gate = NSLock()
  static let shapes: [String: [Int]] = [
    "video_input_proj.weight": [256,4096], "video_input_proj.bias": [256],
    "video_modality_emb": [256], "audio_input_proj.weight": [256,2048],
    "audio_input_proj.bias": [256], "audio_modality_emb": [256],
    "attention_pooler.query_tokens": [1,256],
    "attention_pooler.cross_attn.in_proj_weight": [768,256],
    "attention_pooler.cross_attn.in_proj_bias": [768],
    "attention_pooler.cross_attn.out_proj.weight": [256,256],
    "attention_pooler.cross_attn.out_proj.bias": [256],
    "mlp_hidden.weight": [256,256], "mlp_hidden.bias": [256],
    "mlp_out.weight": [1,256], "mlp_out.bias": [1]]

  /// Header-only admission for the released BF16 head; no tensor payload reads.
  public init(checkpoint: URL) throws {
    guard checkpoint.isFileURL else { throw LTXError.invalid("Duration head must be a local checkpoint.") }
    self.checkpoint = checkpoint.resolvingSymlinksInPath().standardizedFileURL
    let file = try SafeTensorFile(url: self.checkpoint, maximumHeaderBytes: 1024*1024)
    guard file.fileByteCount <= 16*1024*1024,
      ["2.5", "2.5.0"].contains(file.metadata["model_version"] ?? ""),
      let encoded = file.metadata["config"], encoded.utf8.count <= 256*1024,
      let data = encoded.data(using: .utf8),
      let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let transformer = config["transformer"] as? [String: Any],
      let head = config["duration_head"] as? [String: Any] else {
      throw LTXError.invalid("Expected the official LTX 2.5 BF16 duration-head configuration.")
    }
    func dimension(_ values: [String: Any], _ name: String, _ expected: Int) -> Bool {
      guard let raw = values[name] else { return true }
      guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return false }
      return number.doubleValue == Double(expected)
    }
    guard dimension(transformer, "cross_attention_dim", 4096),
      dimension(transformer, "audio_cross_attention_dim", 2048),
      dimension(head, "pooler_hidden_dim", 256), dimension(head, "num_queries", 1),
      dimension(head, "num_pooler_heads", 4), dimension(head, "mlp_hidden", 256),
      Set(file.tensors.keys) == Set(Self.shapes.keys.map { "duration_head." + $0 }),
      Self.shapes.allSatisfy({ name, shape in
        guard let tensor = file.tensors["duration_head." + name] else { return false }
        return tensor.dtype == "BF16" && tensor.shape == shape.map(UInt64.init)
      }) else { throw LTXError.invalid("Duration-head tensor shapes or architecture are incompatible.") }
    self.file = file
    tensorBytes = file.tensors.values.reduce(0) { $0 + $1.byteCount }
    try file.checkUnchanged(at: self.checkpoint)
    let handle = try FileHandle(forReadingFrom: self.checkpoint)
    defer { try? handle.close() }
    guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
      throw LTXError.invalid("Truncated duration-head header.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8*$1.offset) }
    guard length > 0, length <= 1024*1024,
      let header = try handle.read(upToCount: Int(length)), header.count == Int(length) else {
      throw LTXError.invalid("Invalid bounded duration-head header.")
    }
    headerSHA256 = SHA256.hash(data: header).map { String(format:"%02x",$0) }.joined()
    try file.checkUnchanged(at: self.checkpoint)
  }

  /// Performs one small head evaluation, reusing the positive encoder outputs.
  /// Accepts native 2-D streams or a batch of exactly one; computation stays FP32.
  public func predict(video: MLXArray, audio: MLXArray) throws -> Double {
    guard gate.try() else { throw LTXError.invalid("Duration prediction is already active.") }
    defer { gate.unlock() }
    try Task.checkCancellation()
    try file.checkUnchanged(at: checkpoint)
    return try autoreleasepool {
      func stream(_ value: MLXArray, width: Int) throws -> MLXArray {
        let shape = value.shape
        guard value.dtype == .float32,
          (shape.count == 2 && shape[1] == width && (1...1024).contains(shape[0]))
            || (shape.count == 3 && shape[0] == 1 && shape[2] == width && (1...1024).contains(shape[1])) else {
          throw LTXError.invalid("Duration prediction needs one positive FP32 connector stream per modality.")
        }
        return shape.count == 2 ? value.reshaped([1,shape[0],width]) : value
      }
      func weight(_ name: String) throws -> MLXArray {
        try Task.checkCancellation()
        return try MLXWeight.read(file, "duration_head." + name, access: .buffered).asType(.float32)
      }
      func linear(_ value: MLXArray, _ name: String) throws -> MLXArray {
        let w = try weight(name + ".weight"), bias = try weight(name + ".bias")
        return matmul(value, w.transposed()) + bias
      }
      let video = try stream(video, width: 4096), audio = try stream(audio, width: 2048)
      let videoTokens = try linear(video, "video_input_proj") + weight("video_modality_emb")
      let audioTokens = try linear(audio, "audio_input_proj") + weight("audio_modality_emb")
      let tokens = concatenated([videoTokens,audioTokens], axis: 1)
      let query = try weight("attention_pooler.query_tokens").reshaped([1,1,256])
      let qkv = try weight("attention_pooler.cross_attn.in_proj_weight")
      let bias = try weight("attention_pooler.cross_attn.in_proj_bias")
      let q = (matmul(query,qkv[0..<256].transposed()) + bias[0..<256])
        .reshaped([1,1,4,64]).transposed(0,2,1,3)
      let k = (matmul(tokens,qkv[256..<512].transposed()) + bias[256..<512])
        .reshaped([1,tokens.shape[1],4,64]).transposed(0,2,1,3)
      let v = (matmul(tokens,qkv[512..<768].transposed()) + bias[512..<768])
        .reshaped([1,tokens.shape[1],4,64]).transposed(0,2,1,3)
      let scores = matmul(q * Float(0.125), k.transposed(0,1,3,2))
      let pooled = matmul(softmax(scores, axis: -1), v).transposed(0,2,1,3).reshaped([1,1,256])
      let projected = try linear(pooled, "attention_pooler.cross_attn.out_proj")
      let hidden = try linear(projected.reshaped([1,256]), "mlp_hidden")
      let activated = MLXTextMath.gelu(hidden)
      let seconds = exp(try linear(activated, "mlp_out")).reshaped([])
      eval(seconds)
      try Task.checkCancellation()
      try file.checkUnchanged(at: checkpoint)
      let result = Double(seconds.item(Float.self))
      guard result.isFinite, result > 0 else { throw LTXError.invalid("Duration head produced a nonfinite or nonpositive duration.") }
      return result
    }
  }

  /// Python's nearest-even frame rounding followed by floor-to-8k+1.
  /// Bounds with no legal frame-grid point fail instead of returning an off-grid clip.
  public static func frames(seconds: Double, fps: Double, minimumSeconds: Double = 1,
    maximumSeconds: Double = 20) throws -> Int {
    guard seconds.isFinite, seconds > 0, fps.isFinite, (1...120).contains(fps),
      minimumSeconds.isFinite, maximumSeconds.isFinite,
      (0.25...30).contains(minimumSeconds), maximumSeconds >= minimumSeconds,
      maximumSeconds <= 30 else { throw LTXError.invalid("Automatic duration needs finite bounds within 0.25–30 seconds and a valid frame rate.") }
    let minimum = Int((minimumSeconds*fps).rounded(.toNearestOrEven))
    let maximum = Int((maximumSeconds*fps).rounded(.toNearestOrEven))
    let firstGrid = ((max(0,minimum-1)+7)/8)*8+1
    guard firstGrid <= maximum else { throw LTXError.invalid("Automatic duration bounds contain no valid 8k+1 frame count.") }
    let bounded = max(minimumSeconds,min(seconds,maximumSeconds))
    let raw = max(minimum,min(Int((bounded*fps).rounded(.toNearestOrEven)),maximum))
    return max(firstGrid,((raw-1)/8)*8+1)
  }
}
