import Foundation
import TensorIO

/// Native Swift conditioning for the released Gemma4-12B LTX2.5 pack.
/// Weights stay in place. Only one 64 MiB decoded weight slab is passed to Metal
/// at a time. Gemma's states write directly into one normalized/interleaved layout,
/// which is released before connector execution.
/// FP32 execution is the numerical baseline, not BF16 sampler parity.
public final class LTX25TextEncoder {
  public struct Progress: Sendable {
    public let stage: String
    public let completed: Int
    public let total: Int
  }
  public struct Output: Sendable {
    public let video: [Float]
    public let audio: [Float]
    public let tokenIDs: [Int]
    public let tokenCount: Int
    public let videoWidth: Int
    public let audioWidth: Int
    public var videoShape: [Int] { [1, tokenCount, videoWidth] }
    public var audioShape: [Int] { [1, tokenCount, audioWidth] }
  }
  private let executionGate = TextExecutionGate()
  private let fixed: SafeTensorFile
  private let layers: [SafeTensorFile]
  private let connector: SafeTensorFile
  private let connectorPrefix: String
  private let tokenizer: GemmaTokenizer
  private let types: [String]

  public init(gemmaRoot:URL,connectorURL:URL) throws {
    let inventory=try TextCheckpoint(gemmaRoot:gemmaRoot,connectorURL:connectorURL)
    fixed=inventory.fixed; layers=inventory.layers; connector=inventory.connector
    connectorPrefix=inventory.connectorPrefix; tokenizer=inventory.tokenizer; types=inventory.types
  }
  private static func configuration(_ type: String) -> GemmaLayerConfiguration {
    var c = GemmaLayerConfiguration()
    if type == "full_attention" {
      c.kvHeads = 1; c.headWidth = 512; c.window = nil; c.keyEqualsValue = true
      c.theta = 1000000; c.rotaryFraction = 0.25
    }
    return c
  }
  public func tokenize(_ prompt: String, maxLength: Int = 1024) throws -> [Int] {
    try tokenizer.encode(prompt, maxLength: maxLength)
  }
  public func encode(prompt: String, maxLength: Int = 1024,
    configuration: TextEncodingConfiguration = .init(),
    progress: (Progress) throws -> Void = { _ in }) throws -> Output {
    return try executionGate.run {
    let ids = try tokenize(prompt, maxLength: maxLength)
    // Admit the entire component before embeddings, GPU allocation or callbacks.
    _ = try TextEncodingPlan(promptTokens: ids.count, configuration: configuration)
    try Task.checkCancellation()
    let gpu = try TextMatrixGPU()
    let (videoProjection, audioProjection) = try project(ids: ids, gpu: gpu, progress: progress)
    let video = try connect(videoProjection, promptTokens: ids.count, modality: "video", width: 4096, gpu: gpu, progress: progress)
    let audio = try connect(audioProjection, promptTokens: ids.count, modality: "audio", width: 2048, gpu: gpu, progress: progress)
    try Task.checkCancellation()
    return Output(video: video, audio: audio, tokenIDs: ids, tokenCount: 1024, videoWidth: 4096, audioWidth: 2048)
    }
  }
  private func project(ids: [Int], gpu: TextMatrixGPU, progress: (Progress) throws -> Void) throws -> ([Float],[Float]) {
    let weights = TextWeights(file: fixed, prefix: "")
    var hidden: [Float] = []
    for id in ids {
      guard (0..<262144).contains(id) else { throw TextEncodingError.invalid("Invalid Gemma vocabulary ID.") }
      hidden += try weights.rows("model.embed_tokens.weight", id..<(id+1)).map { $0 * Float(3840).squareRoot() }
    }
    let states = try TextStateAccumulator(tokens: ids.count, width: 3840, layers: 49)
    try states.append(hidden)
    for index in layers.indices {
      try Task.checkCancellation()
      hidden = try autoreleasepool {
        try GemmaLayer.evaluate(hidden, tokens: ids.count, configuration: Self.configuration(types[index]),
          weights: TextWeights(file: layers[index], prefix: "model.layers.\(index)."), gpu: gpu)
      }
      try states.append(hidden)
      try progress(Progress(stage: "gemma", completed: index+1, total: layers.count))
    }
    let stacked = try states.take()
    hidden.removeAll()
    let video = try weights.linear("text_embedding_projection.video_aggregate_embed", stacked,
      tokens: ids.count, width: 188160, output: 4096, bias: true, scale: sqrt(4096/Float(3840)), gpu: gpu)
    try progress(Progress(stage: "aggregation", completed: 1, total: 2))
    let audio = try weights.linear("text_embedding_projection.audio_aggregate_embed", stacked,
      tokens: ids.count, width: 188160, output: 2048, bias: true, scale: sqrt(2048/Float(3840)), gpu: gpu)
    try progress(Progress(stage: "aggregation", completed: 2, total: 2))
    return (video,audio)
  }
  private func connect(_ projection: [Float], promptTokens: Int, modality: String, width: Int,
    gpu: TextMatrixGPU, progress: (Progress) throws -> Void) throws -> [Float] {
    let stem = connectorPrefix + modality + "_embeddings_connector."
    let w = TextWeights(file: connector, prefix: stem)
    let registers = try w.rows("learnable_registers", 0..<128)
    var hidden = projection
    for token in promptTokens..<1024 { hidden += registers[(token%128)*width..<((token%128)+1)*width] }
    for index in 0..<8 {
      hidden = try autoreleasepool {
        try TextConnector.evaluate(hidden, tokens: 1024, width: width,
          weights: TextWeights(file: connector, prefix: stem + "transformer_1d_blocks.\(index)."), gpu: gpu)
      }
      try progress(Progress(stage: modality + "_connector", completed: index+1, total: 8))
    }
    return TextMath.rms(hidden, width: width)
  }
}

/// A progress callback may synchronously reenter the encoder. Reject before any
/// second weighted stage or GPU allocation; defer also releases on thrown errors.
final class TextExecutionGate {
  private let lock = NSLock()
  func run<T>(_ body: () throws -> T) throws -> T {
    guard lock.try() else { throw TextEncodingError.invalid("This text encoder is already encoding a prompt.") }
    defer { lock.unlock() }
    return try body()
  }
}
