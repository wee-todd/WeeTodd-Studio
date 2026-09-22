import Foundation
import LLM
import NNC

/// A serial, job-owned 4B runtime. All mutable attention state belongs to one run;
/// only model parameters and explicitly budgeted visual features survive a run.
/// Call run/unload on one executor; the cancellation callback may read an atomic flag.
public final class LocalTextSession {
  private var runtime: ResidentQwenRuntime?
  public var isResident: Bool { runtime != nil }
  public init() {}

  public func unload() {
    let graph = runtime?.graph
    graph?.joined()
    runtime = nil
    graph?.garbageCollect()
  }

  public func run(_ value: [String: Any], progress: ([String: Any]) -> Void,
                  cancelled: () -> Bool) throws -> [String: Any] {
    do {
      guard !cancelled() else { throw LocalTextError("text_cancelled") }
      let request = try LocalTextRequest(value)
      guard request.model == .qwen35_4B else { throw LocalTextError("text_session_model_unsupported") }
      _ = try LocalTextGeneration.promptTokens(request)
      let fingerprint = try ResidentQwenRuntime.modelIdentity(request.modelPath)
      if let runtime = runtime, runtime.fingerprint != fingerprint {
        throw LocalTextError("text_session_model_changed")
      }
      if runtime == nil { runtime = ResidentQwenRuntime(path: request.modelPath, fingerprint: fingerprint) }
      return try runtime!.run(request, progress: progress, cancelled: cancelled)
    } catch {
      // The worker cannot accidentally retain partially initialized weights after failure.
      unload()
      if let error = error as? LocalTextError { throw error }
      throw LocalTextError("text_generation_failed")
    }
  }
}

private final class ResidentQwenRuntime {
  typealias Shape = (cached: Int, length: Int, outputs: Int)
  struct VisionEntry {
    let prepared: LocalVisionInput
    let features: DynamicGraph.Tensor<Float16>
    var bytes: Int {
      prepared.patches.shape.reduce(1, *) * MemoryLayout<Float>.stride
        + features.shape.reduce(1, *) * MemoryLayout<Float16>.stride
    }
  }

  let path: String
  let fingerprint: String
  let graph = DynamicGraph()
  private let stream = StreamContext(.GPU(0))
  private let config = Qwen3_5ModelConfiguration.qwen3_5_4B
  private let visionConfig = Qwen3_5VisionConfiguration.qwen3_5_4B
  private let decoder: ModelBuilder<Shape>
  private let vision: ModelBuilder<[(t: Int, h: Int, w: Int)]>
  private var positionWeights: NNC.AnyTensor?
  private var decoderLoaded = false
  private var visionLoaded = false
  private var cache = LocalVisionCache<VisionEntry>()
  private var requests = 0
  private var decoderLoads = 0
  private var visionLoads = 0
  private var visionEncodes = 0
  private var visionHits = 0

  init(path: String, fingerprint: String) {
    self.path = path; self.fingerprint = fingerprint
    // A single injection-capable decoder handles both request kinds. ModelBuilder
    // retains parameters while replacing the shape-dependent computation graph.
    decoder = ModelBuilder<Shape> { shape, _ in
      Qwen3_5CausalLM(Float16.self, tokenLength: shape.length,
        cachedTokenLength: shape.cached, configuration: .qwen3_5_4B,
        includeLogits: true, outputCacheStates: true, tieEmbedding: true,
        injectEmbeddings: true, lastNumberOfTokens: shape.outputs)
    }
    vision = ModelBuilder<[(t: Int, h: Int, w: Int)]> { grids, _ in
      Qwen3_5VisionTransformer(Float.self, gridThw: grids, configuration: .qwen3_5_4B)
    }
    decoder.maxConcurrency = .limit(4); vision.maxConcurrency = .limit(4)
  }

  static func modelIdentity(_ path: String) throws -> String {
    let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
    guard let file = try? FileHandle(forReadingFrom: url) else { throw LocalTextError("text_model_unavailable") }
    defer { try? file.close() }
    guard try file.read(upToCount: 16) == Data("SQLite format 3\0".utf8) else {
      throw LocalTextError("text_model_invalid_store")
    }
    var records: [[String: Any]] = []
    for component in [url.path, url.path + "-tensordata"] {
      guard let attributes = try? FileManager.default.attributesOfItem(atPath: component),
        attributes[.type] as? FileAttributeType == .typeRegular,
        let bytes = attributes[.size] as? NSNumber,
        let modified = attributes[.modificationDate] as? Date else {
        throw LocalTextError("text_model_unavailable")
      }
      records.append(["path": component, "size": bytes,
        "modified": modified.timeIntervalSinceReferenceDate,
        "inode": attributes[.systemFileNumber] as? NSNumber ?? 0])
    }
    return String(decoding: try JSONSerialization.data(withJSONObject: records, options: [.sortedKeys]), as: UTF8.self)
  }

  private func check(_ cancelled: () -> Bool) throws {
    guard !cancelled() else { throw LocalTextError("text_cancelled") }
  }

  private func milliseconds(_ date: Date) -> Double { Date().timeIntervalSince(date) * 1000 }

  func run(_ request: LocalTextRequest, progress: ([String: Any]) -> Void,
           cancelled: () -> Bool) throws -> [String: Any] {
    try graph.withStream(stream) {
      try graph.withNoGrad {
        let result = try execute(request, progress: progress, cancelled: cancelled)
        let cleanup = Date()
        graph.joined()
        graph.garbageCollect()
        var final = result
        var timing = result["timing"] as? [String: Any] ?? [:]
        timing["requestCleanupMilliseconds"] = milliseconds(cleanup)
        final["timing"] = timing
        final["totalSeconds"] = (result["totalSeconds"] as? Double ?? 0) + Date().timeIntervalSince(cleanup)
        return final
      }
    }
  }

  private func execute(_ request: LocalTextRequest, progress: ([String: Any]) -> Void,
                       cancelled: () -> Bool) throws -> [String: Any] {
    let started = Date()
    var timing: [String: Any] = ["visionLoadAndCompileMilliseconds": 0.0, "visionEncodeMilliseconds": 0.0]
    let reusedDecoder = decoderLoaded
    progress(["stage": "preparing", "modelResident": decoderLoaded])
    let plainTokens = try LocalTextGeneration.promptTokens(request)
    let imageKey = try LocalVisionIdentity.key(images: request.images, model: fingerprint, cancelled: cancelled)
    var visual: VisionEntry?
    var prepared: LocalVisionInput?
    let reusedVision: Bool
    if !request.images.isEmpty {
      visual = cache.value(for: imageKey)
      reusedVision = visual != nil
      if let visual = visual { prepared = visual.prepared; visionHits += 1 }
      else { prepared = try LocalVisionInput.prepare(request.images, cancelled: cancelled) }
    } else { reusedVision = false }
    let expanded = try prepared.map { try LocalTextGeneration.multimodalTokens(request, grids: $0.grids) }
    let tokens = expanded?.tokenIDs ?? plainTokens
    guard tokens.count + request.maxTokens <= 5120 else { throw LocalTextError("text_context_too_long") }
    timing["requestPreparationMilliseconds"] = milliseconds(started)
    let budget: [String: Any] = ["valid": true,
      "inputBytes": request.systemPrompt.utf8.count + request.prompt.utf8.count,
      "textTokens": plainTokens.count, "inputTokens": tokens.count,
      "imageTokens": tokens.count - plainTokens.count, "imagesUsed": request.images.count,
      "outputTokenBudget": request.maxTokens, "inputByteLimit": 24_000,
      "inputTokenLimit": 4096, "totalTokenLimit": 5120]
    try check(cancelled)
    if let prepared = prepared, visual == nil {
      progress(["stage": "vision", "images": request.images.count, "modelResident": decoderLoaded])
      visual = try encode(prepared, timing: &timing, cancelled: cancelled)
      guard imageKey == (try LocalVisionIdentity.key(images: request.images, model: fingerprint, cancelled: cancelled)) else {
        throw LocalTextError("vision_source_changed")
      }
      cache.insert(visual!, key: imageKey, bytes: visual!.bytes)
    }
    let rotary: Tensor<Float16>
    var ropeDelta = 0
    if let expanded = expanded, let prepared = prepared {
      let positions = Qwen3_5MakeMultimodalPositionIDs(tokenTypeIDs: expanded.tokenTypeIDs,
        imageGridThw: prepared.grids, configuration: visionConfig)
      rotary = Qwen3_5RotaryEmbedding(positionIDs: positions, configuration: config, of: Float16.self)
      ropeDelta = positions.ropeDelta
    } else {
      rotary = Qwen3_5RotaryEmbedding(sequenceLength: tokens.count, configuration: config, of: Float16.self)
    }
    let prompt = graph.variable(Tensor<Int32>(tokens, .CPU, .C(tokens.count)).toGPU(0))
    let rotaryGPU = graph.variable(rotary.toGPU(0))
    var mask = Tensor<Float16>(Array(repeating: 1, count: tokens.count), .CPU, .WC(tokens.count, 1))
    var embeddings = graph.variable(.GPU(0), .WC(tokens.count, config.hiddenSize), of: Float16.self)
    embeddings.full(0)
    if let visual = visual {
      let locations = tokens.indices.filter { tokens[$0] == 248_056 }
      guard locations.count == visual.features.shape[0] else { throw LocalTextError("vision_inputs_invalid") }
      for (row, location) in locations.enumerated() {
        mask[location, 0] = 0
        embeddings[location..<(location + 1), 0..<config.hiddenSize] = visual.features[row..<(row + 1), 0..<config.hiddenSize]
      }
    }
    let maskGPU = graph.variable(mask.toGPU(0))
    // Fresh zeroed state for EVERY request: full KV, convolution and FP32 recurrence.
    var state = makeState(capacity: tokens.count + request.maxTokens + 1)
    let chunkSize = visual == nil ? 512 : tokens.count
    var generated: [Int32] = []
    var loadMilliseconds = 0.0
    let prefillStarted = Date()
    progress(["stage": "prefill", "inputTokens": tokens.count, "modelResident": decoderLoaded])
    for offset in stride(from: 0, to: tokens.count, by: chunkSize) {
      try check(cancelled)
      let length = min(chunkSize, tokens.count - offset)
      let last = offset + length == tokens.count
      let shape: Shape = (offset, length, last ? 1 : 0)
      let part = prompt.reshaped(.C(length), offset: [offset], strides: [1])
      let inputs: [DynamicGraph.AnyTensor] = [part,
        maskGPU.reshaped(.WC(length, 1), offset: [offset, 0], strides: [1, 1]).copied(),
        embeddings.reshaped(.WC(length, config.hiddenSize), offset: [offset, 0], strides: [config.hiddenSize, 1]).copied(),
        rotaryGPU.reshaped(.NHWC(1, length, 1, config.attentionHeadDim),
          offset: [0, offset, 0, 0], strides: [tokens.count * config.attentionHeadDim, config.attentionHeadDim, config.attentionHeadDim, 1]).copied()
      ] + attentionInputs(state, length: offset + length)
      if !decoderLoaded {
        progress(["stage": "loading", "tokens": 0, "modelResident": false])
        let loading = Date()
        // Bootstrap with a nonempty logits row even for a cache-only first
        // chunk. Initializing the quantized GEMV graph with zero output rows
        // produces an invalid native kernel shape; later cache-only chunks are
        // safe after the complete decoder parameters/graph have been established.
        let bootstrapShape: Shape = (offset, length, 1)
        decoder.compile(bootstrapShape, inputs: inputs)
        try graph.openStore(path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)) { store in
          try store.read("text_model", model: decoder, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
        }
        decoder.compile(bootstrapShape, inputs: inputs, isEager: true)
        graph.joined()
        decoderLoaded = true; decoderLoads += 1
        loadMilliseconds += milliseconds(loading)
        try check(cancelled)
        progress(["stage": "prefill", "inputTokens": tokens.count, "modelResident": true])
      }
      let outputs = decoder(shape, inputs: inputs[0], Array(inputs.dropFirst()))
      advanceLinearState(&state, outputs: outputs)
      if last { generated.append(selectToken(outputs[0], multimodal: visual != nil)) }
    }
    graph.joined()
    timing["loadAndCompileMilliseconds"] = loadMilliseconds
    timing["prefillMilliseconds"] = max(0, milliseconds(prefillStarted) - loadMilliseconds)
    let unitMask = graph.variable(Tensor<Float16>([1], .CPU, .WC(1, 1)).toGPU(0))
    let unitEmbedding = graph.variable(.GPU(0), .WC(1, config.hiddenSize), of: Float16.self)
    unitEmbedding.full(0)
    let compileStarted = Date()
    if request.maxTokens > 1, let last = generated.last, !LocalTextGeneration.eos.contains(last) {
      try check(cancelled)
      let capacity = tokens.count + request.maxTokens + 1
      let token = graph.variable(Tensor<Int32>([last], .CPU, .C(1)).toGPU(0))
      let rotation = graph.variable(Qwen3_5RotaryEmbedding(sequenceLength: 1,
        cachedTokenLength: capacity - 1 + ropeDelta, configuration: config, of: Float16.self).toGPU(0))
      // Reserve the largest decode shape once; subsequent token steps reuse the
      // resident parameters and bounded execution scratch below this high-water mark.
      decoder.compile((capacity - 1, 1, 1), inputs: [token, unitMask, unitEmbedding, rotation]
        + attentionInputs(state, length: capacity), isEager: true)
      graph.joined()
    }
    timing["decodeCompileMilliseconds"] = milliseconds(compileStarted)
    let decodeStarted = Date()
    progress(["stage": "writing", "tokens": generated.count, "modelResident": true])
    if request.maxTokens > 1, let first = generated.first, !LocalTextGeneration.eos.contains(first) {
      let previousWatermark = DynamicGraph.queueWatermark
      DynamicGraph.queueWatermark = previousWatermark * 16
      defer { DynamicGraph.queueWatermark = previousWatermark }
      // Keep token feedback on the GPU. Reading the preceding result while the
      // next decoder step runs avoids one full CPU/GPU synchronization per token.
      var feedback = graph.variable(Tensor<Int32>([first], .CPU, .C(1)).toGPU(0))
      let rotations = graph.variable(Qwen3_5RotaryEmbedding(sequenceLength: request.maxTokens - 1,
        cachedTokenLength: tokens.count + ropeDelta, configuration: config, of: Float16.self).toGPU(0))
      var stopped = false
      for step in 0..<(request.maxTokens - 1) {
        try check(cancelled)
        let pending = step > 0 ? feedback.toCPU() : nil
        let offset = tokens.count + step
        let rotation = rotations.reshaped(.NHWC(1, 1, 1, config.attentionHeadDim),
          offset: [0, step, 0, 0],
          strides: [(request.maxTokens - 1) * config.attentionHeadDim, config.attentionHeadDim, config.attentionHeadDim, 1]).copied()
        let outputs = decoder((offset, 1, 1), inputs: feedback,
          [unitMask, unitEmbedding, rotation] + attentionInputs(state, length: offset + 1))
        advanceLinearState(&state, outputs: outputs)
        // The baseline uses FP32 only for text prefill; decode argmax is FP16
        // for both modalities. Preserve that numerical selection policy.
        feedback = Functional.argmax(outputs[0].as(of: Float16.self), axis: 1).reshaped(.C(1))
        if let pending = pending {
          let token = pending[0]
          generated.append(token)
          if generated.count % 8 == 0 { progress(["stage": "writing", "tokens": generated.count, "modelResident": true]) }
          if LocalTextGeneration.eos.contains(token) { stopped = true; break }
        }
      }
      // Drain the final pending result at the budget boundary. The old vision
      // one-shot loop omits this token; returning it also makes truncation honest.
      if !stopped, generated.count < request.maxTokens {
        let pending = feedback.toCPU()
        graph.joined()
        generated.append(pending[0])
      }
    }
    graph.joined()
    try check(cancelled)
    timing["decodeLoopMilliseconds"] = milliseconds(decodeStarted)
    timing["tokensPerSecond"] = Double(max(0, generated.count - 1)) / max(0.001, Date().timeIntervalSince(decodeStarted))
    guard imageKey == (try LocalVisionIdentity.key(images: request.images, model: fingerprint, cancelled: cancelled)),
      fingerprint == (try Self.modelIdentity(path)) else { throw LocalTextError("vision_source_changed") }
    let content = Array(generated.prefix(while: { !LocalTextGeneration.eos.contains($0) }))
    let text = LocalTextGeneration.tokenizer.decode(content).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { throw LocalTextError("text_result_empty") }
    requests += 1
    let reuse: [String: Any] = ["decoderReused": reusedDecoder, "visionReused": reusedVision,
      "requests": requests, "decoderLoads": decoderLoads, "visionLoads": visionLoads,
      "visionEncodes": visionEncodes, "visionCacheHits": visionHits,
      "visionCacheEntries": cache.count, "visionCacheBytes": cache.retainedBytes,
      "visionCacheByteLimit": cache.byteLimit, "maxActiveBatchSize": 1]
    return ["text": text, "model": request.model.rawValue, "inputTokens": tokens.count,
      "imagesUsed": request.images.count, "outputTokens": content.count,
      "truncated": !generated.contains(where: LocalTextGeneration.eos.contains) && generated.count >= request.maxTokens,
      "totalSeconds": Date().timeIntervalSince(started), "timing": timing, "preflight": budget,
      "reuse": reuse, "modelResident": true]
  }

  private func encode(_ input: LocalVisionInput, timing: inout [String: Any], cancelled: () -> Bool) throws -> VisionEntry {
    try check(cancelled)
    let load = Date()
    if positionWeights == nil {
      try graph.openStore(path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)) { store in
        guard let weight = store.read("model.visual.pos_embed.weight", codec: [.jit, .i8x, .ezm7, .externalData]) else {
          throw LocalTextError("text_model_invalid_store")
        }
        positionWeights = weight
      }
    }
    let patches = graph.variable(input.patches.toGPU(0))
    let positions = graph.variable(Qwen3_5VisionPositionEmbedding(weight: positionWeights!, gridThw: input.grids,
      configuration: visionConfig, of: Float.self).toGPU(0))
    let rotary = graph.variable(Qwen3_5VisionRotaryEmbedding(gridThw: input.grids, configuration: visionConfig, of: Float16.self).toGPU(0))
    let offsets = graph.variable(Qwen3_5VisionSequenceOffsets(gridThw: input.grids).offsets.toGPU(0))
    if !visionLoaded {
      vision.compile(input.grids, inputs: patches, positions, rotary, offsets)
      try graph.openStore(path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)) { store in
        try store.read("vision_model", model: vision, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
      }
      visionLoaded = true; visionLoads += 1
    }
    timing["visionLoadAndCompileMilliseconds"] = milliseconds(load)
    try check(cancelled)
    let encoding = Date()
    let output = vision(input.grids, inputs: patches, positions, rotary, offsets)[0].as(of: Float.self)
    let features = DynamicGraph.Tensor<Float16>(from: output).copied()
    graph.joined(); try check(cancelled)
    timing["visionEncodeMilliseconds"] = milliseconds(encoding)
    visionEncodes += 1
    return VisionEntry(prepared: input, features: features)
  }

  private func selectToken(_ logits: DynamicGraph.AnyTensor, multimodal: Bool) -> Int32 {
    let typed = logits.as(of: Float16.self)
    // Match the existing backend's distinct text/vision greedy selectors. Changing
    // multimodal precision is a separate qualification from resident weight reuse.
    let token = multimodal ? Functional.argmax(typed, axis: 1).reshaped(.C(1)).toCPU()
      : Functional.argmax(DynamicGraph.Tensor<Float>(from: typed), axis: 1).reshaped(.C(1)).toCPU()
    graph.joined()
    return token[0]
  }

  private func makeState(capacity: Int) -> [DynamicGraph.AnyTensor] {
    var tensors: [DynamicGraph.AnyTensor] = []
    for layer in 0..<config.layers {
      if config.isLinearAttentionLayer(layer) {
        let convolution = graph.variable(.GPU(0), .NHWC(1, config.linearConvKernel - 1, 1, config.linearConvDim), of: Float16.self)
        let recurrence = graph.variable(.GPU(0), .NHWC(1, config.linearNumValueHeads, config.linearValueHeadDim, config.linearKeyHeadDim), of: Float.self)
        convolution.full(0); recurrence.full(0)
        tensors.append(contentsOf: [convolution, recurrence])
      } else {
        for _ in 0..<2 {
          let attention = graph.variable(.GPU(0), .NHWC(1, capacity, config.keyValueHeads, config.attentionHeadDim), of: Float16.self)
          attention.full(0); tensors.append(attention)
        }
      }
    }
    return tensors
  }

  private func attentionInputs(_ state: [DynamicGraph.AnyTensor], length: Int) -> [DynamicGraph.AnyTensor] {
    state.enumerated().map { index, tensor in
      if config.isLinearAttentionLayer(index / 2) { return tensor }
      return tensor.as(of: Float16.self).reshaped(.NHWC(1, length, config.keyValueHeads, config.attentionHeadDim))
    }
  }

  private func advanceLinearState(_ state: inout [DynamicGraph.AnyTensor], outputs: [DynamicGraph.AnyTensor]) {
    var output = 1
    for layer in 0..<config.layers where config.isLinearAttentionLayer(layer) {
      // No prefix/MTP checkpoints: each output is this step's final state. Copies
      // detach the history graph so earlier requests cannot remain reachable.
      state[layer * 2] = outputs[output].as(of: Float16.self).copied()
      state[layer * 2 + 1] = outputs[output + 1].as(of: Float.self).copied()
      output += 2
    }
  }
}
