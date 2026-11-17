import Darwin
import Foundation
import MLX
import MLXRandom
import ImageLayout

public struct MOSSRegionalContextReceipt: Codable, Sendable {
  public let group: MOSSRegionalPrompt
  public let tokenCount: Int
  public let promptSHA256: String
}

public struct MOSSSourceReceipt: Codable, Sendable {
  public let source: MOSSReferenceImage
  public let sourceWidth, sourceHeight: Int
  public let preparedRGBSHA256: String
  public var initialLatentRMSE: Float?
  public var finalLatentRMSE: Float?
}

public struct MOSSRunReceipt: Codable, Sendable {
  public var schema = "weetodd-moss-image-result-v1"
  public var eligibility = "allowed"
  public let requestID, fingerprint: String
  public let width, height: Int
  public let seed: UInt64
  public let normalizedRequest: MOSSImageRequest
  public let effectivePrompt: String
  public let positiveTokens, negativeTokens, estimatedPeakBytes, retainedBlocks: Int
  public var imagePath: String?
  public var imageSHA256: String?
  public var stageTimings: [String: Double] = [:]
  public var peakMLXBytes = 0, peakRSSBytes = 0, modelEvaluations = 0
  public var releasedStages: [String] = []
  public var sigmas: [Float] = []
  public var operatorTimings: [String: MOSSOperatorTiming]?
  public var profilingNote: String?
  public var generationMode: String?
  public var sourceReconstruction: MOSSSourceReceipt?
  public var conditioningMode: String?
  public var regionalContexts: [MOSSRegionalContextReceipt]?
  public var arithmeticPolicy = "BF16 GEMMs, FP32 residuals/CFG/Euler, FP32 small VAE"
  public init(_ prepared: MOSSPreparedRequest) {
    let request = prepared.request
    requestID = request.requestID; fingerprint = prepared.fingerprint
    width = request.configuration.width; height = request.configuration.height; seed = request.configuration.seed
    normalizedRequest = request; effectivePrompt = prepared.effectivePrompt
    positiveTokens = prepared.positiveTokens.count; negativeTokens = prepared.negativeTokens?.count ?? 0
    estimatedPeakBytes = prepared.estimatedPeakBytes; retainedBlocks = prepared.retainedBlocks
    if let groups = prepared.regionalContexts {
      conditioningMode = "regional-attention-v2"
      regionalContexts = groups.map { .init(group: $0.group, tokenCount: $0.tokens.count, promptSHA256: MOSSImagePreflight.digest(Data($0.group.prompt.utf8))) }
    }
  }
}

public enum MOSSImageRunner {
  public typealias Progress = (String, String, Double) -> Void
  public static func render(_ prepared: MOSSPreparedRequest, captureDirectory: URL? = nil,
    vaeRoundtrip: Bool = false,
    progress: @escaping Progress = { _, _, _ in }) throws -> MOSSRunReceipt {
    guard let output = prepared.request.outputPath, output.hasPrefix("/"), !output.utf8.contains(0) else { throw MOSSError("A local output PNG path is required.") }
    guard !FileManager.default.fileExists(atPath: output) else { throw MOSSError("Output exists. Choose a new take path.") }
    if let captureDirectory { try FileManager.default.createDirectory(at: captureDirectory, withIntermediateDirectories: true) }
    let started = Date(), settings = prepared.request.configuration
    let lifetime = MOSSStageLifetime()
    var receipt = MOSSRunReceipt(prepared)
    let profiler = settings.profile == true ? MOSSProfiler() : nil
    receipt.arithmeticPolicy = "\(settings.quantizationBits.map { "Affine Q\($0) transformer / FP32 scales" } ?? "BF16 transformer GEMMs"), FP32 residuals/CFG/\(settings.sampler), FP32 small VAE; native MLX seeded noise"
    Memory.cacheLimit = 256 * 1024 * 1024
    Memory.peakMemory = Memory.activeMemory
    func index(_ directory: URL, role: String) throws -> MOSSCheckpointIndex {
      let value = try MOSSCheckpointIndex(directory: directory)
      guard try value.identity() == prepared.sourceIdentities[role] else { throw MOSSError("\(role) changed before its weighted stage.") }
      return value
    }
    let shared = URL(fileURLWithPath: prepared.request.sharedDirectory)
    guard !vaeRoundtrip || prepared.request.sourceImage != nil else { throw MOSSError("VAE roundtrip requires an explicitly bound source image.") }
    receipt.generationMode = vaeRoundtrip ? "vae-roundtrip-control" : (prepared.request.sourceImage == nil ? "text-and-layout" : "source-reconstruction")
    let sourceLatents: MLXArray?
    if let reference = prepared.request.sourceImage {
      let encodingStart = Date()
      sourceLatents = try lifetime.withStage(.encoding) {
        progress("encoding", "Preparing source reconstruction", 0)
        let image = try reference.prepare(width: settings.width, height: settings.height)
        receipt.sourceReconstruction = .init(source: reference, sourceWidth: image.sourceWidth,
          sourceHeight: image.sourceHeight, preparedRGBSHA256: image.rgbSHA256)
        if let captureDirectory { try MOSSVAEOutput.writePNG(image.pixels, to: captureDirectory.appendingPathComponent("source-resized.png")) }
        let source = MOSSWeightSource(index: try index(shared.appendingPathComponent("vae"), role: "vae"), profiler: profiler, role: "vae-encoder")
        defer { source.release(); progress("encoding_released", "Source VAE encoder released", 0.01) }
        let latent = try MOSSSmallVAE(weights: source).encode(image: image.pixels) { progress("encoding", $0, 0.005) }
        if let captureDirectory { try MLX.save(arrays: ["source": latent], url: captureDirectory.appendingPathComponent("source-latents.safetensors")) }
        return latent
      }
      receipt.stageTimings["encoding"] = Date().timeIntervalSince(encodingStart)
    } else { sourceLatents = nil }
    let latents: MLXArray
    if vaeRoundtrip { latents = sourceLatents! } else {
    let conditioningStart = Date()
    let embeddings = try lifetime.withStage(.conditioning) { () throws -> (MLXArray?, MLXArray?, [MLXArray]?) in
      progress("conditioning", "Loading MOSS Qwen3.5-9B", 0.01)
      let source = MOSSWeightSource(index: try index(shared.appendingPathComponent("text_encoder"), role: "text_encoder"), profiler: profiler, role: "qwen")
      defer { source.release(); progress("conditioning_released", "Text encoder released", 0.12) }
      let model = MOSSQwenTextEncoder(weights: source)
      var positive: MLXArray?, regional: [MLXArray]?
      if let groups = prepared.regionalContexts {
        regional = []
        for (index, group) in groups.enumerated() {
          regional!.append(try model.encode(tokens: group.tokens) { done, total in
            progress("conditioning", "Region \(index+1)/\(groups.count) · Qwen layer \(done)/\(total)", 0.02 + 0.05 * (Double(index)+Double(done)/Double(total))/Double(groups.count))
          })
        }
      } else {
        positive = try model.encode(tokens: prepared.positiveTokens) { done, total in
          progress("conditioning", "MOSS prompt · Qwen layer \(done)/\(total)", 0.02 + 0.05 * Double(done) / Double(total))
        }
      }
      let negative = try prepared.negativeTokens.map { tokens in
        try model.encode(tokens: tokens) { done, total in
          progress("conditioning", "Negative prompt · Qwen layer \(done)/\(total)", 0.07 + 0.04 * Double(done) / Double(total))
        }
      }
      if let captureDirectory {
        var arrays: [String:MLXArray] = [:]
        if let positive { arrays["positive"] = positive }
        if let regional { for (index, value) in regional.enumerated() { arrays["region_\(index)"] = value } }
        if let negative { arrays["negative"] = negative }
        try MLX.save(arrays: arrays, url: captureDirectory.appendingPathComponent("conditioning.safetensors"))
      }
      return (positive, negative, regional)
    }
    receipt.stageTimings["conditioning"] = Date().timeIntervalSince(conditioningStart)
    let samplingStart = Date()
    latents = try lifetime.withStage(.transformer) { () throws -> MLXArray in
      progress("sampling", "Preparing MOSS transformer", 0.13)
      let source = MOSSWeightSource(index: try index(URL(fileURLWithPath: prepared.request.transformerDirectory), role: "transformer"), retainedBlocks: prepared.retainedBlocks, profiler: profiler, role: "transformer")
      defer { source.release(); progress("transformer_released", "Transformer released", 0.85) }
      let active = prepared.request.loras.filter { $0.enabled && $0.strength != 0 }
      let adapters = try active.enumerated().map { i, descriptor in
        let adapter = try MOSSLoRA(descriptor: descriptor, profiler: profiler)
        guard try adapter.source.index.identity() == prepared.sourceIdentities["adapter_\(i)"] else { throw MOSSError("Adapter changed before sampling.") }
        return adapter
      }
      defer { adapters.forEach { $0.release() } }
      let model = MOSSTransformer(weights: MOSSAdaptedWeights(base: source, adapters: adapters))
      let refinementStart = Date()
      let positive: MOSSPreparedConditioning
      if let contexts = embeddings.2, let groups = prepared.regionalContexts {
        positive = try model.prepareRegional(contexts: contexts, groups: groups.map(\.group), height: settings.height/16, width: settings.width/16)
      } else { positive = try model.prepare(context: embeddings.0!, height: settings.height/16, width: settings.width/16) }
      let negative = try embeddings.1.map { try model.prepare(context: $0, height: settings.height / 16, width: settings.width / 16) }
      if profiler != nil { profiler?.record("transformer/text refinement", seconds: Date().timeIntervalSince(refinementStart)) }
      let noise = MLXRandom.normal([1, 128, settings.height / 16, settings.width / 16], dtype: .float32, key: MLXRandom.key(settings.seed))
      var initial = noise
      var sigmas = try settings.schedule == "simple" ? MOSSSchedule.simple(steps: settings.steps, shift: settings.shift) : MOSSSchedule.sigmas(steps: settings.steps, shift: settings.shift)
      if let sourceLatents, let reference = prepared.request.sourceImage {
        initial = try MOSSReferenceImage.initialize(source: sourceLatents, noise: noise, sigma: reference.noiseLevel)
        sigmas = try MOSSReferenceImage.sigmas(sigmas, noiseLevel: reference.noiseLevel)
        receipt.sourceReconstruction?.initialLatentRMSE = sqrt(mean(square(initial - sourceLatents))).item(Float.self)
      }
      receipt.sigmas = sigmas
      if let captureDirectory { try MLX.save(arrays: ["noise": noise, "initial": initial], url: captureDirectory.appendingPathComponent("initial.safetensors")) }
      let predict: (MLXArray, Double, Bool, Set<Int>) throws -> MLXArray = { latent, timestep, isNegative, skip in
          guard !isNegative || negative != nil else { throw MOSSError("Negative conditioning is missing.") }
          let prediction = try model.predict(latents: latent, timestep: timestep, conditioning: isNegative ? negative! : positive, skipLayers: skip)
          receipt.modelEvaluations += 1
          if receipt.modelEvaluations == 1, let captureDirectory {
            try MLX.save(arrays: ["prediction": prediction], url: captureDirectory.appendingPathComponent("first-prediction.safetensors"))
          }
          return prediction
        }
      let onStep: (Int, Int, MLXArray) throws -> Void = { done, total, _ in
          progress("sampling", "MOSS sampling \(done)/\(total)", 0.14 + 0.70 * Double(done) / Double(total))
        }
      let result: MLXArray
      if settings.sampler == "er_sde" {
        result = try MOSSERSDESampler.sample(initial: initial, sigmas: sigmas, guidance: settings.guidance,
          skipLayers: Set(settings.skipLayers), skipStart: settings.skipStart, skipStop: settings.skipStop, shift: settings.shift,
          noise: { i in MLXRandom.normal(initial.shape, dtype: .float32, key: MLXRandom.key(settings.seed &+ UInt64(i) &+ 1)) }, predict: predict, onStep: onStep)
      } else {
        result = try MOSSEulerSampler.sample(initial: initial, sigmas: sigmas, guidance: settings.guidance,
          skipLayers: Set(settings.skipLayers), skipStart: settings.skipStart, skipStop: settings.skipStop, predict: predict, onStep: onStep)
      }
      eval(result)
      if let captureDirectory { try MLX.save(arrays: ["latents": result], url: captureDirectory.appendingPathComponent("final-latents.safetensors")) }
      return result
    }
    receipt.stageTimings["sampling"] = Date().timeIntervalSince(samplingStart)
    }
    if let sourceLatents { receipt.sourceReconstruction?.finalLatentRMSE = sqrt(mean(square(latents - sourceLatents))).item(Float.self) }
    let decodeStart = Date()
    let image = try lifetime.withStage(.decoding) { () throws -> MLXArray in
      progress("decoding", "Loading small FLUX.2 VAE", 0.86)
      let source = MOSSWeightSource(index: try index(shared.appendingPathComponent("vae"), role: "vae"), profiler: profiler, role: "vae")
      defer { source.release(); progress("vae_released", "Image VAE released", 0.98) }
      return try MOSSSmallVAE(weights: source).decode(packedLatents: latents) { message in progress("decoding", message, 0.90) }
    }
    guard image.shape == [1, settings.height, settings.width, 3] else { throw MOSSError("MOSS output geometry differs from the request.") }
    try Task.checkCancellation()
    let outputURL = URL(fileURLWithPath: output)
    try MOSSVAEOutput.writePNG(image, to: outputURL)
    receipt.imagePath = output; receipt.imageSHA256 = MOSSImagePreflight.digest(try Data(contentsOf: outputURL))
    receipt.stageTimings["decoding"] = Date().timeIntervalSince(decodeStart)
    receipt.stageTimings["total"] = Date().timeIntervalSince(started)
    receipt.peakMLXBytes = Memory.peakMemory
    var usage = rusage(); if getrusage(RUSAGE_SELF, &usage) == 0 { receipt.peakRSSBytes = Int(usage.ru_maxrss) }
    receipt.releasedStages = lifetime.released
    receipt.operatorTimings = profiler?.records
    if profiler != nil { receipt.profilingNote = "Synchronized diagnostic operator groups include evaluation overhead; nested groups overlap. Do not sum groups or compare instrumented latency to ordinary runs." }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(receipt).write(to: outputURL.appendingPathExtension("json"), options: .atomic)
    progress("complete", "MOSS image saved", 1)
    return receipt
  }
}
