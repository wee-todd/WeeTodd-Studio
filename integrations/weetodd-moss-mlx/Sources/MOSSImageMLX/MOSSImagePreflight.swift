import CryptoKit
import Darwin
import Foundation
import ImageLayout

public struct MOSSRegionalTokens: Sendable {
  public let group: MOSSRegionalPrompt
  public let tokens: [Int]
}

public struct MOSSPreparedRequest: Sendable {
  public let request: MOSSImageRequest
  public let effectivePrompt: String
  public let positiveTokens: [Int]
  public let negativeTokens: [Int]?
  public let fingerprint: String
  public let retainedBlocks: Int
  public let sourceIdentities: [String: String]
  public let estimatedPeakBytes: Int
  public var regionalContexts: [MOSSRegionalTokens]? = nil
}

public enum MOSSImagePreflight {
  public static func inspect(_ request: MOSSImageRequest) async throws -> MOSSPreparedRequest {
    let prompt = try request.effectivePrompt()
    let transformer = URL(fileURLWithPath: request.transformerDirectory)
    let shared = URL(fileURLWithPath: request.sharedDirectory)
    guard transformer.path.hasPrefix("/"), shared.path.hasPrefix("/"), !request.transformerDirectory.utf8.contains(0),
      !request.sharedDirectory.utf8.contains(0) else { throw MOSSError("Link local MOSS component folders.") }
    let text = shared.appendingPathComponent("text_encoder"), vae = shared.appendingPathComponent("vae")
    let configs = [transformer.appendingPathComponent("config.json"), text.appendingPathComponent("config.json"), vae.appendingPathComponent("config.json")]
    let data = try configs.map { try bounded($0, maximum: 1024 * 1024) }
    try MOSSModelContract.validateTransformerConfig(data[0]); try MOSSModelContract.validateQwenConfig(data[1]); try MOSSModelContract.validateVAEConfig(data[2])
    let transformerIndex = try MOSSCheckpointIndex(directory: transformer), textIndex = try MOSSCheckpointIndex(directory: text), vaeIndex = try MOSSCheckpointIndex(directory: vae)
    guard request.configuration.quantizationBits == transformerIndex.quantization?.bits else { throw MOSSError("Choose the precision recorded in the linked transformer’s native affine manifest.") }
    try validate(transformerIndex, shapes: MOSSModelContract.transformerShapes(), selected: { $0 != "rope.bands" })
    try validate(textIndex, shapes: MOSSModelContract.qwenShapes(), selected: { $0.hasPrefix("language_model.") })
    guard textIndex.descriptors.keys.allSatisfy({ $0.hasPrefix("language_model.") || $0.hasPrefix("visual.") }) else { throw MOSSError("Unknown text encoder component namespace.") }
    try validate(vaeIndex, shapes: MOSSModelContract.vaeShapes(), selected: {
      ($0.hasPrefix("decoder.") || $0.hasPrefix("post_quant_conv.") || $0.hasPrefix("bn.")) && $0 != "bn.num_batches_tracked"
    })
    if let reference = request.sourceImage {
      try MOSSModelContract.validateEncoderConfig(data[2])
      try validate(vaeIndex, shapes: MOSSModelContract.encoderShapes(), selected: { $0.hasPrefix("encoder.") || $0.hasPrefix("quant_conv.") })
      _ = try reference.prepare(width: request.configuration.width, height: request.configuration.height)
    }
    let tokenizerDirectory = shared.appendingPathComponent("tokenizer")
    let tokenizer = try await MOSSQwenTokenizer(directory: tokenizerDirectory)
    let positive = try tokenizer.tokens(prompt)
    let negative = request.configuration.guidance == 1 ? nil : try tokenizer.tokens(request.configuration.negativePrompt)
    var regional: [MOSSRegionalTokens]?
    if request.configuration.regionalLayout == true, let layout = request.layout {
      regional = try MOSSRegionalLayout.groups(layout).map { .init(group: $0, tokens: try tokenizer.tokens($0.prompt)) }
      guard regional!.reduce(0, { $0+$1.tokens.count }) <= 8192 else { throw MOSSError("Regional layout exceeds its combined 8192-token budget. Shorten subject descriptions or lettering.") }
    }
    var identities = ["transformer": try transformerIndex.identity(), "text_encoder": try textIndex.identity(), "vae": try vaeIndex.identity()]
    if let reference = request.sourceImage { identities["sourceImage"] = reference.sha256 }
    let active = request.loras.filter { $0.enabled && $0.strength != 0 }
    guard active.count <= 8, Set(active.map { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path }).count == active.count else { throw MOSSError("Use at most eight distinct MOSS adapters.") }
    var adapterBytes = 0
    for (i, descriptor) in active.enumerated() {
      let adapter = try MOSSLoRA(descriptor: descriptor)
      adapterBytes += adapter.source.index.descriptors.values.reduce(0) { total, tensor in
        total + Int(tensor.byteCount) * (tensor.dtype == "F32" ? 1 : 2)
      }
      identities["adapter_\(i)"] = try adapter.source.index.identity()
      if descriptor.profile == "sft_4step_v2" {
        guard request.modelVariant == "sft", request.configuration.guidance == 1,
          (4...8).contains(request.configuration.steps), request.configuration.shift == 5,
          request.configuration.sampler == "euler", request.configuration.schedule == "release", descriptor.strength == 1 else {
          throw MOSSError("The pinned four-step adapter requires SFT, strength 1, CFG 1, shift 5, 4–8 release Euler steps.")
        }
      }
    }
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    if FileManager.default.isExecutableFile(atPath: executable.path) {
      identities["worker"] = try MOSSLoRA.sha256(executable)
      let metal = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
      if FileManager.default.fileExists(atPath: metal.path) { identities["metal"] = try MOSSLoRA.sha256(metal) }
    }
    for (url, bytes) in zip(configs, data) { identities[url.path] = digest(bytes) }
    for name in ["tokenizer.json", "tokenizer_config.json"] {
      identities[name] = digest(try bounded(tokenizerDirectory.appendingPathComponent(name), maximum: 64 * 1024 * 1024))
    }
    var retainedBlocks = 0, retainedBytes = 0
    let budget = request.configuration.weightCacheGB * 1_073_741_824
    for i in 0..<42 {
      let bytes = transformerIndex.descriptors.filter { $0.key.hasPrefix("layers.\(i).") }.values.reduce(0) { $0 + Int($1.byteCount) }
      if bytes > budget - retainedBytes { break }
      retainedBlocks += 1; retainedBytes += bytes
    }
    let contextRows = max(max(positive.count, negative?.count ?? 0), regional?.reduce(0, { $0+$1.tokens.count }) ?? 0)
    let tokenRows = request.configuration.width * request.configuration.height / 256 + contextRows
    // Conservative staged maximum, with independent room for Qwen/DiT/decoder activations and allocator state.
    let textBytes = textIndex.descriptors.filter { $0.key.hasPrefix("language_model.") }.values.reduce(0) { $0 + Int($1.byteCount) }
    let estimate = estimate(textBytes: textBytes, retainedBytes: retainedBytes, adapterBytes: adapterBytes,
      tokenRows: tokenRows, pixels: request.configuration.width * request.configuration.height)
    guard estimate <= availableMemory() else { throw MOSSError("MOSS needs about \(estimate / 1_073_741_824 + 1) GiB of available working memory for this recipe. Close other jobs or choose a smaller/quantized profile.") }
    var normalized = request; normalized.expectedFingerprint = nil; normalized.outputPath = nil
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var fingerprintData = try encoder.encode(normalized)
    fingerprintData.append(Data(prompt.utf8)); fingerprintData.append(try encoder.encode(identities))
    let fingerprint = digest(fingerprintData)
    if let expected = request.expectedFingerprint, expected != fingerprint { throw MOSSError("MOSS components or settings changed after Check Settings. Check them again.") }
    var prepared = MOSSPreparedRequest(request: request, effectivePrompt: prompt, positiveTokens: positive, negativeTokens: negative,
      fingerprint: fingerprint, retainedBlocks: retainedBlocks, sourceIdentities: identities, estimatedPeakBytes: estimate)
    prepared.regionalContexts = regional
    return prepared
  }
  static func estimate(textBytes: Int, retainedBytes: Int, adapterBytes: Int, tokenRows: Int, pixels: Int) -> Int {
    let working = max(2 * 1_073_741_824, tokenRows * 8192 * 8)
    return max(textBytes + working, max(retainedBytes + adapterBytes + working + 1_073_741_824, pixels * 384 * 4)) + 2 * 1_073_741_824
  }
  static func validate(_ index: MOSSCheckpointIndex, shapes: [String: [Int]], selected: (String) -> Bool) throws {
    if let manifest = index.quantization {
      guard manifest.logicalShapes == shapes else { throw MOSSError("Quantized logical tensor shapes differ from this MOSS component.") }
      var expected = Set(shapes.keys)
      for (name, shape) in shapes {
        if manifest.quantizedNames.contains(name) {
          try index.require(name, shape: [shape[0], shape[1] * manifest.bits / 32], dtypes: ["U32"])
          for suffix in [".scales", ".biases"] {
            try index.require(name + suffix, shape: [shape[0], shape[1] / 64], dtypes: ["F32"]); expected.insert(name + suffix)
          }
        } else { try index.require(name, shape: shape) }
      }
      guard Set(index.descriptors.keys) == expected else { throw MOSSError("Unconsumed native quantization tensors.") }
      try index.checkUnchanged(); return
    }
    for (name, shape) in shapes { try index.require(name, shape: shape) }
    let actual = Set(index.descriptors.keys.filter(selected))
    guard actual == Set(shapes.keys) else { throw MOSSError("Checkpoint has unexpected or unconsumed component tensors: \(actual.subtracting(shapes.keys).sorted().prefix(3))") }
    try index.checkUnchanged()
  }
  static func bounded(_ url: URL, maximum: Int) throws -> Data {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    let bytes = try handle.read(upToCount: maximum + 1) ?? Data()
    guard bytes.count <= maximum else { throw MOSSError("Oversized model metadata.") }; return bytes
  }
  static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
  static func availableMemory() -> Int {
    var stats = vm_statistics64_data_t(), count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    let physical = Int(ProcessInfo.processInfo.physicalMemory)
    guard result == KERN_SUCCESS else { return physical / 2 }
    return min(physical * 3 / 4, Int(UInt64(stats.free_count) + UInt64(stats.inactive_count)) * Int(getpagesize()))
  }
}
