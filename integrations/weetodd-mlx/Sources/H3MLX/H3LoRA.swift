import Foundation
import MLX
import TensorIO

public enum H3LoRAQKVLayout: String, Sendable { case auto, nativeInterleaved = "native_interleaved", contiguousQKV = "contiguous_qkv" }
public enum H3LoRAProfile: String, Sendable { case auto, standard, turbo }

public struct H3LoRAAdapter: Sendable {
  public let url: URL
  public let strength: Float
  public let profile: H3LoRAProfile
  public let qkvLayout: H3LoRAQKVLayout
  public let startAfterEvaluations: Int

  public init(url: URL, strength: Float, profile: H3LoRAProfile = .auto,
    qkvLayout: H3LoRAQKVLayout = .auto, startAfterEvaluations: Int = 0) throws {
    guard url.isFileURL, url.path.hasPrefix("/"), strength.isFinite,
      (-10...10).contains(strength), (0...99).contains(startAfterEvaluations) else {
      throw H3CheckpointError.invalid("Invalid H3 LoRA path or strength.")
    }
    self.url = url
    self.strength = strength
    self.profile = profile; self.qkvLayout = qkvLayout
    self.startAfterEvaluations = startAfterEvaluations
  }
  func validate(requestedSteps: Int, samplingMethod: H3SamplingMethod) throws {
    guard startAfterEvaluations < requestedSteps - 1,
      profile != .turbo || (requestedSteps == 5 && samplingMethod == .euler && startAfterEvaluations == 0) else {
      throw H3CheckpointError.invalid("H3 LoRA activation must occur before the final evaluation; explicit Turbo needs the complete four-evaluation Euler schedule.")
    }
  }
  func isActive(evaluation: Int?) -> Bool {
    evaluation.map { $0 >= startAfterEvaluations } ?? true
  }
}

protocol H3LoRAApplying {
  func prepareBlock(index: Int) throws -> H3LoRABlock?
  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool) throws -> MLXArray
  func applyQueued(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool) throws -> MLXArray
}
extension H3LoRAApplying {
  func prepareBlock(index: Int) throws -> H3LoRABlock? { nil }
  func applyQueued(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool) throws -> MLXArray {
    try apply(base:base,input:input,target:target,reorderQKV:reorderQKV)
  }
}

/// Ordered activation-space updates. Each file maps only the active projection;
/// no merged checkpoint or resident adapter-weight stack is created.
final class H3LoRAStack: H3LoRAApplying {
  private let files: [H3LoRAFile]
  private let adapters: [H3LoRAAdapter]
  var evaluation: Int? = nil

  init(adapters: [H3LoRAAdapter]) throws {
    guard (1...8).contains(adapters.count),
      Set(adapters.map { $0.url.standardizedFileURL.path }).count == adapters.count else {
      throw H3CheckpointError.invalid("H3 supports one to eight distinct ordered LoRAs.")
    }
    self.adapters = adapters
    files = try adapters.map { try H3LoRAFile(url: $0.url,
      strength: $0.strength, qkvLayout: $0.qkvLayout, profile: $0.profile, startAfterEvaluations: $0.startAfterEvaluations) }
  }

  func prepareBlock(index: Int) throws -> H3LoRABlock? {
    try H3LoRABlock(index: index, files: zip(files, adapters).compactMap { entry in
      entry.1.isActive(evaluation: evaluation) ? entry.0 : nil
    })
  }

  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    var value = base
    for (file, adapter) in zip(files, adapters) where adapter.isActive(evaluation: evaluation) {
      value = try file.apply(base: value, input: input,
        target: target, reorderQKV: reorderQKV)
    }
    return value
  }
  func applyQueued(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    var value=base
    for (file,adapter) in zip(files,adapters) where adapter.isActive(evaluation:evaluation) {
      value=try file.applyQueued(base:value,input:input,target:target,reorderQKV:reorderQKV)
    }
    return value
  }

}

/// Lazy adapter pairs for one prepared diffusion block and evaluation snapshot.
/// The enclosing H3PreparedBlock drains GPU consumers before closing this scope.
final class H3LoRABlock: H3LoRAApplying {
  let index: Int
  private let files: [H3LoRAFile]
  private var pairs: [ObjectIdentifier: [String: H3LoRAFile.PreparedPair]] = [:]
  private var retiredTargets: Set<String> = []
  private(set) var isClosed = false
  private(set) var acquiredPairCount = 0

  var storageBytes: Int {
    pairs.values.reduce(0) { total, targets in
      total + targets.values.reduce(0) { $0 + $1.a.nbytes + $1.b.nbytes }
    }
  }
  var residentPairCount: Int { pairs.values.reduce(0) { $0 + $1.count } }

  init(index: Int, files: [H3LoRAFile]) throws {
    guard (0..<50).contains(index) else {
      throw H3CheckpointError.invalid("Invalid prepared H3 LoRA block index.")
    }
    try Task.checkCancellation()
    self.index = index
    self.files = files
  }

  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    try compute(base: base, input: input, target: target, reorderQKV: reorderQKV, queued: false)
  }
  func applyQueued(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    try compute(base: base, input: input, target: target, reorderQKV: reorderQKV, queued: true)
  }
  private func compute(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool, queued: Bool) throws -> MLXArray {
    let prefix = "diffusion_model.blocks.\(index)."
    guard !isClosed, !retiredTargets.contains(target),
      ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"].contains(where: { target == prefix + $0 }),
      input.ndim >= 2, base.ndim >= 2 else {
      throw H3CheckpointError.invalid("Invalid or retired prepared H3 LoRA target.")
    }
    try Task.checkCancellation()
    var value = base
    for file in files {
      try file.checkUnchanged()
      let key = ObjectIdentifier(file)
      let pair: H3LoRAFile.PreparedPair
      if let cached = pairs[key]?[target] { pair = cached }
      else {
        guard let loaded = try file.loadPair(target: target,
          inputWidth: input.shape.last!, outputWidth: base.shape.last!) else { continue }
        // Publish only a complete pair whose source identity still matches.
        try file.checkUnchanged()
        try Task.checkCancellation()
        pairs[key, default: [:]][target] = loaded
        acquiredPairCount += 1
        pair = loaded
      }
      guard pair.a.shape[1] == input.shape.last!, pair.b.shape[0] == base.shape.last! else {
        throw H3CheckpointError.invalid("Prepared H3 LoRA projection widths changed.")
      }
      value = try file.apply(pair: pair, base: value, input: input,
        reorderQKV: reorderQKV, queued: queued)
    }
    try Task.checkCancellation()
    return value
  }

  /// Invoke at the same drained branch boundary as the base projection.
  func retire(target: String) {
    for key in Array(pairs.keys) { pairs[key]?.removeValue(forKey: target) }
    retiredTargets.insert(target)
  }
  /// The owner must drain consumers first; clearing this scope adds no fence.
  func close() {
    pairs.removeAll()
    retiredTargets.removeAll()
    isClosed = true
  }
  deinit { close() }
}

/// Apply a LoRA in activation space. H3's installed transformer projections
/// are weight-decoded one at a time, so merging a 2 GB adapter into the base
/// checkpoint would create another large resident copy.
enum H3LoRAProjection {
  static func apply(base: MLXArray, input: MLXArray,
    a: MLXArray, b: MLXArray, alpha: Float, strength: Float,
    reorderQKV: Bool, qkvHeads: Int = 56,
    qkvHeadSize: Int = 128) -> MLXArray {
    let rank = a.shape[0]
    var outputWeight = b
    if reorderQKV {
      outputWeight = b.reshaped([3, qkvHeads, qkvHeadSize, rank])
        .transposed(1, 0, 2, 3).reshaped([b.shape[0], rank])
    }
    let source = input.asType(a.dtype)
    let lowRank = matmul(source, a.T)
    let delta = matmul(lowRank, outputWeight.T)
      * (alpha * strength / Float(rank))
    return base + delta.asType(base.dtype)
  }
}

/// Compatible ComfyUI H3 adapters are validated once from their safetensors
/// headers. Declared projections are mapped only while their block executes,
/// preserving H3's staged weight residency.
final class H3LoRAFile: H3LoRAApplying {
  private enum ScaleLayout {
    case perTargetAlpha
    case bakedIntoB
  }
  let url: URL
  let strength: Float
  let targetCount: Int
  private let file: SafeTensorFile
  private let targets: [String: (rank: Int, alpha: Float)]
  private let qkvLayout: H3LoRAQKVLayout

  init(url: URL, strength: Float, requestedSteps: Int? = nil,
    samplingMethod: H3SamplingMethod = .euler, qkvLayout: H3LoRAQKVLayout = .auto,
    profile: H3LoRAProfile = .auto, startAfterEvaluations: Int = 0,
    requiresStandardProfile: Bool = false) throws {
    guard url.isFileURL, strength.isFinite, (-10...10).contains(strength) else {
      throw H3CheckpointError.invalid("Invalid H3 LoRA path or strength.")
    }
    let file = try SafeTensorFile(url: url)
    let isTurbo = try Self.validateSampling(metadata: file.metadata, requestedSteps: requestedSteps,
      samplingMethod: samplingMethod, profile: profile, startAfterEvaluations: startAfterEvaluations)
    guard !requiresStandardProfile || !isTurbo else {
      throw H3CheckpointError.invalid("Initialized partial H3 refinement requires ordinary full-schedule adapters, not Turbo metadata.")
    }
    let explicitAlpha = file.metadata["target_format"] == "ComfyUI generic LoRA"
      && file.metadata["qkv_fusion"]?.contains("block diagonal B") == true
    let conversion = file.metadata["conversion"] ?? ""
    let bakedScale = file.metadata["converted_layout"] == "comfyui_minimax_h3"
      && file.metadata["format"] == "pt"
      && file.metadata["floating_dtype"] == "bfloat16"
      && file.metadata["baked_scale"] == "0.125"
      && conversion.contains("mlp.fc1 swiglu halves swapped")
      && (conversion.contains("qkv fused contiguous q|k|v (block-diagonal")
        || conversion.contains("qkv block-diag fused (per-projection ranks)"))
    guard explicitAlpha != bakedScale else {
      throw H3CheckpointError.invalid("H3 LoRA metadata or QKV layout is unsupported.")
    }
    let layout: ScaleLayout = explicitAlpha ? .perTargetAlpha : .bakedIntoB
    var supported: [String: (rows: Int, columns: Int)] = [:]
    for group in 0..<52 {
      let prefix = group < 50
        ? "diffusion_model.blocks.\(group)."
        : "diffusion_model.token_refiner.blocks.\(group - 50)."
      for (suffix, rows, columns) in [
        ("attn.qkv_proj", 21504, 5376),
        ("attn.out_proj", 5376, 7168),
        ("mlp.fc1", 28672, 5376),
        ("mlp.fc2", 5376, 14336),
      ] {
        supported[prefix + suffix] = (rows, columns)
      }
    }
    var targets: [String: (rank: Int, alpha: Float)] = [:]
    for aName in file.tensors.keys where aName.hasSuffix(".lora_A.weight") {
        let target = String(aName.dropLast(".lora_A.weight".count))
        guard let dimensions = supported[target], let a = file.tensors[aName],
          a.dtype == "BF16", a.shape.count == 2,
          let rankValue = a.shape.first, (1...512).contains(rankValue),
          a.shape[1] == UInt64(dimensions.columns) else {
          throw H3CheckpointError.invalid("Unsupported H3 LoRA target or down projection: \(target)")
        }
        let rank = Int(rankValue)
        let bName = target + ".lora_B.weight"
        let alphaName = target + ".alpha"
        guard let b = file.tensors[bName], b.dtype == "BF16",
          b.shape == [UInt64(dimensions.rows), rankValue] else {
          throw H3CheckpointError.invalid("Incomplete H3 LoRA target: \(target)")
        }
        let alpha: Float
        switch layout {
        case .perTargetAlpha:
          guard let descriptor = file.tensors[alphaName],
            descriptor.dtype == "F32", descriptor.shape.isEmpty else {
            throw H3CheckpointError.invalid("Incomplete H3 LoRA target: \(target)")
          }
          alpha = try H3TensorPayload.withTensorBytes(file: file, name: alphaName,
            maximumBufferedBytes: 4 * 1024 * 1024) {
            $0.loadUnaligned(as: Float.self)
          }
        case .bakedIntoB:
          guard file.tensors[alphaName] == nil else {
            throw H3CheckpointError.invalid("Baked-scale H3 LoRA must not add a second alpha: \(target)")
          }
          // The installed conversion already multiplied B by its training
          // scale. alpha == rank makes activation-space application exactly B@A.
          alpha = Float(rank)
        }
        guard alpha.isFinite, alpha >= 0 else {
          throw H3CheckpointError.invalid("Invalid H3 LoRA alpha: \(target)")
        }
        targets[target] = (rank, alpha)
    }
    let tensorsPerTarget = layout == .perTargetAlpha ? 3 : 2
    guard !targets.isEmpty, file.tensors.count == targets.count * tensorsPerTarget else {
      throw H3CheckpointError.invalid("H3 LoRA contains unsupported extra targets.")
    }
    self.url = url
    self.strength = strength
    self.targetCount = targets.count
    self.qkvLayout = qkvLayout
    self.file = file
    self.targets = targets
    try file.checkUnchanged(at: url)
  }

  /// Sampling declarations belong to the adapter header, not its filename or
  /// the historical `turboLoRA` argument name. Request steps count grid points;
  /// adapter inference-step declarations count actual transformer evaluations.
  private static func validateSampling(metadata: [String: String],
    requestedSteps: Int?, samplingMethod: H3SamplingMethod, profile: H3LoRAProfile, startAfterEvaluations: Int) throws -> Bool {
    func normalized(_ key: String) -> String? {
      metadata[key]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
    var profiles = Set<String>()
    for key in ["adapter_profile", "profile", "distillation_profile"] {
      guard let value = normalized(key) else { continue }
      switch value {
      case "standard", "base", "quality": profiles.insert("standard")
      case "turbo": profiles.insert("turbo")
      default: throw H3CheckpointError.invalid("Unsupported H3 LoRA profile metadata: \(key)")
      }
    }
    if let role = normalized("adapter_role") {
      guard ["standard", "transformer_lora", "style", "character", "turbo"].contains(role) else {
        throw H3CheckpointError.invalid("Unsupported H3 LoRA role metadata.")
      }
      if role == "turbo" { profiles.insert("turbo") }
    }
    var evaluations = Set<Int>()
    for key in ["inference_steps", "num_inference_steps", "steps",
      "transformer_evaluations", "schedule_points"] {
      guard let value = normalized(key) else { continue }
      guard value.range(of: "^[1-9][0-9]*$", options: .regularExpression) != nil,
        let count = Int(value) else {
        throw H3CheckpointError.invalid("Invalid H3 LoRA sampling metadata: \(key)")
      }
      let actual = count - (key == "schedule_points" ? 1 : 0)
      guard actual > 0 else {
        throw H3CheckpointError.invalid("H3 LoRA schedule needs at least one evaluation.")
      }
      evaluations.insert(actual)
    }
    guard evaluations.count <= 1 else {
      throw H3CheckpointError.invalid("Conflicting H3 LoRA sampling-count metadata.")
    }
    if let count = evaluations.first, count <= 8 { profiles.insert("turbo") }
    guard profiles.count <= 1 else {
      throw H3CheckpointError.invalid("Conflicting H3 LoRA profile metadata.")
    }
    if profiles.isEmpty && profile != .auto { profiles.insert(profile.rawValue) }
    if profiles.first == "turbo" {
      guard startAfterEvaluations == 0 else {
        throw H3CheckpointError.invalid("H3 Turbo metadata requires all four evaluations; deferred activation is unsupported for this adapter.")
      }
      guard samplingMethod == .euler else {
        throw H3CheckpointError.invalid("H3 Turbo LoRA requires Euler sampling.")
      }
      guard evaluations.first == nil || evaluations.first == 4 else {
        throw H3CheckpointError.invalid("H3 Turbo LoRA supports four evaluations (five schedule points).")
      }
      if let requestedSteps, requestedSteps != 5 {
        throw H3CheckpointError.invalid("H3 Turbo LoRA requires five requested schedule points for four evaluations.")
      }
    }
    return profiles.first == "turbo"
  }

  func apply(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    try compute(base:base,input:input,target:target,reorderQKV:reorderQKV,queued:false)
  }
  func applyQueued(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool = false) throws -> MLXArray {
    try compute(base:base,input:input,target:target,reorderQKV:reorderQKV,queued:true)
  }
  struct PreparedPair {
    let a: MLXArray
    let b: MLXArray
    let alpha: Float
  }

  func prepareBlock(index: Int) throws -> H3LoRABlock? {
    try H3LoRABlock(index: index, files: [self])
  }
  func checkUnchanged() throws { try file.checkUnchanged(at: url) }

  func loadPair(target: String, inputWidth: Int, outputWidth: Int) throws -> PreparedPair? {
    try Task.checkCancellation()
    guard let info = targets[target] else { return nil }
    let aName = target + ".lora_A.weight"
    let bName = target + ".lora_B.weight"
    let aShape = [info.rank, inputWidth]
    let bShape = [outputWidth, info.rank]
    guard file.tensors[aName]?.shape == aShape.map(UInt64.init),
      file.tensors[bName]?.shape == bShape.map(UInt64.init) else {
      throw H3CheckpointError.invalid("H3 LoRA projection widths differ from its checkpoint.")
    }
    let a = try H3TensorPayload.withTensorBytes(file: file, name: aName,
      maximumBufferedBytes: 16 * 1024 * 1024) { bytes in
      MLXArray(bytes, aShape, type: UInt16.self).view(dtype: .bfloat16)
    }
    let b = try H3TensorPayload.withTensorBytes(file: file, name: bName,
      maximumBufferedBytes: 16 * 1024 * 1024) { bytes in
      MLXArray(bytes, bShape, type: UInt16.self).view(dtype: .bfloat16)
    }
    try checkUnchanged()
    try Task.checkCancellation()
    return PreparedPair(a: a, b: b, alpha: info.alpha)
  }

  func apply(pair: PreparedPair, base: MLXArray, input: MLXArray,
    reorderQKV: Bool, queued: Bool) throws -> MLXArray {
    try Task.checkCancellation()
    let output = H3LoRAProjection.apply(base: base, input: input,
      a: pair.a, b: pair.b, alpha: pair.alpha, strength: strength,
      reorderQKV: reorderQKV && qkvLayout != .nativeInterleaved)
    if !queued { eval(output) }
    try checkUnchanged()
    try Task.checkCancellation()
    return output
  }

  private func compute(base: MLXArray, input: MLXArray,
    target: String, reorderQKV: Bool, queued: Bool) throws -> MLXArray {
    try Task.checkCancellation()
    guard targets[target] != nil else { return base }
    guard let pair = try loadPair(target: target,
      inputWidth: input.shape.last!, outputWidth: base.shape.last!) else { return base }
    return try apply(pair: pair, base: base, input: input, reorderQKV: reorderQKV, queued: queued)
  }
}
