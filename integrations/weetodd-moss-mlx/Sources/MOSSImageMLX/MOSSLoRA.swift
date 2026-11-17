import CryptoKit
import Foundation
import ImageLayout
import MLX

final class MOSSLoRA {
  struct Pair { var down: String?; var up: String?; var alpha: String?; var schema: String?; var rank = 0; var scale: Float = 1 }
  static let fastSHA = "154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f"
  let descriptor: MOSSLoRADescriptor
  let source: MOSSWeightSource
  let pairs: [String: Pair]
  init(descriptor: MOSSLoRADescriptor, shapes: [String: [Int]] = MOSSModelContract.transformerShapes(), profiler: MOSSProfiler? = nil) throws {
    guard descriptor.path.hasPrefix("/"), descriptor.path.hasSuffix(".safetensors"), descriptor.strength.isFinite,
      (-10...10).contains(descriptor.strength), ["ordinary", "sft_4step_v2"].contains(descriptor.profile) else { throw MOSSError("Invalid MOSS adapter path, strength or profile.") }
    self.descriptor = descriptor
    let index = try MOSSCheckpointIndex(directory: URL(fileURLWithPath: descriptor.path))
    source = MOSSWeightSource(index: index, profiler: profiler, role: "adapter")
    let metadata = index.files.values.first?.metadata ?? [:]
    for key in ["base_model", "base_model_name_or_path", "ss_base_model_version", "modelspec.architecture"] {
      if let value = metadata[key], !value.isEmpty, !value.lowercased().contains("moss") { throw MOSSError("Adapter metadata declares an incompatible training family: \(key)") }
    }
    var globalAlpha: Float?
    for key in ["ss_network_alpha", "lora_alpha", "network_alpha"] {
      if let raw = metadata[key], !["none", "null", "dynamic", "baked", "baked_scale"].contains(raw.lowercased()) {
        guard let value = Float(raw), value.isFinite, value >= 0, globalAlpha == nil || globalAlpha == value else { throw MOSSError("Invalid or conflicting global LoRA alpha.") }
        globalAlpha = value
      }
    }
    let targets = Set(shapes.filter { $0.value.count == 2 && $0.key.hasSuffix(".weight") }.keys.map { String($0.dropLast(7)) })
    func target(_ raw: String) throws -> String {
      try MOSSAdapterTargets.canonical(raw, allowed: targets)
    }
    var pairs: [String: Pair] = [:]
    let suffixes = [(".lora_A.weight", true, "peft"), (".lora_B.weight", false, "peft"),
      (".lora_A.default.weight", true, "default"), (".lora_B.default.weight", false, "default"),
      (".lora_down.weight", true, "kohya"), (".lora_up.weight", false, "kohya")]
    for (name, tensor) in index.descriptors {
      guard ["F32", "BF16", "F16"].contains(tensor.dtype) else { throw MOSSError("Unsupported MOSS adapter tensor precision.") }
      if let suffix = suffixes.first(where: { name.hasSuffix($0.0) }) {
        let key = try target(String(name.dropLast(suffix.0.count)))
        var pair = pairs[key] ?? Pair()
        guard pair.schema == nil || pair.schema == suffix.2, (suffix.1 ? pair.down : pair.up) == nil else { throw MOSSError("Duplicate MOSS adapter target or pair schema.") }
        pair.schema = suffix.2
        if suffix.1 { pair.down = name } else { pair.up = name }
        pairs[key] = pair
      } else if name.hasSuffix(".alpha") {
        let key = try target(String(name.dropLast(6))); var pair = pairs[key] ?? Pair()
        guard pair.alpha == nil, tensor.shape.isEmpty || tensor.shape == [1] else { throw MOSSError("Adapter alpha must be a unique scalar.") }
        pair.alpha = name; pairs[key] = pair
      } else { throw MOSSError("Unconsumed MOSS adapter tensor: \(name)") }
    }
    guard !pairs.isEmpty else { throw MOSSError("MOSS adapter contains no supported pairs.") }
    for key in pairs.keys {
      guard let a = pairs[key]?.down, let b = pairs[key]?.up,
        let down = index.descriptors[a], let up = index.descriptors[b], let shape = shapes[key + ".weight"],
        down.shape.count == 2, up.shape.count == 2, down.shape[0] > 0, down.shape[0] <= 1024,
        down.shape[0] == up.shape[1], down.shape[1] == UInt64(shape[1]), up.shape[0] == UInt64(shape[0]) else {
        throw MOSSError("Incomplete or incompatible MOSS LoRA pair: \(key)")
      }
      pairs[key]!.rank = Int(down.shape[0])
      let rank = Float(down.shape[0])
      let alpha: Float
      if let alphaName = pairs[key]!.alpha { alpha = try source.tensor(alphaName).item(Float.self) }
      else { alpha = globalAlpha ?? rank }
      guard alpha.isFinite, alpha >= 0 else { throw MOSSError("MOSS adapter has invalid alpha.") }
      pairs[key]!.scale = descriptor.strength * alpha / rank
    }
    if descriptor.profile == "sft_4step_v2" {
      guard pairs.count == 252, try Self.sha256(URL(fileURLWithPath: descriptor.path)) == Self.fastSHA else { throw MOSSError("The four-step profile requires the pinned MOSS SFT v2 adapter bytes.") }
    }
    self.pairs = pairs
  }
  func delta(_ x: MLXArray, target: String) throws -> MLXArray? {
    guard descriptor.enabled, descriptor.strength != 0, let pair = pairs[target] else { return nil }
    let down = try source.tensor(pair.down!).asType(.float32), up = try source.tensor(pair.up!).asType(.float32)
    let result = matmul(matmul(x.asType(.float32), down.transposed()), up.transposed()) * pair.scale
    return source.profiler?.evaluate("adapter/FP32 projection", result) ?? result
  }
  func release() { source.release() }
  static func sha256(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }; var hash = SHA256()
    while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty { try Task.checkCancellation(); hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

final class MOSSAdaptedWeights: MOSSWeightAccess {
  let base: any MOSSWeightAccess, adapters: [MOSSLoRA]
  var profiler: MOSSProfiler? { base.profiler }
  init(base: any MOSSWeightAccess, adapters: [MOSSLoRA]) { self.base = base; self.adapters = adapters }
  func contains(_ name: String) -> Bool { base.contains(name) }
  func tensor(_ name: String) throws -> MLXArray { try base.tensor(name) }
  func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
    var result = try base.linear(x, name)
    for adapter in adapters {
      if let delta = try adapter.delta(x, target: name) {
        result = result.asType(.float32) + delta
        if let profiler { result = profiler.evaluate("adapter/FP32 accumulation", result) }
      }
    }
    return result
  }
}
