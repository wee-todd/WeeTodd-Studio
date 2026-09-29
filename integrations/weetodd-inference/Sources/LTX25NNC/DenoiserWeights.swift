import Foundation
import AdapterRuntime
import LTX25Engine
import TensorIO

final class FixedWeightSource {
  private let file: SafeTensorFile
  private let names: [String: String]
  private let shapes: [String: [Int]]
  var tensorCount: Int { file.tensors.count }
  var tensorBytes: UInt64 { file.tensors.values.reduce(0) { $0 + $1.byteCount } }

  init(url: URL, configuration: AVBlockConfiguration) throws {
    try AVBlockRunner.validateAllocation(configuration: configuration)
    let shapes = DenoiserLayout.weightShapes(configuration)
    let file = try SafeTensorFile(url: url, maximumHeaderBytes: 1024 * 1024)
    var names: [String: String] = [:]
    for key in file.tensors.keys {
      guard let normalized = LTXAdapterCompatibility.normalize(key), shapes[normalized] != nil else { continue }
      guard names.updateValue(key, forKey: normalized) == nil else { throw BlockError.invalid("Duplicate fixed-weight alias: \(normalized)") }
    }
    for (name, shape) in shapes {
      guard let key = names[name], let record = file.tensors[key], record.shape == shape.map(UInt64.init),
        ["F32", "BF16", "F16"].contains(record.dtype), shape.reduce(4, *) <= 768 * 1024 * 1024 else {
        throw BlockError.invalid("Missing, incompatible or oversized fixed weight: \(name)")
      }
    }
    self.file = file; self.names = names; self.shapes = shapes
  }

  func read(_ name: String, shape: [Int]) throws -> [Float] {
    guard shapes[name] == shape, let key = names[name] else { throw BlockError.invalid("Unvalidated fixed-weight request: \(name)") }
    // The 9×4096 AdaLN projection is 576 MiB in Float32. Its one-stage budget
    // permits this explicitly; no other adaptive head is resident concurrently.
    return try file.readFloat32(named: key, maximumBytes: 768 * 1024 * 1024)
  }
}

/// Header-only preflight for the exact supported uniform-timestep LTX 2.5 model.
/// The fixed file's text connectors are intentionally left to the encoder stage.
public final class DenoiserWeights {
  public let blocks: PagedBlockWeights
  private let fixed: FixedWeightSource
  private let adapters: LoRAWeightStack?
  public init(root: URL, configuration: AVBlockConfiguration,adapters: LoRAWeightStack? = nil) throws {
    self.adapters = adapters
    if let adapters {
      var targets = Dictionary(uniqueKeysWithValues: DenoiserLayout.weightShapes(configuration)
        .filter { $0.key.hasSuffix(".weight") }.map { (String($0.key.dropLast(7)),$0.value) })
      let block = try AVBlockRunner.expectedWeightShapes(configuration: configuration)
      for index in 0..<48 {
        for (name,shape) in block where name.hasSuffix(".weight") {
          targets["transformer_blocks.\(index)."+String(name.dropLast(7))] = shape
        }
      }
      try adapters.validateTargets(targets)
    }
    guard root.isFileURL else { throw BlockError.invalid("Denoiser weights must be local.") }
    let root = root.resolvingSymlinksInPath().standardizedFileURL
    let manifest = root.appendingPathComponent("paged_manifest.json")
    let info = try manifest.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard info.isRegularFile == true, let bytes = info.fileSize, bytes <= 1024 * 1024 else {
      throw BlockError.invalid("Denoiser manifest must be a regular file of at most 1 MiB.")
    }
    guard let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any],
      let metadata = raw["metadata"] as? [String: Any], let config = metadata["config"] as? [String: Any],
      let t = config["transformer"] as? [String: Any],
      t["timestep_scale_multiplier"] as? Double == 1000,
      t["av_ca_timestep_scale_multiplier"] as? Double == 1000,
      t["positional_embedding_theta"] as? Double == 10000,
      t["frequencies_precision"] as? String == "float64",
      t["positional_embedding_max_pos"] as? [Int] == [20, 2048, 2048],
      t["audio_positional_embedding_max_pos"] as? [Int] == [20],
      t["in_channels"] as? Int == 128, t["out_channels"] as? Int == 128,
      t["audio_out_channels"] as? Int == 128,
      let record = raw["fixed"] as? [String: Any], let path = record["file"] as? String,
      !path.hasPrefix("/"), !path.split(separator: "/", omittingEmptySubsequences: false).contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
      throw BlockError.invalid("Unsupported top-level LTX timestep, position, latent or fixed-file contract.")
    }
    let fixedURL = root.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
    guard fixedURL.path.hasPrefix(root.path + "/") else { throw BlockError.invalid("Fixed checkpoint escapes its root.") }
    fixed = try FixedWeightSource(url: fixedURL, configuration: configuration)
    guard record["tensor_count"] as? Int == fixed.tensorCount,
      (record["tensor_bytes"] as? NSNumber)?.uint64Value == fixed.tensorBytes else {
      throw BlockError.invalid("Fixed-file header differs from the manifest.")
    }
    blocks = try PagedBlockWeights(root: root, configuration: configuration)
  }
  public func readFixed(_ name: String, shape: [Int]) throws -> [Float] {
    if let adapters { return try adapters.read(name,shape: shape) { try fixed.read(name,shape: shape) } }
    return try fixed.read(name,shape: shape)
  }
  /// Use this adapter-aware provider for sampling; `blocks` exposes base weights only.
  public func readBlock(_ block: Int,name: String,shape: [Int]) throws -> [Float] {
    if let adapters {
      return try adapters.read("transformer_blocks.\(block)."+name,shape: shape) {
        try blocks.read(block: block,name: name,shape: shape)
      }
    }
    return try blocks.read(block: block,name: name,shape: shape)
  }
}
