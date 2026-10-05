import Darwin
import Foundation
import TensorIO

public enum H3CheckpointError: Error, Equatable { case invalid(String) }

public struct H3TensorInfo: Sendable, Equatable {
  public let dtype: String
  public let shape: [UInt64]
  public init(dtype: String, shape: [UInt64]) { self.dtype = dtype; self.shape = shape }
}

/// Inspects the existing direct Comfy checkpoint in place. This does not map or
/// copy the giant transformer weights; a sampler may stream each checked tensor.
public enum H3FastVariant: String, Sendable, Codable {
  case denseV1 = "dense-v1"
  case vsaV1 = "vsa-v1"
}

public struct H3CheckpointLayout: Sendable {
  public let fastVariant: H3FastVariant?
  public let prefix: String
  public let blockCount: Int
  public let quantizedProjections: Int
  public let curveRank: Int?
  private let quantizedBases: [String]

  private struct FileIdentity: Equatable {
    let device: dev_t
    let inode: ino_t
    let size: off_t
    let modifiedSeconds: Int
    let modifiedNanos: Int
    let changedSeconds: Int
    let changedNanos: Int

    init(_ status: stat) {
      device = status.st_dev
      inode = status.st_ino
      size = status.st_size
      modifiedSeconds = status.st_mtimespec.tv_sec
      modifiedNanos = status.st_mtimespec.tv_nsec
      changedSeconds = status.st_ctimespec.tv_sec
      changedNanos = status.st_ctimespec.tv_nsec
    }
  }

  private final class ValidationCache: @unchecked Sendable {
    private let lock = NSLock()
    private var identity: FileIdentity?
    private var layout: H3CheckpointLayout?

    func lookup(_ candidate: FileIdentity) -> H3CheckpointLayout? {
      lock.lock()
      defer { lock.unlock() }
      return identity == candidate ? layout : nil
    }

    func remember(_ candidate: FileIdentity, layout: H3CheckpointLayout) {
      lock.lock()
      defer { lock.unlock() }
      identity = candidate
      self.layout = layout
    }
  }

  private static let validationCache = ValidationCache()

  private static func fileIdentity(at url: URL) throws -> FileIdentity {
    var status = stat()
    guard url.isFileURL, fstatat(AT_FDCWD, url.path, &status, 0) == 0,
      status.st_mode & S_IFMT == S_IFREG else {
      throw H3CheckpointError.invalid("H3 checkpoint is not a local regular file.")
    }
    return FileIdentity(status)
  }

  public init(tensors: [String: H3TensorInfo]) throws {
    try self.init(tensors: tensors, pagedAffine: false, computedRotary: false)
  }

  init(tensors: [String: H3TensorInfo], pagedAffine: Bool, computedRotary: Bool,
    fastVariant: H3FastVariant? = nil) throws {
    let candidates = ["model.diffusion_model.", "diffusion_model.", ""]
      .filter { tensors[$0 + "video_patch_proj.weight"] != nil }
    guard candidates.count == 1 else {
      throw H3CheckpointError.invalid("Expected one MiniMax H3 transformer namespace.")
    }
    let root = candidates[0]
    let curved = root.isEmpty && tensors["adaln_t_table"] ==
      H3TensorInfo(dtype: "F32", shape: [1001, 64])
    guard !root.isEmpty || curved || (fastVariant != nil && pagedAffine && computedRotary) else {
      throw H3CheckpointError.invalid("Rootless H3 requires the released FL2VA AdaLN curve.")
    }
    func require(_ suffix: String, _ shape: [UInt64], _ dtypes: Set<String> = ["BF16", "F16"]) throws {
      guard let tensor = tensors[root + suffix], tensor.shape == shape,
        dtypes.contains(tensor.dtype) else {
        throw H3CheckpointError.invalid("Missing or incompatible H3 tensor: \(suffix)")
      }
    }
    let ioBiasTypes: Set<String> = fastVariant == nil ? ["F32"] : ["BF16"]
    try require("video_patch_proj.weight", [5376, 96], curved ? ["F32"] : ["BF16", "F16"])
    try require("video_patch_proj.bias", [5376], ioBiasTypes)
    try require("audio_patch_proj.weight", [5376, 32], curved ? ["F32"] : ["BF16", "F16"])
    try require("audio_patch_proj.bias", [5376], ioBiasTypes)
    try require("condition_proj.weight", [5376, 5120])
    try require("condition_proj.bias", [5376])
    try require("final_layer.video_out.weight", [96, 5376], curved ? ["F32"] : ["BF16", "F16"])
    try require("final_layer.video_out.bias", [96], ioBiasTypes)
    try require("final_layer.audio_out.weight", [32, 5376], curved ? ["F32"] : ["BF16", "F16"])
    try require("final_layer.audio_out.bias", [32], ioBiasTypes)
    try require("final_layer.norm.weight", [5376])
    try require("final_layer.adaln_proj.linear.weight", [10752, curved ? 64 : 2688],
      curved ? ["F32"] : ["BF16", "F16"])
    try require("final_layer.adaln_proj.linear.bias", [10752],
      curved ? ["F32"] : ["BF16", "F16"])
    if !curved {
      try require("time_embedder.proj_in.weight", [5376, 256])
      try require("time_embedder.proj_in.bias", [5376], ioBiasTypes)
      try require("time_embedder.proj_out.weight", [2688, 5376])
      try require("time_embedder.proj_out.bias", [2688], ioBiasTypes)
    }
    if !computedRotary { try require("rope.inv_freq", [16], ["F32"]) }
    guard !pagedAffine || curved || fastVariant != nil else { throw H3CheckpointError.invalid("Paged affine H3 requires the FL2VA64 architecture.") }
    try require("token_refiner.final_norm.weight", [5376])
    for index in 0..<2 {
      let base = "token_refiner.blocks.\(index)."
      for name in ["attn.q_norm.weight", "attn.k_norm.weight"] {
        try require(base + name, [128])
      }
      for name in ["norm1.weight", "norm2.weight"] {
        try require(base + name, [5376])
      }
      for (name, shape) in [
        ("attn.qkv_proj.weight", [UInt64(21504), 5376]),
        ("attn.out_proj.weight", [5376, 7168]),
        ("mlp.fc1.weight", [28672, 5376]),
        ("mlp.fc2.weight", [5376, 14336]),
      ] {
        if fastVariant != nil {
          let stem = base + String(name.dropLast(".weight".count))
          try require(base + name, [shape[0], shape[1] / 4], ["U32"])
          try require(stem + ".scales", [shape[0], shape[1] / 64], ["BF16"])
          try require(stem + ".biases", [shape[0], shape[1] / 64], ["BF16"])
        } else { try require(base + name, shape) }
      }
    }

    let projections: [(String, [UInt64])] = [
      ("attn.qkv_proj", [21504, 5376]), ("attn.out_proj", [5376, 7168]),
      ("mlp.fc1", [28672, 5376]), ("mlp.fc2", [5376, 14336]),
      ("adaln_proj.linear", [96768, curved ? 64 : 2688]),
    ]
    var quantized: [String] = []
    for index in 0..<50 {
      let block = "blocks.\(index)."
      for name in ["attn.q_norm.weight", "attn.k_norm.weight"] {
        try require(block + name, [128])
      }
      for name in ["norm1.weight", "norm2.weight"] {
        try require(block + name, [5376])
      }
      try require(block + "adaln_proj.linear.bias", [96768],
        curved ? ["F32"] : ["BF16", "F16"])
      if fastVariant == .vsaV1 {
        try require(block + "attn.gate_compress.weight", [7168, 5376], ["BF16"])
      } else if tensors[root + block + "attn.gate_compress.weight"] != nil {
        throw H3CheckpointError.invalid("A dense H3 checkpoint cannot contain a trained VSA branch.")
      }
      for (name, shape) in projections {
        let base = root + "blocks.\(index)." + name
        guard let weight = tensors[base + ".weight"], weight.shape == ((pagedAffine && weight.dtype == "U32") ? [shape[0], shape[1] / 4] : shape) else {
          throw H3CheckpointError.invalid("Missing or incompatible H3 projection: \(base)")
        }
        switch weight.dtype {
        case "BF16" where !curved: break
        case "F16" where !curved: break
        case "BF16" where curved && name != "adaln_proj.linear": break
        case "F32" where curved && name == "adaln_proj.linear": break
        case "U32" where pagedAffine && (name != "adaln_proj.linear" || fastVariant != nil):
          guard tensors[base + ".scales"] == H3TensorInfo(dtype: "BF16", shape: [shape[0], shape[1] / 64]),
            tensors[base + ".biases"] == H3TensorInfo(dtype: "BF16", shape: [shape[0], shape[1] / 64]) else {
            throw H3CheckpointError.invalid("Incomplete paged affine H3 metadata: \(base)")
          }
        case "I8":
          guard tensors[base + ".weight_scale"] == H3TensorInfo(dtype: "F32", shape: [shape[0], 1]),
            let marker = tensors[base + ".comfy_quant"], marker.dtype == "U8",
            marker.shape.count == 1, let count = marker.shape.first, (1...4096).contains(count) else {
            throw H3CheckpointError.invalid("Incomplete Comfy INT8 metadata: \(base)")
          }
          quantized.append(base)
        default:
          throw H3CheckpointError.invalid("Unsupported H3 projection dtype: \(base)")
        }
      }
    }
    self.fastVariant = fastVariant
    prefix = root
    curveRank = curved ? 64 : nil
    blockCount = 50
    quantizedProjections = quantized.count
    quantizedBases = quantized
  }

  public init(url: URL) throws {
    if H3CheckpointSource.isPaged(url) {
      self = try H3CheckpointSource.inspect(url).layout
      return
    }
    let identity = try Self.fileIdentity(at: url)
    if let cached = Self.validationCache.lookup(identity) {
      self = cached
      return
    }
    try self.init(validatingURL: url)
    guard try Self.fileIdentity(at: url) == identity else {
      throw H3CheckpointError.invalid("H3 checkpoint changed during validation.")
    }
    Self.validationCache.remember(identity, layout: self)
  }

  private init(validatingURL url: URL) throws {
    let file = try SafeTensorFile(url: url)
    if file.tensors["adaln_t_table"] != nil {
      guard file.metadata["partition"]?.uppercased() == "FL2VA",
        file.metadata["adaln_curve_grid"] == "1001",
        file.metadata["adaln_curve_rank"] == "64",
        file.metadata["adaln_curve_centered"] == "true" else {
        throw H3CheckpointError.invalid("Unsupported H3 FL2VA AdaLN curve metadata.")
      }
    }
    let headers = file.tensors.mapValues { H3TensorInfo(dtype: $0.dtype, shape: $0.shape) }
    try self.init(tensors: headers)
    for base in quantizedBases {
      let name = base + ".comfy_quant"
      let marker: Data = try file.withTensorBytes(named: name,
        range: 0..<file.tensors[name]!.byteCount, access: .buffered) { Data($0) }
      guard let object = try JSONSerialization.jsonObject(with: marker) as? [String: Any],
        object["format"] as? String == "int8_tensorwise",
        Set(object.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]) else {
        throw H3CheckpointError.invalid("Unsupported Comfy INT8 marker: \(base)")
      }
    }
    try file.checkUnchanged(at: url)
  }
}
