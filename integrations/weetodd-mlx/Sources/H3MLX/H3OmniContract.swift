import CoreFoundation
import CryptoKit
import Foundation

/// The one trained Omni release supported by the native engine. This value can
/// only be constructed by validating the entire official inference sidecar.
public struct H3OmniContract: Sendable {
  public let schema = "fasth3-inference-contract-v1"
  public let modelType = "ref2va"
  public let transformerComponent = "transformer_ref"
  public let baseModelRevision = "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da"
  public let conditioning = "fixed_ordered_references_target_only_flow"
  public let pddSteps = 32
  public let pddStepIndices = [0,4,8,12,16,20,24,28,32]
  public let transformerForwards = 8
  public let gridMaxT = 0.999
  public let videoSchedulerShift = 12.0
  public let audioSchedulerShift = 3.0
  public let guidanceScale = 1.0
  public let attentionBackend = "VIDEO_SPARSE_ATTN_H3"
  public let vsaTileSize = 128
  public let vsaSparsity = 0.9
  public let vsaRefPolicy = "p2_multi_region"
  public let vsaRefKeepRate = 0.1
  private static let maximumBytes = 65_536

  public init(data: Data) throws {
    guard !data.isEmpty, data.count <= Self.maximumBytes,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3GeometryError.invalid("FastH3 Omni requires a bounded inference-contract JSON object.")
    }
    let strings = ["schema": schema, "schema_version": schema, "model_type": modelType,
      "transformer_component": transformerComponent, "base_model_revision": baseModelRevision,
      "conditioning": conditioning, "attention_backend": attentionBackend, "vsa_ref_policy": vsaRefPolicy]
    let integers = ["pdd_steps": pddSteps, "num_inference_steps": transformerForwards,
      "transformer_forwards": transformerForwards, "vsa_tile_size": vsaTileSize]
    let reals = ["grid_max_t": gridMaxT, "video_scheduler_shift": videoSchedulerShift,
      "audio_scheduler_shift": audioSchedulerShift, "guidance_scale": guidanceScale,
      "vsa_sparsity": vsaSparsity, "vsa_ref_keep_rate": vsaRefKeepRate]
    let fields = Set(strings.keys).union(integers.keys).union(reals.keys).union(["pdd_step_indices"])
    guard Set(root.keys) == fields else {
      throw H3GeometryError.invalid("FastH3 Omni inference-contract fields are missing or unsupported.")
    }
    for (key, value) in strings where root[key] as? String != value {
      throw H3GeometryError.invalid("FastH3 Omni contract has an unsupported \(key).")
    }
    for (key, value) in integers {
      guard let number = Self.number(root[key]), !Self.isRealStorage(number), number.doubleValue == Double(value) else {
        throw H3GeometryError.invalid("FastH3 Omni contract requires the trained integer \(key).")
      }
    }
    for (key, value) in reals {
      guard let number = Self.number(root[key]), number.doubleValue == value else {
        throw H3GeometryError.invalid("FastH3 Omni contract requires the trained finite \(key).")
      }
    }
    guard let indices = root["pdd_step_indices"] as? [Any], indices.count == pddStepIndices.count else {
      throw H3GeometryError.invalid("FastH3 Omni contract requires eight trained four-head blocks.")
    }
    for (index, value) in zip(indices, pddStepIndices) {
      guard let number = Self.number(index), !Self.isRealStorage(number), number.doubleValue == Double(value) else {
        throw H3GeometryError.invalid("FastH3 Omni contract has unsupported PDD head boundaries.")
      }
    }
  }

  /// Hash the same bounded bytes that are parsed, binding execution to the
  /// sidecar selected by the saved recipe rather than a nearby filename.
  public static func load(url: URL, expectedSHA256: String) throws -> Self {
    guard url.isFileURL, url.path.hasPrefix("/"), expectedSHA256.utf8.count == 64,
      expectedSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
      throw H3GeometryError.invalid("FastH3 Omni requires an absolute sidecar file and lowercase SHA256.")
    }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: maximumBytes + 1) ?? Data()
    guard data.count <= maximumBytes,
      SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == expectedSHA256 else {
      throw H3GeometryError.invalid("FastH3 Omni inference sidecar is oversized or its SHA256 changed.")
    }
    return try Self(data: data)
  }

  private static func number(_ value: Any?) -> NSNumber? {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.doubleValue.isFinite else { return nil }
    return number
  }

  private static func isRealStorage(_ value: NSNumber) -> Bool {
    ["f", "d"].contains(String(cString: value.objCType))
  }
}
