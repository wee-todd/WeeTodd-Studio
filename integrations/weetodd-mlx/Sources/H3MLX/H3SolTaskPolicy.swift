import CoreFoundation
import Foundation
import TensorIO

/// Saved attention selection. Dense preserves the existing default; the
/// explicit experimental selection uses fixed Sol policy version 1. Changing
/// its settings requires a new saved value/version, never reinterpreting old takes.
public enum H3AttentionPolicy: String, Sendable, Equatable {
  case dense
  case solExperimental = "sol_experimental"

  public static func parse(_ value: Any?) throws -> H3AttentionPolicy {
    guard let value else { return .dense }
    guard let text = value as? String, let policy = Self(rawValue: text) else {
      throw H3CheckpointError.invalid("H3 attention_policy must be dense or sol_experimental.")
    }
    return policy
  }

  static func admitRecipe(_ root: [String: Any], canvasAdmission: H3CanvasAdmission) throws -> Self {
    let config = root["config"] as? [String: Any] ?? [:]
    let policy = try parse(config["attention_policy"])
    if policy == .solExperimental {
      let components = root["components"] as? [String: Any] ?? [:]
      guard let steps = config["steps"] as? NSNumber,
        CFGetTypeID(steps) != CFBooleanGetTypeID(), steps.doubleValue == 5,
        (config["sampling_method"] as? String ?? "euler") == "euler" else {
        throw H3CheckpointError.invalid("Sol policy version 1 requires four Euler evaluations (five schedule points).")
      }
      if let drop = config["drop_adaln"] {
        guard CFGetTypeID(drop as CFTypeRef) == CFBooleanGetTypeID(), drop as? Bool == true else {
          throw H3CheckpointError.invalid("Sol policy version 1 requires Drop AdaLN.")
        }
      }
      let conditioning = root["conditioning"] as? [String: Any] ?? [:]
      let task = components["task"] as? String
      guard (task == "ref2va" && conditioning["task"] as? String == "ref2va" || task == "fl2va" && conditioning["task"] as? String == "fflf"), canvasAdmission == .ordinary,
        (config["transformer_backend"] as? String ?? "mlx") == "mlx",
        components["fun_controlnet"] == nil,
        ["fasth3", "vdn", "motion_fidelity", "joint_refinement", "joint_latents",
          "refinement", "continuation"].allSatisfy({ root[$0] == nil }) else {
        throw H3CheckpointError.invalid("Sol experimental attention supports only ordinary MLX Ref2VA or FL2VA without continuation, refinement or controls.")
      }
    }
    return policy
  }

  static func validateDiagnosticEnvironment(_ environment: [String: String], saved: Self) throws {
    let setting = environment["WEETODD_H3_EXPERIMENTAL_SOL"]
    let tau = environment["WEETODD_H3_EXPERIMENTAL_SOL_TAU"]
    guard setting == nil || setting == "0" || setting == "1" else {
      throw H3CheckpointError.invalid("WEETODD_H3_EXPERIMENTAL_SOL requires 0 or 1.")
    }
    if let setting {
      guard (setting == "1") == (saved == .solExperimental) else {
        throw H3CheckpointError.invalid("Experimental Sol diagnostic selection conflicts with saved attention_policy.")
      }
    }
    if let tau {
      guard setting == "1", saved == .solExperimental, let value = Float(tau), value == 0.5 else {
        throw H3CheckpointError.invalid("Saved Sol policy version 1 fixes tau at 0.5; conflicting diagnostic tau is unsupported.")
      }
    }
  }
}

/// Explicit private diagnostic policy. Ordinary requests do not enable Sol.
struct H3SolTaskPolicy: Sendable, Equatable {
  static let excludedBlocks: Set<Int> = [0, 1, 33, 34, 35]
  let tau: Float
  init(tau: Float = 0.5) throws {
    guard tau.isFinite, (0...16).contains(tau) else {
      throw H3CheckpointError.invalid("Experimental Sol tau must be finite and within 0...16.")
    }
    self.tau = tau
  }
  static func validateTask(task: String, contextFrames: Int, isRefinement: Bool,
    ordinaryCanvas: Bool, mlxBackend: Bool, hasFast: Bool, hasVDN: Bool,
    hasFun: Bool, hasMotion: Bool) throws {
    guard ["ref2va", "fl2va"].contains(task), contextFrames == 0, !isRefinement, ordinaryCanvas,
      mlxBackend, !hasFast, !hasVDN, !hasFun, !hasMotion else {
      throw H3CheckpointError.invalid("Experimental Sol supports only ordinary MLX Ref2VA or FL2VA without continuation, refinement or controls.")
    }
  }
  static func validateSettings(steps: Int, samplingMethod: H3SamplingMethod,
    adapters: [H3LoRAAdapter], packedRows: Int) throws {
    try Task.checkCancellation()
    guard steps == 5, samplingMethod == .euler, (1...40_000).contains(packedRows),
      adapters.count == 1, let adapter = adapters.first,
      adapter.strength == 1, adapter.profile != .standard,
      adapter.qkvLayout != .nativeInterleaved, adapter.startAfterEvaluations == 0 else {
      throw H3CheckpointError.invalid("Sol policy version 1 requires four Euler evaluations, one strength-1 contiguous BF16 Turbo adapter active from evaluation zero and at most 40000 packed rows.")
    }
    try adapter.validate(requestedSteps: steps, samplingMethod: samplingMethod)
  }

  /// CPU headers and existing small marker/alpha admission only, before Qwen.
  /// The shared admission helper does not launch an NNC process or load MLX.
  static func inspectQualifiedFiles(checkpoint: URL, adapter: URL, task: String = "ref2va") throws {
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpoint)
    let file = try SafeTensorFile(url: checkpoint)
    if task == "fl2va" {
      guard layout.curveRank == 64, layout.fastVariant == nil else {
        throw H3CheckpointError.invalid("Sol FL2VA requires the verified FL2VA64 partition.")
      }
      for block in 0..<50 { for (suffix, shape) in [("attn.qkv_proj", [21504,5376]), ("attn.out_proj", [5376,7168]), ("mlp.fc1", [28672,5376]), ("mlp.fc2", [5376,14336])] {
        let name=layout.prefix+"blocks.\(block)."+suffix+".weight"
        guard file.tensors[name]?.dtype == "BF16", file.tensors[name]?.shape == shape.map(UInt64.init) else {
          throw H3CheckpointError.invalid("Sol FL2VA requires complete BF16 core projections: \(name)")
        }
      } }
      let validated=try H3LoRAFile(url:adapter,strength:1,requestedSteps:5,samplingMethod:.euler,qkvLayout:.contiguousQKV,profile:.turbo)
      try validated.checkUnchanged();try file.checkUnchanged(at:checkpoint)
      return
    }
    guard task == "ref2va", layout.prefix == "model.diffusion_model.", layout.fastVariant == nil,
      layout.curveRank == nil else {
      throw H3CheckpointError.invalid("Sol policy version 1 requires the original monolithic ordinary H3 backbone.")
    }
    let core: [(String, [UInt64])] = [
      ("attn.qkv_proj", [21504,5376]), ("attn.out_proj", [5376,7168]),
      ("mlp.fc1", [28672,5376]), ("mlp.fc2", [5376,14336])
    ]
    for block in 0..<50 { for (suffix, shape) in core {
      try Task.checkCancellation()
      let name = layout.prefix + "blocks.\(block)." + suffix + ".weight"
      guard let descriptor = file.tensors[name], descriptor.dtype == "I8", descriptor.shape == shape else {
        throw H3CheckpointError.invalid("Sol policy version 1 requires complete canonical signed-I8 core projections: \(name)")
      }
    } }
    do { try H3NativeBlockAdmission.inspect(checkpoint: checkpoint, adapter: adapter) }
    catch is CancellationError { throw CancellationError() }
    catch { throw H3CheckpointError.invalid("Sol backbone/BF16 Turbo admission failed: \(error)") }
    try file.checkUnchanged(at: checkpoint)
  }

  static func validateState(referenceLayout: Bool, blockCount: Int,
    weightDecoded: Bool, hasNativeWorker: Bool, maximumRows: Int,
    hasFast: Bool, hasCurveRank: Bool, hasVDN: Bool, hasFun: Bool) throws {
    guard (referenceLayout ? !hasCurveRank : hasCurveRank), blockCount == 50, weightDecoded, !hasNativeWorker,
      maximumRows == 40_000, !hasFast, !hasVDN, !hasFun else {
      throw H3CheckpointError.invalid("Experimental Sol requires the complete ordinary prepared Ref2VA or FL2VA backbone.")
    }
  }
  static func generatedVideoRange(indices: [Int], rows: Int) throws -> Range<Int> {
    guard (1...40_000).contains(rows), let start = indices.first,
      start >= 0, start < rows, indices.count == rows - start,
      indices.enumerated().allSatisfy({ $0.element == start + $0.offset }) else {
      throw H3CheckpointError.invalid("Experimental Sol requires a contiguous generated-video suffix in packed coordinates.")
    }
    return start..<rows
  }
  func usesSol(block: Int, completedEvaluations: Int) -> Bool {
    (0..<50).contains(block) && completedEvaluations >= 2 && !Self.excludedBlocks.contains(block)
  }
}

enum H3SolTaskContext {
  @TaskLocal static var policy: H3SolTaskPolicy?
}

/// Private worker diagnostic access; saved recipe admission owns selection.
/// No diagnostic environment setting changes the default or selects a backend.
@_spi(H3SolDiagnostic) public enum H3SolDiagnostic {
  public static var enabled: Bool { H3SolTaskContext.policy != nil }
  public static func validateSavedRecipe(data: Data,
    environment: [String: String]) throws -> H3AttentionPolicy {
    guard data.count <= 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3CheckpointError.invalid("Invalid bounded H3 attention recipe.")
    }
    let policy = try H3AttentionPolicy.admitRecipe(root, canvasAdmission: .ordinary)
    try H3AttentionPolicy.validateDiagnosticEnvironment(environment, saved: policy)
    return policy
  }
  public static func withPolicy<T>(tau: Float = 0.5,
    operation: () throws -> T) throws -> T {
    let policy = try H3SolTaskPolicy(tau: tau)
    return try H3SolTaskContext.$policy.withValue(policy, operation: operation)
  }
}

/// Scalar-only snapshot; completed consumers are counted even if a later block fails.
public struct H3SolReport: Sendable, Equatable {
  public let tau: Float
  public let generatedVideoStart: Int
  public let generatedVideoEnd: Int
  public let completedEvaluations: Int
  public let completedSolConsumers: Int
  public let completedDenseBlocks: Int
  public let selectedExactBlockPairs: UInt64
  public let coarseBlockPairs: UInt64
  public let maximumPreparedLogicalBytes: Int
  public var metadata: [String: Any] {
    ["experimental": true, "algorithm": "sol_original_rows_bf16_f32",
      "attentionPolicy": "sol_experimental", "attentionPolicyVersion": 1, "policyVersion": 1,
      "settingsScope": "ordinary signed-I8 Ref2VA or BF16 FL2VA64; weight-decoded; one BF16 Turbo strength-1 adapter; four Euler evaluations; Drop AdaLN; no controls/context/refinement; at most 40000 rows",
      "tau": tau, "blockSize": 64, "queryBlockSize": 64,
      "denseWarmupEvaluations": 2, "excludedBlocks": [0, 1, 33, 34, 35],
      "generatedVideoRange": [generatedVideoStart, generatedVideoEnd],
      "completedEvaluations": completedEvaluations,
      "completedSolConsumers": completedSolConsumers, "completedDenseBlocks": completedDenseBlocks,
      "selectedExactBlockPairs": selectedExactBlockPairs, "coarseBlockPairs": coarseBlockPairs,
      "maximumPreparedLogicalBytes": maximumPreparedLogicalBytes,
      "precisionScope": "BF16 operands/coarse means/output; Float32 pooling, routing and online softmax accumulation; approximate attention",
      "counterScope": "actual completed Sol consumers and completed dense blocks; route pairs across heads and query groups; metadata releases before out projection/FFN"]
  }
}
