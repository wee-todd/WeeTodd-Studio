import Foundation
import MLX
import MLXNN
import TensorIO

/// A guide consists of normalized 24-channel VAE rows. The branch accepts
/// 49 channels; trailing channels are zero, preserving the released guide-only
/// contract without silently treating the video as an inpainting input.
struct H3FunControlCondition {
  let checkpoint: URL
  let strength: Float
  let guideRows: MLXArray
}

enum H3FunControlMath {
  static func paddedGuideRows(_ rows: MLXArray) throws -> MLXArray {
    guard rows.ndim == 3, rows.shape[0] == 1, rows.shape[1] > 0,
      rows.shape[2] == 96, rows.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("H3 Fun guide must contain normalized channel-major 2×2 VAE rows.")
    }
    return concatenated([rows.asType(.float32),
      MLXArray.zeros([1, rows.shape[1], 100], dtype: .float32)], axis: 2)
  }

  static func initialize(hidden: MLXArray, projected: MLXArray,
    targetIndices: MLXArray, before: (MLXArray) throws -> MLXArray) throws -> MLXArray {
    guard hidden.ndim == 3, hidden.shape[0] == 1,
      projected.shape == [1, targetIndices.size, hidden.shape[2]],
      targetIndices.dtype == .int32 else {
      throw H3CheckpointError.invalid("H3 Fun guide geometry differs from target video rows.")
    }
    let control = hidden.reshaped(hidden.shape)
    control[0..<1, targetIndices, 0..<hidden.shape[2]] = projected.asType(hidden.dtype)
    return try before(control) + hidden
  }

  /// Run base blocks first, then the matching control block and residual.
  /// The branch evolves its own stream across injections rather than taking
  /// each later base activation as its input.
  static func runBlocks(input: MLXArray, blockCount: Int, control initial: MLXArray?,
    injectionLayers: [Int] = H3FunControlLayout.v1InjectionLayers,
    baseBlock: (Int, MLXArray) throws -> MLXArray,
    controlBlock: ((Int, MLXArray) throws -> (MLXArray, MLXArray))?,
    progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
    guard (1...50).contains(blockCount), (initial == nil) == (controlBlock == nil),
      initial == nil || (blockCount == 50 &&
        [H3FunControlLayout.v1InjectionLayers, H3FunControlLayout.v2InjectionLayers].contains(injectionLayers)) else {
      throw H3CheckpointError.invalid("H3 Fun branch needs all 50 base blocks and a paired control stream.")
    }
    try Task.checkCancellation()
    var hidden = input
    var control = initial
    var branch = 0
    for index in 0..<blockCount {
      hidden = try baseBlock(index, hidden)
      try Task.checkCancellation()
      if let current = control, let controlBlock,
        branch < injectionLayers.count,
        index == injectionLayers[branch] {
        let (next, residual) = try controlBlock(branch, current)
        guard next.shape == hidden.shape, residual.shape == hidden.shape else {
          throw H3CheckpointError.invalid("H3 Fun branch residual changed its packed shape.")
        }
        control = next
        hidden = hidden + residual
        eval(hidden)
        branch += 1
      }
      progress(index + 1, blockCount)
      try Task.checkCancellation()
    }
    return hidden
  }

  static func suppressAudio(_ residual: MLXArray, audioIndices: MLXArray) -> MLXArray {
    let result = residual.reshaped(residual.shape)
    if audioIndices.size > 0 {
      result[0..<1, audioIndices, 0..<residual.shape[2]] = MLXArray(0).asType(residual.dtype)
    }
    return result
  }
}

/// The only control activations retained between denoiser evaluations. A
/// single owner releases guide rows and every modulation table together.
final class H3FunControlActivations {
  var guideRows: MLXArray?
  var modulations: [MLXArray] = []
  var residentBytes: Int {
    (guideRows?.nbytes ?? 0) + modulations.reduce(0) { $0 + $1.nbytes }
  }
  func unload() { guideRows = nil; modulations = [] }
}

/// Process-local control modulation and a guide tensor; projections stream from
/// the adapter in place and are released at every operation just like base H3.
final class H3FunControlState {
  private let checkpoint: URL
  private let layout: H3FunControlLayout
  private let file: SafeTensorFile
  private let strength: Float
  private let activations = H3FunControlActivations()
  private var rows: MLXArray? {
    get { activations.guideRows }
    set { activations.guideRows = newValue }
  }
  private var modulations: [MLXArray] {
    get { activations.modulations }
    set { activations.modulations = newValue }
  }

  var injectionLayers: [Int] { layout.injectionLayers }

  var residentActivationBytes: Int {
    activations.residentBytes
  }

  init(condition: H3FunControlCondition, timeEmbeddings: MLXArray,
    base: H3CheckpointLayout) throws {
    checkpoint = condition.checkpoint
    layout = try H3FunControlLayout(url: checkpoint, base: base)
    file = try SafeTensorFile(url: checkpoint)
    guard condition.strength.isFinite, (0...1).contains(condition.strength),
      timeEmbeddings.shape[1] == layout.timeWidth else {
      throw H3CheckpointError.invalid("Invalid H3 Fun control strength or timestep coordinates.")
    }
    strength = condition.strength
    rows = try H3FunControlMath.paddedGuideRows(condition.guideRows)
    for index in 0..<layout.blockCount {
      let activated = layout.timeWidth == 64 ? timeEmbeddings.asType(.float32)
        : silu(timeEmbeddings.asType(.float32))
      modulations.append(try project(activated,
        "control_blocks.\(index).adaln_proj.linear", rows: 96768,
        columns: layout.timeWidth, bias: true).asType(.bfloat16))
      try Task.checkCancellation()
    }
  }

  private func read(_ suffix: String, shape: [Int]) throws -> MLXArray {
    let name = layout.prefix + suffix
    guard let descriptor = file.tensors[name],
      descriptor.shape == shape.map(UInt64.init) else {
      throw H3CheckpointError.invalid("Invalid H3 Fun control tensor: \(suffix)")
    }
    let result = try file.withTensorBytes(named: name) { bytes in
      switch descriptor.dtype {
      case "F32": return MLXArray(bytes, shape, type: Float.self)
      case "F16": return MLXArray(bytes, shape, type: Float16.self)
      case "BF16": return MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      default: throw H3CheckpointError.invalid("Unsupported H3 Fun tensor dtype: \(suffix)")
      }
    }
    eval(result)
    try file.checkUnchanged(at: checkpoint)
    try Task.checkCancellation()
    return result
  }

  private func project(_ input: MLXArray, _ suffix: String,
    rows: Int, columns: Int, bias: Bool = false,
    reorderQKV: Bool = false) throws -> MLXArray {
    let name = layout.prefix + suffix + ".weight"
    var result: MLXArray
    if file.tensors[name]?.dtype == "I8" {
      let weight = try H3ComfyDecodedProjection.load(file: file,
        checkpointURL: checkpoint, name: name, rows: rows,
        columns: columns, reorderQKV: reorderQKV)
      result = matmul(input.asType(.bfloat16), weight.T)
      if bias { result = result + (try read(suffix + ".bias", shape: [rows])).asType(result.dtype) }
    } else {
      let weight = try read(suffix + ".weight", shape: [rows, columns])
      result = matmul(input.asType(weight.dtype), weight.T)
      if bias { result = result + (try read(suffix + ".bias", shape: [rows])).asType(result.dtype) }
    }
    eval(result)
    Memory.clearCache()
    try Task.checkCancellation()
    return result
  }

  func initialize(hidden: MLXArray, targetIndices: MLXArray) throws -> MLXArray {
    guard let rows else { throw H3CheckpointError.invalid("H3 Fun state was unloaded.") }
    let projected = try project(rows, "control_proj_in", rows: 5376,
      columns: 196, bias: true).asType(hidden.dtype)
    return try H3FunControlMath.initialize(hidden: hidden, projected: projected,
      targetIndices: targetIndices) {
        try project($0, "control_blocks.0.before_proj", rows: 5376,
          columns: 5376, bias: true).asType(hidden.dtype)
      }
  }

  func step(index: Int, control: MLXArray, modulationIndices: MLXArray,
    angles: H3RotaryAngles, audioIndices: MLXArray) throws -> (MLXArray, MLXArray) {
    guard modulations.count == layout.blockCount, (0..<layout.blockCount).contains(index) else {
      throw H3CheckpointError.invalid("H3 Fun control branch is unloaded or out of range.")
    }
    let prefix = "control_blocks.\(index)."
    let output = try H3TransformerBlock.evaluateKernel(input: control,
      modulation: modulations[index], modulationIndices: modulationIndices,
      angles: angles,
      read: { suffix, shape in
        let name = layout.splitQKV ? suffix.replacingOccurrences(of: "attn.q_norm", with: "attn.norm_q")
          .replacingOccurrences(of: "attn.k_norm", with: "attn.norm_k") : suffix
        return try read(prefix + name, shape: shape).asType(control.dtype)
      }, project: { input, suffix, rows, columns, qkv in
        if layout.splitQKV && qkv {
          let parts = try ["q", "k", "v"].map { part in
            try project(input, prefix + "attn.to_\(part)", rows: 7168, columns: 5376)
              .reshaped([1, input.shape[1], 56, 128, 1])
          }
          // Raw VideoX stores separate Q/K/V; the shared block consumes
          // head-major QKV rows, never a globally grouped concatenation.
          return concatenated(parts, axis: 4).transposed(0, 1, 2, 4, 3)
            .reshaped([1, input.shape[1], 21504]).asType(control.dtype)
        }
        let raw = ["attn.out_proj": "attn.to_out.0",
          "mlp.fc1": "ff.net.0.proj", "mlp.fc2": "ff.net.2"]
        let name = layout.splitQKV ? (raw[suffix] ?? suffix) : suffix
        var value = try project(input, prefix + name, rows: rows,
          columns: columns, reorderQKV: qkv && !layout.splitQKV).asType(control.dtype)
        if layout.splitQKV && suffix == "mlp.fc1" {
          value = concatenated([value[.ellipsis, 14336..<28672],
            value[.ellipsis, 0..<14336]], axis: -1)
        }
        return value
      })
    let skip = try project(output, prefix + "after_proj", rows: 5376,
      columns: 5376, bias: true).asType(control.dtype)
    return (output, H3FunControlMath.suppressAudio(skip, audioIndices: audioIndices) * strength)
  }

  func unload() { activations.unload(); Stream.gpu.synchronize(); Memory.clearCache() }
  deinit { unload() }
}
