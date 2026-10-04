import Foundation
import MLX
import MLXNN
import TensorIO

/// H3's shared 256→5376→2688 timestep MLP. The released checkpoint stores
/// BF16 matrices, but the reference evaluates this inference-critical stage
/// in Float32 before any block consumes its modulation.
public enum H3TimeEmbedding {
  /// Released FL2VA pruned checkpoints replace the original timestep MLP
  /// with linearly interpolated rank-64 coordinates. No SiLU follows them.
  public static func interpolateCurve(table: [Float], timesteps: [Float]) throws -> [Float] {
    guard table.count == 1001 * 64, table.allSatisfy(\.isFinite),
      timesteps.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("Invalid H3 FL2VA AdaLN curve table.")
    }
    var result = [Float](repeating: 0, count: timesteps.count * 64)
    for (row, timestep) in timesteps.enumerated() {
      let position = min(1, max(0, timestep)) * 1000
      let lower = min(Int(floor(position)), 999)
      let fraction = position - Float(lower)
      for column in 0..<64 {
        let a = table[lower * 64 + column]
        let b = table[(lower + 1) * 64 + column]
        result[row * 64 + column] = a * (1 - fraction) + b * fraction
      }
    }
    return result
  }

  public static func evaluate(checkpointURL: URL,
    timesteps: MLXArray) throws -> MLXArray {
    try evaluate(checkpointURL: checkpointURL, timesteps: timesteps,
      observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, timesteps: MLXArray,
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard timesteps.ndim == 1, (1...4096).contains(timesteps.shape[0]),
      timesteps.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("H3 timesteps require a bounded floating-point vector.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpointURL)
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL)
    let file = try SafeTensorFile(url: tensorURL)
    if layout.curveRank == 64 {
      let table = try file.readFloat32(named: "adaln_t_table")
      let values = try interpolateCurve(table: table,
        timesteps: timesteps.asType(.float32).asArray(Float.self))
      let result = MLXArray(values, [timesteps.shape[0], 64])
      eval(result)
      try observe("curve", result)
      try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
      return result
    }
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    func read(_ suffix: String, shape: [UInt64], dtype: String) throws -> MLXArray {
      let name = layout.prefix + suffix
      guard file.tensors[name].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
        == H3TensorInfo(dtype: dtype, shape: shape) else {
        throw H3CheckpointError.invalid("Missing H3 timestep tensor: \(suffix)")
      }
      return try file.withTensorBytes(named: name) { bytes in
        let dimensions = shape.map(Int.init)
        switch dtype {
        case "BF16":
          return MLXArray(bytes, dimensions, type: UInt16.self)
            .view(dtype: .bfloat16).asType(.float32)
        case "F32": return MLXArray(bytes, dimensions, type: Float.self)
        default: throw H3CheckpointError.invalid("Unsupported H3 timestep dtype.")
        }
      }
    }
    let firstWeight = try read("time_embedder.proj_in.weight", shape: [5376, 256], dtype: "BF16")
    let firstBias = try read("time_embedder.proj_in.bias", shape: [5376], dtype: "F32")
    let secondWeight = try read("time_embedder.proj_out.weight", shape: [2688, 5376], dtype: "BF16")
    let secondBias = try read("time_embedder.proj_out.bias", shape: [2688], dtype: "F32")
    let indices = MLXArray((0..<128).map(Float.init))
    let exponent = (-Float(log(10_000.0)) * indices) / Float(128)
    let frequencies = MLX.exp(exponent)
    let angles = timesteps.asType(.float32).expandedDimensions(axis: 1)
      * frequencies.expandedDimensions(axis: 0)
    let sinusoid = concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    try observe("sinusoid", sinusoid)
    let first = addMM(firstBias, sinusoid, firstWeight.T)
    try observe("fc1", first)
    let activated = silu(first)
    try observe("activated", activated)
    let output = addMM(secondBias, activated, secondWeight.T)
    try observe("output", output)
    eval(output)
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    try Task.checkCancellation()
    return output
  }
}
