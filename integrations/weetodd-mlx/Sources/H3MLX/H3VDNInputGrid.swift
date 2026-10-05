import Foundation
import MLX
import TensorIO

/// Original SiLU timestep coordinates for adapters on the pruned H3 base.
/// Keep only the requested schedule rows; the source grid is released here.
struct H3VDNInputGrid {
  private let file: SafeTensorFile
  private let url: URL
  init(url: URL) throws {
    let file = try SafeTensorFile(url: url)
    guard file.tensors.count == 1,
      let tensor = file.tensors["silu_t_emb_grid"], tensor.dtype == "BF16",
      tensor.shape == [1025, 2688] else {
      throw H3CheckpointError.invalid("VDN AdaLN input grid requires the released 1025 × 2688 BF16 SiLU coordinates.")
    }
    self.file = file; self.url = url
  }
  func evaluate(timesteps: [Float]) throws -> MLXArray {
    try Task.checkCancellation(); try file.checkUnchanged(at: url)
    let grid = try file.withTensorBytes(named: "silu_t_emb_grid") {
      MLXArray($0, [1025, 2688], type: UInt16.self).view(dtype: .bfloat16)
    }
    guard MLX.isFinite(grid).all().item(Bool.self) else {
      throw H3CheckpointError.invalid("VDN AdaLN input grid contains non-finite coordinates.")
    }
    let result = try Self.interpolate(grid: grid, timesteps: timesteps)
    eval(result); try file.checkUnchanged(at: url); try Task.checkCancellation()
    return result
  }
  static func interpolate(grid: MLXArray, timesteps: [Float]) throws -> MLXArray {
    guard grid.ndim == 2, (2...4097).contains(grid.shape[0]),
      (1...2688).contains(grid.shape[1]), grid.dtype.isFloatingPoint,
      (1...128).contains(timesteps.count),
      timesteps.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
      throw H3CheckpointError.invalid("Invalid VDN source-grid interpolation.")
    }
    let position = MLXArray(timesteps) * Float(grid.shape[0] - 1)
    let lower = minimum(floor(position).asType(.int32), MLXArray(Int32(grid.shape[0] - 2)))
    let fraction = (position - lower.asType(.float32)).expandedDimensions(axis: 1)
    let source = grid.asType(.float32)
    return take(source, lower, axis: 0) * (1 - fraction)
      + take(source, lower + 1, axis: 0) * fraction
  }
}
