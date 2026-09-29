import Foundation
import MLX
import TensorIO

/// Absolute bilinear position embedding and spatial rotary coordinates in the
/// patch order produced by Qwen's 2×2 spatial merge.
public enum H3QwenVisionPosition {
  public struct Output {
    public let absolute: MLXArray
    public let rotary: MLXArray
    public let boundaries: [Int]
  }

  public static func make(checkpointURL: URL,
    grids: [H3QwenRequest.Grid]) throws -> Output {
    guard !grids.isEmpty else {
      throw H3CheckpointError.invalid("H3 Qwen vision needs at least one image grid.")
    }
    var indices = [[Int32]](repeating: [], count: 4)
    var weights = [[Float]](repeating: [], count: 4)
    var rotaryRows: [Int32] = []
    var rotaryColumns: [Int32] = []
    var boundaries = [0]
    var maximumSpatialSize = 0
    for grid in grids {
      guard (1...16).contains(grid.temporal), (2...128).contains(grid.height),
        (2...128).contains(grid.width), grid.height.isMultiple(of: 2),
        grid.width.isMultiple(of: 2) else {
        throw H3CheckpointError.invalid("Invalid H3 Qwen vision position grid.")
      }
      let framePatches = grid.height * grid.width
      maximumSpatialSize = max(maximumSpatialSize, grid.height, grid.width)
      guard boundaries.last! + grid.temporal * framePatches <= 16_384 else {
        throw H3CheckpointError.invalid("H3 Qwen vision patch window is too large.")
      }
      for _ in 0..<grid.temporal {
        for blockRow in 0..<(grid.height / 2) {
          for blockColumn in 0..<(grid.width / 2) {
            for intraRow in 0..<2 {
              for intraColumn in 0..<2 {
                let row = blockRow * 2 + intraRow
                let column = blockColumn * 2 + intraColumn
                let h = Float(row) * 47 / Float(grid.height - 1)
                let w = Float(column) * 47 / Float(grid.width - 1)
                let h0 = Int(h), w0 = Int(w)
                let h1 = min(h0 + 1, 47), w1 = min(w0 + 1, 47)
                let dh = h - Float(h0), dw = w - Float(w0)
                let nearest = [h0 * 48 + w0, h0 * 48 + w1,
                  h1 * 48 + w0, h1 * 48 + w1]
                let mix = [(1 - dh) * (1 - dw), (1 - dh) * dw,
                  dh * (1 - dw), dh * dw]
                for index in 0..<4 {
                  indices[index].append(Int32(nearest[index]))
                  weights[index].append(mix[index])
                }
                rotaryRows.append(Int32(row))
                rotaryColumns.append(Int32(column))
              }
            }
          }
        }
        boundaries.append(boundaries.last! + framePatches)
      }
    }
    let file = try SafeTensorFile(url: checkpointURL)
    let name = "visual.pos_embed.weight"
    guard file.tensors[name].map({ H3TensorInfo(dtype: $0.dtype, shape: $0.shape) })
      == H3TensorInfo(dtype: "BF16", shape: [2304, 1152]) else {
      throw H3CheckpointError.invalid("H3 Qwen visual position table is missing.")
    }
    let table = try file.withTensorBytes(named: name) { bytes in
      MLXArray(bytes, [2304, 1152], type: UInt16.self).view(dtype: .bfloat16)
    }
    let count = indices[0].count
    var parts: [MLXArray] = []
    for index in 0..<4 {
      let rows = table.take(MLXArray(indices[index]), axis: 0)
      let blend = MLXArray(weights[index], [count, 1]).asType(.bfloat16)
      parts.append(rows * blend)
    }
    let absolute = parts[0] + parts[1] + parts[2] + parts[3]
    // Qwen's fixed 36-wide vision rotary table. These Float32 values pin the
    // released transformer's 10_000^(-2i/36) table across MLX versions whose
    // pow kernels can differ by one ULP and amplify over 27 BF16 blocks.
    let inverseBits: [UInt32] = [
      0x3f800000, 0x3f1977cc, 0x3eb800d5, 0x3e5c9d35, 0x3e044133, 0x3d9e91b6,
      0x3d3e1e95, 0x3ce3f27e, 0x3c88a69b, 0x3c23d70a, 0x3bc4705e, 0x3b6b8630,
      0x3b0d3168, 0x3aa94938, 0x3a4af7f2, 0x39f35a5c, 0x3991e2e1, 0x392ee9be,
    ]
    let inverse = MLXArray(inverseBits.map(Float.init(bitPattern:)))
    let frequencyTable = outer(MLXArray((0..<maximumSpatialSize).map(Float.init)), inverse)
    let rotaryArray = concatenated([
      frequencyTable.take(MLXArray(rotaryRows), axis: 0),
      frequencyTable.take(MLXArray(rotaryColumns), axis: 0),
    ], axis: -1)
    eval(absolute, rotaryArray)
    try file.checkUnchanged(at: checkpointURL)
    return Output(absolute: absolute, rotary: rotaryArray, boundaries: boundaries)
  }
}
