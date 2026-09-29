import Foundation
import MLX

/// Qwen3-VL still-image patch packing after media preparation has resized
/// the RGB image to an admitted 32-pixel grid. Still images use CLIP channel
/// statistics; Qwen's video processor uses a different normalization.
public enum H3QwenImageProcessor {
  public struct Packed {
    public let pixels: MLXArray
    public let grid: H3QwenRequest.Grid
  }

  public static func packRGB8(image: Data, width: Int, height: Int) throws -> Packed {
    let patch = 16
    let merge = 2
    guard width >= 32, height >= 32, width <= 2048, height <= 2048,
      width.isMultiple(of: patch * merge),
      height.isMultiple(of: patch * merge),
      width * height >= 32 * 32 * 4,
      image.count == width * height * 3 else {
      throw H3CheckpointError.invalid("Qwen still reference requires bounded RGB geometry on a 32-pixel grid.")
    }
    let gridHeight = height / patch
    let gridWidth = width / patch
    let patches = gridHeight * gridWidth
    guard patches / 4 <= 1024 else {
      throw H3CheckpointError.invalid("Qwen still reference exceeds its visual token window.")
    }
    let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    let standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]
    var values = [Float](repeating: 0, count: patches * 1536)
    image.withUnsafeBytes { raw in
      let input = raw.bindMemory(to: UInt8.self)
      var destination = 0
      for blockY in 0..<(gridHeight / merge) {
        for blockX in 0..<(gridWidth / merge) {
          for mergeY in 0..<merge {
            for mergeX in 0..<merge {
              for channel in 0..<3 {
                for _ in 0..<2 {
                  for patchY in 0..<patch {
                    let y = (blockY * merge + mergeY) * patch + patchY
                    for patchX in 0..<patch {
                      let x = (blockX * merge + mergeX) * patch + patchX
                      let source = (y * width + x) * 3 + channel
                      let scaled = Float(input[source]) / 255
                      values[destination] = (scaled - mean[channel])
                        / standardDeviation[channel]
                      destination += 1
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return Packed(pixels: MLXArray(values, [patches, 1536]),
      grid: H3QwenRequest.Grid(temporal: 1,
        height: gridHeight, width: gridWidth))
  }
}
