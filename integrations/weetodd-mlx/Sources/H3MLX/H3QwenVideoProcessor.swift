import Foundation
import MLX

/// Packs pre-sized RGB8 frames in Qwen3-VL's temporal/channel/2×2 spatial
/// patch order. Image resizing is deliberately a separate preparation step.
public enum H3QwenVideoProcessor {
  public struct Packed {
    public let pixels: MLXArray
    public let grid: H3QwenRequest.Grid
  }

  public static func packRGB8(frames: Data, frameCount: Int,
    width: Int, height: Int) throws -> Packed {
    let patch = 16
    let merge = 2
    let factor = patch * merge
    guard (1...32).contains(frameCount), width >= factor, height >= factor,
      width.isMultiple(of: factor), height.isMultiple(of: factor),
      width <= 2048, height <= 2048 else {
      throw H3CheckpointError.invalid("Qwen RGB frames need bounded dimensions divisible by 32.")
    }
    let frameBytes = width * height * 3
    let temporal = (frameCount + 1) / 2
    let gridHeight = height / patch
    let gridWidth = width / patch
    let patchCount = temporal * gridHeight * gridWidth
    guard frames.count == frameCount * frameBytes,
      patchCount > 0, patchCount / 4 <= 1024,
      gridHeight <= 128, gridWidth <= 128 else {
      throw H3CheckpointError.invalid("Qwen RGB frame payload or visual token count is invalid.")
    }
    var values = [Float](repeating: 0, count: patchCount * 1536)
    frames.withUnsafeBytes { raw in
      let input = raw.bindMemory(to: UInt8.self)
      var destination = 0
      for time in 0..<temporal {
        for blockY in 0..<(gridHeight / merge) {
          for blockX in 0..<(gridWidth / merge) {
            for mergeY in 0..<merge {
              for mergeX in 0..<merge {
                for channel in 0..<3 {
                  for temporalPatch in 0..<2 {
                    let frame = min(time * 2 + temporalPatch, frameCount - 1)
                    for patchY in 0..<patch {
                      let y = (blockY * merge + mergeY) * patch + patchY
                      for patchX in 0..<patch {
                        let x = (blockX * merge + mergeX) * patch + patchX
                        let source = frame * frameBytes + (y * width + x) * 3 + channel
                        let normalized = Float(input[source]) / 255
                        values[destination] = (normalized - 0.5) / 0.5
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
    }
    return Packed(pixels: MLXArray(values, [patchCount, 1536]),
      grid: H3QwenRequest.Grid(temporal: temporal,
        height: gridHeight, width: gridWidth))
  }
}
