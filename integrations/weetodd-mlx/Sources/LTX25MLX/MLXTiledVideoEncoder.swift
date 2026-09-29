import Darwin
import Foundation
import MLX
import LTX25Engine

/// Encodes a streamed RGB24 causal guide without creating a whole-video MLX
/// tensor. Only one pixel tile, one VAE workspace, and the final latent buffer
/// are resident. Tile outputs are blended on the CPU before the next load.
public enum MLXTiledVideoEncoder {
  private static func openGuide(_ path: URL, plan: MLXVideoEncodeTilePlan) throws -> Int32 {
    guard path.isFileURL else { throw LTXError.invalid("Video guide must be a local RGB24 file.") }
    let descriptor = Darwin.open(path.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    guard descriptor >= 0 else { throw LTXError.invalid("Cannot open the RGB24 video guide.") }
    var info = stat()
    let expected = Int64(plan.frames) * Int64(plan.height) * Int64(plan.width) * 3
    guard fstat(descriptor, &info) == 0,
      info.st_mode & S_IFMT == S_IFREG, info.st_size == expected else {
      Darwin.close(descriptor)
      throw LTXError.invalid("RGB24 guide byte count differs from its admitted geometry.")
    }
    return descriptor
  }

  private static func readTile(_ descriptor: Int32, plan: MLXVideoEncodeTilePlan,
    tile: MLXVideoEncodeTilePlan.Tile) throws -> MLXArray {
    let rowBytes = tile.width * 3
    var row = [UInt8](repeating: 0, count: rowBytes)
    var pixels = [Float](repeating: 0, count: tile.frames * tile.height * rowBytes)
    for frame in 0..<tile.frames {
      try Task.checkCancellation()
      for y in 0..<tile.height {
        let offset = Int64(((tile.frameStart + frame) * plan.height + tile.yStart + y)
          * plan.width + tile.xStart) * 3
        var received = 0
        while received < rowBytes {
          let count = row.withUnsafeMutableBytes { bytes in
            Darwin.pread(descriptor, bytes.baseAddress!.advanced(by: received),
              rowBytes - received, off_t(offset + Int64(received)))
          }
          if count < 0 && errno == EINTR { continue }
          guard count > 0 else { throw LTXError.invalid("RGB24 guide ended inside a tile.") }
          received += count
        }
        let target = (frame * tile.height + y) * rowBytes
        for i in 0..<rowBytes { pixels[target + i] = Float(row[i]) * (2.0 / 255.0) - 1 }
      }
    }
    return MLXArray(pixels, [tile.frames, tile.height, tile.width, 3])
  }

  static func loadTile(path: URL, plan: MLXVideoEncodeTilePlan,
    tile: MLXVideoEncodeTilePlan.Tile) throws -> MLXArray {
    let descriptor = try openGuide(path, plan: plan)
    defer { Darwin.close(descriptor) }
    return try readTile(descriptor, plan: plan, tile: tile)
  }

  public static func encode(guide: URL, checkpoint: URL,
    plan: MLXVideoEncodeTilePlan,
    progress: (Int, Int) throws -> Void = { _, _ in }) throws -> MLXArray {
    let descriptor = try openGuide(guide, plan: plan)
    defer { Darwin.close(descriptor) }
    let encoder = try MLXVideoEncoder(checkpoint: checkpoint)
    let fCount = plan.latentShape[0], hCount = plan.latentShape[1], wCount = plan.latentShape[2]
    let weights = plan.coverageWeights()
    var accumulated = [Float](repeating: 0, count: fCount * hCount * wCount * 128)
    for (index, tile) in plan.tiles.enumerated() {
      try Task.checkCancellation()
      let pixels = try readTile(descriptor, plan: plan, tile: tile)
      let latent = try encoder.encode(pixels,
        maximumOwnedBufferBytes: plan.maximumOwnedBufferBytes)
      guard latent.shape == [tile.latentFrames, tile.latentHeight, tile.latentWidth, 128] else {
        throw LTXError.invalid("Encoded video tile differs from its planned latent geometry.")
      }
      let values = latent.asArray(Float.self)
      for f in 0..<tile.latentFrames {
        for y in 0..<tile.latentHeight {
          for x in 0..<tile.latentWidth {
            let amount = plan.weight(tile, frame: f, y: y, x: x)
            if amount == 0 { continue }
            let source = ((f * tile.latentHeight + y) * tile.latentWidth + x) * 128
            let target = (((tile.latentFrameStart + f) * hCount + tile.latentYStart + y)
              * wCount + tile.latentXStart + x) * 128
            for channel in 0..<128 {
              accumulated[target + channel] += values[source + channel] * amount
            }
          }
        }
      }
      try progress(index + 1, plan.tiles.count)
    }
    for token in weights.indices {
      let factor = 1 / weights[token]
      for channel in 0..<128 { accumulated[token * 128 + channel] *= factor }
    }
    let output = MLXArray(accumulated, plan.latentShape)
    eval(output)
    try MLXVideoDecoder.checkFinite(output)
    return output
  }
}
