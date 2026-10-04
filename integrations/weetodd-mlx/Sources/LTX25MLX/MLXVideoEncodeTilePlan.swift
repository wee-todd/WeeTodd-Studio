import Foundation
import LTX25Engine

/// Spatial and causal-temporal partitions for an RGB24 video guide. Each tile
/// is admitted by the existing encoder before any source bytes or weights load.
public struct MLXVideoEncodeTilePlan: Sendable {
  /// Whole spatial windows retain the encoder's convolution context. Forced
  /// tiles are useful for qualification and remain the bounded fallback.
  public enum SpatialPolicy: Sendable { case preferWhole, tiles }

  public struct Tile: Sendable {
    public let frameStart: Int, frameEnd: Int
    public let yStart: Int, yEnd: Int
    public let xStart: Int, xEnd: Int
    public let ownedBufferBytes: Int
    public var frames: Int { frameEnd - frameStart }
    public var height: Int { yEnd - yStart }
    public var width: Int { xEnd - xStart }
    public var latentFrames: Int { (frames - 1) / 8 + 1 }
    public var latentHeight: Int { height / 32 }
    public var latentWidth: Int { width / 32 }
    public var latentFrameStart: Int { frameStart / 8 }
    public var latentYStart: Int { yStart / 32 }
    public var latentXStart: Int { xStart / 32 }
  }

  public let frames: Int, width: Int, height: Int
  public let maximumOwnedBufferBytes: Int
  public let latentShape: [Int]
  public let tiles: [Tile]

  public init(frames: Int, width: Int, height: Int, tilePixels: Int = 512,
    spatialPolicy: SpatialPolicy = .preferWhole,
    maximumOwnedBufferBytes: Int = 4 * 1024 * 1024 * 1024) throws {
    guard tilePixels >= 128, tilePixels % 32 == 0, maximumOwnedBufferBytes > 0 else {
      throw LTXError.invalid("Video encoder tiles require a bounded 32-pixel grid.")
    }
    // This validates the complete causal sequence without applying its
    // untiled memory limit. Every tile is then independently admitted below.
    let full = try MLXVideoEncodePlan(frames: frames, width: width, height: height,
      maximumOwnedBufferBytes: Int.max)
    self.frames = frames; self.width = width; self.height = height
    self.maximumOwnedBufferBytes = maximumOwnedBufferBytes
    latentShape = full.latentShape

    func spatial(_ length: Int) -> [(Int, Int)] {
      if length <= tilePixels { return [(0, length)] }
      let stride = tilePixels - 64
      let count = (length + tilePixels - 129) / stride
      return (0..<count).map { index in
        let start = index * stride
        return (start, index == count - 1 ? length : start + tilePixels)
      }
    }
    func temporal() -> [(Int, Int)] {
      if frames <= 33 { return [(0, frames)] }
      let count = (frames + 32 - 32 - 1) / 16
      return (0..<count).map { index in
        let start = index * 16
        return (start, index == count - 1 ? frames : start + 33)
      }
    }
    var result: [Tile] = []
    for (f0, f1) in temporal() {
      let useWhole = spatialPolicy == .preferWhole &&
        (try? MLXVideoEncodePlan(frames: f1 - f0, width: width, height: height,
          maximumOwnedBufferBytes: min(maximumOwnedBufferBytes, 4 * 1024 * 1024 * 1024))) != nil
      for (y0, y1) in useWhole ? [(0, height)] : spatial(height) {
        for (x0, x1) in useWhole ? [(0, width)] : spatial(width) {
          let admission = try MLXVideoEncodePlan(frames: f1 - f0, width: x1 - x0,
            height: y1 - y0, maximumOwnedBufferBytes: maximumOwnedBufferBytes)
          result.append(Tile(frameStart: f0, frameEnd: f1,
            yStart: y0, yEnd: y1, xStart: x0, xEnd: x1,
            ownedBufferBytes: admission.ownedBufferBytes))
        }
      }
    }
    guard !result.isEmpty else { throw LTXError.invalid("Video encoder produced no tiles.") }
    tiles = result
    guard coverageWeights().allSatisfy({ $0 > 0 }) else {
      throw LTXError.invalid("Video encoder tile masks leave uncovered latent pixels.")
    }
  }

  /// Rectangular ramps discard spatial edge latents and the causal prefix of
  /// non-first temporal tiles, then normalize any remaining overlap.
  public func weight(_ tile: Tile, frame: Int, y: Int, x: Int) -> Float {
    guard (0..<tile.latentFrames).contains(frame),
      (0..<tile.latentHeight).contains(y), (0..<tile.latentWidth).contains(x) else { return 0 }
    if tile.frameStart > 0 && frame < 2 { return 0 }
    if tile.yStart > 0 && y == 0 { return 0 }
    if tile.yEnd < height && y == tile.latentHeight - 1 { return 0 }
    if tile.xStart > 0 && x == 0 { return 0 }
    if tile.xEnd < width && x == tile.latentWidth - 1 { return 0 }
    return 1
  }

  public func coverageWeights() -> [Float] {
    let fCount = latentShape[0], hCount = latentShape[1], wCount = latentShape[2]
    var weights = [Float](repeating: 0, count: fCount * hCount * wCount)
    for tile in tiles {
      for f in 0..<tile.latentFrames {
        for y in 0..<tile.latentHeight {
          for x in 0..<tile.latentWidth {
            let index = ((tile.latentFrameStart + f) * hCount + tile.latentYStart + y) * wCount
              + tile.latentXStart + x
            weights[index] += weight(tile, frame: f, y: y, x: x)
          }
        }
      }
    }
    return weights
  }
}
