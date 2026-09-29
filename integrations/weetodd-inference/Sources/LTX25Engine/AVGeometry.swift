import Foundation

/// Batch-one, patch-size-one layout shared by transformer and convolutional VAEs.
/// Audio length follows the released Comfy contract: ceil(duration * 25).
/// Causal audio decoding may finish up to 30 ms early; publishing must preserve
/// actual sample counts rather than mislabeling or silently stretching the audio.
public struct AVGeometry: Sendable {
  public let width: Int, height: Int, frames: Int
  public let fps: Double
  public let latentFrames: Int, latentHeight: Int, latentWidth: Int, audioFrames: Int
  public var videoTokens: Int { latentFrames * latentHeight * latentWidth }
  public var videoShape: [Int] { [1, 128, latentFrames, latentHeight, latentWidth] }
  public init(width: Int, height: Int, frames: Int, fps: Double) throws {
    guard (32...4096).contains(width), (32...4096).contains(height), width % 32 == 0,
      height % 32 == 0,
      (1...4097).contains(frames), (frames - 1) % 8 == 0, fps.isFinite, (1...120).contains(fps)
    else {
      throw LTXError.invalid(
        "Video geometry requires bounded multiples of 32, 8n+1 frames, and 1–120 fps.")
    }
    self.width = width
    self.height = height
    self.frames = frames
    self.fps = fps
    latentFrames = (frames - 1) / 8 + 1
    latentHeight = height / 32
    latentWidth = width / 32
    audioFrames = Int(ceil(Double(frames) / fps * 25))
    guard videoTokens <= 131072, audioFrames <= 1501 else {
      throw LTXError.invalid("Audiovisual latent geometry exceeds allocation limits.")
    }
  }
  public var videoPositions: [Float] {
    var result = [Float]()
    result.reserveCapacity(videoTokens * 3)
    for t in 0..<latentFrames {
      let midpoint = Float(max(0, t * 8 - 7) + t * 8 + 1) / 2 / Float(fps)
      for h in 0..<latentHeight {
        for w in 0..<latentWidth { result += [midpoint, Float(h * 32 + 16), Float(w * 32 + 16)] }
      }
    }
    return result
  }
  public var audioPositions: [Float] {
    (0..<audioFrames).map { index in
      let start = Float(max(0, index * 4 - 3)) * 160 / 16000
      let end = Float(index * 4 + 1) * 160 / 16000
      return (start + end) / 2
    }
  }
  public func unpackVideo(_ packed: [Float]) throws -> [Float] {
    guard packed.count == videoTokens * 128, packed.allSatisfy(\.isFinite) else {
      throw LTXError.invalid("Invalid packed video latent.")
    }
    var output = [Float](repeating: 0, count: packed.count)
    for t in 0..<videoTokens {
      for c in 0..<128 { output[c * videoTokens + t] = packed[t * 128 + c] }
    }
    return output
  }
  public func unpackAudio(_ packed: [Float]) throws -> [Float] {
    guard packed.count == audioFrames * 128, packed.allSatisfy(\.isFinite) else {
      throw LTXError.invalid("Invalid packed audio latent.")
    }
    var output = [Float](repeating: 0, count: packed.count)
    for t in 0..<audioFrames {
      for c in 0..<8 {
        for f in 0..<16 { output[(c * audioFrames + t) * 16 + f] = packed[t * 128 + c * 16 + f] }
      }
    }
    return output
  }
}

/// Versioned native RNG: SplitMix64 uniforms + Box–Muller normals. Reproducible
/// within this backend; seeds are not claimed to match MLX or Draw Things RNGs.
public struct GaussianNoise: Sendable {
  public static let algorithm = "splitmix64-boxmuller-v1"
  private var state: UInt64
  private var spare: Float?
  public init(seed: UInt64) { state = seed }
  private mutating func uniform() -> Double {
    state &+= 0x9e37_79b9_7f4a_7c15
    var z = state
    z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
    z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
    return (Double((z ^ (z >> 31)) >> 11) + 0.5) / 9007199254740992.0
  }
  public mutating func values(count: Int) throws -> [Float] {
    guard (0...16_777_216).contains(count) else {
      throw LTXError.invalid("Noise allocation exceeds its bound.")
    }
    var output = [Float]()
    output.reserveCapacity(count)
    for i in 0..<count {
      if i % 4096 == 0 { try Task.checkCancellation() }
      if let ready = spare {
        output.append(ready)
        spare = nil
      } else {
        let radius = sqrt(-2 * log(uniform()))
        let angle = 2 * Double.pi * uniform()
        output.append(Float(radius * cos(angle)))
        spare = Float(radius * sin(angle))
      }
    }
    return output
  }
}
