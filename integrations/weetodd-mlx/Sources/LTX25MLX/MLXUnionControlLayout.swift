import Foundation
import MLX
import LTX25Engine

/// One full-timeline Union Control guide at half the stage-one canvas size.
/// The guide is appended only while the Union adapter is active in stage one.
public struct MLXUnionControlLayout: Sendable {
  public let geometry: AVGeometry
  public let strength: Float
  public let referenceTokens: Int
  public let videoTokens: Int
  public let positions: [Float]

  public init(geometry: AVGeometry, strength: Float) throws {
    guard geometry.width % 64 == 0, geometry.height % 64 == 0,
      strength.isFinite, (0...1).contains(strength) else {
      throw LTXError.invalid("Union Control needs a half-resolution reference on a 64-pixel target grid.")
    }
    let rows = geometry.latentFrames * (geometry.latentHeight / 2) * (geometry.latentWidth / 2)
    let count = geometry.videoTokens + rows
    guard count <= 131_072 else {
      throw LTXError.invalid("Union Control reference exceeds its admitted video token budget.")
    }
    self.geometry = geometry
    self.strength = strength
    referenceTokens = rows
    videoTokens = count
    var result = geometry.videoPositions
    result.reserveCapacity(count * 3)
    for f in 0..<geometry.latentFrames {
      let time = Float(max(0, f * 8 - 7) + f * 8 + 1) / 2 / Float(geometry.fps)
      for h in 0..<(geometry.latentHeight / 2) {
        for w in 0..<(geometry.latentWidth / 2) {
          result += [time, Float(h * 32 + 16) * 2, Float(w * 32 + 16) * 2]
        }
      }
    }
    positions = result
  }

  public func prepare(generated: MLXArray, reference: MLXArray) throws
    -> (latent: MLXArray, condition: MLXVideoDenoiseCondition) {
    guard generated.dtype == .float32, generated.shape == [geometry.videoTokens, 128],
      reference.dtype == .float32, reference.shape == [referenceTokens, 128],
      MLX.isFinite(generated).all().item(Bool.self),
      MLX.isFinite(reference).all().item(Bool.self) else {
      throw LTXError.invalid("Union Control guide differs from its admitted half-resolution timeline.")
    }
    let latent = concatenated([generated, reference], axis: 0)
    let clean = concatenated([MLXArray.zeros([geometry.videoTokens, 128]), reference], axis: 0)
    let mask = [Float](repeating: 1, count: geometry.videoTokens)
      + [Float](repeating: 1 - strength, count: referenceTokens)
    eval(latent, clean)
    return (latent, try MLXVideoDenoiseCondition(clean: clean, mask: mask))
  }
}
