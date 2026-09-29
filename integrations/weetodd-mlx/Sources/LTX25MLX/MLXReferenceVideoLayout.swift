import Foundation
import MLX
import LTX25Engine

public struct RippleImageAnchor: Sendable {
  public let frame: Int
  public let strength: Float
  public init(frame: Int, strength: Float) {
    self.frame = frame
    self.strength = strength
  }
}

/// Ripple's full-resolution source guide is an appended IC-LoRA video group.
/// The edited first frame is encoded with the source movie as one causal guide.
/// Later edits remain separate timed image anchors, preserving full-rate guide
/// motion instead of splicing them into source frames.
public struct MLXReferenceVideoLayout: Sendable {
  public let geometry: AVGeometry
  public let strength: Float
  public let anchors: [RippleImageAnchor]
  public var referenceMask: Float { 1 - strength }
  public var frameTokens: Int { geometry.latentHeight * geometry.latentWidth }
  public var videoTokens: Int { geometry.videoTokens * 2 + anchors.count * frameTokens }
  public var positions: [Float] {
    var result = geometry.videoPositions + geometry.videoPositions
    result.reserveCapacity(videoTokens * 3)
    for anchor in anchors {
      let time = Float(Double(anchor.frame) + 0.5) / Float(geometry.fps)
      for h in 0..<geometry.latentHeight {
        for w in 0..<geometry.latentWidth {
          result += [time, Float(h * 32 + 16), Float(w * 32 + 16)]
        }
      }
    }
    return result
  }

  public init(geometry: AVGeometry, strength: Float, anchors: [RippleImageAnchor] = []) throws {
    guard strength.isFinite, (0...1).contains(strength), geometry.videoTokens <= 65536 else {
      throw LTXError.invalid("Ripple reference strength or combined video token budget is invalid.")
    }
    guard anchors.count <= 64,
      anchors.allSatisfy({ (1..<geometry.frames).contains($0.frame) &&
        $0.strength.isFinite && (0...1).contains($0.strength) }),
      zip(anchors, anchors.dropFirst()).allSatisfy({ pair in pair.0.frame < pair.1.frame }),
      geometry.videoTokens * 2 + anchors.count * geometry.latentHeight * geometry.latentWidth <= 131072
    else {
      throw LTXError.invalid("Ripple image anchors exceed frame, strength, or token bounds.")
    }
    self.geometry = geometry
    self.strength = strength
    self.anchors = anchors
  }

  public func prepare(generated: MLXArray, reference: MLXArray,
    anchors imageLatents: [MLXArray] = []) throws
    -> (latent: MLXArray, condition: MLXVideoDenoiseCondition) {
    let expected = [geometry.videoTokens, 128]
    guard generated.dtype == .float32, generated.shape == expected,
      reference.dtype == .float32, reference.shape == expected,
      imageLatents.count == anchors.count,
      imageLatents.allSatisfy({ $0.dtype == .float32 &&
        $0.shape == [frameTokens, 128] && MLX.isFinite($0).all().item(Bool.self) }),
      MLX.isFinite(generated).all().item(Bool.self),
      MLX.isFinite(reference).all().item(Bool.self) else {
      throw LTXError.invalid("Ripple guide or image anchors differ from the admitted geometry.")
    }
    let latent = concatenated([generated, reference] + imageLatents, axis: 0)
    let clean = concatenated([MLXArray.zeros(expected), reference] + imageLatents, axis: 0)
    let mask = [Float](repeating: 1, count: geometry.videoTokens)
      + [Float](repeating: referenceMask, count: geometry.videoTokens)
      + anchors.flatMap { [Float](repeating: 1 - $0.strength, count: frameTokens) }
    eval(latent, clean)
    return (latent, try MLXVideoDenoiseCondition(clean: clean, mask: mask))
  }
}
