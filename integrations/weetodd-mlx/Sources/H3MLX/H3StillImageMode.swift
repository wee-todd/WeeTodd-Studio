import Foundation
import MLX

public enum H3ImageFrameSelection: String, Codable, Sendable, CaseIterable {
  case first, middle, last
}

/// Still generation retains H3's joint audiovisual sampler, with an explicit
/// short grid. It never relaxes ordinary video duration admission.
public enum H3StillImageMode: String, Codable, Sendable, CaseIterable {
  case qualityFiveFrames = "quality_five_frames"
  case singleLatent = "single_latent"

  public var frames: Int { self == .qualityFiveFrames ? 5 : 1 }
  public var latentFrames: Int { self == .qualityFiveFrames ? 2 : 1 }
  public var audioLatentFrames: Int { self == .qualityFiveFrames ? 8 : 2 }

  public func validateDecoder(imageOnly: Bool) throws {
    guard imageOnly == (self == .singleLatent) else {
      throw H3CheckpointError.invalid("Use the original VAE for quality stills and an image-only VAE for single-latent stills.")
    }
  }

  public func selectedIndex(_ selection: H3ImageFrameSelection) -> Int {
    switch selection { case .first: return 0; case .middle: return frames / 2; case .last: return frames - 1 }
  }

  public func select(decoded: MLXArray, selection: H3ImageFrameSelection) throws -> MLXArray {
    guard decoded.ndim == 5, decoded.shape[0] == 1,
      decoded.shape[1] == latentFrames * 4 else {
      throw H3CheckpointError.invalid("H3 still decoder returned an unexpected temporal grid.")
    }
    let index = 3 + selectedIndex(selection)
    return decoded[0..<1, index..<(index + 1), 0..<decoded.shape[2],
      0..<decoded.shape[3], 0..<decoded.shape[4]]
  }
}
