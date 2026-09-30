import LTX25Engine

public enum MLXSceneDecodeMode: Sendable {
  case single
  case windowed(maximumFrames: Int?)

  public var maximumWindowFrames: Int? {
    switch self {
    case .single: return nil
    case .windowed(let frames): return frames
    }
  }
  public var publicationMode: String {
    switch self {
    case .single: return "single_decode_native_latent_chain"
    case .windowed: return "windowed_decode_native_latent_chain"
    }
  }
}

/// A bounded video-VAE schedule over the already assembled scene latent.
/// Adjacent ranges share four latent frames, which decode to 25 RGB frames.
public struct MLXSceneDecodeWindowPlan: Sendable {
  public let latentRanges: [Range<Int>]
  public let admittedActivationBytes: Int
  public let overlapFrames: Int

  public init(geometry: AVGeometry, maximumActivationBytes: Int,
    maximumWindowFrames: Int? = nil) throws {
    guard maximumActivationBytes > 0,
      maximumWindowFrames == nil ||
        (maximumWindowFrames! >= 33 && (maximumWindowFrames! - 1) % 8 == 0) else {
      throw LTXError.invalid("Scene video decode needs a positive allowance and aligned windows of at least 33 frames.")
    }
    let total = geometry.latentFrames
    let cap = min(total, maximumWindowFrames.map { ($0 - 1) / 8 + 1 } ?? total)
    let config = MLXMediaPipeline.videoConfiguration(for: geometry,
      activationBytes: maximumActivationBytes)
    func estimate(_ frames: Int) throws -> Int {
      try MLXVideoDecodePlan(shape: [1, 128, frames,
        geometry.latentHeight, geometry.latentWidth],
        configuration: config).admittedActivationBytes
    }
    // Binary search avoids probing every latent frame of a long scene.
    var lower = 4, upper = cap + 1
    while lower + 1 < upper {
      let middle = (lower + upper) / 2
      if (try? estimate(middle)) != nil { lower = middle }
      else { upper = middle }
    }
    guard lower >= 5 else {
      throw LTXError.invalid("Scene video decoder cannot admit a 33-frame window on this Mac.")
    }
    var ranges: [Range<Int>] = []
    var start = 0
    while start < total {
      let end = min(total, start + lower)
      ranges.append(start..<end)
      if end == total { break }
      start = end - 4
    }
    guard ranges.count <= 2 || lower >= 8 else {
      throw LTXError.invalid("Scene video decode needs at least 57 frames in every interior window to carry both 25-frame joins.")
    }
    self.latentRanges = ranges
    self.admittedActivationBytes = try estimate(ranges.map(\.count).max()!)
    self.overlapFrames = 25
  }
}
