import Foundation

public enum FlashVSRVariant: String, Codable, Sendable { case turbo, base }
public enum FlashVSRError: Error, LocalizedError {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

/// Released FlashVSR+ stream architecture; these are model semantics, not UI profiles.
public struct FlashVSRConfig: Sendable {
  public let variant: FlashVSRVariant
  public let layers = 30, width = 1536, heads = 12, latentChannels = 16
  public let textWidth = 4096, frequencyWidth = 256, feedForwardWidth = 8960
  public let projectionLayers = 1, latentFrames = 4
  public let window = [2,8,8]
  public let epsilon: Float = 1e-6
  public init(variant: FlashVSRVariant) { self.variant = variant }
  public func isReplacement(_ layer: Int) -> Bool { variant == .turbo && layer % 6 != 0 }
  public func cacheWindows(layer: Int) -> Int { isReplacement(layer) ? 1 : 3 }
  public func localRange(layer: Int) -> Int { isReplacement(layer) ? 3 : 11 }
  public func topK(spatialBlocks: Int) -> Int { max(Int(Double(spatialBlocks) * Double(spatialBlocks) * 2.5) - 1, 0) }
}
