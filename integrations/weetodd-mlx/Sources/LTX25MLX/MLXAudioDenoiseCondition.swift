import Foundation
import MLX
import LTX25Engine

/// Clean audio tokens and per-token denoise factors for joint audiovisual continuation.
/// One generates a token; zero protects the supplied source token.
public struct MLXAudioDenoiseCondition {
  private let storedClean:MLXArray
  public let mask:[Float]
  public var clean:MLXArray { storedClean.reshaped(storedClean.shape) }
  public init(clean:MLXArray,mask:[Float]) throws {
    guard (1...131072).contains(mask.count),mask.allSatisfy({ $0.isFinite && $0>=0 && $0<=1 }),
      clean.dtype == .float32,clean.shape == [mask.count,128],MLX.isFinite(clean).all().item(Bool.self) else {
      throw LTXError.invalid("Invalid clean audio reference or denoise mask.")
    }
    storedClean=clean.reshaped(clean.shape);self.mask=mask
  }
}
