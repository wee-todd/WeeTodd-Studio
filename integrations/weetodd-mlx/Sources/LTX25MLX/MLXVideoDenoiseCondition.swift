import Foundation
import MLX
import LTX25Engine

/// Immutable snapshot of trained reference tokens and per-token denoise factors.
/// One means generate; zero preserves the clean reference at the terminal step.
public struct MLXVideoDenoiseCondition {
  private let storedClean:MLXArray
  public let mask:[Float]
  public var clean:MLXArray { storedClean.reshaped(storedClean.shape) }
  public init(clean:MLXArray,mask:[Float]) throws {
    guard (1...131072).contains(mask.count),mask.allSatisfy({ $0.isFinite && $0>=0 && $0<=1 }),
      clean.dtype == .float32,clean.shape == [mask.count,128],MLX.isFinite(clean).all().item(Bool.self) else {
      throw LTXError.invalid("Invalid clean video reference or denoise mask.")
    }
    storedClean=clean.reshaped(clean.shape);self.mask=mask
  }
}
