import Foundation

/// Rectified-flow Euler ancestral CFG++ at CFG=1, eta=1 and noise strength=1.
/// The unconditional prediction is raw x0; reference blending applies only to
/// the conditional prediction. The caller supplies noise and owns RNG policy.
public struct CFGPPAncestralStep: Sendable {
  public let sampleScale: Float
  public let predictionScale: Float
  public let unconditionalScale: Float
  public let noiseScale: Float
  public let terminal: Bool

  public init(sigma: Double, nextSigma: Double) throws {
    guard sigma.isFinite, sigma > 0, sigma <= 1, nextSigma.isFinite,
      nextSigma >= 0, nextSigma < sigma else {
      throw LTXError.invalid("CFG++ requires descending finite sigmas in [0, 1].")
    }
    terminal = nextSigma == 0
    if terminal {
      sampleScale = 0; predictionScale = 1; unconditionalScale = 0; noiseScale = 0
      return
    }
    // This normalized-SNR ratio has a finite limit at sigma=1. Computing
    // sigma/(1-sigma) directly would introduce infinity at the first step.
    let ratio = nextSigma * (1 - sigma) / (sigma * (1 - nextSigma))
    let sample = nextSigma / sigma * ratio
    sampleScale = Float(sample)
    predictionScale = Float(1 - nextSigma)
    unconditionalScale = Float(-sample * (1 - sigma))
    noiseScale = Float(nextSigma * sqrt(max(0, 1 - ratio * ratio)))
    guard [sampleScale, predictionScale, unconditionalScale, noiseScale].allSatisfy(\.isFinite) else {
      throw LTXError.invalid("CFG++ coefficients overflow Float32.")
    }
  }
}
