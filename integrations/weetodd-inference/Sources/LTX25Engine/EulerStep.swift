import Foundation

/// Variance-preserving rectified-flow Euler coefficients. Noise is supplied by the
/// caller, keeping seed policy separate and allowing identical-noise backend comparisons.
public struct EulerStep: Sendable {
  public let sampleScale: Float
  public let predictionScale: Float
  public let noiseScale: Float
  public let terminal: Bool
  public let ancestral: Bool

  public init(sigma: Double, nextSigma: Double, eta: Double = 1, noiseStrength: Double = 1) throws {
    guard sigma.isFinite, sigma > 0, sigma <= 1, nextSigma.isFinite,
          nextSigma >= 0, nextSigma < sigma, eta.isFinite, (0...1).contains(eta),
          noiseStrength.isFinite, noiseStrength >= 0 else {
      throw LTXError.invalid("LTX Euler requires descending sigmas in [0, 1] and eta in [0, 1].")
    }
    terminal = nextSigma == 0
    ancestral = eta > 0 && !terminal
    if terminal {
      sampleScale = 0; predictionScale = 1; noiseScale = 0
      return
    }
    let down = nextSigma * (1 + (nextSigma / sigma - 1) * eta)
    let ratio = down / sigma
    let alpha = ancestral ? (1 - nextSigma) / (1 - down) : 1
    sampleScale = Float(alpha * ratio)
    predictionScale = Float(alpha * (1 - ratio))
    let variance = max(nextSigma * nextSigma - down * down * alpha * alpha, 0)
    noiseScale = ancestral ? Float(noiseStrength * sqrt(variance)) : 0
    guard sampleScale.isFinite, predictionScale.isFinite, noiseScale.isFinite else {
      throw LTXError.invalid("LTX Euler coefficients overflow Float32.")
    }
  }
}
