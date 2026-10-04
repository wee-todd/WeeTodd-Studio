import Foundation
import MLX

public enum H3SamplingMethod: String, Sendable, Codable {
  case euler
  case resMultistep = "res_multistep"
}

/// Run-local, target-only history. This independently implements the owned
/// Python H3 scheduler's float32 coefficients and operation order; it adds no
/// stochastic draws and does not change the historical Euler update.
struct H3SamplingStepper {
  let schedule: H3Schedule
  let method: H3SamplingMethod
  private var nextIndex = 0
  private var oldDenoised: MLXArray?
  private var oldSigmaDown: Float?

  init(schedule: H3Schedule, method: H3SamplingMethod) {
    self.schedule = schedule
    self.method = method
  }

  mutating func advance(sample: MLXArray, velocity: MLXArray,
    index: Int) throws -> MLXArray {
    guard index == nextIndex, schedule.timesteps.indices.contains(index),
      sample.shape == velocity.shape, sample.dtype == .float32,
      velocity.dtype == .float32,
      oldDenoised == nil || oldDenoised!.shape == sample.shape else {
      throw H3CheckpointError.invalid("Invalid H3 sequential sampling state or target rows.")
    }
    try Task.checkCancellation()
    if method == .euler {
      let result = try schedule.advance(sample: sample, velocity: velocity, index: index)
      nextIndex += 1
      return result
    }
    let sigma = schedule.sigmas[index]
    let sigmaNext = schedule.sigmas[index + 1]
    let denoised = sample + (Float(1) - schedule.timesteps[index]) * velocity
    let result: MLXArray
    if sigmaNext == 0 {
      result = denoised
    } else if let oldDenoised, let oldSigmaDown {
      let t = -logf(sigma)
      let tOld = -logf(oldSigmaDown)
      let tNext = -logf(sigmaNext)
      let tPrevious = -logf(schedule.sigmas[index - 1])
      let h = tNext - t
      let c2 = (tPrevious - tOld) / h
      let minusH = -h
      let phi1 = expm1f(minusH) / minusH
      let phi2 = (phi1 - Float(1)) / minusH
      // np.nan_to_num's float32 scalar policy, including signed infinities.
      func finite(_ value: Float) -> Float {
        if value.isNaN { return 0 }
        if value == .infinity { return .greatestFiniteMagnitude }
        if value == -.infinity { return -.greatestFiniteMagnitude }
        return value
      }
      let b1 = finite(phi1 - phi2 / c2)
      let b2 = finite(phi2 / c2)
      let decay = expf(-h)
      result = decay * sample + h * (b1 * denoised + b2 * oldDenoised)
    } else {
      let derivative = (sample - denoised) / sigma
      result = sample + derivative * (sigmaNext - sigma)
    }
    self.oldDenoised = denoised
    self.oldSigmaDown = sigmaNext
    nextIndex += 1
    return result.asType(sample.dtype)
  }
}
