import Foundation
import MLX

/// Independent VP ER-SDE Taylor solver (Cui et al., arXiv:2309.06169).
/// The paper's data prediction equations use alpha=1-sigma for rectified flow.
enum MOSSERSDESampler {
  static func sample(initial: MLXArray, sigmas requested: [Float], guidance: Float, skipLayers: Set<Int>,
    skipStart: Float, skipStop: Float, shift: Float = 5,
    noise: (Int) throws -> MLXArray,
    predict: (MLXArray, Double, Bool, Set<Int>) throws -> MLXArray,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
    var sigmas = requested
    // A strictly positive signal coefficient is necessary for lambda=sigma/alpha.
    if sigmas[0] >= 1 { let t: Float = 1 - 0.0001; sigmas[0] = shift * t / (1 + (shift - 1) * t) }
    let lambdas = sigmas.map { $0 / (1 - $0) }
    func phi(_ value: Float) -> Float { value * (exp(pow(value, 0.3)) + 10) }
    var x = initial.asType(.float32), previous: MLXArray?, derivative: MLXArray?
    for i in 0..<(sigmas.count - 1) {
      try Task.checkCancellation()
      let s = sigmas[i], t = sigmas[i + 1]
      let positive = try predict(x, Double(s * 1000), false, []).asType(.float32)
      var velocity = positive
      if guidance != 1 {
        let progress = Float(i) / Float(sigmas.count - 1)
        let negative = try predict(x, Double(s * 1000), true, skipStart <= progress && progress < skipStop ? skipLayers : []).asType(.float32)
        velocity = negative + guidance * (positive - negative)
      }
      let clean = x - s * velocity
      if t == 0 { x = clean }
      else {
        let ls = lambdas[i], lt = lambdas[i + 1], alpha = 1 - t
        let ratio = phi(lt) / phi(ls), dt = lt - ls
        var next = x * (alpha / (1 - s) * ratio) + clean * (alpha * (1 - ratio))
        if let previous {
          var integral: Float = 0, moment: Float = 0
          let h = (ls - lt) / 200
          for j in 0..<200 {
            let p = lt + Float(j) * h, reciprocal = 1 / phi(p)
            integral += reciprocal; moment += (p - ls) * reciprocal
          }
          integral *= h; moment *= h
          let first = (clean - previous) / (ls - lambdas[i - 1])
          next = next + first * (alpha * (dt + integral * phi(lt)))
          if let derivative {
            let second = (first - derivative) * (2 / (ls - lambdas[i - 2]))
            next = next + second * (alpha * (dt * dt / 2 + moment * phi(lt)))
          }
          derivative = first
        }
        let deviation = alpha * sqrt(max(0, lt * lt - ls * ls * ratio * ratio))
        next = next + (try noise(i)) * deviation
        x = next
      }
      previous = clean
      guard isFinite(x).all().item(Bool.self) else { throw MOSSError("ER-SDE produced nonfinite latents.") }
      eval(x); try onStep?(i + 1, sigmas.count - 1, x)
    }
    return x
  }
}
