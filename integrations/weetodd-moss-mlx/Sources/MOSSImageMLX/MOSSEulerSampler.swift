import Foundation
import MLX

enum MOSSEulerSampler {
  static func sample(initial: MLXArray, sigmas: [Float], guidance: Float, skipLayers: Set<Int>,
    skipStart: Float, skipStop: Float,
    predict: (MLXArray, Double, Bool, Set<Int>) throws -> MLXArray,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
    var latents = initial.asType(.float32)
    let steps = sigmas.count - 1
    for i in 0..<steps {
      try Task.checkCancellation()
      let timestep = Double(sigmas[i] * 1000)
      let positive = try predict(latents, timestep, false, []).asType(.float32)
      var prediction = positive
      if guidance != 1 {
        let progress = Float(i) / Float(steps)
        let skip = skipStart <= progress && progress < skipStop ? skipLayers : []
        let negative = try predict(latents, timestep, true, skip).asType(.float32)
        prediction = negative + guidance * (positive - negative)
      }
      guard isFinite(prediction).all().item(Bool.self) else { throw MOSSError("MOSS produced nonfinite predictions; the image was not saved.") }
      latents = latents + (sigmas[i + 1] - sigmas[i]) * prediction
      eval(latents)
      try onStep?(i + 1, steps, latents)
    }
    return latents
  }
}
