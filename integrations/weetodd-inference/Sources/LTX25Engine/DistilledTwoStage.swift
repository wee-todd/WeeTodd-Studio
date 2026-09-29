import Foundation

public struct DistilledTwoStageRecipe: Sendable {
  public let low: AVGeometry
  public let high: AVGeometry
  public let seed: UInt64
  public let first: SamplingSchedule
  public let second: SamplingSchedule
  public static let identifier = "ltx25-distilled-two-stage-native-rng-v1"
  public init(width: Int,height: Int,frames: Int,fps: Double,seed: UInt64) throws {
    guard width >= 64,height >= 64,width % 64 == 0,height % 64 == 0 else {
      throw LTXError.invalid("Two-stage LTX requires final dimensions divisible by 64.")
    }
    high = try AVGeometry(width: width,height: height,frames: frames,fps: fps)
    low = try AVGeometry(width: width/2,height: height/2,frames: frames,fps: fps)
    self.seed = seed
    first = try SamplingSchedule(sigmas: [1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0],eta: 1)
    second = try SamplingSchedule(sigmas: [0.909375,0.725,0.421875,0],eta: 0)
  }
}

/// Owns stage ordering and versioned native RNG, while each synchronous callback
/// owns and releases its weighted component before returning. No second sampler.
public final class TwoStageTrajectory {
  public typealias Sample = (Int,AVGeometry,AVLatents,SamplingSchedule,EulerTrajectory.Noise?) throws -> AVLatents
  private let lock = NSLock()
  public init() {}
  public func evaluate(recipe: DistilledTwoStageRecipe,sample: Sample,
    upscale: ([Float],[Int]) throws -> [Float],
    checkCancelled: () throws -> Void = { try Task.checkCancellation() }) throws -> AVLatents {
    guard lock.try() else { throw LTXError.invalid("Two-stage trajectory is already running.") }
    defer { lock.unlock() }
    try checkCancelled()
    // Scope drops every low-resolution latent and upscaling temporary before
    // creating the larger stage-two transformer. Only materialized mixed arrays exit.
    let refinement: AVLatents = try autoreleasepool {
      var initial = GaussianNoise(seed: recipe.seed)
      let state = AVLatents(video: try initial.values(count: recipe.low.videoTokens*128),
        audio: try initial.values(count: recipe.low.audioFrames*128))
      var ancestral = GaussianNoise(seed: recipe.seed &+ 10000)
      let first = try sample(1,recipe.low,state,recipe.first,{ _,_,count in try ancestral.values(count: count) })
      try Self.validate(first,geometry: recipe.low); try checkCancelled()
      let high = try upscale(first.video,[recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128])
      guard high.count == recipe.high.videoTokens*128, high.allSatisfy(\.isFinite) else {
        throw LTXError.invalid("Upscaler returned invalid high-resolution latents.")
      }
      try checkCancelled()
      var random = GaussianNoise(seed: recipe.seed &+ 2)
      let sigma = Float(recipe.second.sigmas[0])
      func mix(_ clean: [Float]) throws -> [Float] {
        let noise = try random.values(count: clean.count)
        var output = clean
        for i in output.indices {
          if i % 4096 == 0 { try checkCancelled() }
          output[i] = (1-sigma)*clean[i]+sigma*noise[i]
        }
        return output
      }
      return AVLatents(video: try mix(high),audio: try mix(first.audio))
    }
    try checkCancelled()
    let result = try sample(2,recipe.high,refinement,recipe.second,nil)
    try Self.validate(result,geometry: recipe.high); try checkCancelled()
    return result
  }
  private static func validate(_ x: AVLatents,geometry g: AVGeometry) throws {
    guard x.video.count == g.videoTokens*128,x.audio.count == g.audioFrames*128,
      x.video.allSatisfy(\.isFinite),x.audio.allSatisfy(\.isFinite) else { throw LTXError.invalid("Invalid two-stage sampler output.") }
  }
}
