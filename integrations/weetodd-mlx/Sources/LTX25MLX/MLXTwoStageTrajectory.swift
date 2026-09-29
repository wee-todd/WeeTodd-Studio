import Foundation
import MLX
import LTX25Engine

/// Shared released recipe and native RNG; evaluated states remain MLX-resident.
/// Callbacks are synchronous and must release their weighted stage before return.
public final class MLXTwoStageTrajectory {
  public typealias Sample=(Int,AVGeometry,[String:MLXArray],SamplingSchedule,MLXSamplingRunner.Noise?) throws -> [String:MLXArray]
  public typealias Upscale=(MLXArray,[Int]) throws -> MLXArray
  private let gate=NSLock()
  public init() {}

  public func evaluate(recipe:DistilledTwoStageRecipe,noisePolicy:MLXNoisePolicy = .native,
    frozenAudio:MLXArray?=nil,sample:Sample,upscale:Upscale) throws -> [String:MLXArray] {
    guard gate.try() else { throw LTXError.invalid("Two-stage MLX trajectory is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    try Task.checkCancellation()
    guard frozenAudio == nil || (recipe.low.audioFrames == recipe.high.audioFrames &&
      frozenAudio!.dtype == .float32 && frozenAudio!.shape == [recipe.low.audioFrames,128] &&
      MLX.isFinite(frozenAudio!).all().item(Bool.self)) else {
      throw LTXError.invalid("Frozen audio tokens must match both LTX stages before sampling.")
    }
    let sourceAudio=frozenAudio?.reshaped(frozenAudio!.shape)
    let refinement=try autoreleasepool {
      var initial=GaussianNoise(seed:recipe.seed)
      let initialVideo=noisePolicy == .releasedMLX ? MLXNoisePolicy.seeded(recipe.seed,tokens:recipe.low.videoTokens).asType(.float32) : try Self.random(&initial,tokens:recipe.low.videoTokens)
      let initialAudio=try sourceAudio ?? (noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed &+ 1,tokens:recipe.low.audioFrames).asType(.float32)
        : Self.random(&initial,tokens:recipe.low.audioFrames))
      let state=["video":initialVideo,"audio":initialAudio]
      // Fix evaluation/draw order explicitly; dictionary iteration never owns RNG.
      var ancestral=GaussianNoise(seed:recipe.seed &+ 10000)
      var ancestralKey=MLXRandom.key(recipe.seed &+ 10000)
      let first=try Self.validated(sample(1,recipe.low,state,recipe.first,{ _,_,shape in
        try Task.checkCancellation()
        if noisePolicy == .releasedMLX {
          let (next,draw)=MLXRandom.split(key:ancestralKey); ancestralKey=next
          return MLXRandom.normal([1]+shape,key:draw).reshaped(shape)
        }
        return MLXArray(try ancestral.values(count:shape.reduce(1,*)),shape)
      }),geometry:recipe.low)
      try Task.checkCancellation()
      let high=try upscale(first["video"]!.reshaped(first["video"]!.shape),
        [recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128])
      try Task.checkCancellation()
      guard high.dtype == .float32, high.shape == [recipe.high.videoTokens,128] else {
        throw LTXError.invalid("Upscaler returned invalid high-resolution geometry or dtype.")
      }
      let clean=high.reshaped(high.shape)
      try Self.finish(clean)
      var random=GaussianNoise(seed:recipe.seed &+ 2)
      let vn=noisePolicy == .releasedMLX ? MLXNoisePolicy.seeded(recipe.seed &+ 2,tokens:recipe.high.videoTokens) : try Self.random(&random,tokens:recipe.high.videoTokens)
      let an=try sourceAudio == nil ? (noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed &+ 2,tokens:recipe.high.audioFrames)
        : Self.random(&random,tokens:recipe.high.audioFrames)) : nil
      let sigma=Float(recipe.second.sigmas[0])
      let video:MLXArray,audio:MLXArray
      if noisePolicy == .releasedMLX {
        // Video legacy scalar blend rounds the noise product before promotion.
        video=vn*sigma+clean*(1-sigma)
        // Audio uses BF16 mask-aware blending, and the same seed+2 as video.
        let mask=MLXArray.ones([recipe.high.audioFrames,1],dtype:.bfloat16)*sigma
        audio=sourceAudio ?? (an!*mask+first["audio"]!.asType(.bfloat16)*(1-mask)).asType(.float32)
      } else { video=(1-sigma)*clean+sigma*vn;audio=sourceAudio ?? (1-sigma)*first["audio"]!+sigma*an! }
      try Self.finish(video); try Self.finish(audio)
      return ["video":video,"audio":audio]
    }
    Memory.clearCache(); try Task.checkCancellation()
    return try Self.validated(sample(2,recipe.high,refinement,recipe.second,nil),geometry:recipe.high)
  }
  private static func random(_ random:inout GaussianNoise,tokens:Int) throws -> MLXArray {
    let output=MLXArray(try random.values(count:tokens*128),[tokens,128])
    eval(output); try Task.checkCancellation()
    return output
  }
  private static func finish(_ x:MLXArray) throws {
    try Task.checkCancellation(); eval(x)
    guard MLX.isFinite(x).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite two-stage latents.") }
    try Task.checkCancellation()
  }
  private static func validated(_ values:[String:MLXArray],geometry:AVGeometry) throws -> [String:MLXArray] {
    try Task.checkCancellation()
    guard Set(values.keys) == ["video","audio"] else { throw LTXError.invalid("Invalid two-stage output keys.") }
    var result:[String:MLXArray]=[:]
    for (name,tokens) in [("video",geometry.videoTokens),("audio",geometry.audioFrames)] {
      let x=values[name]!
      guard x.dtype == .float32, x.shape == [tokens,128] else { throw LTXError.invalid("Invalid two-stage output shape/dtype.") }
      let snapshot=x.reshaped(x.shape)
      try finish(snapshot); result[name]=snapshot
    }
    return result
  }
}
