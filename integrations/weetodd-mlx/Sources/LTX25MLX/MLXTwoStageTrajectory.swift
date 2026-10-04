import Foundation
import MLX
import LTX25Engine

/// Shared released recipe and native RNG; evaluated states remain MLX-resident.
/// Callbacks are synchronous and must release their weighted stage before return.
public final class MLXTwoStageTrajectory {
  public typealias Sample=(Int,AVGeometry,[String:MLXArray],SamplingSchedule,MLXSamplingRunner.Noise?) throws -> [String:MLXArray]
  public typealias GuideSample=(Int,AVGeometry,[String:MLXArray],[String:MLXArray],SamplingSchedule,MLXSamplingRunner.Noise?) throws -> [String:MLXArray]
  public typealias Upscale=(MLXArray,[Int]) throws -> MLXArray
  private let gate=NSLock()
  public init() {}

  public func evaluate(recipe:DistilledTwoStageRecipe,noisePolicy:MLXNoisePolicy = .native,
    frozenAudio:MLXArray?=nil,firstSchedule:SamplingSchedule?=nil,
    stageOneVideoObserver:((MLXArray) throws -> Void)?=nil,
    sample:Sample,upscale:Upscale) throws -> [String:MLXArray] {
    try evaluateInternal(recipe:recipe,noisePolicy:noisePolicy,frozenAudio:frozenAudio,guideTokens:nil,firstSchedule:firstSchedule,
      stageOneVideoObserver:stageOneVideoObserver,
      sample:{ stage,g,state,_,schedule,noise in try sample(stage,g,state,schedule,noise) },upscale:upscale)
  }

  /// Extension draws target and appended source-guide noise in one shaped MLX
  /// tensor. Threefry results are shape-dependent, so separate draws would
  /// silently change the target sequence for the same seed.
  public func evaluateWithGuides(recipe:DistilledTwoStageRecipe,videoGuideFrames:Int,
    audioGuideTokens:Int,stageOneVideoObserver:((MLXArray) throws -> Void)?=nil,
    sample:GuideSample,upscale:Upscale) throws -> [String:MLXArray] {
    guard videoGuideFrames>0,audioGuideTokens>0 else { throw LTXError.invalid("LTX extension needs positive audiovisual guide lengths.") }
    return try evaluateInternal(recipe:recipe,noisePolicy:.releasedMLX,frozenAudio:nil,
      guideTokens:(videoFrames:videoGuideFrames,audio:audioGuideTokens),firstSchedule:nil,
      stageOneVideoObserver:stageOneVideoObserver,sample:sample,upscale:upscale)
  }

  private func evaluateInternal(recipe:DistilledTwoStageRecipe,noisePolicy:MLXNoisePolicy,
    frozenAudio:MLXArray?,guideTokens:(videoFrames:Int,audio:Int)?,firstSchedule:SamplingSchedule?,
    stageOneVideoObserver:((MLXArray) throws -> Void)?,
    sample:GuideSample,upscale:Upscale) throws -> [String:MLXArray] {
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
      let lowGuideVideo=guideTokens.map { $0.videoFrames*recipe.low.latentHeight*recipe.low.latentWidth } ?? 0
      let initialVideoFull=noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed,tokens:recipe.low.videoTokens+lowGuideVideo).asType(.float32)
        : try Self.random(&initial,tokens:recipe.low.videoTokens)
      let initialVideo=initialVideoFull[0..<recipe.low.videoTokens]
      let initialAudioFull=try sourceAudio ?? (noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed &+ 1,tokens:recipe.low.audioFrames+(guideTokens?.audio ?? 0)).asType(.float32)
        : Self.random(&initial,tokens:recipe.low.audioFrames))
      let initialAudio=initialAudioFull[0..<recipe.low.audioFrames]
      let firstGuide:[String:MLXArray]=guideTokens == nil ? [:] : [
        "video":initialVideoFull[recipe.low.videoTokens...],
        "audio":initialAudioFull[recipe.low.audioFrames...]]
      var state=["video":initialVideo,"audio":initialAudio]
      if let schedule=firstSchedule,schedule.sigmas[0] != 1 {
        state["video"]=initialVideo*Float(schedule.sigmas[0])
        if sourceAudio == nil { state["audio"]=initialAudio*Float(schedule.sigmas[0]) }
      }
      // Fix evaluation/draw order explicitly; dictionary iteration never owns RNG.
      var ancestral=GaussianNoise(seed:recipe.seed &+ 10000)
      var ancestralKey=MLXRandom.key(recipe.seed &+ 10000)
      let first=try Self.validated(sample(1,recipe.low,state,firstGuide,firstSchedule ?? recipe.first,{ _,_,shape in
        try Task.checkCancellation()
        if noisePolicy == .releasedMLX {
          let (next,draw)=MLXRandom.split(key:ancestralKey); ancestralKey=next
          return MLXRandom.normal([1]+shape,key:draw).reshaped(shape)
        }
        return MLXArray(try ancestral.values(count:shape.reduce(1,*)),shape)
      }),geometry:recipe.low)
      try Task.checkCancellation()
      try stageOneVideoObserver?(first["video"]!)
      let high=try upscale(first["video"]!.reshaped(first["video"]!.shape),
        [recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128])
      try Task.checkCancellation()
      guard high.dtype == .float32, high.shape == [recipe.high.videoTokens,128] else {
        throw LTXError.invalid("Upscaler returned invalid high-resolution geometry or dtype.")
      }
      let clean=high.reshaped(high.shape)
      try Self.finish(clean)
      var random=GaussianNoise(seed:recipe.seed &+ 2)
      let highGuideVideo=guideTokens.map { $0.videoFrames*recipe.high.latentHeight*recipe.high.latentWidth } ?? 0
      let vnFull=noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed &+ 2,tokens:recipe.high.videoTokens+highGuideVideo)
        : try Self.random(&random,tokens:recipe.high.videoTokens)
      let vn=vnFull[0..<recipe.high.videoTokens]
      let anFull=try sourceAudio == nil ? (noisePolicy == .releasedMLX
        ? MLXNoisePolicy.seeded(recipe.seed &+ 2,tokens:recipe.high.audioFrames+(guideTokens?.audio ?? 0))
        : Self.random(&random,tokens:recipe.high.audioFrames)) : nil
      let an=anFull?[0..<recipe.high.audioFrames]
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
      let secondGuide:[String:MLXArray]=guideTokens == nil ? [:] : [
        "video":vnFull[recipe.high.videoTokens...].asType(.float32),
        "audio":anFull![recipe.high.audioFrames...].asType(.float32)]
      return (state:["video":video,"audio":audio],guides:secondGuide)
    }
    Memory.clearCache(); try Task.checkCancellation()
    return try Self.validated(sample(2,recipe.high,refinement.state,refinement.guides,recipe.second,nil),geometry:recipe.high)
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
