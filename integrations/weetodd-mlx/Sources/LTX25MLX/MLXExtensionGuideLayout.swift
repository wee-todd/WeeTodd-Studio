import Foundation
import MLX
import LTX25Engine

/// Appended audiovisual source guides for one LTX extension sampling stage.
/// The generated timeline keeps its original token order; source history uses
/// matching leading positions and is cropped from the sampler result.
public struct MLXExtensionGuideLayout {
  public let geometry:AVGeometry
  public let contextFrames:Int
  public let videoGuideTokens:Int
  public let audioGuideTokens:Int
  public let videoGuideLatentFrames:Int
  public let strength:Float
  public var videoTokens:Int { geometry.videoTokens+videoGuideTokens }
  public var audioTokens:Int { geometry.audioFrames+audioGuideTokens }

  public init(geometry:AVGeometry,contextFrames:Int,
    videoGuideLatentFrames:Int?=nil,audioGuideTokens:Int?=nil,
    strength:Float=0.5) throws {
    guard contextFrames>=9,(contextFrames-1)%8==0,contextFrames<geometry.frames,
      strength.isFinite,(0...1).contains(strength) else {
      throw LTXError.invalid("LTX extension guide needs aligned source context shorter than the sampled window.")
    }
    let latentFrames=(contextFrames-1)/8+1
    let selectedVideoFrames=videoGuideLatentFrames ?? latentFrames
    let videoCount=selectedVideoFrames*geometry.latentHeight*geometry.latentWidth
    let audioCount=audioGuideTokens ?? Int(ceil(Double(contextFrames)/geometry.fps*25))
    guard (1...latentFrames).contains(selectedVideoFrames),audioCount>0 else {
      throw LTXError.invalid("LTX scene guide lengths exceed their causal overlap.")
    }
    guard videoCount<=geometry.videoTokens,audioCount<=geometry.audioFrames else {
      throw LTXError.invalid("LTX extension source guides exceed the sampled timeline.")
    }
    _ = try AVBlockConfiguration(videoTokens:geometry.videoTokens+videoCount,
      audioTokens:geometry.audioFrames+audioCount,textTokens:1024)
    self.geometry=geometry;self.contextFrames=contextFrames
    videoGuideTokens=videoCount;self.audioGuideTokens=audioCount
    self.videoGuideLatentFrames=selectedVideoFrames;self.strength=strength
  }

  public var videoPositions:[Float] {
    let target=geometry.videoPositions
    return target+Array(target.prefix(videoGuideTokens*3))
  }
  public var audioPositions:[Float] {
    let target=geometry.audioPositions
    return target+Array(target.prefix(audioGuideTokens))
  }

  public func prepare(targetVideo:MLXArray,targetVideoCondition:MLXVideoDenoiseCondition?=nil,
    targetAudio:MLXArray,targetAudioCondition:MLXAudioDenoiseCondition?=nil,
    sourceVideo:MLXArray,sourceAudio:MLXArray,
    guideVideoNoise:MLXArray,guideAudioNoise:MLXArray,sigma:Float) throws ->
    (video:MLXArray,audio:MLXArray,videoCondition:MLXVideoDenoiseCondition,audioCondition:MLXAudioDenoiseCondition) {
    _ = try DenoiserMath.timestep(sigma)
    for (value,count,label) in [(targetVideo,geometry.videoTokens,"target video"),
      (targetAudio,geometry.audioFrames,"target audio"),
      (sourceVideo,videoGuideTokens,"source video"),(sourceAudio,audioGuideTokens,"source audio"),
      (guideVideoNoise,videoGuideTokens,"guide video noise"),
      (guideAudioNoise,audioGuideTokens,"guide audio noise")] {
      guard value.dtype == .float32,value.shape == [count,128],MLX.isFinite(value).all().item(Bool.self) else {
        throw LTXError.invalid("Invalid LTX extension \(label) tokens.")
      }
    }
    if let targetVideoCondition {
      guard targetVideoCondition.clean.shape == targetVideo.shape,
        targetVideoCondition.mask.count == geometry.videoTokens else {
        throw LTXError.invalid("LTX image anchor does not match the audiovisual scene target.")
      }
    }
    if let targetAudioCondition {
      guard targetAudioCondition.clean.shape == targetAudio.shape,
        targetAudioCondition.mask.count == geometry.audioFrames else {
        throw LTXError.invalid("LTX audio driver does not match the audiovisual scene target.")
      }
    }
    let guideSigma=sigma*(1-strength)
    let videoGuide=sourceVideo*(1-guideSigma)+guideVideoNoise*guideSigma
    let audioGuide=sourceAudio*(1-guideSigma)+guideAudioNoise*guideSigma
    let video=concatenated([targetVideo,videoGuide],axis:0)
    let audio=concatenated([targetAudio,audioGuide],axis:0)
    let videoClean=concatenated([targetVideoCondition?.clean ?? MLXArray.zeros(targetVideo.shape),sourceVideo],axis:0)
    let audioClean=concatenated([targetAudioCondition?.clean ?? MLXArray.zeros(targetAudio.shape),sourceAudio],axis:0)
    let videoMask=(targetVideoCondition?.mask ?? Array(repeating:Float(1),count:geometry.videoTokens)) +
      Array(repeating:1-strength,count:videoGuideTokens)
    let audioMask=(targetAudioCondition?.mask ?? Array(repeating:Float(1),count:geometry.audioFrames)) +
      Array(repeating:1-strength,count:audioGuideTokens)
    return (video,audio,try MLXVideoDenoiseCondition(clean:videoClean,mask:videoMask),
      try MLXAudioDenoiseCondition(clean:audioClean,mask:audioMask))
  }
}
