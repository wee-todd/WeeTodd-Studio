import Foundation
import MLX
import LTX25Engine

/// Combines scene-only image rows with the existing AV history blend. History
/// precedes appended images, matching the primary chain conditioning order.
/// There are no generated slots, marker rows or attention-isolation groups.
public struct MLXSceneKeyframeLayout {
  public struct Guides {
    public let video:MLXArray,audio:MLXArray,videoNoise:MLXArray,audioNoise:MLXArray
    public init(video:MLXArray,audio:MLXArray,videoNoise:MLXArray,audioNoise:MLXArray) {
      self.video=video;self.audio=audio;self.videoNoise=videoNoise;self.audioNoise=audioNoise
    }
  }
  public let ordinary:MLXOrdinaryKeyframeLayout
  public let extensionGuide:MLXExtensionGuideLayout?
  public var videoTokens:Int { ordinary.videoTokens+(extensionGuide?.videoGuideTokens ?? 0) }
  public var audioTokens:Int { extensionGuide?.audioTokens ?? ordinary.geometry.audioFrames }
  public var videoPositions:[Float] {
    (extensionGuide?.videoPositions ?? ordinary.geometry.videoPositions)+Array(ordinary.positions.dropFirst(ordinary.geometry.videoTokens*3))
  }
  public var audioPositions:[Float] { extensionGuide?.audioPositions ?? ordinary.geometry.audioPositions }
  public var videoAttentionGroups:[Int] { [] }
  public init(geometry:AVGeometry,anchors:[MLXOrdinaryKeyframeLayout.Anchor],
    extensionGuide:MLXExtensionGuideLayout?=nil) throws {
    ordinary=try MLXOrdinaryKeyframeLayout(geometry:geometry,anchors:anchors,generatedCount:0,maximumAnchors:32)
    guard extensionGuide == nil || (extensionGuide!.geometry.width == geometry.width &&
      extensionGuide!.geometry.height == geometry.height && extensionGuide!.geometry.frames == geometry.frames &&
      extensionGuide!.geometry.fps == geometry.fps) else {
      throw LTXError.invalid("Scene keyframes and history must share exact geometry.")
    }
    self.extensionGuide=extensionGuide
    _ = try AVBlockConfiguration(videoTokens:videoTokens,audioTokens:audioTokens,textTokens:1024)
  }
  /// The first driven window uses the sampler's exact frozen-audio branch.
  /// Only later windows need per-token audio masks for appended history rows.
  func sourceAudioConditionForHistory(_ source:MLXArray?) throws -> MLXAudioDenoiseCondition? {
    guard let source else {return nil}
    guard source.dtype == .float32,source.shape == [ordinary.geometry.audioFrames,128],
      MLX.isFinite(source).all().item(Bool.self) else {
      throw LTXError.invalid("Scene source audio differs from its admitted target rows.")
    }
    guard extensionGuide != nil else {return nil}
    return try MLXAudioDenoiseCondition(clean:source,mask:Array(repeating:0,count:ordinary.geometry.audioFrames))
  }
  public func prepare(generated:MLXArray,anchors:[MLXArray],targetAudio:MLXArray,
    targetAudioCondition:MLXAudioDenoiseCondition?=nil,guides:Guides?=nil,sigma:Float) throws ->
    (video:MLXArray,audio:MLXArray,videoCondition:MLXVideoDenoiseCondition,audioCondition:MLXAudioDenoiseCondition?) {
    try Task.checkCancellation()
    _ = try DenoiserMath.timestep(sigma)
    let geometry=ordinary.geometry
    guard (extensionGuide == nil)==(guides == nil),targetAudio.dtype == .float32,
      targetAudio.shape == [geometry.audioFrames,128],MLX.isFinite(targetAudio).all().item(Bool.self),
      targetAudioCondition == nil || targetAudioCondition!.clean.shape == targetAudio.shape else {
      throw LTXError.invalid("Scene audio or history differs from its admitted layout.")
    }
    let target=try ordinary.prepare(generated:generated,anchors:anchors)
    guard let guide=extensionGuide,let guides else {
      return (target.latent,targetAudio.reshaped(targetAudio.shape),target.condition,targetAudioCondition)
    }
    let main=geometry.videoTokens
    let mainCondition=try MLXVideoDenoiseCondition(clean:target.condition.clean[0..<main],
      mask:Array(target.condition.mask.prefix(main)))
    let base=try guide.prepare(targetVideo:target.latent[0..<main],targetVideoCondition:mainCondition,
      targetAudio:targetAudio,targetAudioCondition:targetAudioCondition,
      sourceVideo:guides.video,sourceAudio:guides.audio,
      guideVideoNoise:guides.videoNoise,guideAudioNoise:guides.audioNoise,sigma:sigma)
    let video:MLXArray,condition:MLXVideoDenoiseCondition
    if ordinary.anchorTokens>0 {
      video=concatenated([base.video,target.latent[main..<ordinary.videoTokens]],axis:0)
      condition=try MLXVideoDenoiseCondition(clean:concatenated([base.videoCondition.clean,
        target.condition.clean[main..<ordinary.videoTokens]],axis:0),
        mask:base.videoCondition.mask+Array(target.condition.mask.dropFirst(main)))
    } else { video=base.video;condition=base.videoCondition }
    eval(video,base.audio)
    try Task.checkCancellation()
    return(video,base.audio,condition,base.audioCondition)
  }
}
