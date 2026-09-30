import Foundation
import MLX
import LTX25Engine

/// Encodes the two video grids and synchronized source audio with one weighted
/// component resident at a time. RGB24 source frames stay on disk between tiles.
public struct MLXExtensionGuideEncoder {
  public let window:LTX25ExtensionWindow
  public let videoCheckpoint:URL
  public let audioCheckpoint:URL
  public let maximumOwnedBufferBytes:Int

  public init(window:LTX25ExtensionWindow,videoCheckpoint:URL,audioCheckpoint:URL,
    maximumOwnedBufferBytes:Int=4*1024*1024*1024) throws {
    guard videoCheckpoint.isFileURL,audioCheckpoint.isFileURL,
      maximumOwnedBufferBytes>0 else {
      throw LTXError.invalid("LTX extension VAE checkpoints and buffer budget must be local and explicit.")
    }
    let g=window.geometry
    _ = try MLXVideoEncodeTilePlan(frames:window.contextFrames,width:g.width/2,height:g.height/2,
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    _ = try MLXVideoEncodeTilePlan(frames:window.contextFrames,width:g.width,height:g.height,
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    _ = try MLXVideoEncoder(checkpoint:videoCheckpoint)
    _ = try MLXAudioEncoder(checkpoint:audioCheckpoint)
    self.window=window;self.videoCheckpoint=videoCheckpoint
    self.audioCheckpoint=audioCheckpoint;self.maximumOwnedBufferBytes=maximumOwnedBufferBytes
  }

  public func encode(_ source:MLXSourceMovieInterval.Prepared,
    progress:(String,Int,Int) throws -> Void = { _,_,_ in }) throws -> MLXDistilledSamplingRunner.ExtensionGuides {
    let g=window.geometry
    let low=try MLXVideoEncodeTilePlan(frames:window.contextFrames,width:g.width/2,height:g.height/2,
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    let high=try MLXVideoEncodeTilePlan(frames:window.contextFrames,width:g.width,height:g.height,
      maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    let stageOne=try autoreleasepool {
      let tokens=try MLXTiledVideoEncoder.encode(guide:source.lowRGB24,
        checkpoint:videoCheckpoint,plan:low) { try progress("extension_video_low",$0,$1) }
      return tokens.reshaped([low.latentShape[0]*low.latentShape[1]*low.latentShape[2],128])
    }
    Stream.gpu.synchronize();Memory.clearCache();try progress("extension_video_low_released",1,1)
    let stageTwo=try autoreleasepool {
      let tokens=try MLXTiledVideoEncoder.encode(guide:source.rgb24,
        checkpoint:videoCheckpoint,plan:high) { try progress("extension_video_high",$0,$1) }
      return tokens.reshaped([high.latentShape[0]*high.latentShape[1]*high.latentShape[2],128])
    }
    Stream.gpu.synchronize();Memory.clearCache();try progress("extension_video_high_released",1,1)
    let audio=try autoreleasepool {
      let mel=try MLXAudioMel.encode(wav:source.audio16k)
      let encoder=try MLXAudioEncoder(checkpoint:audioCheckpoint)
      let encoded=try encoder.encode(mel:mel) { try progress("extension_audio",$0,$1) }
      let expected=Int(ceil(Double(window.contextFrames)/g.fps*25))
      let fitted=encoded.shape[0]<expected
        ? concatenated([encoded,MLXArray.zeros([expected-encoded.shape[0],128])],axis:0)
        : encoded[0..<expected]
      eval(fitted)
      return fitted
    }
    Stream.gpu.synchronize();Memory.clearCache();try progress("extension_audio_released",1,1)
    return MLXDistilledSamplingRunner.ExtensionGuides(stageOneVideo:stageOne.asType(.float32),
      stageTwoVideo:stageTwo.asType(.float32),audio:audio.asType(.float32))
  }
}
