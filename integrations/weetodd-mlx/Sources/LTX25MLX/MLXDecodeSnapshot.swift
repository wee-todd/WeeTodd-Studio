import Foundation
import MLX
import LTX25Engine

public enum MLXVideoBackend:String,Sendable { case mps,mlx }

/// Opt-in developer capture in the job's private, atomically published directory.
/// Stores decoder layouts, not packed transformer tokens, for sampling-free tests.
enum MLXDecodeSnapshot {
  static func write(_ latents:AVLatents,geometry:AVGeometry,to url:URL) throws {
    try Task.checkCancellation()
    guard url.isFileURL,url.pathExtension == "safetensors",!FileManager.default.fileExists(atPath:url.path) else {
      throw LTXError.invalid("Decoder snapshot requires a new local safetensors file.")
    }
    let video=try geometry.unpackVideo(latents.video),audio=try geometry.unpackAudio(latents.audio)
    try MLX.save(arrays:["video":MLXArray(video,geometry.videoShape),"audio":MLXArray(audio,[8,geometry.audioFrames,16])],
      metadata:["format":"weetodd-decoder-latents-v1","video_layout":"BCFHW","audio_layout":"CTF","fps":String(geometry.fps)],url:url)
    try Task.checkCancellation()
  }
}
