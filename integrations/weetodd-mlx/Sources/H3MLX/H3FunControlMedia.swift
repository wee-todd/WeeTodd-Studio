import Darwin
import Foundation
import MLX

public struct H3FunControlGuide: Sendable {
  public let checkpoint: URL
  public let strength: Float
  public let video: H3VideoReference

  public init(checkpoint: URL, strength: Float, video: H3VideoReference) throws {
    guard checkpoint.isFileURL, checkpoint.path.hasPrefix("/"),
      checkpoint.pathExtension.lowercased() == "safetensors",
      strength.isFinite, (0...1).contains(strength) else {
      throw H3CheckpointError.invalid("H3 Fun control needs a local checkpoint and finite strength from 0 to 1.")
    }
    self.checkpoint = checkpoint; self.strength = strength; self.video = video
  }

  public func validate(geometry: H3Geometry) throws {
    guard geometry.width <= 2048, geometry.height <= 2048,
      video.width == geometry.width, video.height == geometry.height,
      video.frameCount == geometry.frames, video.audio == nil,
      video.rgb8.count == geometry.frames * geometry.width * geometry.height * 3,
      video.rgb8.count <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("H3 Fun guide needs the exact output canvas and aligned frame count, under 1 GiB of RGB8 pixels.")
    }
  }

  func encode(videoVAE: URL, geometry: H3Geometry,
    progress: (Int, Int) -> Void = { _, _ in }) throws -> H3FunControlCondition {
    try validate(geometry: geometry)
    let metadata = try H3VideoVAELayout(url: videoVAE)
    let latent = try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: videoVAE,
      rgb8: Array(video.rgb8), frameCount: geometry.frames,
      width: geometry.width, height: geometry.height, progress: progress)
    let rows = try H3LatentCodec.videoEncoderRows(latents: latent,
      mean: metadata.latentsMean, standardDeviation: metadata.latentsStandardDeviation)
    guard rows.shape == [1, geometry.videoRows, 96] else {
      throw H3CheckpointError.invalid("H3 Fun encoded guide changed its admitted target geometry.")
    }
    return H3FunControlCondition(checkpoint: checkpoint, strength: strength, guideRows: rows)
  }
}

public enum H3FunControlMedia {
  /// Preprocessed control pixels are resampled to 24 fps, cover-cropped and
  /// held at the last source frame when short. They never contribute sound.
  public static func load(path: String, ffmpeg: URL,
    geometry: H3Geometry) throws -> H3VideoReference {
    let count = geometry.frames * geometry.width * geometry.height * 3
    guard path.hasPrefix("/"), !path.utf8.contains(0),
      geometry.width <= 2048, geometry.height <= 2048,
      count <= 1024 * 1024 * 1024, ffmpeg.isFileURL,
      FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw H3CheckpointError.invalid("H3 Fun guide needs a bounded local movie and executable FFmpeg.")
    }
    let process = Process()
    process.executableURL = ffmpeg
    let filter = "fps=24,scale=\(geometry.width):\(geometry.height):force_original_aspect_ratio=increase:flags=lanczos,crop=\(geometry.width):\(geometry.height),tpad=stop_mode=clone:stop_duration=16"
    process.arguments = ["-v", "error", "-nostdin", "-i", path,
      "-map", "0:v:0", "-an", "-vf", filter, "-frames:v", String(geometry.frames),
      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    let pipe = Pipe(); process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning {
        process.terminate(); usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    var data = Data()
    while let part = try pipe.fileHandleForReading.read(upToCount: 1024 * 1024), !part.isEmpty {
      try Task.checkCancellation()
      guard data.count + part.count <= count else {
        throw H3CheckpointError.invalid("H3 Fun guide exceeded its admitted RGB decode window.")
      }
      data.append(part)
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10_000) }
    guard process.terminationStatus == 0, data.count == count else {
      throw H3CheckpointError.invalid("H3 Fun guide could not decode the admitted 24 fps frame count.")
    }
    return H3VideoReference(rgb8: data, frameCount: geometry.frames,
      width: geometry.width, height: geometry.height)
  }
}
