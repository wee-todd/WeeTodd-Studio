import AVFoundation
import Foundation

/// Decode only a bounded first segment of a local movie to H3's 24 fps grid.
/// The caller verifies source identity before and after this operation. FFmpeg
/// is the same configured, local media utility used to mux the finished take.
public enum H3VideoReferenceMedia {
  public struct Loaded {
    public let reference: H3VideoReference
    public let decodedFrames: Int
  }

  public static func load(path: String, ffmpeg: URL) throws -> Loaded {
    guard path.hasPrefix("/"), !path.utf8.contains(0),
      ffmpeg.isFileURL,
      FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw H3CheckpointError.invalid("H3 video reference needs a local movie and executable FFmpeg.")
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let hasAudio = !asset.tracks(withMediaType: .audio).isEmpty
    let side = 256
    let frameBytes = side * side * 3
    let maximumFrames = 175
    let maximumBytes = maximumFrames * frameBytes
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-nostdin", "-i", path,
      "-map", "0:v:0", "-vf",
      "fps=24,scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2:color=gray",
      "-frames:v", String(maximumFrames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try Task.checkCancellation()
    try process.run()
    defer {
      if process.isRunning {
        process.terminate()
        usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    var bytes = Data()
    bytes.reserveCapacity(min(maximumBytes, 22 * frameBytes))
    while let part = try output.fileHandleForReading.read(upToCount: 1024 * 1024),
      !part.isEmpty {
      try Task.checkCancellation()
      guard bytes.count + part.count <= maximumBytes else {
        throw H3CheckpointError.invalid("H3 reference movie exceeds its bounded decode window.")
      }
      bytes.append(part)
    }
    while process.isRunning {
      try Task.checkCancellation()
      usleep(10_000)
    }
    guard process.terminationStatus == 0, bytes.count.isMultiple(of: frameBytes) else {
      throw H3CheckpointError.invalid("H3 reference movie could not be decoded on the 24 fps grid.")
    }
    let decodedFrames = bytes.count / frameBytes
    guard decodedFrames >= 5 else {
      throw H3CheckpointError.invalid("H3 reference movie needs at least five decoded frames.")
    }
    let selected = (decodedFrames - 5) / 17 * 17 + 5
    bytes.count = selected * frameBytes
    let soundtrack = try hasAudio ? H3AudioReferenceMedia.load(path: path,
      ffmpeg: ffmpeg, maximumSeconds: Double(selected) / 24) : nil
    return Loaded(reference: H3VideoReference(rgb8: bytes,
      frameCount: selected, width: side, height: side,
      audio: soundtrack),
      decodedFrames: decodedFrames)
  }
}
