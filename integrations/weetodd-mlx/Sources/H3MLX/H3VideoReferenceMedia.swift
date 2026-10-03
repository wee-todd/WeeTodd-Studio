import AVFoundation
import Foundation

/// Decode only a bounded first segment of a local movie to H3's 24 fps grid.
/// The caller verifies source identity before and after this operation. FFmpeg
/// is the same configured, local media utility used to mux the finished take.
public enum H3VideoReferenceMedia {
  public struct Loaded {
    public let reference: H3VideoReference
    public let decodedFrames: Int
    public let lastFrame: H3StillReference
  }

  public static func load(path: String, ffmpeg: URL,
    retainCompleteAudio: Bool = false,
    preserveTail: Bool = false) throws -> Loaded {
    guard path.hasPrefix("/"), !path.utf8.contains(0),
      ffmpeg.isFileURL,
      FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw H3CheckpointError.invalid("H3 video reference needs a local movie and executable FFmpeg.")
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let hasAudio = !asset.tracks(withMediaType: .audio).isEmpty
    let side = 256
    let frameBytes = side * side * 3
    let maximumFrames = preserveTail ? 361 : 175
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
    guard decodedFrames >= 5, !preserveTail || decodedFrames <= 360 else {
      throw H3CheckpointError.invalid("H3 extension source needs 5–360 decoded frames.")
    }
    let paddedLastFrame = H3StillReference(rgb8: bytes.subdata(in:
      ((decodedFrames - 1) * frameBytes)..<(decodedFrames * frameBytes)),
      width: side, height: side)
    let selected = preserveTail
      ? decodedFrames + (5 - decodedFrames % 17 + 17) % 17
      : (decodedFrames - 5) / 17 * 17 + 5
    if preserveTail {
      for _ in decodedFrames..<selected { bytes.append(paddedLastFrame.rgb8) }
    } else { bytes.count = selected * frameBytes }
    // The square buffer is analysis/reference content. Its gray letterbox is
    // not the endpoint image: a timed frame-zero guide must retain source aspect.
    let lastFrame = try preserveTail ? seamFrame(path: path, ffmpeg: ffmpeg,
      asset: asset, frame: decodedFrames - 1) : paddedLastFrame
    let soundtrack = try hasAudio ? H3AudioReferenceMedia.load(path: path,
      ffmpeg: ffmpeg,
      maximumSeconds: Double(retainCompleteAudio ? decodedFrames : selected) / 24) : nil
    return Loaded(reference: H3VideoReference(rgb8: bytes,
      frameCount: selected, width: side, height: side,
      audio: soundtrack),
      decodedFrames: decodedFrames, lastFrame: lastFrame)
  }

  private static func seamFrame(path: String, ffmpeg: URL,
    asset: AVURLAsset, frame: Int) throws -> H3StillReference {
    guard let track = asset.tracks(withMediaType: .video).first else {
      throw H3CheckpointError.invalid("H3 extension source has no video track.")
    }
    // FFmpeg applies the display transform before scaling. Use the same display
    // geometry to choose the bounded grid, including quarter-turn rotation.
    let display = CGRect(origin: .zero, size: track.naturalSize)
      .applying(track.preferredTransform)
    let sourceWidth = abs(display.width), sourceHeight = abs(display.height)
    guard sourceWidth.isFinite, sourceHeight.isFinite,
      (32...20_000).contains(sourceWidth), (32...20_000).contains(sourceHeight),
      sourceWidth * sourceHeight <= 100_000_000,
      (0.25...4).contains(sourceWidth / sourceHeight) else {
      throw H3CheckpointError.invalid("H3 extension seam needs bounded source display geometry.")
    }
    let aspect = sourceWidth / sourceHeight
    var width = 0, height = 0, bestError = Double.infinity, bestArea = 0
    for h in stride(from: 32, through: min(256, Int(sourceHeight)), by: 32) {
      for w in stride(from: 32, through: min(256, Int(sourceWidth)), by: 32) {
        let area = w * h
        guard area >= 4096 else { continue }
        let error = abs(log((Double(w) / Double(h)) / aspect))
        if error < bestError - 1e-12 ||
          (abs(error - bestError) <= 1e-12 && area > bestArea) {
          width = w; height = h; bestError = error; bestArea = area
        }
      }
    }
    try H3StillReferenceMedia.validateCanvas(width: width, height: height)
    let expected = width * height * 3
    let process = Process(); process.executableURL = ffmpeg
    // Select on the exact same 24 fps grid as the reference decode, before its
    // temporal clone padding. Only a single bounded RGB frame reaches the host.
    process.arguments = ["-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
      "-vf", "fps=24,select=eq(n\\,\(frame)),scale=\(width):\(height)",
      "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    let output = Pipe(); process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning {
        process.terminate(); usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    var pixels = Data()
    while let part = try output.fileHandleForReading.read(upToCount: 64 * 1024), !part.isEmpty {
      try Task.checkCancellation()
      guard pixels.count + part.count <= expected else {
        throw H3CheckpointError.invalid("H3 extension seam exceeds its single-frame bound.")
      }
      pixels.append(part)
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10_000) }
    guard process.terminationStatus == 0, pixels.count == expected else {
      throw H3CheckpointError.invalid("Cannot decode the true final H3 extension source frame.")
    }
    try Task.checkCancellation()
    return H3StillReference(rgb8: pixels, width: width, height: height)
  }

}
