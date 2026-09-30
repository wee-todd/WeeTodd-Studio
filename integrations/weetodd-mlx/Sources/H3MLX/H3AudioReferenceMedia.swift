import Foundation

/// Bounded local decode into the H3 audio VAE's 32 kHz stereo contract.
/// Source identity is verified by the worker before and after this operation.
public enum H3AudioReferenceMedia {
  public static func load(path: String, ffmpeg: URL,
    maximumSeconds: Double = 15) throws -> H3AudioReference {
    guard path.hasPrefix("/"), !path.utf8.contains(0), ffmpeg.isFileURL,
      FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      maximumSeconds.isFinite, (0.2...15).contains(maximumSeconds) else {
      throw H3CheckpointError.invalid("H3 audio reference needs a local source and FFmpeg.")
    }
    let maximumFrames = min(480_000, Int(ceil(maximumSeconds * 32_000)))
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-nostdin", "-i", path,
      "-map", "0:a:0", "-t", String(maximumSeconds),
      "-ac", "2", "-ar", "32000",
      "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
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
    bytes.reserveCapacity(2 * 32_000 * 4)
    while let part = try output.fileHandleForReading.read(upToCount: 256 * 1024),
      !part.isEmpty {
      try Task.checkCancellation()
      guard bytes.count + part.count <= (maximumFrames + 1024) * 2 * 4 else {
        throw H3CheckpointError.invalid("H3 reference audio exceeds its 15-second limit.")
      }
      bytes.append(part)
    }
    while process.isRunning {
      try Task.checkCancellation()
      usleep(10_000)
    }
    guard process.terminationStatus == 0, bytes.count.isMultiple(of: 8),
      bytes.count >= 800 * 2 * 4 else {
      throw H3CheckpointError.invalid("H3 reference audio could not be decoded to 32 kHz stereo PCM.")
    }
    let frames = min(bytes.count / 8, maximumFrames)
    var samples = [Float](repeating: 0, count: frames * 2)
    bytes.withUnsafeBytes { raw in
      for frame in 0..<frames {
        samples[frame] = Float(bitPattern: UInt32(littleEndian:
          raw.loadUnaligned(fromByteOffset: frame * 8, as: UInt32.self)))
        samples[frames + frame] = Float(bitPattern: UInt32(littleEndian:
          raw.loadUnaligned(fromByteOffset: frame * 8 + 4, as: UInt32.self)))
      }
    }
    guard samples.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 reference audio contains non-finite samples.")
    }
    return H3AudioReference(samples: samples, frames: frames)
  }
}
