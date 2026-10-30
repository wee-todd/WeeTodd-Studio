import Foundation
import InferenceMedia
import LTX25Engine

/// Concatenates verified decoded frames and rebuilds their source frame clock.
/// FFV1 is lossless; one global RGB hash independently checks chunk ordering.
enum MLXMovieLosslessPublication {
  static func concatenate(ffmpeg: URL, chunks: [URL], output: URL, width: Int, height: Int,
    frames: Int, fps: Double) throws -> LosslessVideoVerifier.Proof {
    guard !chunks.isEmpty, chunks.count <= 20408, !FileManager.default.fileExists(atPath: output.path) else {
      throw LTXError.invalid("Lossless movie publication needs completed chunks and a fresh output.")
    }
    let fm = FileManager.default, root = output.deletingLastPathComponent().appendingPathComponent(".lossless-join-" + UUID().uuidString)
    try fm.createDirectory(at: root, withIntermediateDirectories: false); defer { try? fm.removeItem(at: root) }
    var names: [String] = []
    for (index, chunk) in chunks.enumerated() {
      try Task.checkCancellation()
      let name = String(format: "part-%06d.mkv", index)
      try fm.linkItem(at: chunk, to: root.appendingPathComponent(name)); names.append(name)
    }
    let list = root.appendingPathComponent("chunks.ffconcat"), hash = root.appendingPathComponent("rgb.sha256")
    try Data((names.map { "file '" + $0 + "'" }.joined(separator: "\n") + "\n").utf8).write(to: list, options: .withoutOverwriting)
    // Never accumulate Matroska's millisecond chunk-duration rounding. The
    // source frame ordinal determines PTS, independently of packet timestamps.
    try MLXMovieFiles.run(ffmpeg, ["-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1", "-i", list.path,
      "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "hash", "-hash", "sha256", hash.path,
      "-map", "0:v:0", "-an", "-vf", "settb=expr=1/(\(fps)),setpts=N", "-enc_time_base", "0", "-fps_mode", "passthrough",
      "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", output.path], log: root.appendingPathComponent("join.log"))
    let bytes = try MLXMovieFiles.metadata(hash, maximumBytes: 128)
    let text = String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    guard text.hasPrefix("SHA256="), text.count == 71 else { throw LTXError.invalid("Lossless join has no ordered RGB proof.") }
    return try LosslessVideoVerifier.verify(ffmpeg: ffmpeg, video: output, width: width, height: height,
      frames: frames, fps: fps, expectedRGBSHA256: String(text.dropFirst(7)))
  }
}
