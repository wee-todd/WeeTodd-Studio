import Darwin
import Foundation
import InferenceContracts

/// Original-canvas media for Motion Fidelity. This never uses the square
/// analysis/reference canvas and never changes the source or its trim.
public enum H3MotionFidelityMedia {
  public struct Source: Sendable {
    public let identity: NativeMediaSource
    public let width: Int
    public let height: Int
    public let frames: Int
    public let startSeconds: Double
    public let hasAudio: Bool
  }
  public struct Decoded {
    public let rgb8: [UInt8]
    public let audio: H3AudioReference
  }

  public static func inspect(path: String, sha256: String, ffprobe: URL,
    startSeconds: Double, durationSeconds: Double,
    settings: H3MotionFidelitySettings) throws -> Source {
    try settings.validate()
    guard startSeconds.isFinite, startSeconds >= 0, startSeconds <= 86400,
      durationSeconds.isFinite, (2.5...Double(345) / 24).contains(durationSeconds),
      abs(startSeconds * 24 - (startSeconds * 24).rounded(.toNearestOrEven)) <= 0.001,
      abs(durationSeconds * 24 - (durationSeconds * 24).rounded(.toNearestOrEven)) <= 0.001 else {
      throw H3CheckpointError.invalid("Motion Fidelity trims must use native 24 fps frame boundaries.")
    }
    let source = try NativeMediaSource(path: path, sha256: sha256)
    try source.verify()
    let data = try capture(ffprobe, ["-v", "error", "-show_streams", "-show_format", "-of", "json", path], limit: 1024 * 1024)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let streams = object["streams"] as? [[String: Any]],
      let video = streams.first(where: { $0["codec_type"] as? String == "video" }),
      var width = video["width"] as? Int, var height = video["height"] as? Int,
      let rate = video["avg_frame_rate"] as? String,
      let durationString = (object["format"] as? [String: Any])?["duration"] as? String,
      let total = Double(durationString), total.isFinite,
      startSeconds + durationSeconds <= total + 0.025 else {
      throw H3CheckpointError.invalid("Motion Fidelity source does not cover its selected native interval.")
    }
    let fraction = rate.split(separator: "/").compactMap { Double($0) }
    guard fraction.count == 2, fraction[1] > 0, abs(fraction[0] / fraction[1] - 24) <= 0.0001 else {
      throw H3CheckpointError.invalid("Motion Fidelity requires native 24 fps source video.")
    }
    let sideData = video["side_data_list"] as? [[String: Any]] ?? []
    let rotation = sideData.compactMap { $0["rotation"] as? Double }.first ??
      ((video["tags"] as? [String: String])?["rotate"].flatMap(Double.init) ?? 0)
    guard rotation.isFinite, abs(rotation / 90 - (rotation / 90).rounded()) < 0.0001 else {
      throw H3CheckpointError.invalid("Motion Fidelity source rotation must be a right angle.")
    }
    if Int(abs(rotation / 90).rounded()).isMultiple(of: 2) == false { swap(&width, &height) }
    let frames = Int((durationSeconds * 24).rounded(.toNearestOrEven))
    let budget = settings.mode == .uniform ? try H3MotionFidelityPlan.alignedFrames(frames * settings.maxHold) : settings.maxFrames
    guard (60...345).contains(frames), (32...2048).contains(width), (32...2048).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      width * height <= H3Geometry.maximumCanvasPixels,
      budget <= settings.maxFrames, budget * width * height <= 160_000_000,
      budget * width * height * 3 <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Motion Fidelity exceeds the canvas, temporal or RGB memory budget.")
    }
    let timestamps = try capture(ffprobe, ["-v", "error", "-select_streams", "v:0", "-read_intervals",
      "\(startSeconds)%+\(durationSeconds + 1)", "-show_frames", "-show_entries",
      "frame=best_effort_timestamp_time", "-of", "json", path], limit: 1024 * 1024)
    guard let clock = try JSONSerialization.jsonObject(with: timestamps) as? [String: Any],
      let items = clock["frames"] as? [[String: Any]] else {
      throw H3CheckpointError.invalid("Motion Fidelity source timestamp inspection failed.")
    }
    let times = items.compactMap { ($0["best_effort_timestamp_time"] as? String).flatMap(Double.init) }
    // ffprobe seeks to a preceding keyframe; select the actual edited interval.
    let visible = times.filter { $0 >= startSeconds - 0.0001 && $0 < startSeconds + durationSeconds - 0.0001 }
    guard visible.count == frames, visible.allSatisfy(\.isFinite),
      abs(visible[0] - startSeconds) <= 0.0001,
      zip(visible, visible.dropFirst()).allSatisfy({ abs($0.1 - $0.0 - 1.0 / 24) <= 0.0001 }) else {
      throw H3CheckpointError.invalid("Motion Fidelity requires constant 24 fps timestamps without gaps.")
    }
    try source.verify()
    return Source(identity: source, width: width, height: height, frames: frames,
      startSeconds: startSeconds, hasAudio: streams.contains { $0["codec_type"] as? String == "audio" })
  }

  public static func decode(_ source: Source, ffmpeg: URL) throws -> Decoded {
    try source.identity.verify()
    let count = source.frames * source.width * source.height * 3
    let pixels = try capture(ffmpeg, ["-v", "error", "-nostdin", "-i", source.identity.path,
      "-ss", String(source.startSeconds), "-map", "0:v:0", "-an", "-frames:v", String(source.frames),
      "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], limit: count)
    guard pixels.count == count else { throw H3CheckpointError.invalid("Motion source RGB frame count changed.") }
    let samples = H3MotionFidelityPlan.audioSampleBoundary(frame: source.frames)
    let audio: H3AudioReference
    if source.hasAudio {
      // Input seeking selects the interval before the exact sample-count filter.
      // Output -ss would discard samples after atrim and shorten nonzero trims.
      let bytes = try capture(ffmpeg, ["-v", "error", "-nostdin", "-ss", String(source.startSeconds),
        "-i", source.identity.path, "-map", "0:a:0", "-af", "aresample=32000,apad,atrim=end_sample=\(samples)",
        "-ac", "2", "-ar", "32000", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], limit: samples * 8)
      audio = try pcm(bytes, frames: samples)
    } else { audio = H3AudioReference(samples: [Float](repeating: 0, count: samples * 2), frames: samples) }
    try source.identity.verify()
    return Decoded(rgb8: Array(pixels), audio: audio)
  }

  /// The caller supplies a fresh scratch folder inside its atomic output stage.
  /// Only pitch-preserved analysis audio is expanded; publication uses original audio.
  public static func expandedAudio(original: H3AudioReference,
    plan: H3MotionFidelityPlan, ffmpeg: URL, scratch: URL) throws -> H3AudioReference {
    guard original.frames == H3MotionFidelityPlan.audioSampleBoundary(frame: plan.sourceFrames),
      original.samples.count == original.frames * 2, original.samples.allSatisfy(\.isFinite),
      !FileManager.default.fileExists(atPath: scratch.path) else {
      throw H3CheckpointError.invalid("Motion audio expansion requires a fresh stage and exact source interval.")
    }
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: scratch) }
    let input = scratch.appendingPathComponent("source.f32")
    guard FileManager.default.createFile(atPath: input.path, contents: nil) else {
      throw H3CheckpointError.invalid("Cannot create bounded motion PCM stage.")
    }
    let handle = try FileHandle(forWritingTo: input); defer { try? handle.close() }
    var bytes = Data(); bytes.reserveCapacity(8192 * 8)
    for frame in 0..<original.frames {
      if frame.isMultiple(of: 8192) { try Task.checkCancellation() }
      for value in [original.samples[frame], original.samples[original.frames + frame]] {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
      }
      if bytes.count >= 8192 * 8 { try handle.write(contentsOf: bytes); bytes.removeAll(keepingCapacity: true) }
    }
    try handle.write(contentsOf: bytes); try handle.close()
    let samples = H3MotionFidelityPlan.audioSampleBoundary(frame: plan.paddedFrames)
    let result = try capture(ffmpeg, ["-v", "error", "-nostdin", "-f", "f32le", "-ar", "32000", "-ac", "2",
      "-i", input.path, "-filter_complex", plan.audioFilter, "-map", "[out]", "-ac", "2", "-ar", "32000",
      "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], limit: samples * 8)
    return try pcm(result, frames: samples)
  }

  static func pcm(_ data: Data, frames: Int) throws -> H3AudioReference {
    guard frames > 0, data.count == frames * 8 else { throw H3CheckpointError.invalid("Motion PCM length changed.") }
    var result = [Float](repeating: 0, count: frames * 2)
    data.withUnsafeBytes { raw in
      for frame in 0..<frames { for channel in 0..<2 {
        result[channel * frames + frame] = Float(bitPattern: UInt32(littleEndian:
          raw.loadUnaligned(fromByteOffset: frame * 8 + channel * 4, as: UInt32.self)))
      } }
    }
    guard result.allSatisfy(\.isFinite) else { throw H3CheckpointError.invalid("Motion PCM is non-finite.") }
    return H3AudioReference(samples: result, frames: frames)
  }
  private static func capture(_ executable: URL, _ arguments: [String], limit: Int) throws -> Data {
    guard executable.isFileURL, FileManager.default.isExecutableFile(atPath: executable.path),
      limit > 0, limit <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Motion media requires an executable native media utility and bounded output.")
    }
    let process = Process(), pipe = Pipe()
    process.executableURL = executable; process.arguments = arguments
    process.standardOutput = pipe; process.standardError = FileHandle.nullDevice
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning {
        process.terminate(); usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit(); try? pipe.fileHandleForReading.close()
    }
    var result = Data()
    while let chunk = try pipe.fileHandleForReading.read(upToCount: 1024 * 1024), !chunk.isEmpty {
      try Task.checkCancellation()
      guard result.count <= limit - chunk.count else { throw H3CheckpointError.invalid("Motion media exceeded its admitted output size.") }
      result.append(chunk)
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10_000) }
    guard process.terminationStatus == 0 else { throw H3CheckpointError.invalid("Native motion media preparation failed.") }
    return result
  }
}
