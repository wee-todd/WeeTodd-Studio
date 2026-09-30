import AVFoundation
import CryptoKit
import Darwin
import Foundation

/// The visible tail of an accepted take, or an explicitly chosen extension movie.
/// Preparation publishes a short CFR audiovisual excerpt so the worker never scans
/// or hashes a long source movie during weighted inference.
struct NativeLTXMovieSource {
  let url: URL
  let start: Double
  let duration: Double?
  let sourceClipID: UUID?
  let takeID: UUID?
  let signature: [Int64]

  init(project: StudioProject, clip: Clip) throws {
    let source: Clip?
    if clip.continuityMode == "motion" {
      source = try project.continuitySource(for: clip)
    } else if let id = clip.extensionClipID {
      source = project.earlierContinuitySources(for: clip).first { $0.id == id }
      guard source != nil else { throw StudioError.invalid("The extension source clip is no longer earlier in the timeline.") }
    } else {
      source = nil
    }
    let path = source?.sourcePath ?? clip.extensionSource
    guard !path.isEmpty, path.hasPrefix("/"), !path.utf8.contains(0) else {
      throw StudioError.invalid("Select an accepted movie to extend.")
    }
    url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    signature = try Self.signature(url)
    if let source {
      guard source.sourceIn.isFinite, source.sourceIn >= 0,
        source.duration.isFinite, source.duration > 0,
        (source.sourceIn + source.duration).isFinite else {
        throw StudioError.invalid("The source clip needs a finite visible interval.")
      }
      start = source.sourceIn
      duration = source.duration
      sourceClipID = source.id
      takeID = source.activeRenderVersion?.id
    } else {
      start = 0
      duration = nil
      sourceClipID = nil
      takeID = nil
    }
  }

  private static func signature(_ url: URL) throws -> [Int64] {
    var status = stat()
    guard url.path.withCString({ Darwin.lstat($0, &status) }) == 0,
      status.st_mode & S_IFMT == S_IFREG, status.st_size > 0,
      FileManager.default.isReadableFile(atPath: url.path) else {
      throw StudioError.invalid("Render and accept the source movie, or relink it.")
    }
    return [Int64(status.st_dev), Int64(bitPattern: UInt64(status.st_ino)), status.st_size,
      Int64(status.st_mtimespec.tv_sec), Int64(status.st_mtimespec.tv_nsec),
      Int64(status.st_ctimespec.tv_sec), Int64(status.st_ctimespec.tv_nsec)]
  }

  func verify() throws {
    guard try Self.signature(url) == signature else {
      throw StudioError.invalid("The extension source changed. Prepare the clip again.")
    }
  }

  var report: [String: Any] {
    var result: [String: Any] = ["version": 1, "mode": sourceClipID == nil ? "extension" : "motion",
      "engine": "ltx25", "sourcePath": url.path, "sourceIn": start,
      "sourceStat": signature]
    if let duration { result["duration"] = duration }
    if let sourceClipID { result["sourceClipID"] = sourceClipID.uuidString }
    if let takeID { result["sourceTakeID"] = takeID.uuidString }
    return result
  }

  /// FFmpeg receives bounded trim times and writes exactly contextFrames video
  /// frames plus their matching stereo audio; no whole-source frame array exists.
  func extract(to destination: URL, ffmpeg: URL, fps: Double,
    width: Int, height: Int, contextFrames: Int = 25) async throws -> String {
    try verify()
    try Task.checkCancellation()
    guard ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      fps.isFinite, fps > 0, width > 0, height > 0, contextFrames > 0,
      !FileManager.default.fileExists(atPath: destination.path) else {
      throw StudioError.invalid("The extension media destination or FFmpeg executable is invalid.")
    }
    let asset = AVURLAsset(url: url)
    let mediaDuration = try await asset.load(.duration).seconds
    let tracks = try await asset.loadTracks(withMediaType: .video)
    let audio = try await asset.loadTracks(withMediaType: .audio)
    guard tracks.count == 1, audio.count == 1, mediaDuration.isFinite, mediaDuration > 0 else {
      throw StudioError.invalid("LTX motion extension needs one video track and one embedded audio track.")
    }
    let end = duration.map { start + $0 } ?? mediaDuration
    let contextDuration = Double(contextFrames) / fps
    guard end.isFinite, end <= mediaDuration + 0.000001,
      end - start + 0.000001 >= contextDuration else {
      throw StudioError.invalid("The visible extension source needs at least (contextFrames) frames at the selected rate.")
    }
    let begin = end - contextDuration
    // Accurate input seeking decodes only the nearby GOP rather than every
    // preceding frame of a long accepted take.
    let videoFilter = "setpts=PTS-STARTPTS,fps=\(String(format: "%.12f", fps))," +
      "tpad=stop_mode=clone:stop_duration=1,trim=end_frame=\(contextFrames)," +
      "scale=\(width):\(height):force_original_aspect_ratio=increase,crop=\(width):\(height)"
    let audioFilter = "asetpts=PTS-STARTPTS,aresample=48000,apad," +
      "atrim=end=\(String(format: "%.12f", contextDuration))"
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-nostdin", "-ss", String(format: "%.12f", begin),
      "-i", url.path,
      "-filter_complex", "[0:v:0]\(videoFilter)[v];[0:a:0]\(audioFilter)[a]",
      "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "10",
      "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
      "-ac", "2", "-frames:v", String(contextFrames), "-y", destination.path]
    let log = destination.deletingPathExtension().appendingPathExtension("ffmpeg.log")
    guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
      throw StudioError.invalid("Cannot create extension preparation log.")
    }
    let handle = try FileHandle(forWritingTo: log)
    defer { try? handle.close() }
    process.standardOutput = handle
    process.standardError = handle
    try process.run()
    defer {
      if process.isRunning { process.terminate() }
      process.waitUntilExit()
    }
    while process.isRunning {
      try Task.checkCancellation()
      usleep(20_000)
    }
    guard process.terminationStatus == 0 else {
      throw StudioError.invalid("Cannot extract the visible audiovisual source tail for LTX extension.")
    }
    try verify()
    let fd = Darwin.open(destination.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    guard fd >= 0 else { throw StudioError.invalid("Cannot read the prepared extension source.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var digest = SHA256()
    while let chunk = try file.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
      try Task.checkCancellation()
      digest.update(data: chunk)
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
