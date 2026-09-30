import AVFoundation
import Darwin
import Foundation
import InferenceContracts
import LTX25Engine

/// Prepares only the exact decoded audiovisual tail used by an LTX extension.
/// Frame inspection precedes any weighted model load; raw RGB stays on disk.
public struct MLXSourceMovieInterval: Sendable {
  public struct Prepared: Sendable {
    public let rgb24: URL
    public let lowRGB24: URL
    public let audio16k: URL
    public let sourceFrames: Int
    public let sourceRange: Range<Int>
  }

  public let source: URL
  public let window: LTX25ExtensionWindow
  private let identity: NativeMediaSource

  public init(source: URL, sha256: String, window: LTX25ExtensionWindow) throws {
    guard source.isFileURL else {
      throw LTXError.invalid("LTX extension source must be a local movie file.")
    }
    self.source = source
    self.window = window
    identity = try NativeMediaSource(path: source.path, sha256: sha256)
  }

  public func validateSource() throws { try identity.verify() }

  private func inspect() async throws -> (Int, Range<Int>) {
    try validateSource()
    let asset = AVURLAsset(url: source)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    guard videoTracks.count == 1, audioTracks.count == 1 else {
      throw LTXError.invalid("LTX extension source needs one video track and one embedded audio track.")
    }
    let video = videoTracks[0], audio = audioTracks[0]
    let nominalRate = try await video.load(.nominalFrameRate)
    guard nominalRate.isFinite, abs(Double(nominalRate) - window.geometry.fps) < 0.001 else {
      throw LTXError.invalid("LTX extension source frame rate must match the selected output rate.")
    }
    let descriptions = try await audio.load(.formatDescriptions)
    guard let format = descriptions.first,
      let description = CMAudioFormatDescriptionGetStreamBasicDescription(format),
      (1...2).contains(description.pointee.mChannelsPerFrame) else {
      throw LTXError.invalid("LTX extension source audio must be mono or stereo.")
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: video, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    guard reader.canAdd(output) else {
      throw LTXError.invalid("Cannot inspect LTX extension source frame timing.")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw LTXError.invalid("Cannot read LTX extension source frame timing.")
    }
    // Read one decoded sample at a time. The reader retains no whole movie;
    // sorting bounded timestamps tolerates source B-frame decode order.
    var times: [Double] = []
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
      guard time.isFinite else { throw LTXError.invalid("LTX extension source has nonfinite frame timing.") }
      times.append(time)
      guard times.count <= 4097 else { throw LTXError.invalid("LTX extension source exceeds the frame inspection limit.") }
    }
    guard reader.status == .completed else {
      throw LTXError.invalid("LTX extension source frame inspection did not complete.")
    }
    times.sort()
    guard let firstTime = times.first,
      times.enumerated().allSatisfy({ index, time in
        abs((time - firstTime) * window.geometry.fps - Double(index)) < 0.05
      }) else {
      throw LTXError.invalid("LTX extension source has variable frame timing. Use a constant-rate take.")
    }
    let count = times.count
    let range = try window.sourceRange(sourceFrames: count)
    let audioRange = try await audio.load(.timeRange)
    guard audioRange.start.seconds <= Double(range.lowerBound) / window.geometry.fps + 0.01,
      audioRange.end.seconds + 0.01 >= Double(range.upperBound) / window.geometry.fps else {
      throw LTXError.invalid("LTX extension source audio does not cover the exact video context.")
    }
    try validateSource()
    return (count, range)
  }

  public func preflight() async throws -> (sourceFrames:Int,sourceRange:Range<Int>) {
    let (frames,range)=try await inspect()
    return (frames,range)
  }

  private static func run(_ ffmpeg: URL, args: [String], log: URL) throws {
    guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
      throw LTXError.invalid("Cannot create LTX extension media preparation log.")
    }
    let handle = try FileHandle(forWritingTo: log)
    defer { try? handle.close() }
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = args
    process.standardOutput = handle
    process.standardError = handle
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
    while process.isRunning { try Task.checkCancellation(); usleep(10_000) }
    guard process.terminationStatus == 0 else {
      throw LTXError.invalid("FFmpeg could not prepare the exact LTX extension source tail.")
    }
  }

  public func prepare(ffmpeg: URL, directory: URL) async throws -> Prepared {
    guard ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      directory.isFileURL, !FileManager.default.fileExists(atPath: directory.path) else {
      throw LTXError.invalid("LTX extension needs executable FFmpeg and a new local preparation directory.")
    }
    let (sourceFrames, range) = try await inspect()
    let parent = directory.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".ltx-extension-source-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: staging) }
    let rgb = staging.appendingPathComponent("context.rgb24")
    let lowRGB = staging.appendingPathComponent("context-low.rgb24")
    let audio = staging.appendingPathComponent("context-16k.wav")
    let width = window.geometry.width, height = window.geometry.height
    let fps = window.geometry.fps
    let videoFilter = "trim=start_frame=\(range.lowerBound):end_frame=\(range.upperBound)," +
      "setpts=PTS-STARTPTS,scale=\(width):\(height):force_original_aspect_ratio=increase," +
      "crop=\(width):\(height),format=rgb24"
    try Self.run(ffmpeg, args: ["-v", "error", "-nostdin", "-n", "-i", source.path,
      "-map", "0:v:0", "-vf", videoFilter, "-vsync", "0", "-frames:v",
      String(window.contextFrames), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", rgb.path],
      log: staging.appendingPathComponent("video.log"))
    let expectedVideoBytes = window.contextFrames * width * height * 3
    guard (try FileManager.default.attributesOfItem(atPath: rgb.path)[.size] as? Int) == expectedVideoBytes else {
      throw LTXError.invalid("LTX extension video tail differs from the admitted frame count.")
    }
    let lowWidth=width/2,lowHeight=height/2
    let lowFilter = "trim=start_frame=\(range.lowerBound):end_frame=\(range.upperBound)," +
      "setpts=PTS-STARTPTS,scale=\(lowWidth):\(lowHeight):force_original_aspect_ratio=increase," +
      "crop=\(lowWidth):\(lowHeight),format=rgb24"
    try Self.run(ffmpeg, args: ["-v", "error", "-nostdin", "-n", "-i", source.path,
      "-map", "0:v:0", "-vf", lowFilter, "-vsync", "0", "-frames:v",
      String(window.contextFrames), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", lowRGB.path],
      log: staging.appendingPathComponent("video-low.log"))
    guard (try FileManager.default.attributesOfItem(atPath: lowRGB.path)[.size] as? Int) ==
      window.contextFrames * lowWidth * lowHeight * 3 else {
      throw LTXError.invalid("LTX extension low-resolution video tail differs from the admitted frame count.")
    }
    let audioSamples = Int((Double(window.contextFrames) / fps * 16000).rounded(.toNearestOrEven))
    let audioFilter = "atrim=start=\(Double(range.lowerBound) / fps):end=\(Double(range.upperBound) / fps)," +
      "asetpts=PTS-STARTPTS,aresample=16000,apad=whole_len=\(audioSamples)," +
      "atrim=end_sample=\(audioSamples)"
    try Self.run(ffmpeg, args: ["-v", "error", "-nostdin", "-n", "-i", source.path,
      "-map", "0:a:0", "-af", audioFilter, "-ac", "2", "-c:a", "pcm_f32le", audio.path],
      log: staging.appendingPathComponent("audio.log"))
    let preparedAudio = try AVAudioFile(forReading: audio)
    guard preparedAudio.length == audioSamples,
      preparedAudio.fileFormat.sampleRate == 16000,
      preparedAudio.fileFormat.channelCount == 2 else {
      throw LTXError.invalid("LTX extension audio tail differs from the admitted sample count.")
    }
    try Task.checkCancellation()
    try validateSource()
    try FileManager.default.moveItem(at: staging, to: directory)
    return Prepared(rgb24: directory.appendingPathComponent("context.rgb24"),
      lowRGB24: directory.appendingPathComponent("context-low.rgb24"),
      audio16k: directory.appendingPathComponent("context-16k.wav"),
      sourceFrames: sourceFrames, sourceRange: range)
  }
}
