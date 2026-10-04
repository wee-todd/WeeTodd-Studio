import AVFoundation
import Darwin
import Foundation
import InferenceContracts
import LTX25Engine

/// Model-free source PCM extraction. Publication retains source sample rate/count;
/// the independently resampled 16k file is solely audio-VAE conditioning.
public struct MLXMovieSourceAudio: Sendable {
  public struct Prepared: Sendable {
    public let publication: URL, conditioning: URL
    public let contract: MLXMovieSourceAudioContract
  }
  public let source: URL?
  public let sourceStartSeconds: Double, sourceDurationSeconds: Double?
  public let frames: Int, fps: Double, maximumDriftSeconds: Double
  private let identity: NativeMediaSource?
  public init(source: URL?, sha256: String? = nil, sourceStartSeconds: Double = 0,
    sourceDurationSeconds: Double? = nil, frames: Int, fps: Double,
    maximumDriftSeconds: Double = 0.05) throws {
    _ = try MLXMovieSourceAudioContract(videoFrames: frames, fps: fps,
      maximumDriftSeconds: maximumDriftSeconds)
    guard sourceStartSeconds.isFinite, sourceStartSeconds >= 0,
      sourceDurationSeconds.map({ $0.isFinite && $0 > 0 }) ?? true,
      source == nil ? (sha256 == nil && sourceStartSeconds == 0 && sourceDurationSeconds == nil)
        : (source!.isFileURL && source!.path.hasPrefix("/") && source!.path.utf8.count <= 4096 && !source!.path.utf8.contains(0) && sha256 != nil) else {
      throw LTXError.invalid("Movie audio source needs a frozen local path/hash and valid interval, or explicit silence.")
    }
    self.source = source; self.sourceStartSeconds = sourceStartSeconds
    self.sourceDurationSeconds = sourceDurationSeconds; self.frames = frames; self.fps = fps
    self.maximumDriftSeconds = maximumDriftSeconds
    identity = try source.map { try NativeMediaSource(path: $0.path, sha256: sha256!) }
  }
  private static func run(ffmpeg: URL, arguments: [String], log: URL) throws {
    guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
      throw LTXError.invalid("Cannot create movie audio preparation log.")
    }
    let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
    let process = Process(); process.executableURL = ffmpeg; process.arguments = arguments
    process.standardOutput = handle; process.standardError = handle
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning { process.terminate(); usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
      process.waitUntilExit()
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10_000) }
    guard process.terminationStatus == 0 else { throw LTXError.invalid("FFmpeg could not prepare source movie PCM.") }
  }
  static func inspectPCM(_ path: URL) throws -> (rate: Int, samples: Int, channels: Int) {
    let file = try AVAudioFile(forReading: path, commonFormat: .pcmFormatFloat32, interleaved: false)
    let rate = file.fileFormat.sampleRate
    guard rate.isFinite, rate > 0, rate < Double(Int.max), rate.rounded() == rate,
      file.length > 0, file.length < Int64.max, (1...2).contains(file.processingFormat.channelCount),
      let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 16384) else {
      throw LTXError.invalid("Movie PCM has invalid sample rate/count/channels.")
    }
    while file.framePosition < file.length {
      try Task.checkCancellation(); try file.read(into: buffer)
      guard buffer.frameLength > 0, let channels = buffer.floatChannelData else {
        throw LTXError.invalid("Movie PCM ended before its declared sample count.")
      }
      for channel in 0..<Int(buffer.format.channelCount) {
        for index in 0..<Int(buffer.frameLength) where !channels[channel][index].isFinite {
          throw LTXError.invalid("Movie source PCM contains nonfinite samples.")
        }
      }
    }
    return (Int(rate), Int(file.length), Int(file.processingFormat.channelCount))
  }
  public func prepare(ffmpeg: URL, directory: URL,
    progress: @Sendable (String) throws -> Void = { _ in }) async throws -> Prepared {
    guard ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      directory.isFileURL, directory.path.hasPrefix("/"), !directory.path.utf8.contains(0),
      !FileManager.default.fileExists(atPath: directory.path) else {
      throw LTXError.invalid("Movie audio needs executable FFmpeg and a new local preparation directory.")
    }
    try identity?.verify(); try Task.checkCancellation()
    let seconds = Double(frames) / fps
    var hasAudio = false, originalChannels = 2, originalRate = 48000
    if let source {
      let asset = AVURLAsset(url: source)
      if let track = try await asset.loadTracks(withMediaType: .audio).first {
        let descriptions = try await track.load(.formatDescriptions)
        guard let format = descriptions.first,
          let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format),
          (1...2).contains(asbd.pointee.mChannelsPerFrame) else {
          throw LTXError.invalid("Movie source audio must be mono or stereo.")
        }
        let interval = try await track.load(.timeRange)
        let available = interval.end.seconds - sourceStartSeconds
        let selected = sourceDurationSeconds ?? available
        guard interval.start.seconds.isFinite, interval.end.seconds.isFinite,
          sourceStartSeconds >= interval.start.seconds - 0.000001, selected > 0,
          selected <= available + 0.000001, abs(selected - seconds) <= maximumDriftSeconds + 0.001 else {
          throw LTXError.invalid("Movie source audio interval does not match the visible video duration; no source PCM repair is allowed.")
        }
        let rate = asbd.pointee.mSampleRate
        guard rate.isFinite, rate > 0, rate < Double(Int.max), rate.rounded() == rate else {
          throw LTXError.invalid("Movie source audio sample rate is invalid.")
        }
        originalChannels = Int(asbd.pointee.mChannelsPerFrame); originalRate = Int(rate)
        hasAudio = true
      }
    }
    let parent = directory.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".movie-audio-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: staging) }
    let publication = staging.appendingPathComponent("publication.wav")
    let conditioning = staging.appendingPathComponent("conditioning-16k.wav")
    if hasAudio, let source {
      let length = sourceDurationSeconds.map { ":duration=\($0)" } ?? ""
      try Self.run(ffmpeg: ffmpeg, arguments: ["-v", "error", "-nostdin", "-n", "-i", source.path,
        "-map", "0:a:0", "-af", "atrim=start=\(sourceStartSeconds)\(length),asetpts=PTS-STARTPTS" +
          (originalChannels == 1 ? ",pan=stereo|c0=c0|c1=c0" : ""),
        "-ac", "2", "-c:a", "pcm_f32le", publication.path], log: staging.appendingPathComponent("publication.log"))
    } else {
      let silence = try MLXMovieSourceAudioContract(videoFrames: frames, fps: fps)
      try Self.run(ffmpeg: ffmpeg, arguments: ["-v", "error", "-nostdin", "-n", "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo", "-af", "atrim=end_sample=\(silence.publicationSamples)",
        "-c:a", "pcm_f32le", publication.path], log: staging.appendingPathComponent("publication.log"))
    }
    let pcm = try Self.inspectPCM(publication)
    let contract = hasAudio ? try MLXMovieSourceAudioContract(videoFrames: frames, fps: fps,
      sampleRate: originalRate, samples: pcm.samples, channels: originalChannels, maximumDriftSeconds: maximumDriftSeconds)
      : try MLXMovieSourceAudioContract(videoFrames: frames, fps: fps)
    guard pcm.channels == 2, pcm.rate == contract.publicationSampleRate,
      pcm.samples == contract.publicationSamples else { throw LTXError.invalid("Movie publication PCM changed its admitted sample contract.") }
    try progress("movie_audio_publication_ready"); try Task.checkCancellation()
    // Match Python max_duration cropping for model context only. Publication is untouched.
    try Self.run(ffmpeg: ffmpeg, arguments: ["-v", "error", "-nostdin", "-n", "-i", publication.path,
      "-af", "atrim=duration=\(seconds),aresample=16000", "-ac", "2", "-c:a", "pcm_f32le", conditioning.path],
      log: staging.appendingPathComponent("conditioning.log"))
    let modelPCM = try Self.inspectPCM(conditioning)
    guard modelPCM.rate == 16000, modelPCM.channels == 2 else { throw LTXError.invalid("Movie model context must be stereo16k PCM.") }
    try progress("movie_audio_conditioning_ready"); try identity?.verify(); try Task.checkCancellation()
    try FileManager.default.moveItem(at: staging, to: directory)
    return Prepared(publication: directory.appendingPathComponent("publication.wav"),
      conditioning: directory.appendingPathComponent("conditioning-16k.wav"), contract: contract)
  }
}
