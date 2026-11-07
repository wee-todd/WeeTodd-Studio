@preconcurrency import AVFoundation
import Foundation
import MLX
import XCTest
@testable import RelayVSR

final class RelayMovieTimingTests: XCTestCase {
  func testFractionalVariableTimestampsAndExplicitTailSurviveEncoding() async throws {
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent("relay-timing-\(UUID().uuidString).mov")
    defer { try? FileManager.default.removeItem(at: file) }
    let geometry = try RelayGeometry(width: 32, height: 32)
    let scale: CMTimeScale = 24_000
    let writer = try RelayMovieWriter(url: file, geometry: geometry, transform: .identity,
      timeScale: scale)
    defer { writer.cancelIfWriting() }
    let image = MLXArray.zeros([64, 64, 3])
    let start: [Int64] = [0, 1001, 3003]
    let duration: [Int64] = [1001, 2002, 3003]
    for index in start.indices {
      try writer.append(image, at: CMTime(value: start[index], timescale: scale),
        duration: CMTime(value: duration[index], timescale: scale))
    }
    let end = CMTime(value: 6006, timescale: scale)
    try await writer.finish(at: end)

    let asset = AVURLAsset(url: file, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    let tracks = try await asset.loadTracks(withMediaType: .video)
    let track = try XCTUnwrap(tracks.first)
    let trackRange = try await track.load(.timeRange)
    XCTAssertEqual(trackRange.end, end)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    XCTAssertTrue(reader.startReading())
    var encoded: [CMTime] = []
    while let sample = output.copyNextSampleBuffer() { encoded.append(sample.presentationTimeStamp) }
    XCTAssertEqual(encoded, start.map { CMTime(value: $0, timescale: scale) })
    XCTAssertEqual(reader.status, .completed)
  }

  func testMuxKeepsAudioThatLeadsVideo() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("relay-mux-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let video = directory.appendingPathComponent("video.mov")
    let audio = directory.appendingPathComponent("audio.caf")
    let output = directory.appendingPathComponent("output.mov")
    let geometry = try RelayGeometry(width: 32, height: 32)
    let writer = try RelayMovieWriter(url: video, geometry: geometry, transform: .identity,
      timeScale: 24_000)
    defer { writer.cancelIfWriting() }
    let frameDuration = CMTime(value: 1001, timescale: 24_000)
    try writer.append(MLXArray.zeros([64, 64, 3]), at: .zero, duration: frameDuration)
    try await writer.finish(at: frameDuration)
    let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1))
    let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 9_600))
    buffer.frameLength = buffer.frameCapacity
    buffer.floatChannelData?[0].update(repeating: 0.125, count: Int(buffer.frameLength))
    do {
      let file = try AVAudioFile(forWriting: audio, settings: format.settings)
      try file.write(from: buffer)
    }
    let sourceAudio = AVURLAsset(url: audio)
    let audioTracks = try await sourceAudio.loadTracks(withMediaType: .audio)
    try await RelayMovie.mux(video, audio: audioTracks, sourceStart: frameDuration,
      previewEnd: nil, output: output)
    let result = AVURLAsset(url: output, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    let videoTracks = try await result.loadTracks(withMediaType: .video)
    let resultAudio = try await result.loadTracks(withMediaType: .audio)
    let audioRange = try await XCTUnwrap(resultAudio.first).load(.timeRange)
    let track = try XCTUnwrap(videoTracks.first)
    let trackRange = try await track.load(.timeRange)
    let content = try await RelayMovie.videoContentRange(track, fallback: trackRange)
    XCTAssertEqual(content.start, frameDuration)
    XCTAssertEqual(content.duration, frameDuration)
    let reader = try AVAssetReader(asset: result)
    reader.timeRange = content
    let frames = AVAssetReaderTrackOutput(track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(frames)
    XCTAssertTrue(reader.startReading())
    let first = try XCTUnwrap(frames.copyNextSampleBuffer())
    XCTAssertEqual(first.presentationTimeStamp, frameDuration)
    XCTAssertNil(frames.copyNextSampleBuffer())
    XCTAssertEqual(audioRange.start, .zero)
    XCTAssertEqual(audioRange.duration, CMTime(value: 1, timescale: 5))
  }
}
