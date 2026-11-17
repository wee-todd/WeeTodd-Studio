import XCTest
import AVFoundation
import VideoToolbox
@testable import InferenceMedia

final class NativeUpscaleMovieTests: XCTestCase {
  func testWriterPinsHEVCQualityWithoutAnImplicitRateControlTarget() throws {
    let properties = NativeUpscaleMovieWriter.hevcCompressionProperties
    // Use the VT compressed-video contract; AV's still-image symbol aliases this key.
    let quality = try XCTUnwrap(properties[kVTCompressionPropertyKey_Quality as String] as? NSNumber)
    XCTAssertEqual(quality.floatValue, 0.9, accuracy: 0.000001)
    XCTAssertEqual(properties[AVVideoAllowFrameReorderingKey] as? Bool, false)
    for key in [kVTCompressionPropertyKey_AverageBitRate,
      kVTCompressionPropertyKey_ExpectedFrameRate, kVTCompressionPropertyKey_DataRateLimits,
      kVTCompressionPropertyKey_MinAllowedFrameQP, kVTCompressionPropertyKey_MaxAllowedFrameQP] {
      XCTAssertNil(properties[key as String], "Quality export must not impose a bitrate, nominal clock or QP clamp.")
    }
  }
  func testWriterPreservesEveryVariableTimestampAndExplicitTail() async throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = folder.appendingPathComponent("movie.mov")
    let writer = try NativeUpscaleMovieWriter(url: url, width: 64, height: 48, timeScale: 24000)
    let bytes = Array(repeating: UInt8(40), count: 64 * 48 * 3)
    let starts: [Int64] = [0, 1001, 3003], ends: [Int64] = [1001, 3003, 6006]
    for i in starts.indices {
      try writer.appendRGB(bytes, at: CMTime(value: starts[i], timescale: 24000),
        duration: CMTime(value: ends[i] - starts[i], timescale: 24000))
    }
    try await writer.finish(at: CMTime(value: 6006, timescale: 24000))
    let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    let tracks = try await asset.loadTracks(withMediaType: .video)
    let track = try XCTUnwrap(tracks.first)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    reader.add(output); XCTAssertTrue(reader.startReading())
    var samples: [CMSampleBuffer] = []
    while let sample = output.copyNextSampleBuffer() { if sample.numSamples > 0 { samples.append(sample) } }
    XCTAssertEqual(samples.count, 3)
    for i in starts.indices {
      XCTAssertEqual(CMTimeCompare(samples[i].presentationTimeStamp, CMTime(value: starts[i], timescale: 24000)), 0)
      XCTAssertEqual(CMTimeCompare(samples[i].duration, CMTime(value: ends[i] - starts[i], timescale: 24000)), 0)
    }
  }
  func testEncoderRejectsInvalidDimensionsBeforeStarting() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    XCTAssertThrowsError(try NativeUpscaleMovieWriter(url: url, width: 0, height: 32, timeScale: 24000))
    XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
  }
  func testPartialSDRColorTagsDoNotRejectAnOtherwiseSupportedRaster() async throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    defer { try? FileManager.default.removeItem(at: url) }
    let writer = try NativeUpscaleMovieWriter(url: url, width: 1024, height: 576, timeScale: 24000,
      colorProperties: [AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2])
    try writer.appendRGB(Array(repeating: 80, count: 1024 * 576 * 3), at: .zero, duration: CMTime(value: 1001, timescale: 24000))
    try await writer.finish(at: CMTime(value: 1001, timescale: 24000))
    XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
  }
  func testOddOutputRasterIsPreservedExactlyOrRejectedBeforeInference() async throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    defer { try? FileManager.default.removeItem(at: url) }
    let writer: NativeUpscaleMovieWriter
    do { writer = try NativeUpscaleMovieWriter(url: url, width: 97, height: 73, timeScale: 24000) }
    catch { return } // Encoder admission may reject unsupported exact dimensions.
    try writer.appendRGB(Array(repeating: 80, count: 97 * 73 * 3), at: .zero, duration: CMTime(value: 1001, timescale: 24000))
    try await writer.finish(at: CMTime(value: 1001, timescale: 24000))
    let tracks = try await AVURLAsset(url: url).loadTracks(withMediaType: .video)
    let track = try XCTUnwrap(tracks.first)
    let size = try await track.load(.naturalSize)
    XCTAssertEqual(size, CGSize(width: 97, height: 73))
  }
}
