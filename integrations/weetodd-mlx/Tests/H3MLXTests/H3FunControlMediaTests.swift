import Foundation
import CryptoKit
import MLX
import XCTest
@testable import H3MLX

final class H3FunControlMediaTests: XCTestCase {
  /// Opt-in encoder-only qualification. Both paths read identical pixels and
  /// weights; no Qwen, transformer, audio VAE or sampler is invoked.
  func testInstalledControlEncoderSequentialTermsMatchWholeVolumeWhenRequested() throws {
    let env = ProcessInfo.processInfo.environment
    guard let movie = env["H3_FUN_CONTROL_ENCODER_PROBE_GUIDE"],
      let checkpoint = env["H3_VIDEO_VAE_CHECKPOINT"],
      let destination = env["H3_FUN_CONTROL_ENCODER_PROBE_DIRECTORY"] else {
      throw XCTSkip("Set guide, installed video VAE and new output directory for encoder-only comparison.")
    }
    let directory = URL(fileURLWithPath: destination)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    let geometry = try H3Geometry(width: 384, height: 256, durationSeconds: 2.5)
    let guide = try H3FunControlMedia.load(path: movie,
      ffmpeg: URL(fileURLWithPath: env["WEETODD_FFMPEG"] ?? "/opt/homebrew/bin/ffmpeg"), geometry: geometry)
    var measurements: [[String: Any]] = []
    func encode(_ name: String, sequential: Bool) throws -> [Float] {
      Stream.gpu.synchronize(); Memory.clearCache(); Memory.peakMemory = Memory.activeMemory
      let start = Date()
      let values = try autoreleasepool {
        try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: URL(fileURLWithPath: checkpoint),
          rgb8: Array(guide.rgb8), frameCount: geometry.frames,
          width: geometry.width, height: geometry.height,
          releaseTemporalConvolutionTerms: sequential).asArray(Float.self)
      }
      Stream.gpu.synchronize(); Memory.clearCache()
      let bytes = values.withUnsafeBytes { Data($0) }
      try bytes.write(to: directory.appendingPathComponent(name + ".f32"), options: .withoutOverwriting)
      measurements.append(["path": name, "seconds": Date().timeIntervalSince(start),
        "peakMLXBytes": Memory.peakMemory, "activeMLXBytesAfter": Memory.activeMemory,
        "sha256": SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()])
      return values
    }
    let whole = try encode("whole-volume", sequential: false)
    let bounded = try encode("sequential-terms", sequential: true)
    XCTAssertEqual(bounded.count, whole.count)
    var squared: Double = 0
    var maximum: Float = 0
    for (a, b) in zip(whole, bounded) {
      XCTAssertTrue(a.isFinite && b.isFinite)
      maximum = Swift.max(maximum, abs(a - b))
      squared += Double(a - b) * Double(a - b)
    }
    let rmse = sqrt(squared / Double(whole.count))
    // Predeclared same-engine FP16 convolution tolerances; retained raw
    // outputs permit independent comparison rather than relying on hashes.
    XCTAssertLessThanOrEqual(maximum, 0.01)
    XCTAssertLessThanOrEqual(rmse, 0.001)
    let report: [String: Any] = ["geometry": [384, 256, 73], "guide": movie,
      "checkpoint": checkpoint, "measurements": measurements,
      "maximumAbsoluteDifference": maximum, "rmse": rmse,
      "maximumTolerance": 0.01, "rmseTolerance": 0.001]
    let bytes = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try bytes.write(to: directory.appendingPathComponent("encoder-comparison.json"), options: .withoutOverwriting)
    print(String(decoding: bytes, as: UTF8.self))
  }

  func testInstalledControlEncoderCancellationReleasesStageWhenRequested() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let movie = env["H3_FUN_CONTROL_ENCODER_PROBE_GUIDE"],
      let checkpoint = env["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set guide and installed video VAE for bounded cancellation qualification.")
    }
    let geometry = try H3Geometry(width: 384, height: 256, durationSeconds: 2.5)
    let guide = try H3FunControlMedia.load(path: movie,
      ffmpeg: URL(fileURLWithPath: env["WEETODD_FFMPEG"] ?? "/opt/homebrew/bin/ffmpeg"), geometry: geometry)
    let report = await Task.detached { () -> (Bool, Int, Int, Int) in
      Stream.gpu.synchronize(); Memory.clearCache()
      let before = Memory.activeMemory
      var calls = 0
      var cancelled = false
      do {
        try autoreleasepool {
          _ = try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: URL(fileURLWithPath: checkpoint),
            rgb8: Array(guide.rgb8), frameCount: geometry.frames,
            width: geometry.width, height: geometry.height, progress: { completed, _ in
              calls = completed
              if completed == 1 { withUnsafeCurrentTask { $0?.cancel() } }
            })
        }
      } catch is CancellationError { cancelled = true }
      catch { return (false, calls, before, -1) }
      Stream.gpu.synchronize(); Memory.clearCache()
      return (cancelled, calls, before, Memory.activeMemory)
    }.value
    XCTAssertTrue(report.0); XCTAssertEqual(report.1, 1)
    XCTAssertEqual(report.3, report.2)
    print("H3 installed guide encoder cancellation after one clip: activeMLXBefore=\(report.2), after=\(report.3)")
  }

  func testGuideResamplesAndCoverCropsThenHoldsLastFrameWithoutAudio() throws {
    let ffmpeg = URL(fileURLWithPath: ProcessInfo.processInfo.environment["WEETODD_FFMPEG"]
      ?? "/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw XCTSkip("Set WEETODD_FFMPEG for the bounded guide decode test.")
    }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("guide.mkv")
    let process = Process(); process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=30",
      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-t", "0.5",
      "-c:v", "ffv1", "-c:a", "pcm_s16le", movie.path]
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let guide = try H3FunControlMedia.load(path: movie.path, ffmpeg: ffmpeg, geometry: geometry)
    XCTAssertEqual(guide.width, 32); XCTAssertEqual(guide.height, 32)
    XCTAssertEqual(guide.frameCount, 73); XCTAssertNil(guide.audio)
    XCTAssertEqual(guide.rgb8.count, 73 * 32 * 32 * 3)
    let frameBytes = 32 * 32 * 3
    XCTAssertEqual(guide.rgb8.subdata(in: 12 * frameBytes..<13 * frameBytes),
      guide.rgb8.suffix(frameBytes))
    XCTAssertNotEqual(guide.rgb8.prefix(frameBytes), guide.rgb8.suffix(frameBytes))
  }

  func testInvalidGuideDecodeCanvasFailsBeforeLaunchingFFmpeg() throws {
    let geometry = try H3Geometry(width: 4096, height: 32, durationSeconds: 2.5)
    XCTAssertThrowsError(try H3FunControlMedia.load(path: "/tmp/unavailable.mp4",
      ffmpeg: URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"), geometry: geometry))
  }
}
