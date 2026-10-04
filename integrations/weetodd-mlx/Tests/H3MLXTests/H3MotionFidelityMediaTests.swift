import CryptoKit
import Foundation
import XCTest
@testable import H3MLX

final class H3MotionFidelityMediaTests: XCTestCase {
  private func executable(_ name: String) throws -> URL {
    let dirs = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":")
    for directory in dirs {
      let file = URL(fileURLWithPath: String(directory)).appendingPathComponent(name)
      if FileManager.default.isExecutableFile(atPath: file.path) { return file }
    }
    throw XCTSkip("Native \(name) media utility is unavailable.")
  }
  private func generate(_ movie: URL, ffmpeg: URL, audio: Bool) throws {
    let process = Process(); process.executableURL = ffmpeg
    var args = ["-v","error","-nostdin","-n","-f","lavfi","-i","testsrc2=size=64x32:rate=24:duration=3"]
    if audio { args += ["-f","lavfi","-i","sine=frequency=440:sample_rate=32000:duration=3"] }
    args += ["-c:v","libx264","-pix_fmt","yuv420p"]
    if audio { args += ["-c:a","aac","-ac","2"] }
    args += [movie.path]; process.arguments = args
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
  }
  private func hash(_ file: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: file)).map { String(format: "%02x", $0) }.joined()
  }
  func testRealNonzeroTrimPreservesOriginalAspectAudioAndSourceIdentity() throws {
    let ffmpeg = try executable("ffmpeg"), ffprobe = try executable("ffprobe")
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: folder) }
    let movie = folder.appendingPathComponent("source.mp4")
    try generate(movie, ffmpeg: ffmpeg, audio: true)
    let original = try hash(movie)
    let settings = try H3MotionFidelitySettings(mode: .uniform, maxHold: 2)
    let source = try H3MotionFidelityMedia.inspect(path: movie.path, sha256: original,
      ffprobe: ffprobe, startSeconds: 0.125, durationSeconds: 2.5, settings: settings)
    XCTAssertEqual(source.width, 64); XCTAssertEqual(source.height, 32); XCTAssertEqual(source.frames, 60)
    let decoded = try H3MotionFidelityMedia.decode(source, ffmpeg: ffmpeg)
    XCTAssertEqual(decoded.rgb8.count, 60 * 64 * 32 * 3)
    XCTAssertEqual(decoded.audio.frames, 80_000)
    XCTAssertTrue(decoded.audio.samples.allSatisfy(\.isFinite))
    XCTAssertTrue(decoded.audio.samples.contains { abs($0) > 0.001 })
    // Independent oracle: decode the entire soundtrack without a seek or trim,
    // then select exactly the edited interval on the native sample clock.
    let audioProcess = Process(), audioPipe = Pipe()
    audioProcess.executableURL = ffmpeg
    audioProcess.arguments = ["-v", "error", "-nostdin", "-i", movie.path,
      "-map", "0:a:0", "-ac", "2", "-ar", "32000", "-f", "f32le",
      "-acodec", "pcm_f32le", "pipe:1"]
    audioProcess.standardOutput = audioPipe; audioProcess.standardError = FileHandle.nullDevice
    try audioProcess.run()
    let fullPCM = audioPipe.fileHandleForReading.readDataToEndOfFile()
    audioProcess.waitUntilExit(); try audioPipe.fileHandleForReading.close()
    XCTAssertEqual(audioProcess.terminationStatus, 0)
    let firstSample = H3MotionFidelityPlan.audioSampleBoundary(frame: 3)
    let sampleEnd = firstSample + decoded.audio.frames
    guard fullPCM.count >= sampleEnd * 8 else {
      XCTFail("Full-source PCM does not cover the independently selected interval."); return
    }
    let selectedPCM = fullPCM.subdata(in: firstSample * 8..<sampleEnd * 8)
    let expectedAudio = try H3MotionFidelityMedia.pcm(selectedPCM, frames: decoded.audio.frames)
    XCTAssertEqual(decoded.audio.samples, expectedAudio.samples)
    let plan = try H3MotionFidelityPlan(sourceFrames: 60, settings: settings)
    let expanded = try H3MotionFidelityMedia.expandedAudio(original: decoded.audio,
      plan: plan, ffmpeg: ffmpeg, scratch: folder.appendingPathComponent("scratch"))
    XCTAssertEqual(expanded.frames, H3MotionFidelityPlan.audioSampleBoundary(frame: plan.paddedFrames))
    XCTAssertTrue(expanded.samples.allSatisfy(\.isFinite))
    XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("scratch").path))
    XCTAssertEqual(try hash(movie), original)
    try Data("mutated".utf8).write(to: movie)
    XCTAssertThrowsError(try H3MotionFidelityMedia.decode(source, ffmpeg: ffmpeg))
  }
  func testSilentNativeSourceHasExactZeroPCMWithoutModelOrPython() throws {
    let ffmpeg = try executable("ffmpeg"), ffprobe = try executable("ffprobe")
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: folder) }
    let movie = folder.appendingPathComponent("silent.mp4")
    try generate(movie, ffmpeg: ffmpeg, audio: false)
    let source = try H3MotionFidelityMedia.inspect(path: movie.path, sha256: hash(movie),
      ffprobe: ffprobe, startSeconds: 0, durationSeconds: 2.5,
      settings: H3MotionFidelitySettings(mode: .uniform))
    XCTAssertFalse(source.hasAudio)
    let decoded = try H3MotionFidelityMedia.decode(source, ffmpeg: ffmpeg)
    XCTAssertEqual(decoded.audio.samples, Array(repeating: Float(0), count: 160_000))
    XCTAssertThrowsError(try H3MotionFidelityMedia.inspect(path: movie.path, sha256: hash(movie),
      ffprobe: ffprobe, startSeconds: 0.01, durationSeconds: 2.5,
      settings: H3MotionFidelitySettings()))
  }
  func testPCMEndianStereoAndInvalidFiniteAdmission() throws {
    let values: [Float] = [1,-1,0.5,-0.5]
    var data = Data()
    for value in values { var bits = value.bitPattern.littleEndian; withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) } }
    let audio = try H3MotionFidelityMedia.pcm(data, frames: 2)
    XCTAssertEqual(audio.samples, [1,0.5,-1,-0.5])
    XCTAssertThrowsError(try H3MotionFidelityMedia.pcm(data, frames: 3))
    var bad = Float.nan.bitPattern.littleEndian
    withUnsafeBytes(of: &bad) { data.replaceSubrange(0..<4, with: $0) }
    XCTAssertThrowsError(try H3MotionFidelityMedia.pcm(data, frames: 2))
  }
}
