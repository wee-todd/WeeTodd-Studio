import Foundation
import XCTest
@testable import StudioCore

/// Only model inference is replaced. Media, hashes, receipts and cache I/O remain real.
actor VideoProcessingFixture: FlashVSRBackend {
  let root: URL
  var renders = 0
  var failures: Int
  var waitForCancellation: Bool
  init(root: URL, failures: Int = 0, waitForCancellation: Bool = false) throws {
    self.root = root; self.failures = failures; self.waitForCancellation = waitForCancellation
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let engine = root.appendingPathComponent("fixture-engine")
    if !FileManager.default.fileExists(atPath: engine.path) { try Data("model-free fixture".utf8).write(to: engine) }
  }
  func allowCompletion() { waitForCancellation = false }
  nonisolated static func ffmpeg(_ args: [String]) throws {
    guard FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/ffmpeg") else { throw XCTSkip("FFmpeg media fixture unavailable") }
    let child = Process(); child.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
    child.arguments = ["-v", "error", "-nostdin", "-n"] + args
    child.standardOutput = FileHandle.nullDevice; child.standardError = FileHandle.nullDevice
    try child.run(); child.waitUntilExit()
    guard child.terminationStatus == 0 else { throw StudioError.invalid("Media fixture failed") }
  }
  nonisolated func source(fps: Int = 24) throws -> FlashVSRFile {
    let file = root.appendingPathComponent("original.mp4")
    try Self.ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=\(fps):duration=1", "-f", "lavfi", "-i", "sine=sample_rate=32000:duration=1",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", file.path])
    return try FlashVSRFile.capture(file)
  }
  nonisolated func plan(decoder: FlashVSRSettings.Decoder = .tiny) throws -> NativeVideoProcessingPlan {
    var settings = FlashVSRSettings(); settings.width = 128; settings.height = 128; settings.decoder = decoder
    let recipe = VideoProcessingRecipe(settings: settings)
    var runtime = FlashVSRRuntime(); runtime.ffmpegPath = "/opt/homebrew/bin/ffmpeg"; runtime.ffprobePath = "/opt/homebrew/bin/ffprobe"
    return NativeVideoProcessingPlan(recipe: recipe, runtime: runtime, resources: ["restore": try resources()])
  }
  nonisolated func resources() throws -> FlashVSRResources {
    let file = try FlashVSRFile.capture(root.appendingPathComponent("fixture-engine"))
    return FlashVSRResources(components: .init(root: root.path, dit: file, projection: file, context: file, decoder: file), engineFiles: [file])
  }
  func preflight(request: VideoUpscaleRequest) async throws -> FlashVSRResolvedPlan {
    let directory = root.appendingPathComponent("inspect-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = try FlashVSRFile.capture(URL(fileURLWithPath: request.sourcePath))
    let media = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: source.path), runtime: request.runtime,
      directory: directory, cancellation: NativeHeadlessCancellation())
    let resources = try resources()
    return .init(request: request, source: source, media: media, components: resources.components,
      engineFiles: resources.engineFiles, frames: media.frames, temporaryBytes: 1024)
  }
  func run(plan: FlashVSRResolvedPlan, directory: URL, emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> FlashVSRReceipt {
    renders += 1
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    if waitForCancellation { try await Task.sleep(nanoseconds: 30_000_000_000) }
    if failures > 0 { failures -= 1; throw StudioError.invalid("Fixture restoration failed") }
    let raw = directory.appendingPathComponent("restored.nut"), audio = directory.appendingPathComponent("source-audio.wav")
    let video = directory.appendingPathComponent("output.mp4")
    try Self.ffmpeg(["-i", plan.source.path, "-vf", "scale=128:128", "-an", "-c:v", "rawvideo", "-pix_fmt", "rgb24", raw.path])
    try Self.ffmpeg(["-i", plan.source.path, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", audio.path])
    try Self.ffmpeg(["-r", plan.media.fps.argument, "-i", raw.path, "-i", audio.path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video.path])
    let receipt = FlashVSRReceipt(requestFingerprint: try plan.fingerprint, plan: plan, video: try .capture(video),
      restored: try .capture(raw), audio: try .capture(audio), elapsedSeconds: 0.01)
    try JSONEncoder().encode(receipt).write(to: directory.appendingPathComponent("receipt.json"), options: .atomic)
    return receipt
  }
}
