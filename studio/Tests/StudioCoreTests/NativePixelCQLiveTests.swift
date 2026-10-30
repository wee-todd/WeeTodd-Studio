import Foundation
import XCTest
@testable import StudioCore

/// Opt-in prepares a real frozen job for the packaged CLI; ordinary validation never calls models.
final class NativePixelCQLiveTests: XCTestCase {
  func testExportRealNativeProcessingJob() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let folder = env["WEETODD_PIXEL_CQ_QUALIFICATION"], let sourcePath = env["WEETODD_PIXEL_CQ_SOURCE"],
      let worker = env["WEETODD_PIXEL_CQ_WORKER"], let profiles = env["WEETODD_PIXEL_CQ_PROFILES"],
      let cq = env["WEETODD_PIXEL_CQ_ADAPTER"] else { throw XCTSkip("Explicit native recipe qualification paths required") }
    let root = URL(fileURLWithPath: folder), source = URL(fileURLWithPath: sourcePath)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    var settings = NativePixelCQSettings(); settings.cqAdapterPath = cq
    let runtime = try NativePixelCQRuntime.resolve(settings: settings, profilesDirectory: profiles, workerPath: worker,
      ffmpegPath: "/opt/homebrew/bin/ffmpeg", ffprobePath: "/opt/homebrew/bin/ffprobe")
    let media = try await NativeLTXMoviePreparation.inspectSource(source)
    var project = StudioProject(); project.settings.width = (media["width"] as! Int) * 2
    project.settings.height = (media["height"] as! Int) * 2; project.settings.fps = media["fps"] as! Double
    var clip = Clip(name: "Native Pixel Spatial + CQ qualification", engine: .h3)
    clip.sourcePath = sourcePath; clip.duration = media["duration"] as! Double
    clip.generationWidth = media["width"] as! Int; clip.generationHeight = media["height"] as! Int
    clip.prompt = "Preserve the woman and her joke."
    clip.videoProcessing = VideoProcessingRecipe(pixelCQ: settings)
    clip.versions = [.init(path: sourcePath, seed: 20261028, prompt: clip.prompt, recipePath: "")]
    project.clips = [clip]
    let job = try NativeHeadlessJob(project: project, recipes: [:], workers: [:], ffmpeg: runtime.ffmpegPath, nativeProcessingRuntime: runtime)
    try job.write(to: root.appendingPathComponent("native-recipe-job.json"))
    var ltx = clip; ltx.id = UUID(); ltx.engine = .ltx25; ltx.name = "LTX native upscale controls"
    project.clips.append(ltx)
    try ProjectStorage.write(project, to: root.appendingPathComponent("Native-Recipe-Controls.weetodd"))
    XCTAssertEqual(try NativeHeadlessJob.read(from: root.appendingPathComponent("native-recipe-job.json")).nativeProcessingRuntime, runtime)
  }
}
