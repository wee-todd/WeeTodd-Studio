import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingLiveTests: XCTestCase {
  func testExistingH3AndLTXTakesThroughRecipeAndResume() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let manifest = environment["WEE_TODD_PROCESSING_LIVE_REQUESTS"],
      let destination = environment["WEE_TODD_PROCESSING_LIVE_OUTPUT"] else {
      throw XCTSkip("Opt-in existing H3/LTX takes with the separately installed FlashVSR runtime")
    }
    let requests = try JSONDecoder().decode([VideoUpscaleRequest].self, from: Data(contentsOf: URL(fileURLWithPath: manifest)))
    XCTAssertEqual(requests.count, 4, "H3/LTX × Tiny/Full")
    var results: [[String: Any]] = []
    for (index, request) in requests.enumerated() {
      var project = StudioProject(); project.settings.width = request.settings.width
      project.settings.height = request.settings.height
      let inspection = URL(fileURLWithPath: destination).appendingPathComponent("source-" + String(index))
      try FileManager.default.createDirectory(at: inspection, withIntermediateDirectories: true)
      let clock = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: request.sourcePath), runtime: request.runtime,
        directory: inspection, cancellation: .init())
      project.settings.fps = clock.fps.value
      var clip = Clip(engine: .movie); clip.sourcePath = request.sourcePath
      clip.duration = 17.0 / clock.fps.value; clip.videoProcessing = VideoProcessingRecipe(settings: request.settings)
      project.clips = [clip]
      let job = try NativeHeadlessJob(project: project, recipes: [:], workers: [:],
        ffmpeg: request.runtime.ffmpegPath, processingRuntime: request.runtime)
      let output = URL(fileURLWithPath: destination).appendingPathComponent(String(index))
      if let cache = environment["WEE_TODD_PROCESSING_LIVE_CACHE"] {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: output.appendingPathComponent("processing"),
          withDestinationURL: URL(fileURLWithPath: cache).appendingPathComponent(String(index) + "/processing"))
      }
      let start = Date()
      let result = try await NativeHeadlessExecutor.run(job: job, output: output)
      let elapsed = Date().timeIntervalSince(start)
      let resumed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true)
      XCTAssertEqual(resumed["newlyGenerated"] as? Int, 0)
      XCTAssertEqual(resumed["reusedProcessingSteps"] as? Int, 1)
      let accepted = try ProjectStorage.read(URL(fileURLWithPath: result["project"] as! String))
      XCTAssertEqual(accepted.clips[0].versions.last?.processingLineage?.source.path,
        URL(fileURLWithPath: request.sourcePath).resolvingSymlinksInPath().path)
      let receiptFile = try XCTUnwrap(accepted.clips[0].versions.last?.processingLineage?.receiptPath)
      let receipt = try JSONDecoder().decode(FlashVSRReceipt.self, from: Data(contentsOf: URL(fileURLWithPath: receiptFile)))
      XCTAssertNotNil(receipt.audio); XCTAssertEqual(receipt.plan.frames, 17)
      XCTAssertEqual(receipt.plan.media.fps.value, clock.fps.value)
      results.append(["source": request.sourcePath, "decoder": request.settings.decoder.rawValue,
        "upscaleSeconds": receipt.elapsedSeconds, "jobSeconds": elapsed, "video": result["video"]!, "receipt": receiptFile])
    }
    try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: destination).appendingPathComponent("qualification.json"), options: .atomic)
  }
}
