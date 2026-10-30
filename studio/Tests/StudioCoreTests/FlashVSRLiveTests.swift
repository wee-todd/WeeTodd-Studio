import Foundation
import XCTest
@testable import StudioCore

final class FlashVSRLiveTests: XCTestCase {
  func testConfiguredTurboDecodersPreserveSourceFrames() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let executable = env["WEETODD_FLASHVSR_VPIPE"], let modelRoot = env["WEETODD_FLASHVSR_MODELS"],
      let destination = env["WEETODD_FLASHVSR_QUALIFICATION"] else { throw XCTSkip("Opt-in official-weight Turbo decoder qualification") }
    let root = URL(fileURLWithPath: destination)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let input = root.appendingPathComponent("source.mkv")
    if !FileManager.default.fileExists(atPath: input.path) {
      try NativeMovieFrozenMedia.run(URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
        ["-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=24", "-frames:v", "49", "-c:v", "ffv1", input.path], log: root.appendingPathComponent("source.log"))
    }
    for decoder in FlashVSRSettings.Decoder.allCases {
      var settings = FlashVSRSettings(); settings.backend = .vpipe; settings.decoder = decoder; settings.width = 256; settings.height = 256
      let files = try FlashVSRModelManifest.resolve(root: URL(fileURLWithPath: modelRoot), settings: settings)
      try FlashVSRModelManifest.verifyPublished(files, settings: settings)
      let output = root.appendingPathComponent(decoder.rawValue + ".nut")
      let graph = try VpipeFlashVSRPipeline.make(settings: settings, modelRoot: modelRoot, input: input.path, output: output.path, fps: 24, frames: 49)
      let recipe = root.appendingPathComponent(decoder.rawValue + ".json"); try graph.write(to: recipe)
      let process = Process(); process.executableURL = URL(fileURLWithPath: executable)
      process.currentDirectoryURL = root
      process.arguments = ["--config", "{\"memory_cap_mb\":12288,\"wired_pool_mb\":4096,\"swap_allowance_mb\":0}", "--launch", recipe.path]
      let log = root.appendingPathComponent(decoder.rawValue + ".log")
      FileManager.default.createFile(atPath: log.path, contents: nil)
      let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
      process.standardInput = FileHandle.nullDevice; process.standardOutput = handle; process.standardError = handle
      try process.run(); process.waitUntilExit()
      XCTAssertEqual(process.terminationStatus, 0)
      let probe = Process(); probe.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffprobe")
      probe.arguments = ["-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", output.path]
      let pipe = Pipe(); probe.standardOutput = pipe
      try probe.run(); let data = pipe.fileHandleForReading.readDataToEndOfFile(); probe.waitUntilExit()
      XCTAssertEqual(probe.terminationStatus, 0)
      let body = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      let stream = (body["streams"] as! [[String: Any]])[0]
      XCTAssertEqual(stream["width"] as? Int, 256); XCTAssertEqual(stream["height"] as? Int, 256)
      XCTAssertEqual(stream["nb_read_frames"] as? String, "49", "Every source frame must survive " + decoder.rawValue)
      XCTAssertEqual(stream["avg_frame_rate"] as? String, "24/1")
    }
  }
}
