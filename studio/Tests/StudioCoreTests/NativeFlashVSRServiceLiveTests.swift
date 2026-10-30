import Foundation
import XCTest
@testable import StudioCore

/// Explicit opt-in only. Runs actual installed models through the shipped
/// service/worker protocol, media pipeline, audio verification and CLI.
final class NativeFlashVSRServiceLiveTests: XCTestCase {
  private func qualify(variant: FlashVSRSettings.Variant, decoder: FlashVSRSettings.Decoder, cli: Bool) async throws {
    let env = ProcessInfo.processInfo.environment
    guard let source = env["WEETODD_NATIVE_FLASH_SOURCE"], let models = env["WEETODD_NATIVE_FLASH_MODELS"],
      let worker = env["WEETODD_NATIVE_FLASH_WORKER"], let destination = env["WEETODD_NATIVE_FLASH_QUALIFICATION"] else {
      throw XCTSkip("Explicit source/model/worker/qualification paths required for actual native FlashVSR+ service qualification")
    }
    let name = variant.rawValue + "-" + decoder.rawValue, root = URL(fileURLWithPath: destination)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    var settings = FlashVSRSettings(); settings.backend = .swiftMLX; settings.variant = variant; settings.decoder = decoder
    settings.width = Int(env["WEETODD_NATIVE_FLASH_WIDTH"] ?? "512") ?? 512
    settings.height = Int(env["WEETODD_NATIVE_FLASH_HEIGHT"] ?? "288") ?? 288
    settings.seed = 42; settings.colorFix = .wavelet; settings.temporalCompression = 4
    var runtime = FlashVSRRuntime(); runtime.modelRoot = models; runtime.swiftWorkerPath = worker
    runtime.ffmpegPath = env["WEETODD_NATIVE_FLASH_FFMPEG"] ?? "/opt/homebrew/bin/ffmpeg"
    runtime.ffprobePath = env["WEETODD_NATIVE_FLASH_FFPROBE"] ?? "/opt/homebrew/bin/ffprobe"
    runtime.supervisorPath = env["WEETODD_NATIVE_FLASH_CLI"] ?? ""
    let request = VideoUpscaleRequest(sourcePath: source, settings: settings, runtime: runtime)
    let plan = try await FlashVSRService.preflight(request: request)
    XCTAssertTrue(plan.media.hasAudio); XCTAssertEqual(plan.frames, 17)
    let output = root.appendingPathComponent(name)
    let requestURL = root.appendingPathComponent(name + "-request.json"), encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
    let verifyOnly = env["WEETODD_NATIVE_FLASH_VERIFY_ONLY"] == "1"
    if verifyOnly {
      XCTAssertEqual(try JSONDecoder().decode(VideoUpscaleRequest.self, from: Data(contentsOf: requestURL)), request)
    } else { try encoder.encode(request).write(to: requestURL, options: .withoutOverwriting) }
    let receipt: FlashVSRReceipt
    if verifyOnly { receipt = try FlashVSRService.resume(plan: plan, directory: output) }
    else if cli {
      guard let executable = env["WEETODD_NATIVE_FLASH_CLI"] else { throw XCTSkip("Packaged native CLI path required for the actual CLI case") }
      let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: executable),
        arguments: ["upscale", "--request", requestURL.path, "--output-directory", output.path], directory: root,
        log: root.appendingPathComponent(name + "-cli.log"), cancellation: .init(), maximumOutput: 8*1024*1024)
      XCTAssertEqual(result.exitCode, 0, String(decoding: result.output.suffix(8192), as: UTF8.self))
      receipt = try FlashVSRService.resume(plan: plan, directory: output)
    } else { receipt = try await FlashVSRService.run(plan: plan, directory: output) }
    XCTAssertEqual(receipt.plan.frames, 17); XCTAssertNotNil(receipt.audio)
    XCTAssertEqual(receipt.plan.request.settings.backend, .swiftMLX)
    XCTAssertTrue(receipt.plan.request.runtime.vpipePath.isEmpty)
    let checked = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: receipt.video.path), runtime: runtime, directory: root, cancellation: .init())
    XCTAssertEqual(checked.frames, 17); XCTAssertEqual(checked.fps, plan.media.fps)
    XCTAssertEqual(checked.width, settings.width); XCTAssertEqual(checked.height, settings.height); XCTAssertTrue(checked.hasAudio)
    func pcmHash(_ file: String) throws -> Data {
      let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath),
        arguments: ["-v", "error", "-nostdin", "-i", file, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "hash", "-hash", "sha256", "-"],
        directory: root, log: root.appendingPathComponent(UUID().uuidString + "-pcm.log"), cancellation: .init())
      XCTAssertEqual(result.exitCode, 0); return result.output
    }
    XCTAssertEqual(try pcmHash(source), try pcmHash(receipt.video.path))
    let workerResult = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("native-result.json"))) as? [String: Any])
    XCTAssertEqual(workerResult["nativeRuntime"] as? String, "swift-mlx")
    let metrics = try XCTUnwrap(workerResult["metrics"] as? [String: Any])
    XCTAssertEqual(metrics["visibleFrames"] as? Int, 17)
    let released = try XCTUnwrap(metrics["releasedStageActiveBytes"] as? [String: Int])
    XCTAssertEqual(released.count, 3); XCTAssertTrue(released.values.allSatisfy { $0 == 0 })
    XCTAssertGreaterThan(try XCTUnwrap(metrics["peakMLXBytes"] as? Int), 0)
    XCTAssertGreaterThan(try XCTUnwrap(metrics["peakProcessFootprintBytes"] as? Int), 0)
    XCTAssertEqual(metrics["outputSHA256"] as? String, try FlashVSRFile.capture(output.appendingPathComponent("native-worker/restored.rgb")).sha256)
    let reused = try FlashVSRService.resume(plan: plan, directory: output)
    XCTAssertEqual(reused.video, receipt.video); try receipt.restored.verify(); try receipt.video.verify()
    print("native-flash-service \(name) frames=17 fps=\(checked.fps.argument) output=\(receipt.video.path) elapsed=\(receipt.elapsedSeconds)")
  }
  func testActualTurboTinyServiceAudioAndCadence() async throws { try await qualify(variant: .turbo, decoder: .tiny, cli: false) }
  func testActualBaseWanCLIServiceAudioAndCadence() async throws { try await qualify(variant: .base, decoder: .vae, cli: true) }
}
