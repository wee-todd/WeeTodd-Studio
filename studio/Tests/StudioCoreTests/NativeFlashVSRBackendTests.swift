import Foundation
import XCTest
@testable import StudioCore

final class NativeFlashVSRBackendTests: XCTestCase {
  func testProcessingExecutorRoutesFrozenNativeSelectionWithoutExternalOverride() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try VideoProcessingFixture(root: root)
    let source = try fixture.source()
    var base = try fixture.plan()
    var recipe = base.recipe; recipe.steps[0].settings.backend = .swiftMLX
    var runtime = base.runtime; runtime.modelRoot = "/models"
    runtime.swiftWorkerPath = "/missing/WeeToddFlashVSRWorker"
    base = .init(recipe: recipe, runtime: runtime, resources: base.resources)
    do {
      _ = try await NativeVideoProcessingExecutor.run(plan: base, source: source,
        directory: root.appendingPathComponent("cache"))
      XCTFail("The missing selected native helper must fail")
    } catch { XCTAssertTrue(error.localizedDescription.contains("native FlashVSR+ helper")) }
  }
  func testRuntimeRejectsIgnoredInterpreterSettingsAndReadsOlderOptionalFields() throws {
    let data = try JSONEncoder().encode(FlashVSRRuntime())
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    object["pythonPath"] = "/unexpected/python"
    XCTAssertThrowsError(try JSONDecoder().decode(FlashVSRRuntime.self,
      from: JSONSerialization.data(withJSONObject: object)))
    object.removeValue(forKey: "pythonPath"); object.removeValue(forKey: "leasePath")
    XCTAssertNoThrow(try JSONDecoder().decode(FlashVSRRuntime.self,
      from: JSONSerialization.data(withJSONObject: object)))
  }
  func testWrongBackendRejectsBeforeCreatingAnAttempt() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    var settings = FlashVSRSettings(); settings.backend = .swiftMLX
    let file = FlashVSRFile(path: "/missing", sha256: String(repeating: "a", count: 64), bytes: 1)
    let media = FlashVSRMediaInfo(width: 128, height: 128, frames: 1,
      fps: try .init(numerator: 24, denominator: 1), firstTimestamp: 0,
      hasAudio: false, audioRate: nil, audioChannels: nil)
    let plan = FlashVSRResolvedPlan(request: .init(sourcePath: file.path, settings: settings, runtime: .init()),
      source: file, media: media, components: .init(root: "/models", dit: file, projection: file, context: file, decoder: file),
      engineFiles: [], frames: 1, temporaryBytes: 1)
    do { _ = try await VpipeFlashVSRBackend().run(plan: plan, directory: directory, emit: { _ in }); XCTFail("Wrong backend must fail") }
    catch { }
    XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
  }
  func testNativeProfilesAdmitBothVariantsAndKeepDecoderSelectionExplicit() throws {
    for variant in FlashVSRSettings.Variant.allCases {
      for decoder in FlashVSRSettings.Decoder.allCases {
        var settings = FlashVSRSettings(); settings.backend = .swiftMLX
        settings.variant = variant; settings.decoder = decoder
        XCTAssertNoThrow(try settings.validateExecution())
      }
    }
    var legacy = FlashVSRSettings(); legacy.backend = .vpipe; legacy.variant = .base
    XCTAssertThrowsError(try legacy.validateExecution())
    var independent = FlashVSRSettings(); independent.backend = .swiftMLX; independent.continuity = .clips
    XCTAssertThrowsError(try independent.validateExecution())
  }
}
