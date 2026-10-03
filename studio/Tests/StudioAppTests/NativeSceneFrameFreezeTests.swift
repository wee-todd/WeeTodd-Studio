import AppKit
import CryptoKit
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeSceneFrameFreezeTests: XCTestCase {
  @MainActor func testNativeFrameSnapshotUsesLastVisiblePTSForH3AndLTX() async throws {
    let root = try directory(), movie = try colorMovie(root)
    for engine in [Engine.h3, .ltx25] {
      var source = Clip(engine: .movie)
      source.sourcePath = movie.path; source.sourceIn = 0.4; source.duration = 1.6
      var target = Clip(engine: engine)
      target.continuity = ClipContinuity(mode: "frame", sourceClipID: source.id)
      var project = StudioProject(); project.clips = [source, target]
      let output = root.appendingPathComponent(engine.rawValue)
      let result = try await NativeContinuityFrame.freeze(project: project, clip: target, destination: output)
      XCTAssertEqual(result["sourceFrameTime"] as! Double, 1, accuracy: 0.000001)
      XCTAssertEqual(result["sourceIn"] as? Double, 0.4)
      XCTAssertEqual(result["sourceTimeEnd"] as? Double, 2)
      XCTAssertEqual(result["sourceClipID"] as? String, source.id.uuidString)
      XCTAssertEqual(result["engine"] as? String, engine.rawValue)
      XCTAssertEqual(result["pythonExecuted"] as? Bool, false)
      let image = URL(fileURLWithPath: try XCTUnwrap(result["path"] as? String))
      try assertGreen(image)
      XCTAssertEqual(result["sourceSHA256"] as? String, try hash(movie))
      XCTAssertEqual(result["sourceFrozenSHA256"] as? String, try hash(image))
      let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("continuity.json"))) as! NSDictionary
      XCTAssertEqual(receipt, result as NSDictionary)
      XCTAssertEqual(project.clips, [source, target])
    }
  }

  @MainActor func testDefaultNativeSceneConversionFreezesExplicitTrimWithoutPythonAndReopens() async throws {
    let root = try directory(), movie = try colorMovie(root)
    var calls: [String] = []
    let store = StudioStore(dataDirectory: root.appendingPathComponent("store"), restoreSession: false,
      invocation: { command, _, _, _ in
        calls.append(command); throw StudioError.invalid("Unexpected Python bridge call")
      })
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: root.path)
    var source = Clip(name: "Accepted source", engine: .ltx25)
    source.sourcePath = movie.path; source.sourceIn = 0.4; source.duration = 1.6
    var middle = Clip(name: "Other shot", engine: .ltx25)
    middle.sourcePath = movie.path; middle.sourceIn = 2; middle.duration = 1
    var target = Clip(name: "Connected shot", engine: .ltx25)
    target.seed = 42; target.continuity = ClipContinuity(mode: "frame", sourceClipID: source.id)
    target.generationSelection = GenerationSelection(task: "i2v"); target.generationSelection?.steps = 12
    let original = MediaAsset(name: "Original first", kind: .image, path: root.appendingPathComponent("original.png").path)
    try Data("preserved-original".utf8).write(to: URL(fileURLWithPath: original.path))
    target.attachments = [Attachment(assetID: original.id, role: .first)]
    store.project.clips = [source, middle, target]; store.project.assets = [original]
    store.select(target.id)
    await store.connectContinuousScene(clipID: target.id, preserveFrameMatch: true)
    XCTAssertNil(store.error)
    XCTAssertTrue(calls.isEmpty)
    guard store.error == nil else { return }
    XCTAssertEqual(Array(store.project.clips.prefix(2)), [source, middle])
    XCTAssertEqual(store.project.clips[2].continuityMode, "scene")
    XCTAssertEqual(store.project.clips[2].continuity?.sourceClipID, middle.id)
    XCTAssertEqual(store.project.clips[2].seed, 42)
    XCTAssertEqual(store.project.clips[2].generationSelection?.steps, 12)
    XCTAssertTrue(store.project.assets.contains(original))
    XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: original.path)), Data("preserved-original".utf8))
    let attachment = try XCTUnwrap(store.project.clips[2].attachments.first { $0.role == .first })
    let frozen = try XCTUnwrap(store.project.assets.first { $0.id == attachment.assetID })
    let image = URL(fileURLWithPath: frozen.path); try assertGreen(image)
    let provenance = try JSONSerialization.jsonObject(with: Data(contentsOf: image.deletingLastPathComponent().appendingPathComponent("continuity.json"))) as! [String: Any]
    XCTAssertEqual(provenance["sourceClipID"] as? String, source.id.uuidString)
    XCTAssertEqual(provenance["sourceFrozenSHA256"] as? String, try hash(image))
    let saved = root.appendingPathComponent("accepted.weetodd")
    try ProjectStorage.write(store.project, to: saved)
    let reopenedDirectory = root.appendingPathComponent("reopened")
    try FileManager.default.createDirectory(at: reopenedDirectory, withIntermediateDirectories: true)
    let reopened = StudioStore(dataDirectory: reopenedDirectory, restoreSession: false)
    reopened.load(saved)
    XCTAssertNil(reopened.error)
    XCTAssertEqual(reopened.project, store.project)
    let reopenedAsset = try XCTUnwrap(reopened.project.assets.last)
    XCTAssertEqual(try hash(URL(fileURLWithPath: reopenedAsset.path)), provenance["sourceFrozenSHA256"] as? String)
  }

  @MainActor func testCancelledNativeFreezePublishesNothingAndDoesNotOverwrite() async throws {
    let root = try directory(), movie = try colorMovie(root)
    var source = Clip(engine: .movie); source.sourcePath = movie.path; source.duration = 2
    var target = Clip(engine: .ltx25); target.continuity = ClipContinuity(mode: "frame")
    var project = StudioProject(); project.clips = [source, target]
    let output = root.appendingPathComponent("cancelled")
    let task = Task { try await NativeContinuityFrame.freeze(project: project, clip: target, destination: output) }
    task.cancel()
    do { _ = try await task.value; XCTFail("Cancelled freeze must fail") }
    catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let sentinel = output.appendingPathComponent("first-frame.png")
    try Data("do-not-overwrite".utf8).write(to: sentinel)
    do { _ = try await NativeContinuityFrame.freeze(project: project, clip: target, destination: output); XCTFail("Existing destination must fail") }
    catch { XCTAssertTrue(error.localizedDescription.contains("already exists")) }
    XCTAssertEqual(try Data(contentsOf: sentinel), Data("do-not-overwrite".utf8))
    XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".frame-") })
  }

  private func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-frame-freeze-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    return root
  }
  private func colorMovie(_ root: URL) throws -> URL {
    let executable = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].first { FileManager.default.isExecutableFile(atPath: $0) }
    guard let executable else { throw XCTSkip("Direct FFmpeg required for the bounded media-only fixture") }
    for frame in 0..<3 {
      var ppm = Data("P6\n64 64\n255\n".utf8)
      var pixel = [UInt8](repeating: 0, count: 3); pixel[frame] = 255
      for _ in 0..<(64 * 64) { ppm.append(contentsOf: pixel) }
      try ppm.write(to: root.appendingPathComponent("frame-\(frame).ppm"))
    }
    let movie = root.appendingPathComponent("colors.mp4"), process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = ["-v", "error", "-nostdin", "-n", "-framerate", "1", "-i",
      root.appendingPathComponent("frame-%d.ppm").path, "-frames:v", "3", "-c:v", "libx264", "-crf", "12",
      "-vf", "scale=out_color_matrix=bt709", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
      "-color_trc", "bt709", "-colorspace", "bt709", movie.path]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run(); process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0)
    return movie
  }
  private func hash(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
  }
  private func assertGreen(_ image: URL) throws {
    let bitmap = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: image)))
    let color = try XCTUnwrap(bitmap.colorAt(x: 32, y: 32)?.usingColorSpace(.deviceRGB))
    XCTAssertGreaterThan(color.greenComponent, 0.9)
    XCTAssertLessThan(color.redComponent, 0.1); XCTAssertLessThan(color.blueComponent, 0.1)
  }
}
