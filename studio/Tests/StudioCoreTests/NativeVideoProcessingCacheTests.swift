import Foundation
import XCTest
@testable import StudioCore

final class NativeVideoProcessingCacheTests: XCTestCase {
  private func root() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }; return root
  }
  func testCompletedUpscaleReusesVerifiedMediaAndDecoderChangeKeepsOriginal() async throws {
    let root = try root(), backend = try VideoProcessingFixture(root: root)
    let source = try backend.source(fps: 25), cache = root.appendingPathComponent("cache"), plan = try backend.plan()
    let first = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend)
    let reused = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend)
    XCTAssertEqual(reused.receipt.video, first.receipt.video); XCTAssertEqual(reused.reusedSteps, 1)
    let full = try await NativeVideoProcessingExecutor.run(plan: backend.plan(decoder: .vae), source: source, directory: cache, backend: backend)
    XCTAssertNotEqual(full.receipt.video.path, first.receipt.video.path)
    XCTAssertEqual(full.source, source); XCTAssertEqual(full.receipt.fps.value, 25)
    XCTAssertEqual(full.receipt.frames, 25)
    let renders = await backend.renders; XCTAssertEqual(renders, 2)
    try source.verify(); try first.receipt.video.verify()
  }
  func testFailedAndDamagedUpscalesRetryInNewAttemptsWithoutChangingSource() async throws {
    let root = try root(), backend = try VideoProcessingFixture(root: root, failures: 1)
    let source = try backend.source(), cache = root.appendingPathComponent("cache"), plan = try backend.plan()
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend); XCTFail("Expected failure") }
    catch { XCTAssertTrue(error.localizedDescription.contains("Fixture")) }
    let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend)
    try Data("damaged output".utf8).write(to: URL(fileURLWithPath: result.receipt.video.path))
    let repaired = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend)
    XCTAssertNotEqual(repaired.receipt.video.path, result.receipt.video.path)
    XCTAssertEqual(try String(contentsOfFile: result.receipt.video.path, encoding: .utf8), "damaged output")
    let renders = await backend.renders; XCTAssertEqual(renders, 3); try source.verify()
  }
  func testChangedFrozenSourceRejectsBeforeAnotherRestoration() async throws {
    let root = try root(), backend = try VideoProcessingFixture(root: root)
    let source = try backend.source(), cache = root.appendingPathComponent("cache"), plan = try backend.plan()
    _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend)
    let file = URL(fileURLWithPath: source.path); var bytes = try Data(contentsOf: file); bytes.append(0); try bytes.write(to: file)
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend); XCTFail("Changed original accepted") }
    catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
    let renders = await backend.renders; XCTAssertEqual(renders, 1)
  }
  func testChangedAdmittedComponentRejectsBeforeRestoration() async throws {
    let root = try root(), backend = try VideoProcessingFixture(root: root)
    let source = try backend.source(), plan = try backend.plan()
    try Data("different component".utf8).write(to: root.appendingPathComponent("fixture-engine"))
    do { _ = try await NativeVideoProcessingExecutor.run(plan: plan, source: source,
      directory: root.appendingPathComponent("cache"), backend: backend); XCTFail("Changed component admitted") }
    catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
    let renders = await backend.renders; XCTAssertEqual(renders, 0)
  }
  func testCancellingAsyncUpscalePreservesSourceAndDoesNotPublishCompletion() async throws {
    let root = try root(), backend = try VideoProcessingFixture(root: root, waitForCancellation: true)
    let source = try backend.source(), cache = root.appendingPathComponent("cache"), plan = try backend.plan()
    let cancellation = NativeHeadlessCancellation()
    let task = Task { try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: cache, backend: backend, cancellation: cancellation) }
    try await Task.sleep(nanoseconds: 100_000_000); cancellation.cancel()
    do { _ = try await task.value; XCTFail("Cancelled upscale completed") } catch { XCTAssertTrue(error is CancellationError) }
    try source.verify()
    let files = FileManager.default.enumerator(at: cache, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
    XCTAssertFalse(files.contains { $0.lastPathComponent == "completed.json" })
  }
}
