import Foundation
import NNC
import XCTest
@testable import DrawThingsTransport

final class LocalTextSessionTests: XCTestCase {
  private var request: [String: Any] {
    ["modelPath": "/missing/qwen_3.5_4b_i8x.ckpt", "systemPrompt": "Describe precisely.",
     "prompt": "Name one color.", "maxTokens": 16]
  }

  func testSessionCancellationPrecedesModelLoadingAndUnloadIsIdempotent() {
    let session = LocalTextSession()
    XCTAssertThrowsError(try session.run(request, progress: { _ in }, cancelled: { true })) {
      XCTAssertEqual(($0 as? LocalTextError)?.code, "text_cancelled")
    }
    session.unload(); session.unload()
    XCTAssertFalse(session.isResident)
  }

  func testSessionPreservesNineBAsOneShotOnly() {
    var value = request; value["modelPath"] = "/missing/qwen_3.5_9b_i5x.ckpt"
    XCTAssertThrowsError(try LocalTextSession().run(value, progress: { _ in }, cancelled: { false })) {
      XCTAssertEqual(($0 as? LocalTextError)?.code, "text_session_model_unsupported")
    }
  }

  func testSessionRejectsUnavailableStoreWithoutGPUAllocation() {
    let session = LocalTextSession()
    XCTAssertThrowsError(try session.run(request, progress: { _ in }, cancelled: { false })) {
      XCTAssertEqual(($0 as? LocalTextError)?.code, "text_model_unavailable")
    }
    XCTAssertFalse(session.isResident)
  }

  func testVisionCacheBoundsIncludePreparedAndFeatureBytesAndAreLRU() {
    var cache = LocalVisionCache<String>(entryLimit: 2, byteLimit: 12)
    cache.insert("A", key: "a", bytes: 5)
    cache.insert("B", key: "b", bytes: 5)
    XCTAssertEqual(cache.value(for: "a"), "A")
    cache.insert("C", key: "c", bytes: 5)
    XCTAssertNil(cache.value(for: "b"))
    XCTAssertEqual(cache.count, 2)
    XCTAssertEqual(cache.retainedBytes, 10)
    cache.insert("Too large", key: "huge", bytes: 13)
    XCTAssertNil(cache.value(for: "huge"))
    XCTAssertEqual(cache.count, 2)
    cache.insert("replacement", key: "a", bytes: 9)
    XCTAssertEqual(cache.retainedBytes, 9)
    XCTAssertNil(cache.value(for: "c"))
    cache.removeAll()
    XCTAssertEqual(cache.retainedBytes, 0)
    XCTAssertEqual(cache.count, 0)
  }

  func testImageIdentityIncludesOrderLabelsAndFileContentNotJustPathOrSize() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let first = directory.appendingPathComponent("first.png")
    let second = directory.appendingPathComponent("second.png")
    try Data("aaaa".utf8).write(to: first)
    try Data("bbbb".utf8).write(to: second)
    let a = LocalPromptImage(path: first.path, label: "Character")
    let b = LocalPromptImage(path: second.path, label: "Face")
    let original = try LocalVisionIdentity.key(images: [a, b], model: "model-v1", cancelled: { false })
    XCTAssertEqual(original, try LocalVisionIdentity.key(images: [a, b], model: "model-v1", cancelled: { false }))
    XCTAssertNotEqual(original, try LocalVisionIdentity.key(images: [b, a], model: "model-v1", cancelled: { false }))
    XCTAssertNotEqual(original, try LocalVisionIdentity.key(images: [a, b], model: "model-v2", cancelled: { false }))
    XCTAssertNotEqual(original, try LocalVisionIdentity.key(images: [LocalPromptImage(path: first.path, label: "Style"), b], model: "model-v1", cancelled: { false }))
    try Data("zzzz".utf8).write(to: first)
    XCTAssertNotEqual(original, try LocalVisionIdentity.key(images: [a, b], model: "model-v1", cancelled: { false }))
    XCTAssertThrowsError(try LocalVisionIdentity.key(images: [a], model: "model-v1", cancelled: { true }))
  }

  func testLiveResidentVisionReuseAndSourceReplacement() throws {
    guard let model = ProcessInfo.processInfo.environment["WEETODD_TEST_QWEN_MODEL"] else {
      throw XCTSkip("Set WEETODD_TEST_QWEN_MODEL for local resident vision qualification")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    func writeImage(_ color: Int, size: Int = 64) throws -> String {
      if FileManager.default.fileExists(atPath: root.path) { try FileManager.default.removeItem(at: root) }
      let writer = try ArtifactWriter(root: root, requestID: "vision", operation: "image",
        expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
      var pixels = Tensor<Float>(.CPU, .NHWC(1, size, size, 3))
      for y in 0..<size { for x in 0..<size { for c in 0..<3 { pixels[0,y,x,c] = c == color ? 1 : -1 } } }
      try writer.image(pixels)
      return root.appendingPathComponent("frames/00000000.png").path
    }
    let image = try writeImage(0)
    var value = request
    value["modelPath"] = model
    value["systemPrompt"] = "Name the dominant image color in one word."
    value["prompt"] = "What color is this image?"
    value["images"] = [["path": image, "label": "Reference"]]
    let baseline = try LocalTextGeneration.run(value, progress: { _ in })
    let session = LocalTextSession()
    defer { session.unload() }
    let first = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(first["text"] as? String, baseline["text"] as? String)
    let second = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(first["text"] as? String, second["text"] as? String)
    let reuse = try XCTUnwrap(second["reuse"] as? [String: Any])
    XCTAssertEqual(reuse["decoderLoads"] as? Int, 1)
    XCTAssertEqual(reuse["visionLoads"] as? Int, 1)
    XCTAssertEqual(reuse["visionEncodes"] as? Int, 1)
    XCTAssertEqual(reuse["visionCacheHits"] as? Int, 1)
    XCTAssertGreaterThan(reuse["visionCacheBytes"] as? Int ?? 0, 0)
    _ = try writeImage(2)
    let replaced = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertNotEqual(first["text"] as? String, replaced["text"] as? String)
    XCTAssertEqual((replaced["reuse"] as? [String: Any])?["visionEncodes"] as? Int, 2)
    // A shape change must reuse the same vision weights, not allocate another encoder.
    _ = try writeImage(1, size: 96)
    let resized = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual((resized["reuse"] as? [String: Any])?["visionLoads"] as? Int, 1)
    XCTAssertEqual((resized["reuse"] as? [String: Any])?["decoderLoads"] as? Int, 1)
    XCTAssertTrue((resized["text"] as? String ?? "").lowercased().contains("green"))
    var cancelled = false
    XCTAssertThrowsError(try session.run(value, progress: { event in
      if event["stage"] as? String == "writing" { cancelled = true }
    }, cancelled: { cancelled })) {
      XCTAssertEqual(($0 as? LocalTextError)?.code, "text_cancelled")
    }
    XCTAssertFalse(session.isResident)
  }

  func testLiveVisionTruncationConsumesTheRequestedBudget() throws {
    guard let model = ProcessInfo.processInfo.environment["WEETODD_TEST_QWEN_MODEL"] else {
      throw XCTSkip("Set WEETODD_TEST_QWEN_MODEL for output-budget regression")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try ArtifactWriter(root: root, requestID: "budget", operation: "image",
      expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
    var pixels = Tensor<Float>(.CPU, .NHWC(1, 64, 64, 3))
    _ = pixels.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) }
    try writer.image(pixels)
    var value = request; value["modelPath"] = model; value["maxTokens"] = 2
    value["systemPrompt"] = "Follow the instruction exactly."
    value["prompt"] = "Count from one to ten using comma-separated digits and no other text."
    value["images"] = [["path": root.appendingPathComponent("frames/00000000.png").path, "label": "Reference"]]
    let session = LocalTextSession(); defer { session.unload() }
    let result = try session.run(value, progress: { _ in }, cancelled: { false })
    // Unlike the upstream pipelined vision loop, the session drains the final
    // token at the cap. Do not invoke the old short-budget path here: it can
    // assert natively during eager compilation, outside Swift error handling.
    XCTAssertEqual(result["outputTokens"] as? Int, 2)
    XCTAssertEqual(result["truncated"] as? Bool, true)
    XCTAssertTrue((result["text"] as? String ?? "").hasPrefix("1"))
    for budget in [1, 3, 8] {
      value["maxTokens"] = budget
      let bounded = try session.run(value, progress: { _ in }, cancelled: { false })
      XCTAssertEqual(bounded["outputTokens"] as? Int, budget)
      XCTAssertEqual(bounded["truncated"] as? Bool, true)
      XCTAssertTrue((bounded["text"] as? String ?? "").hasPrefix("1"))
    }
  }

  func testLiveLongTextPrefillThenVisionAndTextKeepsWeights() throws {
    guard let model = ProcessInfo.processInfo.environment["WEETODD_TEST_QWEN_MODEL"] else {
      throw XCTSkip("Set WEETODD_TEST_QWEN_MODEL for mixed chunked-prefill qualification")
    }
    var value = request; value["modelPath"] = model; value["maxTokens"] = 128
    value["systemPrompt"] = "Follow the final instruction. Reply with the requested single word only."
    value["prompt"] = String(repeating: "Background record: the room has a table, a lamp, and a wooden floor. ", count: 70)
      + "Final instruction: write amber."
    let budget = try LocalTextGeneration.preflight(value)
    XCTAssertGreaterThan(budget["inputTokens"] as? Int ?? 0, 1024)
    let session = LocalTextSession(); defer { session.unload() }
    let long = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(long["text"] as? String, "amber")

    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try ArtifactWriter(root: root, requestID: "mixed", operation: "image",
      expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
    var pixels = Tensor<Float>(.CPU, .NHWC(1, 64, 64, 3))
    for y in 0..<64 { for x in 0..<64 { for c in 0..<3 { pixels[0,y,x,c] = c == 2 ? 1 : -1 } } }
    try writer.image(pixels)
    var vision = value
    vision["prompt"] = "Name the image color in one word."
    vision["images"] = [["path": root.appendingPathComponent("frames/00000000.png").path, "label": "Color reference"]]
    let color = try session.run(vision, progress: { _ in }, cancelled: { false })
    XCTAssertEqual((color["text"] as? String)?.lowercased(), "blue")
    var short = value; short["prompt"] = "Write violet."
    let isolated = try session.run(short, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(isolated["text"] as? String, "violet")
    let repeated = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(repeated["text"] as? String, "amber")
    XCTAssertEqual((repeated["reuse"] as? [String: Any])?["decoderLoads"] as? Int, 1)
    XCTAssertEqual((repeated["reuse"] as? [String: Any])?["visionLoads"] as? Int, 1)
    XCTAssertEqual((repeated["reuse"] as? [String: Any])?["requests"] as? Int, 4)
  }

  /// Opt-in qualification uses installed weights, never downloads or ordinary CI GPU work.
  func testLiveResidentTextIsolationAndUnload() throws {
    guard let model = ProcessInfo.processInfo.environment["WEETODD_TEST_QWEN_MODEL"] else {
      throw XCTSkip("Set WEETODD_TEST_QWEN_MODEL for local resident parity qualification")
    }
    let session = LocalTextSession()
    defer { session.unload() }
    var value = request; value["modelPath"] = model
    value["systemPrompt"] = "Reply with the requested word only."
    value["prompt"] = "Write amber."
    let baseline = try LocalTextGeneration.run(value, progress: { _ in })
    let first = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(first["text"] as? String, baseline["text"] as? String)
    value["prompt"] = "Write violet."
    _ = try session.run(value, progress: { _ in }, cancelled: { false })
    value["prompt"] = "Write amber."
    let repeated = try session.run(value, progress: { _ in }, cancelled: { false })
    XCTAssertEqual(repeated["text"] as? String, first["text"] as? String)
    XCTAssertEqual((repeated["reuse"] as? [String: Any])?["decoderLoads"] as? Int, 1)
    XCTAssertEqual((repeated["reuse"] as? [String: Any])?["requests"] as? Int, 3)
    session.unload()
    XCTAssertFalse(session.isResident)
  }
}
