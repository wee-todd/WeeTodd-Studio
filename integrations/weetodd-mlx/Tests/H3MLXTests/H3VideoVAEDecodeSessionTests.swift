import Foundation
import CryptoKit
import Darwin
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3VideoVAEDecodeSessionTests: XCTestCase {
  func testInvalidCheckpointRejectedBeforeAnyDecoderTensorRead() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    let header = Data("{}".utf8)
    var length = UInt64(header.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }
    bytes.append(header)
    try bytes.write(to: url)
    XCTAssertThrowsError(try H3VideoVAEDecodeSession(checkpointURL: url)) { error in
      XCTAssertTrue(String(describing: error).contains("Missing H3 video VAE checkpoint metadata"))
    }
  }

  func testUnsupportedSpatialBatchFailsBeforeCheckpointAccess() throws {
    let latent = MLXArray.zeros([1, 7, 2, 2, 24], dtype: .float32)
    let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for batch in [0, 5] {
      XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL: missing,
        latent: latent, retainWeights: true, spatialBatchSize: batch,
        onChunk: { _ in XCTFail("Invalid batch must not decode or publish frames.") })) {
        XCTAssertTrue(String(describing: $0).contains("temporal decode geometry"))
      }
    }
  }
  private func installed() throws -> URL {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_H3_VIDEO_VAE_Q8"] else {
      throw XCTSkip("Set installed video VAE for decoder residency parity.")
    }
    return URL(fileURLWithPath: checkpoint)
  }

  private func tileLatent() -> MLXArray {
    MLXArray((0..<192).map { Float($0 % 29 - 14) / 9 }, [1, 2, 2, 2, 24])
  }

  private func processMemory() -> [String: UInt64] {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS else { return [:] }
    return ["currentPhysicalFootprintBytes": info.phys_footprint,
      "processLifetimePeakPhysicalFootprintBytes": UInt64(max(0, info.ledger_phys_footprint_peak))]
  }

  func testScopedSessionRejectsReplacedCheckpointWithoutReadingTensorPayload() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_HEADER_ONLY"] else {
      throw XCTSkip("Set installed video VAE header-only path for immutable-session admission.")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let alias = root.appendingPathComponent("checkpoint.safetensors")
    try FileManager.default.createSymbolicLink(at: alias,
      withDestinationURL: URL(fileURLWithPath: checkpoint))
    let session = try H3VideoVAEDecodeSession(checkpointURL: alias)
    defer { session.close() }
    try session.checkUnchanged()
    try FileManager.default.removeItem(at: alias)
    try Data("replaced".utf8).write(to: alias)
    XCTAssertThrowsError(try session.checkUnchanged())
    XCTAssertEqual(session.tensorLoads, 0)
    XCTAssertEqual(session.projectionLoads, 0)
    XCTAssertEqual(session.residentBytes, 0)
  }

  func testResidentTileMatchesLegacyAtEveryObservedBoundaryAndReusesWeights() throws {
    let checkpoint = try installed()
    let latent = tileLatent()
    var expected: [String: [Float]] = [:]
    _ = try H3VideoVAETileDecoder.decode(checkpointURL: checkpoint, latent: latent,
      observe: { name, value in expected[name] = value.asArray(Float.self) })
    Stream.gpu.synchronize(); Memory.clearCache()
    let activeBeforeSession = Memory.activeMemory
    let session = try H3VideoVAEDecodeSession(checkpointURL: checkpoint)
    defer { session.close() }
    for _ in 0..<2 {
      _ = try H3VideoVAETileDecoder.decode(checkpointURL: checkpoint, latent: latent,
        session: session, observe: { name, value in
          XCTAssertEqual(value.asArray(Float.self), expected[name], name)
        })
    }
    XCTAssertEqual(session.projectionLoads, 144)
    XCTAssertEqual(session.tensorLoads, 297)
    XCTAssertEqual(session.residentBytes, 2_582_138_032)
    session.close()
    XCTAssertEqual(session.residentBytes, 0)
    XCTAssertEqual(Memory.activeMemory, activeBeforeSession,
      "A closed session must release actual MLX arrays, including while the session object is retained.")
    XCTAssertTrue(session.isClosed)
    XCTAssertThrowsError(try session.read("post_quant_conv.bias", shape: [24]))
    session.close() // Release is idempotent, including a deferred second close.
  }

  func testThrowingTileObserverReleasesScopedSession() throws {
    let checkpoint = try installed()
    let latent = tileLatent()
    enum Stop: Error { case requested }
    // Deferred modes can still have an unevaluated block output at this
    // boundary. Error and cancellation must release its graph and weights.
    for memoryMode: H3VideoDecodeMemoryMode? in [nil, .normal, .lowMemoryBF16] {
      var retained: H3VideoVAEDecodeSession?
      let previousCacheLimit = Memory.cacheLimit
      XCTAssertThrowsError(try H3VideoVAEDecodeSession.withSession(
        checkpointURL: checkpoint, memoryMode: memoryMode) { session in
        retained = session
        return try H3VideoVAETileDecoder.decode(checkpointURL: checkpoint, latent: latent,
          session: session, observe: { name, _ in
            if name == "vit0" { throw Stop.requested }
          })
      }) { XCTAssertTrue($0 is Stop) }
      XCTAssertTrue(try XCTUnwrap(retained).isClosed)
      XCTAssertEqual(retained?.residentBytes, 0)
      XCTAssertEqual(retained?.projectionLoads, 4)
      XCTAssertEqual(Memory.cacheLimit, previousCacheLimit)
      XCTAssertThrowsError(try H3VideoVAEDecodeSession.withSession(
        checkpointURL: checkpoint, memoryMode: memoryMode) { session in
        retained = session
        return try H3VideoVAETileDecoder.decode(checkpointURL: checkpoint, latent: latent,
          session: session, observe: { name, _ in
            if name == "vit0" { throw CancellationError() }
          })
      }) { XCTAssertTrue($0 is CancellationError) }
      XCTAssertTrue(try XCTUnwrap(retained).isClosed)
      XCTAssertEqual(retained?.residentBytes, 0)
      XCTAssertEqual(Memory.cacheLimit, previousCacheLimit)
    }
  }

  /// Decode-only qualification: same immutable saved sampler output, no text
  /// encoder, denoiser, audio VAE or model generation in either measured pass.
  func testSavedActualLatentFullDecodeMatchesLegacyPixelsAndMeasuresStage() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_H3_VIDEO_VAE_Q8"],
      let rawPath = env["WEETODD_H3_VIDEO_VAE_SAVED_LATENT"],
      let expectedSHA = env["WEETODD_H3_VIDEO_VAE_SAVED_LATENT_SHA256"] else {
      throw XCTSkip("Set pinned actual raw video latents for full decoder-only parity and timing.")
    }
    let rawURL = URL(fileURLWithPath: rawPath)
    let size = try rawURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard (1..<(8 * 1024 * 1024)).contains(size) else {
      throw H3CheckpointError.invalid("Saved raw latent exceeds the pinned fixture byte bound.")
    }
    let bytes = try Data(contentsOf: rawURL)
    XCTAssertLessThan(bytes.count, 8 * 1024 * 1024)
    let actualSHA = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    guard actualSHA == expectedSHA else { throw H3CheckpointError.invalid("Saved raw latent SHA256 differs.") }
    let rawFile = try SafeTensorFile(url: rawURL)
    let descriptor = try XCTUnwrap(rawFile.tensors["video_latents"])
    guard descriptor.dtype == "F32", descriptor.shape == [1, 24, 37, 28, 48] else {
      throw H3CheckpointError.invalid("Saved raw video latent layout differs.")
    }
    let raw = try rawFile.withTensorBytes(named: "video_latents") {
      MLXArray($0, [1, 24, 37, 28, 48], type: Float.self)
    }
    let layout = try H3VideoVAELayout(url: URL(fileURLWithPath: checkpoint))
    // Repack normalized channel-first raw latents into the worker's 2x2
    // channel-major rows, then use its exact denormalization entry point.
    let rows = raw.transposed(0, 2, 3, 4, 1)
      .reshaped([1, 37, 14, 2, 24, 2, 24])
      .transposed(0, 1, 2, 4, 6, 3, 5).reshaped([1, 12_432, 96])
    let latent = try H3LatentCodec.videoDecoderInput(rows: rows,
      latentFrames: 37, latentHeight: 28, latentWidth: 48,
      mean: layout.latentsMean, standardDeviation: layout.latentsStandardDeviation)
    var evidence: [[String: Any]] = []
    var expected: [String] = []
    var expectedRGB: [String] = []
    for resident in [false, true] {
      var shapes: [[Int]] = []
      var digests: [String] = []
      var rgbDigests: [String] = []
      var frames = 0
      var hostValidationSeconds = 0.0
      var closed: H3VideoVAEDecodeSession.Statistics?
      var rawOutput: FileHandle?
      if let directory = env["WEETODD_H3_VIDEO_VAE_DECODE_MEDIA_DIRECTORY"] {
        let root = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let output = root.appendingPathComponent(resident ? "resident.rgb24" : "legacy.rgb24")
        guard !FileManager.default.fileExists(atPath: output.path),
          FileManager.default.createFile(atPath: output.path, contents: nil) else {
          throw H3CheckpointError.invalid("Decoder qualification media output already exists.")
        }
        rawOutput = try FileHandle(forWritingTo: output)
      }
      defer { try? rawOutput?.close() }
      Stream.gpu.synchronize(); Memory.clearCache()
      let before = Memory.activeMemory
      let processBefore = processMemory()
      Memory.peakMemory = before
      let started = ProcessInfo.processInfo.systemUptime
      try H3VideoVAEDecoder.decodeChunks(checkpointURL: URL(fileURLWithPath: checkpoint),
        latent: latent, retainWeights: resident, onSessionClosed: { closed = $0 }) { chunk in
        let hostStarted = ProcessInfo.processInfo.systemUptime
        let values = chunk.asArray(Float.self)
        let digest = values.withUnsafeBytes {
          SHA256.hash(data: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: $0.baseAddress!),
            count: $0.count, deallocator: .none))
        }
          .map { String(format: "%02x", $0) }.joined()
        shapes.append(chunk.shape)
        digests.append(digest)
        let rgb = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
        try rgb.withUnsafeBytes { buffer in
          let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
            count: buffer.count, deallocator: .none)
          rgbDigests.append(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
          try rawOutput?.write(contentsOf: data)
        }
        frames += chunk.shape[1]
        hostValidationSeconds += ProcessInfo.processInfo.systemUptime - hostStarted
      }
      let seconds = ProcessInfo.processInfo.systemUptime - started
      XCTAssertEqual(frames, 124)
      XCTAssertTrue(shapes.allSatisfy { $0[0] == 1 && Array($0.suffix(3)) == [448, 768, 3] })
      if resident {
        XCTAssertEqual(digests, expected, "Every streamed float32 pixel must match the legacy decoder.")
        XCTAssertEqual(rgbDigests, expectedRGB)
        let stats = try XCTUnwrap(closed)
        XCTAssertEqual(stats.projectionLoads, 144)
        XCTAssertEqual(stats.tensorLoads, 297)
        XCTAssertEqual(stats.maximumResidentBytes, 2_582_138_032)
        XCTAssertTrue(stats.closed)
        XCTAssertEqual(stats.remainingResidentBytes, 0)
      } else { expected = digests; expectedRGB = rgbDigests }
      evidence.append(["resident": resident, "seconds": seconds,
        "hostValidationSeconds": hostValidationSeconds,
        "frames": frames, "chunkShapes": shapes, "float32ChunkSHA256": digests,
        "rgb8ChunkSHA256": rgbDigests,
        "mlxActiveBeforeBytes": before, "mlxActiveAfterBytes": Memory.activeMemory,
        "mlxPeakBytes": Memory.peakMemory,
        "processBefore": processBefore, "processAfter": processMemory(),
        "residentDecoderBytes": closed?.maximumResidentBytes ?? 0])
    }
    try rawFile.checkUnchanged(at: rawURL)
    if let output = env["WEETODD_H3_VIDEO_VAE_DECODE_EVIDENCE"] {
      let object: [String: Any] = ["scope": "video-decoder-only",
        "rawLatentSHA256": actualSHA, "spatialTileBatch": 4,
        "generationExecuted": false, "audioDecoded": false, "passes": evidence]
      try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: output), options: .atomic)
    }
  }

  func testSavedActualLatentBatchOneMatchesFrozenBatchFour() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_H3_VIDEO_VAE_Q8"],
      let rawPath = env["WEETODD_H3_VIDEO_VAE_SAVED_LATENT"],
      let rawSHA = env["WEETODD_H3_VIDEO_VAE_SAVED_LATENT_SHA256"],
      let baselinePath = env["WEETODD_H3_VIDEO_VAE_BATCH1_BASELINE"],
      let baselineSHA = env["WEETODD_H3_VIDEO_VAE_BATCH1_BASELINE_SHA256"],
      let outputPath = env["WEETODD_H3_VIDEO_VAE_BATCH1_EVIDENCE"] else {
      throw XCTSkip("Set pinned saved latents and completed batch4 evidence for explicit batch1 trial.")
    }
    func sha(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    let baselineData = try Data(contentsOf: URL(fileURLWithPath: baselinePath))
    guard sha(baselineData) == baselineSHA,
      let baseline = try JSONSerialization.jsonObject(with: baselineData) as? [String: Any],
      baseline["spatialTileBatch"] as? Int == 4,
      baseline["rawLatentSHA256"] as? String == rawSHA,
      let passes = baseline["passes"] as? [[String: Any]],
      let reference = passes.first(where: { $0["resident"] as? Bool == true }),
      let expectedFloat = reference["float32ChunkSHA256"] as? [String],
      let expectedRGB = reference["rgb8ChunkSHA256"] as? [String] else {
      throw H3CheckpointError.invalid("Frozen batch4 decoder evidence differs.")
    }
    let rawURL = URL(fileURLWithPath: rawPath)
    let size = try rawURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard (1..<(8 * 1024 * 1024)).contains(size), sha(try Data(contentsOf: rawURL)) == rawSHA else {
      throw H3CheckpointError.invalid("Saved batch1 trial latent differs.")
    }
    let file = try SafeTensorFile(url: rawURL)
    guard file.tensors["video_latents"]?.dtype == "F32",
      file.tensors["video_latents"]?.shape == [1, 24, 37, 28, 48] else {
      throw H3CheckpointError.invalid("Batch1 trial requires the qualified124-frame raw latent.")
    }
    let raw = try file.withTensorBytes(named: "video_latents") {
      MLXArray($0, [1, 24, 37, 28, 48], type: Float.self)
    }
    let layout = try H3VideoVAELayout(url: URL(fileURLWithPath: checkpoint))
    let rows = raw.transposed(0, 2, 3, 4, 1).reshaped([1, 37, 14, 2, 24, 2, 24])
      .transposed(0, 1, 2, 4, 6, 3, 5).reshaped([1, 12_432, 96])
    let latent = try H3LatentCodec.videoDecoderInput(rows: rows,
      latentFrames: 37, latentHeight: 28, latentWidth: 48,
      mean: layout.latentsMean, standardDeviation: layout.latentsStandardDeviation)
    var floatHashes: [String] = [], rgbHashes: [String] = []
    var frames = 0, hostSeconds = 0.0
    var closed: H3VideoVAEDecodeSession.Statistics?
    let outputURL = URL(fileURLWithPath: outputPath)
    let rgbURL = outputURL.deletingPathExtension().appendingPathExtension("rgb24")
    guard !FileManager.default.fileExists(atPath: outputURL.path),
      !FileManager.default.fileExists(atPath: rgbURL.path),
      FileManager.default.createFile(atPath: rgbURL.path, contents: nil) else {
      throw H3CheckpointError.invalid("Batch1 trial output must be fresh.")
    }
    let output = try FileHandle(forWritingTo: rgbURL)
    defer { try? output.close() }
    Stream.gpu.synchronize(); Memory.clearCache()
    let before = Memory.activeMemory, processBefore = processMemory()
    Memory.peakMemory = before
    let started = ProcessInfo.processInfo.systemUptime
    try H3VideoVAEDecoder.decodeChunks(checkpointURL: URL(fileURLWithPath: checkpoint),
      latent: latent, retainWeights: true, spatialBatchSize: 1,
      onSessionClosed: { closed = $0 }) { chunk in
      let hostStarted = ProcessInfo.processInfo.systemUptime
      let values = chunk.asArray(Float.self)
      floatHashes.append(values.withUnsafeBytes { sha(Data(bytesNoCopy:
        UnsafeMutableRawPointer(mutating: $0.baseAddress!), count: $0.count, deallocator: .none)) })
      let rgb = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
      try rgb.withUnsafeBytes {
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: $0.baseAddress!),
          count: $0.count, deallocator: .none)
        rgbHashes.append(sha(data))
        try output.write(contentsOf: data)
      }
      frames += chunk.shape[1]
      hostSeconds += ProcessInfo.processInfo.systemUptime - hostStarted
    }
    let seconds = ProcessInfo.processInfo.systemUptime - started
    let stats = try XCTUnwrap(closed)
    XCTAssertEqual(frames, 124)
    XCTAssertEqual(floatHashes, expectedFloat)
    XCTAssertEqual(rgbHashes, expectedRGB)
    XCTAssertEqual(stats.projectionLoads, 144)
    XCTAssertEqual(stats.tensorLoads, 297)
    XCTAssertTrue(stats.closed)
    XCTAssertEqual(stats.remainingResidentBytes, 0)
    XCTAssertEqual(Memory.activeMemory, before)
    try file.checkUnchanged(at: rawURL)
    let result: [String: Any] = ["scope": "video-decoder-only-batch1-trial",
      "productionBatchUnchanged": 4, "spatialTileBatch": 1, "frames": frames,
      "generationExecuted": false, "audioDecoded": false,
      "rawLatentSHA256": rawSHA, "baselineEvidenceSHA256": baselineSHA,
      "exactFloat32Chunks": floatHashes == expectedFloat, "exactRGB8Chunks": rgbHashes == expectedRGB,
      "float32ChunkSHA256": floatHashes, "rgb8ChunkSHA256": rgbHashes,
      "seconds": seconds, "hostValidationSeconds": hostSeconds,
      "residentDecoderBytes": stats.maximumResidentBytes,
      "mlxPeakBytes": Memory.peakMemory, "mlxActiveBeforeBytes": before,
      "mlxActiveAfterBytes": Memory.activeMemory,
      "processBefore": processBefore, "processAfter": processMemory()]
    try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
      .write(to: outputURL, options: .atomic)
  }
}
