import Foundation
import CryptoKit
import Darwin
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3VideoVAEDecodeSessionTests: XCTestCase {
  func testDecoderPrecisionAdmissionAndDefaultAreIndependentOfMemoryMode() throws {
    XCTAssertEqual(H3VideoDecodePrecision.defaultPrecision,.float32)
    XCTAssertNil(H3VideoDecodePrecision(rawValue:"bfloat16"))
    try Device.withDefaultDevice(.cpu) {
      let input = MLXArray([Float(-65_504),0,65_504])
      XCTAssertEqual(try H3VideoDecodePrecision.float32.transformerInput(input).dtype,.float32)
      XCTAssertEqual(try H3VideoDecodePrecision.float16.transformerInput(input).dtype,.float16)
      for value: Float in [65_505,-65_505,.infinity,-.infinity,.nan] {
        XCTAssertThrowsError(try H3VideoDecodePrecision.float16.transformerInput(MLXArray([value])))
      }
      XCTAssertThrowsError(try H3VideoDecodePrecision.float16.validatePixels(MLXArray([Float.infinity])))
      XCTAssertThrowsError(try H3VideoDecodePrecision.float16.validatePixels(MLXArray([Float.nan])))
    }
  }

  func testFP16DecoderOverflowFailsBeforeCheckpointAccess() throws {
    try Device.withDefaultDevice(.cpu) {
      let unavailable = URL(fileURLWithPath:"/nonexistent/weetodd-fp16-precision.safetensors")
      let latent = MLXArray.full([1,7,2,2,24],values:MLXArray(Float(65_505)),dtype:.float32)
      XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL:unavailable,
        latent:latent,retainWeights:true,memoryMode:.lowMemoryBF16,precision:.float16,
        onChunk:{ _ in XCTFail("Overflow cannot publish a chunk.") })) {
        XCTAssertEqual($0 as? H3CheckpointError,
          .invalid("FP16 H3 video input is nonfinite or exceeds the finite half range."))
      }
    }
  }

  func testFP16DecoderCancellationPrecedesPrecisionReadsAndCheckpointAccess() async throws {
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        try H3VideoDecodePrecision.float16.validateHalfCastInput(MLXArray(Float(1)))
        return false
      } catch is CancellationError { return true }
      catch { return false }
    }
    let cancelled = await task.value
    XCTAssertTrue(cancelled)
  }


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

  private func fakeGrid(batch: Int, depth: Int, height: Int, width: Int,
    dtype: DType) -> H3VideoVAEDecodeSession.Grid {
    let rows = depth * height * width + 5
    return H3VideoVAEDecodeSession.Grid(
      positions: MLXArray.zeros([batch, rows, 3], dtype: .float32),
      rotary: H3VideoVAERotary(
        cosine: MLXArray.ones([batch, rows, 1, 48], dtype: dtype),
      sine: MLXArray.zeros([batch, rows, 1, 48], dtype: dtype)))
  }

  func testAdaptiveNormalBoundariesCheckCancellationForEveryActualBatch() async throws {
    let checkpoint = try installed()
    for batch in 1...4 {
      for firstResidual in [true,false] {
        let task = Task { () throws -> Bool in
          try Device.withDefaultDevice(.cpu) {
            let value = MLXArray.zeros([batch,2,8],dtype:.float32)
            var retained: H3VideoVAEDecodeSession?
            var cancelled = false
            do {
              try H3VideoVAEDecodeSession.withSession(checkpointURL:checkpoint,memoryMode:.normal) { session in
                retained = session
                withUnsafeCurrentTask { $0?.cancel() }
                if firstResidual { try session.materializeFirstResidual(value,blockIndex:0) }
                else { try session.materializeBlockOutput(value,blockIndex:0) }
              }
            } catch is CancellationError { cancelled = true }
            return cancelled && retained?.isClosed == true && retained?.residentBytes == 0
              && retained?.gridCacheBytes == 0 && retained?.projectionLoads == 0
          }
        }
        let released = try await task.value
        XCTAssertTrue(released,"Batch\(batch) must cancel before either eager or lazy boundary and close its session.")
      }
    }
  }

  func testGridCacheReusesExactKeyAndCloseDropsRetainedArrays() throws {
    let checkpoint = try installed()
    try Device.withDefaultDevice(.cpu) {
      let session = try H3VideoVAEDecodeSession(checkpointURL: checkpoint, memoryMode: .lowMemoryBF16)
      defer { session.close() }
      weak var weakPositions: MLXArray?
      weak var weakCosine: MLXArray?
      weak var weakSine: MLXArray?
      func populateAndCheck() throws {
        var builds = 0
        func build() -> H3VideoVAEDecodeSession.Grid {
          builds += 1
          return fakeGrid(batch: 1, depth: 7, height: 16, width: 16, dtype: .float32)
        }
        let first = try session.preparedGrid(batch: 1, depth: 7, height: 16,
          width: 16, dtype: .float32, build: build)
        let second = try session.preparedGrid(batch: 1, depth: 7, height: 16,
          width: 16, dtype: .float32, build: build)
        XCTAssertTrue(first.positions === second.positions)
        XCTAssertTrue(first.rotary.cosine === second.rotary.cosine)
        XCTAssertTrue(first.rotary.sine === second.rotary.sine)
        XCTAssertEqual(builds, 1)
        XCTAssertEqual(session.gridCacheEntries, 1)
        XCTAssertEqual(session.gridCacheBytes, first.storageBytes)
        XCTAssertEqual(session.residentBytes, first.storageBytes)
        weakPositions = first.positions; weakCosine = first.rotary.cosine; weakSine = first.rotary.sine
      }
      try populateAndCheck()
      XCTAssertNotNil(weakPositions); XCTAssertNotNil(weakCosine); XCTAssertNotNil(weakSine)
      session.close(); session.close()
      XCTAssertEqual(session.gridCacheBytes, 0)
      XCTAssertEqual(session.gridCacheEntries, 0)
      XCTAssertEqual(session.residentBytes, 0)
      XCTAssertNil(weakPositions); XCTAssertNil(weakCosine); XCTAssertNil(weakSine)
      XCTAssertThrowsError(try session.preparedGrid(batch: 1, depth: 7, height: 16,
        width: 16, dtype: .float32, build: { XCTFail("Closed session cannot build a grid"); return self.fakeGrid(
          batch: 1, depth: 7, height: 16, width: 16, dtype: .float32) }))
    }
  }

  func testGridCacheBoundFallsBackWithoutEvictionAndPrecisionIsPartOfKey() throws {
    let checkpoint = try installed()
    try Device.withDefaultDevice(.cpu) {
      let session = try H3VideoVAEDecodeSession(checkpointURL: checkpoint)
      defer { session.close() }
      let first = try session.preparedGrid(batch: 4, depth: 16, height: 16, width: 16,
        dtype: .float32, build: { self.fakeGrid(batch: 4, depth: 16, height: 16, width: 16, dtype: .float32) })
      let retained = session.gridCacheBytes
      var builds = 0
      func fallback() -> H3VideoVAEDecodeSession.Grid {
        builds += 1
        return fakeGrid(batch: 4, depth: 16, height: 16, width: 16, dtype: .float16)
      }
      let a = try session.preparedGrid(batch: 4, depth: 16, height: 16, width: 16,
        dtype: .float16, build: fallback)
      let b = try session.preparedGrid(batch: 4, depth: 16, height: 16, width: 16,
        dtype: .float16, build: fallback)
      XCTAssertEqual(builds, 2)
      XCTAssertFalse(a.rotary.cosine === b.rotary.cosine)
      XCTAssertEqual(session.gridCacheBytes, retained)
      XCTAssertEqual(session.gridCacheEntries, 1)
      XCTAssertLessThanOrEqual(retained, H3VideoVAEDecodeSession.maximumGridCacheBytes)
      let hit = try session.preparedGrid(batch: 4, depth: 16, height: 16, width: 16,
        dtype: .float32, build: { XCTFail("Original key must remain cached"); return first })
      XCTAssertTrue(hit.rotary.cosine === first.rotary.cosine)
      XCTAssertThrowsError(try session.preparedGrid(batch: 0, depth: 16, height: 16, width: 16,
        dtype: .float32, build: { XCTFail("Invalid geometry cannot build"); return first }))
      XCTAssertThrowsError(try session.preparedGrid(batch: 1, depth: 7, height: 16, width: 16,
        dtype: .float32, build: { first }))
      XCTAssertEqual(session.gridCacheBytes, retained)
    }
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
    XCTAssertGreaterThan(session.gridCacheBytes, 0)
    XCTAssertEqual(session.gridCacheEntries, 1)
    XCTAssertEqual(session.residentBytes, 2_582_138_032 + session.gridCacheBytes)
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
      XCTAssertEqual(retained?.gridCacheBytes, 0)
      XCTAssertEqual(retained?.gridCacheEntries, 0)
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
      XCTAssertEqual(retained?.gridCacheBytes, 0)
      XCTAssertEqual(retained?.gridCacheEntries, 0)
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
        XCTAssertGreaterThan(stats.maximumGridCacheBytes, 0)
        XCTAssertLessThanOrEqual(stats.maximumGridCacheBytes, H3VideoVAEDecodeSession.maximumGridCacheBytes)
        XCTAssertEqual(stats.maximumResidentBytes, 2_582_138_032 + stats.maximumGridCacheBytes)
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

  func testAllocationCachePolicyRejectsInvalidOptionsBeforeCheckpointAccess() throws {
    let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let before = Memory.cacheLimit
    for mode: H3VideoDecodeMemoryMode? in [nil,.normal,.lowMemoryBF16] {
      XCTAssertNoThrow(try H3VideoVAEDecodeSession.validateAllocationCacheLimit(128 * 1024 * 1024,
        memoryMode:mode))
      for bytes in [-1,0,64 * 1024 * 1024,256 * 1024 * 1024,Int.max] {
        XCTAssertThrowsError(try H3VideoVAEDecodeSession(checkpointURL:missing,
          memoryMode:mode,allocationCacheLimitBytes:bytes)) {
          XCTAssertTrue(String(describing:$0).contains("allocation cache policy"))
        }
      }
    }
    for mode: H3VideoDecodeMemoryMode? in [nil,.normal] {
      XCTAssertThrowsError(try H3VideoVAEDecodeSession(checkpointURL:missing,
        memoryMode:mode,allocationCacheLimitBytes:512 * 1024 * 1024)) {
        XCTAssertTrue(String(describing:$0).contains("allocation cache policy"))
      }
    }
    XCTAssertNoThrow(try H3VideoVAEDecodeSession.validateAllocationCacheLimit(512 * 1024 * 1024,
      memoryMode:.lowMemoryBF16))
    XCTAssertEqual(Memory.cacheLimit,before)
    try Device.withDefaultDevice(.cpu) {
      let input = MLXArray.zeros([1,7,2,2,24],dtype:.float32)
      XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL:missing,
        latent:input,retainWeights:true,memoryMode:.normal,
        allocationCacheLimitBytes:512 * 1024 * 1024,
        onChunk:{ _ in XCTFail("Invalid cache cannot publish frames.") })) {
        XCTAssertTrue(String(describing:$0).contains("allocation cache policy"))
      }
      XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL:missing,
        latent:input,retainWeights:false,groupedStageLoading:true,
        onChunk:{ _ in XCTFail("Grouped nonresident decode cannot publish frames.") })) {
        XCTAssertTrue(String(describing:$0).contains("Grouped H3 video loading requires"))
      }
      XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL:missing,
        latent:input,retainWeights:true,memoryMode:.lowMemoryBF16,
        allocationCacheLimitBytes:512 * 1024 * 1024,groupedStageLoading:true,
        onChunk:{ _ in XCTFail("Grouped cache mismatch cannot publish frames.") })) {
        XCTAssertTrue(String(describing:$0).contains("Grouped H3 video loading requires"))
      }
      XCTAssertEqual(Memory.cacheLimit,before)
    }
  }

  func testExperimentalLowCacheRestoresScopeAfterFailureAndCancellationBeforePayloadRead() throws {
    let checkpoint = try installed()
    enum Stop: Error { case requested }
    for cancel in [false,true] {
      let before = Memory.cacheLimit
      var owner: H3VideoVAEDecodeSession?
      XCTAssertThrowsError(try H3VideoVAEDecodeSession.withSession(checkpointURL:checkpoint,
        memoryMode:.lowMemoryBF16,allocationCacheLimitBytes:512 * 1024 * 1024) { session -> Void in
        owner = session
        XCTAssertEqual(Memory.cacheLimit,512 * 1024 * 1024)
        if cancel { throw CancellationError() }
        throw Stop.requested
      }) { if cancel { XCTAssertTrue($0 is CancellationError) } else { XCTAssertTrue($0 is Stop) } }
      XCTAssertTrue(try XCTUnwrap(owner).isClosed)
      XCTAssertEqual(owner?.residentBytes,0)
      XCTAssertEqual(owner?.projectionLoads,0)
      XCTAssertEqual(owner?.tensorLoads,0)
      XCTAssertEqual(Memory.cacheLimit,before)
      owner?.close()
      XCTAssertEqual(Memory.cacheLimit,before)
    }
  }

  private struct AllocationCacheFixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    let checkpoint: String
    let checkpointIdentity: [Int64]
    let checkpointHeaderSHA256: String
    let latentPath: String
    let latentSHA256: String
    let inputProvenance: String
    let metalLibrary: String
    let metalSHA256: String
    let testExecutable: String
    let testExecutableSHA256: String
    let sourcePins: [Pin]
    let output: String
    let maximumPeakMLXBytes: Int?
    let maximumPhysicalPeakBytes: UInt64?
    let experiment: String?
    let memoryMode: String?
  }

  func testGroupedStagePlanAdmitsOnlyExactDecoderArraysBeforePayloadWork() throws {
    let groups = H3VideoVAEDecodeSession.groupedStagePlan()
    XCTAssertEqual(groups.count,37); XCTAssertEqual(groups[0].count,9)
    XCTAssertTrue(groups.dropFirst().allSatisfy { $0.count == 20 })
    let requests = groups.flatMap { $0 }
    XCTAssertEqual(requests.count,729)
    XCTAssertEqual(Set(requests.map(\.name)).count,729)
    XCTAssertTrue(requests.allSatisfy { $0.name.hasPrefix("decoder.") || $0.name.hasPrefix("post_quant_conv.") })
    XCTAssertEqual(groups[0].reduce(0) { $0 + $1.byteCount },12_717_232)
    XCTAssertTrue(groups.dropFirst().allSatisfy { $0.reduce(0) { $0 + $1.byteCount } == 71_372_800 })
    var info = Dictionary(uniqueKeysWithValues:requests.map {
      ($0.name,H3TensorInfo(dtype:$0.dtype,shape:$0.shape.map(UInt64.init)))
    })
    XCTAssertEqual(try H3VideoVAEDecodeSession.validateGroupedStageAdmission { info[$0] },2_582_138_032)
    let last = try XCTUnwrap(requests.last)
    info.removeValue(forKey:last.name)
    XCTAssertThrowsError(try H3VideoVAEDecodeSession.validateGroupedStageAdmission { info[$0] })
    info[last.name] = H3TensorInfo(dtype:"BF16",shape:last.shape.map(UInt64.init))
    XCTAssertThrowsError(try H3VideoVAEDecodeSession.validateGroupedStageAdmission { info[$0] })
    info[last.name] = H3TensorInfo(dtype:last.dtype,shape:[1])
    XCTAssertThrowsError(try H3VideoVAEDecodeSession.validateGroupedStageAdmission { info[$0] })
  }

  func testGroupedStageAdmissionCancellationPrecedesDescriptorReads() async throws {
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try H3VideoVAEDecodeSession.validateGroupedStageAdmission { _ in
          XCTFail("Cancelled preparation cannot read even a descriptor.")
          return nil
        }
        return false
      } catch is CancellationError { return true }
      catch { return false }
    }
    let cancelled = await task.value
    XCTAssertTrue(cancelled)
  }

  func testInstalledGroupedStageBodyFailureAndCancellationReleaseResidentArrays() throws {
    let checkpoint = try installed()
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_CACHE_FIXTURE"] else {
      throw XCTSkip("Set the pinned video fixture for grouped-stage GPU lifecycle qualification.")
    }
    let fixture = try JSONDecoder().decode(AllocationCacheFixture.self,
      from:Data(contentsOf:URL(fileURLWithPath:manifest)))
    guard let compiled = Bundle(for:Self.self).executableURL else {
      throw H3CheckpointError.invalid("Missing grouped video lifecycle test executable.")
    }
    let executable = compiled.resolvingSymlinksInPath()
    let metal = URL(fileURLWithPath:fixture.metalLibrary)
    _ = try cacheFileIdentity(metal)
    guard checkpoint.standardizedFileURL == URL(fileURLWithPath:fixture.checkpoint).standardizedFileURL,
      executable == URL(fileURLWithPath:fixture.testExecutable).resolvingSymlinksInPath(),
      try cacheFileDigest(executable) == fixture.testExecutableSHA256,
      try cacheFileIdentity(checkpoint) == fixture.checkpointIdentity,
      try cacheHeaderDigest(checkpoint) == fixture.checkpointHeaderSHA256,
      try cacheFileDigest(metal) == fixture.metalSHA256,
      Set(fixture.sourcePins.map { URL(fileURLWithPath:$0.path).lastPathComponent })
        .isSuperset(of:["H3VideoVAEDecodeSession.swift","H3VideoVAEDecoder.swift","H3VideoVAEDecodeSessionTests.swift"]) else {
      throw H3CheckpointError.invalid("Grouped video lifecycle fixture identity changed.")
    }
    for pin in fixture.sourcePins {
      let source = URL(fileURLWithPath:pin.path)
      _ = try cacheFileIdentity(source)
      guard try cacheFileDigest(source) == pin.sha256 else {
        throw H3CheckpointError.invalid("Grouped video lifecycle source changed.")
      }
    }
    GPU.metallib = metal
    let expectedFile = try SafeTensorFile(url:checkpoint)
    enum Stop: Error { case requested }
    for cancel in [false,true] {
      Stream.gpu.synchronize(); Memory.clearCache()
      let baseline = Memory.activeMemory,previousLimit = Memory.cacheLimit
      var owner: H3VideoVAEDecodeSession?
      XCTAssertThrowsError(try H3VideoVAEDecodeSession.withSession(checkpointURL:checkpoint,
        memoryMode:.lowMemoryBF16,groupedStageLoading:true) { session -> Void in
        owner = session
        XCTAssertEqual(session.groupedPreparationGroups,37)
        XCTAssertEqual(session.residentBytes,2_582_138_032)
        if !cancel {
          // Check all297 direct F16 arrays (15.2MB) without copying packed
          // projections or changing any decode activation or timer.
          for request in H3VideoVAEDecodeSession.groupedStagePlan().flatMap({ $0 })
            where request.dtype == "F16" && !request.name.hasSuffix(".scales")
              && !request.name.hasSuffix(".biases") {
            let words = try session.read(request.name,shape:request.shape)
              .view(dtype:.uint16).asArray(UInt16.self)
            let expected = try expectedFile.withTensorBytes(named:request.name) {
              Array($0.bindMemory(to:UInt16.self))
            }
            XCTAssertTrue(words == expected,request.name)
          }
        }
        if cancel { throw CancellationError() }
        throw Stop.requested
      }) { if cancel { XCTAssertTrue($0 is CancellationError) } else { XCTAssertTrue($0 is Stop) } }
      let retained = try XCTUnwrap(owner)
      XCTAssertTrue(retained.isClosed); XCTAssertEqual(retained.residentBytes,0)
      XCTAssertEqual(retained.projectionLoads,144); XCTAssertEqual(retained.tensorLoads,297)
      XCTAssertEqual(retained.gridCacheBytes,0)
      XCTAssertEqual(Memory.activeMemory,baseline,"Closed retained owner must release every prepared MLX array.")
      XCTAssertEqual(Memory.cacheMemory,0); XCTAssertEqual(Memory.cacheLimit,previousLimit)
      XCTAssertThrowsError(try retained.read("post_quant_conv.bias",shape:[24]))
      XCTAssertThrowsError(try retained.projection("decoder.transformer_blocks.0.attn.to_qkv.weight"))
      retained.close()
      XCTAssertEqual(Memory.activeMemory,baseline); XCTAssertEqual(Memory.cacheLimit,previousLimit)
    }
    try expectedFile.checkUnchanged(at:checkpoint)
  }

  private func cacheDigest(_ data: Data) -> String {
    SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
  }

  private func cacheFileDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom:url)
    defer { try? file.close() }
    var hasher = SHA256()
    while let bytes = try file.read(upToCount:4 * 1024 * 1024), !bytes.isEmpty { hasher.update(data:bytes) }
    return hasher.finalize().map { String(format:"%02x",$0) }.joined()
  }

  private func cacheFileIdentity(_ url: URL) throws -> [Int64] {
    var value = stat()
    guard url.path.withCString({ Darwin.lstat($0,&value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG,FileManager.default.isReadableFile(atPath:url.path) else {
      throw H3CheckpointError.invalid("Video cache fixture requires readable regular files.")
    }
    return [Int64(value.st_dev),Int64(value.st_ino),value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }

  private func cacheHeaderDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom:url)
    defer { try? file.close() }
    guard let prefix = try file.read(upToCount:8),prefix.count == 8 else {
      throw H3CheckpointError.invalid("Missing video cache fixture checkpoint header.")
    }
    let count = prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << ($1.offset * 8)) }
    guard count >= 2,count <= 64 * 1024 * 1024,
      let header = try file.read(upToCount:Int(count)),header.count == Int(count) else {
      throw H3CheckpointError.invalid("Invalid video cache fixture checkpoint header.")
    }
    return cacheDigest(header)
  }

  /// Allocation/component parity only. The fixture may declare deterministic
  /// input; it is not generation, media-quality approval or a whole-job forecast.
  func testInstalledLowVideoAllocationCacheClipABBA() throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_CACHE_FIXTURE"] else {
      throw XCTSkip("Set a pinned one-clip video cache fixture; default does not load weights.")
    }
    let manifestURL = URL(fileURLWithPath:manifest)
    let manifestData = try Data(contentsOf:manifestURL)
    let fixture = try JSONDecoder().decode(AllocationCacheFixture.self,from:manifestData)
    let groupedExperiment = fixture.experiment == "grouped_stage_loading"
    let precisionExperiment = fixture.experiment == "decoder_float16"
    guard let mode = H3VideoDecodeMemoryMode(rawValue:fixture.memoryMode ?? "low_memory_bf16"),
      fixture.experiment == nil || groupedExperiment || precisionExperiment,
      groupedExperiment || mode == .lowMemoryBF16 else {
      throw H3CheckpointError.invalid("Unsupported video decoder component experiment or memory mode.")
    }
    if precisionExperiment {
      guard mode == .lowMemoryBF16,
        ["actual_saved_decoder_input","cropped_saved_video_latents"].contains(fixture.inputProvenance),
        Set(fixture.sourcePins.map { URL(fileURLWithPath:$0.path).lastPathComponent })
          .isSuperset(of:["H3VideoDecodePrecision.swift","H3VideoVAETileDecoder.swift","H3VideoVAEBlock.swift"]) else {
        throw H3CheckpointError.invalid("FP16 quality comparison requires actual saved input, low-memory mode and every precision source pin.")
      }
    }
    let outputURL = URL(fileURLWithPath:fixture.output)
    guard !FileManager.default.fileExists(atPath:outputURL.path),
      FileManager.default.fileExists(atPath:outputURL.deletingLastPathComponent().path),
      ["actual_saved_decoder_input","deterministic_allocation_fixture","cropped_saved_video_latents"].contains(fixture.inputProvenance),
      Set(fixture.sourcePins.map { URL(fileURLWithPath:$0.path).lastPathComponent })
        .isSuperset(of:["H3VideoVAEDecodeSession.swift","H3VideoVAEDecoder.swift","H3VideoVAEDecodeSessionTests.swift"]) else {
      throw H3CheckpointError.invalid("Video cache fixture requires new output, input provenance and complete source pins.")
    }
    guard let compiled = Bundle(for:Self.self).executableURL else {
      throw H3CheckpointError.invalid("Missing video cache qualification test executable.")
    }
    let executable = compiled.resolvingSymlinksInPath()
    guard executable == URL(fileURLWithPath:fixture.testExecutable).resolvingSymlinksInPath(),
      try cacheFileDigest(executable) == fixture.testExecutableSHA256 else {
      throw H3CheckpointError.invalid("Video cache fixture test executable changed.")
    }
    let checkpoint = URL(fileURLWithPath:fixture.checkpoint)
    let latentURL = URL(fileURLWithPath:fixture.latentPath)
    let metal = URL(fileURLWithPath:fixture.metalLibrary)
    func verifyPins() throws {
      for pin in fixture.sourcePins {
        let url = URL(fileURLWithPath:pin.path)
        _ = try cacheFileIdentity(url)
        guard try cacheFileDigest(url) == pin.sha256 else {
          throw H3CheckpointError.invalid("Video cache fixture source changed: \(pin.path)")
        }
      }
      _ = try cacheFileIdentity(metal)
      let latentIdentity = try cacheFileIdentity(latentURL)
      guard (1..<(8 * 1024 * 1024)).contains(latentIdentity[2]),
        try cacheFileIdentity(checkpoint) == fixture.checkpointIdentity,
        try cacheHeaderDigest(checkpoint) == fixture.checkpointHeaderSHA256,
        try cacheFileDigest(metal) == fixture.metalSHA256,
        try cacheFileDigest(latentURL) == fixture.latentSHA256,
        try cacheFileDigest(executable) == fixture.testExecutableSHA256,
        try Data(contentsOf:manifestURL) == manifestData else {
        throw H3CheckpointError.invalid("Video cache fixture checkpoint/input/library/binary/manifest changed.")
      }
    }
    try verifyPins()
    GPU.metallib = metal
    let inputFile = try SafeTensorFile(url:latentURL)
    func makeInput() throws -> MLXArray {
      if fixture.inputProvenance == "cropped_saved_video_latents" {
        guard inputFile.tensors["video_latents"]?.dtype == "F32",
          inputFile.tensors["video_latents"]?.shape == [1,24,37,28,48] else {
          throw H3CheckpointError.invalid("Video cache saved raw fixture must match the retained124-frame layout.")
        }
        let raw = try inputFile.withTensorBytes(named:"video_latents") {
          MLXArray($0,[1,24,37,28,48],type:Float.self)
        }
        let layout = try H3VideoVAELayout(url:checkpoint)
        let rows = raw.transposed(0,2,3,4,1).reshaped([1,37,14,2,24,2,24])
          .transposed(0,1,2,4,6,3,5).reshaped([1,12_432,96])
        let decoded = try H3LatentCodec.videoDecoderInput(rows:rows,
          latentFrames:37,latentHeight:28,latentWidth:48,
          mean:layout.latentsMean,standardDeviation:layout.latentsStandardDeviation)
        let selected = contiguous(decoded[0..<1,0..<7,0..<24,0..<42,0..<24])
        eval(selected)
        return selected
      }
      guard inputFile.tensors["video_decoder_input"]?.dtype == "F32",
        inputFile.tensors["video_decoder_input"]?.shape == [1,7,24,42,24] else {
        throw H3CheckpointError.invalid("Video cache fixture requires F32[1,7,24,42,24] decoder input.")
      }
      return try inputFile.withTensorBytes(named:"video_decoder_input") {
        MLXArray($0,[1,7,24,42,24],type:Float.self)
      }
    }
    let input = try makeInput()
    eval(input)
    let inputValues = input.asArray(Float.self)
    let inputFloatSHA = inputValues.withUnsafeBytes { cacheDigest(Data($0)) }
    if precisionExperiment {
      XCTAssertEqual(inputFloatSHA,"fe6abbf1dbade44d982e1557a8c2d9392f8dfa86fff4b9a4176e099ed77046b4",
        "FP16 qualification must use the frozen actual22-frame decoder input.")
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    let inputBaseline = Memory.activeMemory
    let previousLimit = Memory.cacheLimit
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    var expectedFloat: [String] = [],expectedRGB: [String] = []
    var referenceFloat: [[Float]] = [],referenceRGB: [[UInt8]] = []
    var reports: [[String:Any]] = []
    let limits = (precisionExperiment ? [128,128,128,128,128,128]
      : groupedExperiment ? [128,128,128,128] : [128,512,512,128]).map { $0 * 1024 * 1024 }
    for (pass,limit) in limits.enumerated() {
      let grouped = precisionExperiment || (groupedExperiment && [1,2].contains(pass))
      // Warm both precision families before the four measured ABBA passes.
      let half = precisionExperiment && [1,3,4].contains(pass)
      let precision: H3VideoDecodePrecision = half ? .float16 : .float32
      let report: [String:Any] = try autoreleasepool {
        Stream.gpu.synchronize(); Memory.clearCache()
        XCTAssertEqual(Memory.activeMemory,inputBaseline)
        Memory.peakMemory = inputBaseline
        var chunks: [MLXArray] = []
        var closed: H3VideoVAEDecodeSession.Statistics?
        let started = ProcessInfo.processInfo.systemUptime
        try H3VideoVAEDecoder.decodeChunks(checkpointURL:checkpoint,latent:input,
          retainWeights:true,memoryMode:mode,precision:precision,
          allocationCacheLimitBytes:limit,groupedStageLoading:grouped,
          onSessionClosed:{ closed = $0 },
          onChunk:{ chunks.append($0) })
        eval(chunks); Stream.gpu.synchronize()
        let seconds = ProcessInfo.processInfo.systemUptime - started
        // Capture decode peak before host validation creates RGB/host buffers.
        let peak = Memory.peakMemory
        let physical = processMemory()
        let cacheAfterClose = Memory.cacheMemory
        let stats = try XCTUnwrap(closed)
        XCTAssertTrue(stats.closed); XCTAssertEqual(stats.remainingResidentBytes,0)
        XCTAssertEqual(stats.computePrecision,precision.rawValue)
        XCTAssertEqual(stats.allocationCacheLimitBytes,limit)
        XCTAssertEqual(stats.groupedPreparationGroups,grouped ? 37 : 0)
        XCTAssertEqual(stats.projectionLoads,144); XCTAssertEqual(stats.tensorLoads,297)
        XCTAssertEqual(stats.maximumResidentBytes,2_582_138_032 + stats.maximumGridCacheBytes)
        XCTAssertEqual(Memory.cacheMemory,0); XCTAssertEqual(Memory.cacheLimit,previousLimit)
        XCTAssertEqual(chunks.map { $0.shape[1] },[17,5])
        XCTAssertTrue(chunks.allSatisfy { $0.shape == [1,$0.shape[1],384,672,3] && $0.dtype == .float32 })
        let validationStart = ProcessInfo.processInfo.systemUptime
        var floatHashes: [String] = [],rgbHashes: [String] = []
        var floatSSE = 0.0,rgbSSE = 0.0,maxFloatDifference = 0.0
        var maxRGBDifference = 0,comparedSamples = 0,finite = true
        for (chunkIndex,chunk) in chunks.enumerated() {
          let values = chunk.asArray(Float.self)
          floatHashes.append(values.withUnsafeBytes { cacheDigest(Data($0)) })
          let rgb = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
          rgbHashes.append(rgb.withUnsafeBytes { cacheDigest(Data($0)) })
          if precisionExperiment {
            finite = finite && values.allSatisfy(\.isFinite)
            if pass == 0 { referenceFloat.append(values);referenceRGB.append(rgb) }
            XCTAssertEqual(values.count,referenceFloat[chunkIndex].count)
            XCTAssertEqual(rgb.count,referenceRGB[chunkIndex].count)
            for i in values.indices {
              let delta = Double(values[i])-Double(referenceFloat[chunkIndex][i])
              floatSSE += delta*delta;maxFloatDifference = max(maxFloatDifference,abs(delta))
              let difference = abs(Int(rgb[i])-Int(referenceRGB[chunkIndex][i]))
              rgbSSE += Double(difference*difference);maxRGBDifference = max(maxRGBDifference,difference)
            }
            comparedSamples += values.count
          }
        }
        if pass == 0 {
          expectedFloat = floatHashes; expectedRGB = rgbHashes
          if precisionExperiment {
            XCTAssertEqual(floatHashes,[
              "f4ab52275d0c952b9f0c73840a4a5a08b99d44fdc3e11878ce35ffae114b498f",
              "04b083d1694057a7a8df2872dcc9cc1bd5dbbec2c1fc0c25d789efea1e3633ef"])
            XCTAssertEqual(rgbHashes,[
              "6a83c4d7c04f02008ee86e374ce6b7957a7cf938bf20ca245fef1eb048d78a1a",
              "121ee428f09d611a5fb92757ed4ce931903c4fcc2659eca05989ce110f828e89"])
          }
        }
        if precisionExperiment {
          XCTAssertTrue(finite,"Precision comparison requires every Float32 pixel finite.")
          if !half {
            XCTAssertEqual(floatHashes,expectedFloat,"All FP32 reference passes must remain exact.")
            XCTAssertEqual(rgbHashes,expectedRGB)
          }
        } else {
          XCTAssertEqual(floatHashes,expectedFloat,"Every full Float32 chunk must match default128MiB.")
          XCTAssertEqual(rgbHashes,expectedRGB,"Every RGB byte must match default128MiB.")
        }
        let rgbMSE = comparedSamples > 0 ? rgbSSE/Double(comparedSamples) : 0
        let quality: [String:Any] = ["float32Finite":finite,"samples":comparedSamples,
          "float32MSE":comparedSamples > 0 ? floatSSE/Double(comparedSamples) : 0,
          "maxFloat32PixelDifference":maxFloatDifference,
          "rgb8MSE":rgbMSE,"rgb8NormalizedMSE":rgbMSE/(255*255),
          "maxRGB8PixelDifference":maxRGBDifference,
          "rgb8PSNRdB":rgbMSE > 0 ? 10*log10((255*255)/rgbMSE) : NSNull() as Any,
          "zeroRGB8Error":rgbMSE == 0,"qualityApproval":false]
        if let maximum = fixture.maximumPeakMLXBytes { XCTAssertLessThanOrEqual(peak,maximum) }
        return ["pass":pass,"warmup":precisionExperiment && pass < 2,
          "computePrecision":precision.rawValue,"precisionPolicy":precision.diagnostics,
          "qualityAgainstFP32":quality,
          "allocationCacheLimitBytes":limit,"groupedStageLoading":grouped,
          "preparationGroups":stats.groupedPreparationGroups,"seconds":seconds,
          "hostValidationSeconds":ProcessInfo.processInfo.systemUptime - validationStart,
          "frames":22,"chunkShapes":chunks.map(\.shape),"float32ChunkSHA256":floatHashes,
          "rgb8ChunkSHA256":rgbHashes,"mlxPeakBeforeHostHashBytes":peak,
          "physicalBeforeHostHash":physical,"allocationCacheBytesAfterSessionClose":cacheAfterClose,
          "residentDecoderBytes":stats.maximumResidentBytes,
          "sessionClosed":stats.closed,"remainingResidentBytes":stats.remainingResidentBytes]
      }
      Stream.gpu.synchronize(); Memory.clearCache()
      XCTAssertEqual(Memory.activeMemory,inputBaseline,
        "Closed owner and all returned chunks must release back to the retained input baseline.")
      XCTAssertEqual(Memory.cacheMemory,0); XCTAssertEqual(Memory.cacheLimit,previousLimit)
      var released = report
      released["mlxActiveAfterChunkReleaseBytes"] = Memory.activeMemory
      released["remainingOwnerAndChunkActiveBytes"] = Memory.activeMemory - inputBaseline
      reports.append(released)
    }
    try inputFile.checkUnchanged(at:latentURL)
    try verifyPins()
    let physical = processMemory()
    if let maximum = fixture.maximumPhysicalPeakBytes {
      XCTAssertLessThanOrEqual(try XCTUnwrap(physical["processLifetimePeakPhysicalFootprintBytes"]),maximum)
    }
    func warmMedian(_ precision: String) -> Double? {
      let values = reports.filter { ($0["warmup"] as? Bool) != true
        && ($0["computePrecision"] as? String) == precision }
        .compactMap { $0["seconds"] as? Double }.sorted()
      guard !values.isEmpty else { return nil }
      return values.count.isMultiple(of:2)
        ? (values[values.count/2-1]+values[values.count/2])/2 : values[values.count/2]
    }
    let fp32Median = warmMedian("float32"),fp16Median = warmMedian("float16")
    let gain = fp32Median.flatMap { full in fp16Median.map { full - $0 } }
    let result: [String:Any] = ["scope":precisionExperiment
      ? "video-decoder-saved-latent-FP16-quality-component" : "video-decoder-one-clip-loading-allocation-component-parity",
      "warmFP32MedianSeconds":fp32Median.map { $0 as Any } ?? NSNull(),
      "warmFP16MedianSeconds":fp16Median.map { $0 as Any } ?? NSNull(),
      "warmAbsoluteGainSeconds":gain.map { $0 as Any } ?? NSNull(),
      "experiment":fixture.experiment ?? "allocation_cache_limit","memoryMode":mode.rawValue,
      "precisionDefault":"float32","precisionOptIn":"float16",
      "qualityReference":"same saved latent decoded by unchanged FP32 policy; no bitwise FP16 claim",
      "qualityValidationMemoryScope":"Retained untimed FP32/RGB host reference is outside MLX decode timer/peak but included in shared process physical high-water.",
      "inputProvenance":fixture.inputProvenance,"inputSourceSHA256":fixture.latentSHA256,
      "decoderInputFloat32SHA256":inputFloatSHA,
      "fixtureSHA256":cacheDigest(manifestData),"sourcePins":fixture.sourcePins.map { ["path":$0.path,"sha256":$0.sha256] },
      "testExecutable":executable.path,"testExecutableSHA256":fixture.testExecutableSHA256,
      "metalLibrary":metal.path,"metalSHA256":fixture.metalSHA256,
      "checkpointIdentity":fixture.checkpointIdentity,"checkpointHeaderSHA256":fixture.checkpointHeaderSHA256,
      "shape":[1,7,24,42,24],"spatialBatch":H3VideoDecodeMemoryMode.spatialBatchSize(for:mode),
      "spatialTiles":8,"tokensPerTile":1797,
      "passes":reports,"order":precisionExperiment ? "warm A/B then ABBA" : "ABBA",
      "warmupExecuted":precisionExperiment,
      "timingScope":"decode/load/close/complete-chunk eval; excludes all host and RGB hash validation",
      "physicalPeakScope":"shared test-process lifetime highwater; includes untimed validation; cannot compare per-mode highwaters",
      "cacheLimitScope":"allocator policy limit; not a hard instantaneous active-plus-cache or physical byte cap",
      "physicalAfter":physical,"inputBaselineActiveMLXBytes":inputBaseline,
      "generationExecuted":false,"audioDecoded":false,"visualQualityQualified":false,
      "wholeJobSpeedOrMemoryQualified":false,"productionCacheDefaultBytes":128 * 1024 * 1024]
    try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
      .write(to:outputURL,options:.atomic)
  }
}
