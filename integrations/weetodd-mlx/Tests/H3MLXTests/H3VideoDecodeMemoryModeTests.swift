import Foundation
import CryptoKit
import Darwin
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3VideoDecodeMemoryModeTests: XCTestCase {
  func testLowMemoryDecoderReportsOneSpatialTileWhileNormalRetainsFour() {
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for:.lowMemoryBF16)["spatialBatch"] as? Int,1)
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for:.normal)["spatialBatch"] as? Int,4)
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for:nil)["spatialBatch"] as? Int,4)
  }
  func testResidentModesBoundPendingTileBlocksAndKeepDirectEagerBehavior() {
    for mode: H3VideoDecodeMemoryMode in [.normal, .lowMemoryBF16] {
      let completed = (0..<36).filter {
        H3VideoDecodeMemoryMode.materializesBlockOutput(for: mode, blockIndex:$0)
      }
      XCTAssertEqual(completed, mode == .normal
        ? Array(0..<36) : [])
      if mode == .normal {
        XCTAssertEqual(completed.last,35,"Normal batches must retire before the tile head.")
      }
      var pending = 0
      var maximumPending = 0
      for index in 0..<36 {
        pending += 1
        maximumPending = max(maximumPending,pending)
        if H3VideoDecodeMemoryMode.materializesBlockOutput(for:mode,blockIndex:index) {
          pending = 0
        }
      }
      XCTAssertEqual(maximumPending,mode == .normal ? 1 : 36)
      XCTAssertEqual(pending,mode == .normal ? 0 : 36)
      let diagnostics = H3VideoDecodeMemoryMode.diagnostics(for:mode)
      if mode == .normal {
        XCTAssertEqual(diagnostics["materializesBlockOutput"] as? Bool,false)
      } else { XCTAssertEqual(diagnostics["materializesBlockOutput"] as? Bool,false) }
      XCTAssertEqual(diagnostics["blockOutputMaterializationPolicy"] as? String,
        mode == .normal ? "batches_1_2_every_block_batches_3_4_tile_head" : "deferred_until_tile_head")
      // Diagnostics describe the maximum production batch, including smaller tail batches.
      XCTAssertEqual(diagnostics["blockOutputExecutionWindow"] as? Int,36)
      XCTAssertEqual(diagnostics["maximumPendingBlockOutputs"] as? Int,36)
      XCTAssertEqual(diagnostics["maximumPendingSpatialTileBlocks"] as? Int,mode == .normal ? 144 : 36)
    }
    for index in 0..<36 {
      XCTAssertTrue(H3VideoDecodeMemoryMode.materializesBlockOutput(for:nil,blockIndex:index))
    }
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for: nil)["materializesBlockOutput"] as? Bool, true)
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for:nil)["blockOutputExecutionWindow"] as? Int,1)
    XCTAssertEqual(H3VideoDecodeMemoryMode.diagnostics(for:nil)["maximumPendingBlockOutputs"] as? Int,1)
  }

  func testNormalPolicyUsesActualBatchAndKeepsLowAndDirectPolicies() {
    for batch in 1...4 {
      let small = batch <= 2
      XCTAssertEqual(H3VideoDecodeMemoryMode.materializesFirstResidual(for:.normal,
        spatialBatchSize:batch),small)
      XCTAssertEqual(H3VideoDecodeMemoryMode.blockOutputExecutionWindow(for:.normal,
        spatialBatchSize:batch),small ? 1 : 36)
      for index in 0..<36 {
        XCTAssertEqual(H3VideoDecodeMemoryMode.materializesBlockOutput(for:.normal,
          blockIndex:index,spatialBatchSize:batch),small)
        XCTAssertFalse(H3VideoDecodeMemoryMode.materializesBlockOutput(for:.lowMemoryBF16,
          blockIndex:index,spatialBatchSize:batch))
        XCTAssertTrue(H3VideoDecodeMemoryMode.materializesBlockOutput(for:nil,
          blockIndex:index,spatialBatchSize:batch))
      }
      XCTAssertTrue(H3VideoDecodeMemoryMode.materializesFirstResidual(for:nil,spatialBatchSize:batch))
      XCTAssertTrue(H3VideoDecodeMemoryMode.materializesFirstResidual(for:.lowMemoryBF16,spatialBatchSize:batch))
    }
    let report = H3VideoDecodeMemoryMode.diagnostics(for:.normal)
    XCTAssertEqual(report["materializationPolicy"] as? String,"adaptive_normal_spatial_batch")
    XCTAssertEqual(report["normalSmallBatchExecutionWindow"] as? Int,1)
    XCTAssertEqual(report["normalLargeBatchExecutionWindow"] as? Int,36)
    XCTAssertEqual(report["normalFirstResidualPolicy"] as? String,"batches_1_2_eager_batches_3_4_tile_head")
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject:report))
    // Unsupported batch values cannot enable the lazy large-batch route.
    for invalid in [0,5,Int.max] {
      XCTAssertTrue(H3VideoDecodeMemoryMode.materializesFirstResidual(for:.normal,spatialBatchSize:invalid))
      XCTAssertTrue(H3VideoDecodeMemoryMode.materializesBlockOutput(for:.normal,blockIndex:0,spatialBatchSize:invalid))
    }
  }

  func testSelectedModesRequireResidentSessionBeforeCheckpointAccess() throws {
    try Device.withDefaultDevice(.cpu) {
      let latent = MLXArray.zeros([1, 7, 2, 2, 24], dtype: .float32)
      for mode: H3VideoDecodeMemoryMode in [.normal, .lowMemoryBF16] {
        XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(
          checkpointURL: URL(fileURLWithPath: "/unavailable/decoder.safetensors"),
          latent: latent, retainWeights: false, memoryMode: mode,
          onChunk: { _ in XCTFail("Rejected mode must not publish.") })) {
          XCTAssertTrue(String(describing: $0).contains("requires a resident decoder session"))
        }
      }
    }
  }

  private struct Fixture: Decodable {
    let mode: String
    let metalLibrary: String
    let metalSHA256: String
    let checkpoint: String
    let checkpointStat: [Int64]
    let rawPath: String
    let rawSHA256: String
    let expectedReceipt: String
    let expectedReceiptSHA256: String
    let output: String
    let maximumMLXBytes: Int
    let maximumPhysicalBytes: UInt64
    /// An explicit decoder-only comparison; nil follows the production delegate.
    let spatialBatchSize: Int?
  }

  private func sha(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private func signature(_ file: String) throws -> [Int64] {
    var value = stat()
    guard file.withCString({ Darwin.lstat($0, &value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG else {
      throw H3CheckpointError.invalid("Decoder checkpoint must be a regular file.")
    }
    return [Int64(value.st_dev), Int64(value.st_ino), value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }

  private func physicalPeak() throws -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS else {
      throw H3CheckpointError.invalid("Cannot measure decoder process memory.")
    }
    return UInt64(max(0, info.ledger_phys_footprint_peak))
  }

  /// One fresh process per existing recipe mode; no experimental observers or sampler execution.
  func testInstalledPublicMemoryModeSavedLatentParity() throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_VAE_MEMORY_MODE_FIXTURE"] else {
      throw XCTSkip("Set a pinned saved-latent fixture; default does not load weights.")
    }
    let fixture = try JSONDecoder().decode(Fixture.self,
      from: Data(contentsOf: URL(fileURLWithPath: manifest)))
    // Pin the production kernel library before any array or weight is loaded.
    // Test build systems may otherwise select a different default.metallib.
    let metalURL = URL(fileURLWithPath: fixture.metalLibrary)
    var metalStat = stat()
    guard fixture.metalLibrary.withCString({ Darwin.lstat($0, &metalStat) }) == 0,
      metalStat.st_mode & S_IFMT == S_IFREG,
      FileManager.default.isReadableFile(atPath: fixture.metalLibrary),
      fixture.metalSHA256.count == 64,
      sha(try Data(contentsOf: metalURL)) == fixture.metalSHA256 else {
      throw H3CheckpointError.invalid("Missing or changed pinned decoder Metal library.")
    }
    GPU.metallib = metalURL
    guard let mode = H3VideoDecodeMemoryMode(rawValue: fixture.mode),
      fixture.maximumMLXBytes == 6 * 1024 * 1024 * 1024,
      fixture.maximumPhysicalBytes == 8 * 1024 * 1024 * 1024,
      fixture.spatialBatchSize == nil || [1, 4].contains(fixture.spatialBatchSize!),
      try signature(fixture.checkpoint) == fixture.checkpointStat,
      !FileManager.default.fileExists(atPath: fixture.output) else {
      throw H3CheckpointError.invalid("Changed or previously executed decoder fixture.")
    }
    let rawURL = URL(fileURLWithPath: fixture.rawPath)
    let rawData = try Data(contentsOf: rawURL)
    let oracleData = try Data(contentsOf: URL(fileURLWithPath: fixture.expectedReceipt))
    guard rawData.count < 8 * 1024 * 1024, sha(rawData) == fixture.rawSHA256,
      sha(oracleData) == fixture.expectedReceiptSHA256 else {
      throw H3CheckpointError.invalid("Pinned latent or decoder oracle differs.")
    }
    let file = try SafeTensorFile(url: rawURL)
    guard file.tensors["video_latents"]?.dtype == "F32",
      file.tensors["video_latents"]?.shape == [1, 24, 37, 28, 48],
      let oracle = try JSONSerialization.jsonObject(with: oracleData) as? [String: Any],
      oracle["rawLatentSHA256"] as? String == fixture.rawSHA256,
      let passes = oracle["passes"] as? [[String: Any]],
      let reference = passes.first(where: { $0["resident"] as? Bool == true }),
      let expectedFloat = reference["float32ChunkSHA256"] as? [String],
      let expectedRGB = reference["rgb8ChunkSHA256"] as? [String],
      let expectedShapes = reference["chunkShapes"] as? [[Int]],
      reference["frames"] as? Int == 124 else {
      throw H3CheckpointError.invalid("Requires actual124-frame768x448 decoder oracle.")
    }
    let raw = try file.withTensorBytes(named: "video_latents") {
      MLXArray($0, [1, 24, 37, 28, 48], type: Float.self)
    }
    let checkpointURL = URL(fileURLWithPath: fixture.checkpoint)
    let layout = try H3VideoVAELayout(url: checkpointURL)
    let rows = raw.transposed(0, 2, 3, 4, 1).reshaped([1, 37, 14, 2, 24, 2, 24])
      .transposed(0, 1, 2, 4, 6, 3, 5).reshaped([1, 12_432, 96])
    let latent = try H3LatentCodec.videoDecoderInput(rows: rows, latentFrames: 37,
      latentHeight: 28, latentWidth: 48, mean: layout.latentsMean,
      standardDeviation: layout.latentsStandardDeviation)
    Stream.gpu.synchronize()
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = ProcessInfo.processInfo.systemUptime
    var floats: [String] = [], rgb: [String] = [], shapes: [[Int]] = []
    var frames = 0
    var closed: H3VideoVAEDecodeSession.Statistics?
    var failure: Error?
    do {
      // Same overload used by the public decodeChunks delegate, with close statistics only.
      try H3VideoVAEDecoder.decodeChunks(checkpointURL: checkpointURL,
        latent: latent, retainWeights: true, memoryMode: mode,
        spatialBatchSize: fixture.spatialBatchSize,
        onSessionClosed: { closed = $0 }) { chunk in
        guard chunk.dtype == .float32 else {
          throw H3CheckpointError.invalid("Production decoder output precision changed.")
        }
        let values = chunk.asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else {
          throw H3CheckpointError.invalid("Nonfinite production decoder output.")
        }
        floats.append(values.withUnsafeBytes { sha(Data($0)) })
        let pixels = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
        rgb.append(pixels.withUnsafeBytes { sha(Data($0)) })
        shapes.append(chunk.shape)
        frames += chunk.shape[1]
        guard Memory.peakMemory <= fixture.maximumMLXBytes,
          try physicalPeak() <= fixture.maximumPhysicalBytes else {
          throw H3CheckpointError.invalid("Production decoder fixture exceeded fixed memory budget.")
        }
      }
    } catch { failure = error }
    let elapsed = ProcessInfo.processInfo.systemUptime - started
    Stream.gpu.synchronize()
    Memory.clearCache()
    try file.checkUnchanged(at: rawURL)
    guard try signature(fixture.checkpoint) == fixture.checkpointStat else {
      throw H3CheckpointError.invalid("Decoder checkpoint changed during qualification.")
    }
    let exact = frames == 124 && floats == expectedFloat && rgb == expectedRGB && shapes == expectedShapes
    let released = closed?.closed == true && closed?.remainingResidentBytes == 0
    let report: [String: Any] = ["status": failure == nil && exact && released ? "passed" : "failed",
      "memoryMode": mode.rawValue, "videoDecode": H3VideoDecodeMemoryMode.diagnostics(for: mode),
      "metalLibrary": fixture.metalLibrary, "metalSHA256": fixture.metalSHA256,
      "explicitSpatialBatchSize": fixture.spatialBatchSize as Any? ?? NSNull(),
      "secondsInclusiveHostValidation": elapsed, "frames": frames,
      "chunkShapes": shapes, "float32ChunkSHA256": floats, "rgb8ChunkSHA256": rgb,
      "rawLatentSHA256": fixture.rawSHA256, "exactFrozenPixels": exact,
      "closed": closed?.closed ?? false, "remainingResidentBytes": closed?.remainingResidentBytes ?? -1,
      "maximumResidentBytes": closed?.maximumResidentBytes ?? -1,
      "maximumGridCacheBytes": closed?.maximumGridCacheBytes ?? -1,
      "mlxPeakBytes": Memory.peakMemory, "mlxCacheAfterBytes": Memory.cacheMemory,
      "physicalLifetimePeakBytes": try physicalPeak(),
      "failure": failure.map { String(describing: $0) } ?? "",
      "scope": "production video decoder only; caller latent remains alive; no sampler/audio/FFmpeg",
      "generationExecuted": false]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    if let failure { throw failure }
    XCTAssertTrue(exact, "All Float32 chunks and RGB8 bytes must match the frozen eager oracle.")
    XCTAssertTrue(released)
    let statistics = try XCTUnwrap(closed)
    XCTAssertGreaterThan(statistics.maximumGridCacheBytes,0)
    XCTAssertLessThanOrEqual(statistics.maximumGridCacheBytes,H3VideoVAEDecodeSession.maximumGridCacheBytes)
    XCTAssertEqual(statistics.maximumResidentBytes,2_582_138_032+statistics.maximumGridCacheBytes)
    XCTAssertEqual(Memory.cacheMemory, 0)
  }
}
