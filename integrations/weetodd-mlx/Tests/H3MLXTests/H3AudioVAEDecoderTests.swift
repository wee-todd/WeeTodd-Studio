import Foundation
import CryptoKit
import Darwin
import TensorIO
import MLX
import XCTest
@testable import H3MLX

final class H3AudioVAEDecoderTests: XCTestCase {
  func testInstalledTwoFrameStereoDecoderMatchesReferenceBoundaries() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_VAE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_ORACLE"] else {
      throw XCTSkip("Set installed H3 audio VAE and audio oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("latent.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 2, 32], type: Float.self) }
    let output = try H3AudioVAEDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent,
      observe: { name, value in
        let expectedShape: [Int]
        switch name {
        case "decin": expectedShape = [2, 2, 2048]
        case "convpre": expectedShape = [2, 2, 1024]
        case let step where step.hasPrefix("up") || step.hasPrefix("res"):
          guard let stage = Int(step.dropFirst(step.hasPrefix("up") ? 2 : 3)),
            (0..<7).contains(stage) else {
            XCTFail("Unknown audio stage \(name)"); return
          }
          let lengths = [10, 50, 100, 200, 400, 800, 1600]
          expectedShape = [2, lengths[stage], 512 >> stage]
        case "postact": expectedShape = [2, 1600, 8]
        case "postconv", "wave": expectedShape = [2, 1600, 1]
        default: XCTFail("Unknown audio observation \(name)"); return
        }
        let expected = try Data(contentsOf: root.appendingPathComponent("\(name).f32"))
          .withUnsafeBytes { MLXArray($0, expectedShape, type: Float.self) }
        XCTAssertEqual(value.shape, expectedShape)
        XCTAssertLessThan(max(abs(value - expected)).item(Float.self), 1e-4, name)
      })
    XCTAssertEqual(output.shape, [2, 1600, 1])
  }


  private struct SavedAudioFixture: Decodable, Sendable {
    let checkpoint: String
    let checkpointStat: [Int64]
    let rawPath: String
    let rawSHA256: String
    let output: String
    let expectedReceipt: String?
    let expectedReceiptSHA256: String?
    let maximumMLXBytes: Int
    let maximumPhysicalBytes: UInt64
  }

  private static func audioSHA(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func checkpointSignature(_ filename: String) throws -> [Int64] {
    var value = stat()
    guard filename.withCString({ Darwin.lstat($0, &value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG else {
      throw H3CheckpointError.invalid("Audio checkpoint must be a regular file.")
    }
    return [Int64(value.st_dev), Int64(value.st_ino), value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }

  private static func audioPhysicalPeak() throws -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS else {
      throw H3CheckpointError.invalid("Cannot measure audio decoder process memory.")
    }
    return UInt64(max(0, info.ledger_phys_footprint_peak))
  }

  private static func audioFixture(environment: String) throws -> SavedAudioFixture {
    guard let manifest = ProcessInfo.processInfo.environment[environment] else {
      throw XCTSkip("Set a pinned saved207-frame audio fixture; ordinary tests load no weights.")
    }
    let fixture = try JSONDecoder().decode(SavedAudioFixture.self,
      from: Data(contentsOf: URL(fileURLWithPath: manifest)))
    guard fixture.maximumMLXBytes == 6 * 1024 * 1024 * 1024,
      fixture.maximumPhysicalBytes == 8 * 1024 * 1024 * 1024,
      fixture.rawSHA256.count == 64,
      (fixture.expectedReceipt == nil) == (fixture.expectedReceiptSHA256 == nil),
      fixture.expectedReceiptSHA256 == nil || fixture.expectedReceiptSHA256!.count == 64,
      try checkpointSignature(fixture.checkpoint) == fixture.checkpointStat,
      !FileManager.default.fileExists(atPath: fixture.output) else {
      throw H3CheckpointError.invalid("Changed or already executed saved audio fixture.")
    }
    return fixture
  }

  private static func audioInput(_ fixture: SavedAudioFixture) throws -> (SafeTensorFile, MLXArray) {
    let rawURL = URL(fileURLWithPath: fixture.rawPath)
    let values = try rawURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, let count = values.fileSize, count < 8 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Saved audio latent file exceeds the fixture bound.")
    }
    let source = try SafeTensorFile(url: rawURL)
    let bytes = try Data(contentsOf: rawURL)
    guard audioSHA(bytes) == fixture.rawSHA256 else {
      throw H3CheckpointError.invalid("Saved audio latent SHA256 differs.")
    }
    try source.checkUnchanged(at:rawURL)
    guard source.tensors["audio_latents"]?.dtype == "F32",
      source.tensors["audio_latents"]?.shape == [2, 32, 207] else {
      throw H3CheckpointError.invalid("Requires the actual207-frame stereo Float32 audio latent.")
    }
    let normalized = try source.withTensorBytes(named: "audio_latents") {
      MLXArray($0, [2, 32, 207], type: Float.self)
    }
    let layout = try H3AudioVAELayout(url: URL(fileURLWithPath: fixture.checkpoint))
    let rows = normalized.transposed(0, 2, 1).reshaped([1, 414, 32])
    let input = try H3LatentCodec.audioDecoderInput(rows: rows, latentFrames: 207,
      mean: layout.latentsMean, standardDeviation: layout.latentsStandardDeviation)
    guard input.asArray(Float.self).allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("Nonfinite saved audio decoder input.")
    }
    try source.checkUnchanged(at: rawURL)
    return (source, input)
  }

  /// Decoder-only baseline/candidate witness; no sampler, video or FFmpeg work.
  func testInstalledSaved207FrameAudioExactParityAndMeasuresStage() throws {
    let fixture = try Self.audioFixture(environment: "WEETODD_H3_AUDIO_VAE_SAVED_FIXTURE")
    var expectedWaveSHA: String?
    if let filename = fixture.expectedReceipt, let digest = fixture.expectedReceiptSHA256 {
      let data = try Data(contentsOf: URL(fileURLWithPath: filename))
      guard data.count < 64 * 1024, Self.audioSHA(data) == digest,
        let receipt = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        receipt["status"] as? String == "passed",
        receipt["rawLatentSHA256"] as? String == fixture.rawSHA256,
        receipt["checkpointStat"] as? [Int64] == fixture.checkpointStat,
        receipt["shape"] as? [Int] == [2, 165_600, 1],
        receipt["samples"] as? Int == 331_200,
        let waveSHA = receipt["float32WaveformSHA256"] as? String, waveSHA.count == 64 else {
        throw H3CheckpointError.invalid("Pinned full-wave audio receipt differs.")
      }
      expectedWaveSHA = waveSHA
    }
    if ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_VAE_REQUIRE_EXACT_RECEIPT"] == "1",
      expectedWaveSHA == nil {
      throw H3CheckpointError.invalid("Audio optimization qualification requires a pinned prior waveform receipt.")
    }
    let (source, input) = try Self.audioInput(fixture)
    Stream.gpu.synchronize()
    Memory.clearCache()
    let originalCacheLimit = Memory.cacheLimit
    let activeBefore = Memory.activeMemory
    Memory.peakMemory = activeBefore
    let started = ProcessInfo.processInfo.systemUptime
    var waveSHA = "", shape: [Int] = [], count = 0, finite = false
    var cacheAfterDecoder = -1, progressStages: [Int] = []
    var failure: Error?
    do {
      try autoreleasepool {
        let waveform = try H3AudioVAEDecoder.decode(
          checkpointURL: URL(fileURLWithPath: fixture.checkpoint), latent: input,
          progress: { completed, total in
            if total == 7 { progressStages.append(completed) }
          })
        shape = waveform.shape
        let samples = waveform.asArray(Float.self)
        finite = samples.allSatisfy(\.isFinite)
        count = samples.count
        waveSHA = samples.withUnsafeBytes { Self.audioSHA(Data($0)) }
        cacheAfterDecoder = Memory.cacheMemory
      }
    } catch { failure = error }
    let elapsed = ProcessInfo.processInfo.systemUptime - started
    Stream.gpu.synchronize()
    let peak = Memory.peakMemory
    let physical = try Self.audioPhysicalPeak()
    let restored = Memory.cacheLimit == originalCacheLimit
    Memory.clearCache()
    let activeAfter = Memory.activeMemory
    try source.checkUnchanged(at: URL(fileURLWithPath: fixture.rawPath))
    guard try Self.checkpointSignature(fixture.checkpoint) == fixture.checkpointStat else {
      throw H3CheckpointError.invalid("Audio checkpoint changed during qualification.")
    }
    let valid = shape == [2, 165_600, 1] && count == 331_200 && finite
      && progressStages == Array(1...7)
    let exact = expectedWaveSHA.map { $0 == waveSHA } ?? true
    let released = restored && cacheAfterDecoder == 0 && activeAfter == activeBefore
    let budget = peak <= fixture.maximumMLXBytes && physical <= fixture.maximumPhysicalBytes
    let report: [String: Any] = [
      "status": failure == nil && valid && exact && released && budget ? "passed" : "failed",
      "rawLatentSHA256": fixture.rawSHA256, "checkpointStat": fixture.checkpointStat,
      "shape": shape, "samples": count, "finite": finite,
      "float32WaveformSHA256": waveSHA, "comparisonRequested": expectedWaveSHA != nil,
      "exactPriorWaveform": exact, "secondsInclusiveHostValidation": elapsed,
      "materializationPolicy": "resident weights; evaluate convolution, activation, residual and stage boundaries",
      "mlxPeakBytes": peak, "physicalLifetimePeakBytes": physical,
      "activeMLXBeforeBytes": activeBefore, "activeMLXAfterBytes": activeAfter,
      "mlxCacheAfterDecoderBytes": cacheAfterDecoder,
      "cacheLimitRestored": restored, "progressStages": progressStages,
      "failure": failure.map { String(describing: $0) } ?? "",
      "scope": "audio decoder only; caller latent retained; no sampler/video/FFmpeg",
      "generationExecuted": false]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    if let failure { throw failure }
    XCTAssertTrue(valid)
    XCTAssertTrue(exact, "Complete Float32 waveform must match the pinned baseline SHA256.")
    XCTAssertTrue(released, "Scoped weights and allocation cache must release; cache limit must restore.")
    XCTAssertTrue(budget)
  }

  func testInstalledSaved207FrameAudioCancellationRestoresStage() async throws {
    let fixture = try Self.audioFixture(environment: "WEETODD_H3_AUDIO_VAE_CANCEL_FIXTURE")
    let data = try await Task.detached {
      let (source, input) = try Self.audioInput(fixture)
      Stream.gpu.synchronize()
      Memory.clearCache()
      let activeBefore = Memory.activeMemory
      let originalCacheLimit = Memory.cacheLimit
      Memory.peakMemory = activeBefore
      var callbacks = 0, cancelled = false, completed = false
      let started = ProcessInfo.processInfo.systemUptime
      do {
        _ = try H3AudioVAEDecoder.decode(
          checkpointURL: URL(fileURLWithPath: fixture.checkpoint), latent: input,
          progress: { _, _ in
            callbacks += 1
            if callbacks == 1 { withUnsafeCurrentTask { $0?.cancel() } }
          })
        completed = true
      } catch is CancellationError { cancelled = true }
      Stream.gpu.synchronize()
      let cacheAfter = Memory.cacheMemory
      let activeAfter = Memory.activeMemory
      let restored = Memory.cacheLimit == originalCacheLimit
      let peak = Memory.peakMemory
      let physical = try Self.audioPhysicalPeak()
      try source.checkUnchanged(at: URL(fileURLWithPath: fixture.rawPath))
      guard try Self.checkpointSignature(fixture.checkpoint) == fixture.checkpointStat else {
        throw H3CheckpointError.invalid("Audio checkpoint changed during cancellation qualification.")
      }
      let valid = cancelled && !completed && callbacks == 1 && restored
        && cacheAfter == 0 && activeAfter == activeBefore
        && peak <= fixture.maximumMLXBytes && physical <= fixture.maximumPhysicalBytes
      let report: [String: Any] = [
        "status": valid ? "passed" : "failed", "cancelled": cancelled,
        "decoderCompleted": completed, "progressCallbacks": callbacks,
        "rawLatentSHA256": fixture.rawSHA256, "checkpointStat": fixture.checkpointStat,
        "secondsUntilCancellation": ProcessInfo.processInfo.systemUptime - started,
        "mlxPeakBytes": peak, "physicalLifetimePeakBytes": physical,
        "activeMLXBeforeBytes": activeBefore, "activeMLXAfterBytes": activeAfter,
        "mlxCacheAfterBytes": cacheAfter, "cacheLimitRestored": restored,
        "scope": "cancellation after first audio upsample/residual stage; no publication",
        "generationExecuted": false]
      return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    }.value
    let report = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    try data.write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    XCTAssertEqual(report["status"] as? String, "passed")
    XCTAssertEqual(report["cancelled"] as? Bool, true)
    XCTAssertEqual(report["decoderCompleted"] as? Bool, false)
    XCTAssertEqual(report["progressCallbacks"] as? Int, 1)
    XCTAssertEqual(report["mlxCacheAfterBytes"] as? Int, 0)
    XCTAssertEqual(report["cacheLimitRestored"] as? Bool, true)
  }


  /// Cancel after the first raw convolution. The stage must
  /// reject the next cached read and release prefetched weights without
  /// waiting for a progress callback or completing the waveform.
  func testInstalledSavedAudioEarlyBranchCancellationReleasesPrefill() async throws {
    let fixture = try Self.audioFixture(environment: "WEETODD_H3_AUDIO_VAE_LAZY_CANCEL_FIXTURE")
    let data = try await Task.detached {
      let (source, input) = try Self.audioInput(fixture)
      Stream.gpu.synchronize()
      Memory.clearCache()
      let activeBefore = Memory.activeMemory
      let originalCacheLimit = Memory.cacheLimit
      Memory.peakMemory = activeBefore
      var observed = false, cancelled = false, completed = false, callbacks = 0
      let started = ProcessInfo.processInfo.systemUptime
      do {
        _ = try H3AudioVAEDecoder.decode(
          checkpointURL: URL(fileURLWithPath: fixture.checkpoint), latent: input,
          progress: { _, _ in callbacks += 1 }, observe: { name, _ in
            if name == "decin" {
              observed = true
              withUnsafeCurrentTask { $0?.cancel() }
            }
          })
        completed = true
      } catch is CancellationError { cancelled = true }
      Stream.gpu.synchronize()
      let cacheAfter = Memory.cacheMemory, activeAfter = Memory.activeMemory
      let restored = Memory.cacheLimit == originalCacheLimit
      let peak = Memory.peakMemory, physical = try Self.audioPhysicalPeak()
      try source.checkUnchanged(at: URL(fileURLWithPath: fixture.rawPath))
      guard try Self.checkpointSignature(fixture.checkpoint) == fixture.checkpointStat else {
        throw H3CheckpointError.invalid("Audio checkpoint changed during lazy cancellation qualification.")
      }
      let valid = observed && cancelled && !completed && callbacks == 0 && restored
        && cacheAfter == 0 && activeAfter == activeBefore
        && peak <= fixture.maximumMLXBytes && physical <= fixture.maximumPhysicalBytes
      let report: [String: Any] = [
        "status": valid ? "passed" : "failed", "cancelled": cancelled,
        "decoderCompleted": completed, "progressCallbacks": callbacks,
        "rawLatentSHA256": fixture.rawSHA256, "checkpointStat": fixture.checkpointStat,
        "secondsUntilCancellation": ProcessInfo.processInfo.systemUptime - started,
        "mlxPeakBytes": peak, "physicalLifetimePeakBytes": physical,
        "activeMLXBeforeBytes": activeBefore, "activeMLXAfterBytes": activeAfter,
        "mlxCacheAfterBytes": cacheAfter, "cacheLimitRestored": restored,
        "scope": "cancellation after decin before first upsample; caller latent retained",
        "generationExecuted": false]
      return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    }.value
    let report = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    try data.write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    XCTAssertEqual(report["status"] as? String, "passed")
    XCTAssertEqual(report["cancelled"] as? Bool, true)
    XCTAssertEqual(report["decoderCompleted"] as? Bool, false)
    XCTAssertEqual(report["progressCallbacks"] as? Int, 0)
    XCTAssertEqual(report["mlxCacheAfterBytes"] as? Int, 0)
    XCTAssertEqual(report["cacheLimitRestored"] as? Bool, true)
  }

  func testResidentDecoderAdmitsExactAggregateBoundaryWithoutWeights() throws {
    var bytes = Array(repeating: UInt64(4), count: 779)
    XCTAssertEqual(try H3AudioVAEDecoder.admittedResidentDecoderByteCount(bytes), 3116)
    for index in 0..<4 { bytes[index] = 64 * 1024 * 1024 }
    bytes[4] = 300 * 1024 * 1024 - bytes.enumerated()
      .filter { $0.offset != 4 }.reduce(UInt64(0)) { $0 + $1.element }
    XCTAssertEqual(try H3AudioVAEDecoder.admittedResidentDecoderByteCount(bytes),
      300 * 1024 * 1024)
    bytes[4] += 4
    XCTAssertThrowsError(try H3AudioVAEDecoder.admittedResidentDecoderByteCount(bytes))
  }

  func testResidentDecoderRejectsIncompleteZeroOversizedAndOverflowPayloads() {
    XCTAssertThrowsError(try H3AudioVAEDecoder.admittedResidentDecoderByteCount(
      Array(repeating: UInt64(4), count: 778)))
    let invalidCounts:[UInt64] = [0,64 * 1024 * 1024 + 4,UInt64.max]
    for invalid in invalidCounts {
      var bytes = Array(repeating: UInt64(4), count: 779)
      bytes[0] = invalid
      XCTAssertThrowsError(try H3AudioVAEDecoder.admittedResidentDecoderByteCount(bytes))
    }
  }
}
