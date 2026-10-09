import Foundation
import CryptoKit
import Darwin
import MLX
import XCTest
@testable import H3MLX

/// Opt-in exact witness for the frozen whole Qwen stage; no sampler or new generation.
final class H3QwenStageFixtureTests: XCTestCase {
  private struct PinnedFile: Decodable, Sendable {
    let path: String
    let signature: [Int64]
  }
  private struct Fixture: Decodable, Sendable {
    let checkpointRoot: String
    let tokenizer: String
    let prompt: String
    let promptSHA256: String
    let files: [PinnedFile]
    let output: String
    let expectedReceipt: String?
    let expectedReceiptSHA256: String?
  }
  private static func sha(_ bytes: Data) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
  }
  private static func signature(_ filename: String) throws -> [Int64] {
    var value = stat()
    guard filename.withCString({ Darwin.lstat($0, &value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG else {
      throw H3CheckpointError.invalid("Qwen fixture requires resolved regular files.")
    }
    return [Int64(value.st_dev), Int64(value.st_ino), value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }
  private static func checkFiles(_ fixture: Fixture) throws {
    guard fixture.files.count == 53,
      Set(fixture.files.map(\.path)).count == 53 else {
      throw H3CheckpointError.invalid("Qwen fixture must pin manifest, embedding, 50 pages and tokenizer.")
    }
    for file in fixture.files {
      guard try signature(file.path) == file.signature else {
        throw H3CheckpointError.invalid("Qwen fixture file identity changed.")
      }
    }
  }
  private static func fixture(_ environment: String) throws -> Fixture {
    guard let filename = ProcessInfo.processInfo.environment[environment] else {
      throw XCTSkip("Set frozen Qwen fixture; ordinary tests load no installed weights.")
    }
    let bytes = try Data(contentsOf: URL(fileURLWithPath: filename))
    guard bytes.count <= 256 * 1024 else {
      throw H3CheckpointError.invalid("Qwen fixture manifest exceeds its bound.")
    }
    let fixture = try JSONDecoder().decode(Fixture.self, from: bytes)
    guard !fixture.prompt.isEmpty, fixture.prompt.utf8.count <= 65_536,
      sha(Data(fixture.prompt.utf8)) == fixture.promptSHA256,
      (fixture.expectedReceipt == nil) == (fixture.expectedReceiptSHA256 == nil),
      !FileManager.default.fileExists(atPath: fixture.output) else {
      throw H3CheckpointError.invalid("Qwen fixture prompt or unused output differs.")
    }
    try checkFiles(fixture)
    return fixture
  }
  private static func physicalPeak() throws -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else {
      throw H3CheckpointError.invalid("Cannot measure Qwen process memory.")
    }
    return UInt64(max(0, info.ledger_phys_footprint_peak))
  }

  func testInstalledFrozenPromptExactWholeEncoderWitness() throws {
    let fixture = try Self.fixture("WEETODD_H3_QWEN_STAGE_FIXTURE")
    var prior: [String: Any]?
    if let filename = fixture.expectedReceipt, let digest = fixture.expectedReceiptSHA256 {
      let bytes = try Data(contentsOf: URL(fileURLWithPath: filename))
      guard bytes.count < 64 * 1024, Self.sha(bytes) == digest,
        let report = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
        report["status"] as? String == "passed",
        report["promptSHA256"] as? String == fixture.promptSHA256 else {
        throw H3CheckpointError.invalid("Pinned prior Qwen witness differs.")
      }
      prior = report
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    let originalCacheLimit = Memory.cacheLimit
    let activeBefore = Memory.activeMemory
    Memory.peakMemory = activeBefore
    let started = ProcessInfo.processInfo.systemUptime
    var encodeSeconds = 0.0, preludeSeconds = 0.0, previousProgress = started
    var layerSeconds: [Double] = [], progressStages: [Int] = []
    var hiddenSHA = "", shape: [Int] = [], tokenIDs: [Int32] = [], tags: [Int32] = []
    var finite = false, cacheAfterEncoder = -1
    var failure: Error?
    do {
      try autoreleasepool {
        let output = try H3QwenTextEncoder.encode(prompt: fixture.prompt,
          checkpointRoot: URL(fileURLWithPath: fixture.checkpointRoot),
          tokenizerURL: URL(fileURLWithPath: fixture.tokenizer)) { completed, total in
          guard total == 50 else { return }
          let now = ProcessInfo.processInfo.systemUptime
          if completed == 0 { preludeSeconds = now - started }
          else { layerSeconds.append(now - previousProgress) }
          previousProgress = now
          progressStages.append(completed)
        }
        encodeSeconds = ProcessInfo.processInfo.systemUptime - started
        cacheAfterEncoder = Memory.cacheMemory
        shape = output.hidden.shape; tokenIDs = output.tokenIDs; tags = output.tags
        let bits = output.hidden.view(dtype: .uint16).asArray(UInt16.self)
        hiddenSHA = bits.withUnsafeBytes { Self.sha(Data($0)) }
        finite = output.hidden.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite)
      }
    } catch { failure = error }
    Stream.gpu.synchronize()
    let peak = Memory.peakMemory, physical = try Self.physicalPeak()
    let restored = Memory.cacheLimit == originalCacheLimit
    Memory.clearCache()
    let activeAfter = Memory.activeMemory
    try Self.checkFiles(fixture)
    let valid = !tokenIDs.isEmpty && tokenIDs.count <= 1024
      && shape == [tokenIDs.count, 5120] && tags.count == tokenIDs.count
      && finite && layerSeconds.count == 50 && progressStages == Array(0...50)
    let exact = prior.map { report in
      report["hiddenBF16SHA256"] as? String == hiddenSHA
        && report["shape"] as? [Int] == shape
        && report["tokenIDs"] as? [Int] == tokenIDs.map(Int.init)
        && report["tags"] as? [Int] == tags.map(Int.init)
    } ?? true
    let released = restored && cacheAfterEncoder == 0 && activeAfter == activeBefore
    let budget = peak < 2 * 1024 * 1024 * 1024 && physical < 4 * 1024 * 1024 * 1024
    let report: [String: Any] = [
      "status": failure == nil && valid && exact && released && budget ? "passed" : "failed",
      "promptSHA256": fixture.promptSHA256, "hiddenBF16SHA256": hiddenSHA,
      "shape": shape, "tokenIDs": tokenIDs.map(Int.init), "tags": tags.map(Int.init),
      "finite": finite, "exactPriorWitness": exact, "comparisonRequested": prior != nil,
      "encodeSeconds": encodeSeconds, "preludeSeconds": preludeSeconds,
      "layerLoadAndComputeSeconds": layerSeconds,
      "secondsInclusiveHostValidation": ProcessInfo.processInfo.systemUptime - started,
      "mlxPeakBytes": peak, "physicalLifetimePeakBytes": physical,
      "activeMLXBeforeBytes": activeBefore, "activeMLXAfterBytes": activeAfter,
      "mlxCacheAfterEncoderBytes": cacheAfterEncoder, "cacheLimitRestored": restored,
      "failure": failure.map { String(describing: $0) } ?? "",
      "scope": "50-layer Qwen stage including tokenizer/layout/embedding and final unloading; excludes fixture validation and host SHA validation; no sampler/video/audio",
      "generationExecuted": false]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    if let failure { throw failure }
    XCTAssertTrue(valid); XCTAssertTrue(exact); XCTAssertTrue(released); XCTAssertTrue(budget)
  }
}
