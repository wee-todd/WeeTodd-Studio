import CryptoKit
import Darwin
import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

/// Diagnostic only: one fresh process, one loading mode, fifty distinct pages.
/// This does not execute a projection, transformer block, attention, or sampler.
final class H3PreparedFastAllPagesTests: XCTestCase {
  private enum Mode: String { case serial, grouped }
  private struct Fixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    struct HeaderPin: Decodable {
      let path: String; let headerSHA256: String; let identity: [Int64]
    }
    let checkpoint: String
    let mode: String
    let blockIndices: [Int]
    let cacheLimitBytes: Int
    let hashSmallNorms: Bool
    let metalLibrary: String
    let metalSHA256: String
    let testExecutable: String
    let testExecutableSHA256: String
    let sourcePins: [Pin]
    let checkpointPins: [HeaderPin]
    let output: String
  }
  private static let productMetal = "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9"
  private static let cacheBytes = 128 * 1024 * 1024

  private static func admit(mode: String, indices: [Int], cache: Int, hashes: Bool) throws -> Mode {
    guard let selected = Mode(rawValue: mode), indices == Array(0..<50),
      cache == Self.cacheBytes, hashes else {
      throw H3CheckpointError.invalid("Fast all-page diagnostic requires one declared mode, blocks 0–49, small norm hashes, and a 128 MiB cache.")
    }
    try Task.checkCancellation()
    return selected
  }

  func testInvalidDiagnosticSelectionRejectsBeforeDeviceAccess() {
    for mode in ["", "automatic", "Grouped"] {
      XCTAssertThrowsError(try Self.admit(mode: mode, indices: Array(0..<50), cache: Self.cacheBytes, hashes: true))
    }
    XCTAssertThrowsError(try Self.admit(mode: "grouped", indices: [0, 1], cache: Self.cacheBytes, hashes: true))
    XCTAssertThrowsError(try Self.admit(mode: "grouped", indices: Array(0..<50), cache: 0, hashes: true))
    XCTAssertThrowsError(try Self.admit(mode: "grouped", indices: Array(0..<50), cache: Self.cacheBytes, hashes: false))
  }

  func testDiagnosticCancellationRejectsBeforeDeviceAccess() async throws {
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try Self.admit(mode: "grouped", indices: Array(0..<50), cache: Self.cacheBytes, hashes: true)
        return false
      } catch is CancellationError { return true }
      catch { return false }
    }
    let cancelled = await task.value
    XCTAssertTrue(cancelled)
  }

  private func digest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let data = try file.read(upToCount: 4 << 20), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }

  private func identity(_ url: URL) throws -> [Int64] {
    var value = stat()
    guard url.path.withCString({ Darwin.lstat($0, &value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG else {
      throw H3CheckpointError.invalid("Diagnostic inputs must be readable regular files.")
    }
    return [Int64(value.st_dev), Int64(value.st_ino), value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }

  private func headerDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    guard let bytes = try file.read(upToCount: 8), bytes.count == 8 else {
      throw H3CheckpointError.invalid("Missing diagnostic safetensors header.")
    }
    let count = bytes.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << ($1.offset * 8)) }
    guard (2...64 * 1024 * 1024).contains(count),
      let header = try file.read(upToCount: Int(count)), header.count == Int(count) else {
      throw H3CheckpointError.invalid("Invalid diagnostic safetensors header.")
    }
    return SHA256.hash(data: header).map { String(format: "%02x", $0) }.joined()
  }

  private func physical() throws -> [String: UInt64] {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { throw H3CheckpointError.invalid("Diagnostic physical memory query failed.") }
    return ["currentBytes": info.phys_footprint,
      "lifetimePeakBytes": UInt64(max(0, info.ledger_phys_footprint_peak))]
  }

  private func factorHash(_ value: MLXArray) throws -> String {
    // All arrays are already materialized by the constructor. This host
    // inspection is outside load timers and creates no new MLX array.
    let bytes = value.asData(access: .noCopy)
    var expectedStride = 1
    for axis in value.shape.indices.reversed() {
      guard bytes.strides[axis] == expectedStride else {
        throw H3CheckpointError.invalid("Diagnostic factor is not contiguous.")
      }
      expectedStride *= value.shape[axis]
    }
    guard bytes.data.count == value.nbytes else {
      throw H3CheckpointError.invalid("Diagnostic factor byte count differs.")
    }
    return withExtendedLifetime(value) {
      SHA256.hash(data: bytes.data).map { String(format: "%02x", $0) }.joined()
    }
  }

  private func inspect(_ owner: H3PreparedBlock, prefix: String) throws
    -> (hashes: [String: String], factors: [String: [String: Any]], bytes: Int) {
    var hashes: [String: String] = [:]
    var factors: [String: [String: Any]] = [:]
    var bytes = 0
    for (suffix, shape) in [("norm1.weight", [5376]), ("norm2.weight", [5376]),
      ("attn.q_norm.weight", [128]), ("attn.k_norm.weight", [128]),
      ("attn.gate_compress.weight", [7168, 5376])] {
      let value = try owner.read(prefix + suffix, shape: shape)
      if suffix != "attn.gate_compress.weight" { hashes[suffix] = try factorHash(value) }
      factors[suffix] = ["shape": value.shape, "dtype": String(describing: value.dtype), "bytes": value.nbytes]
      bytes += value.nbytes
    }
    for suffix in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
      let values = try owner.readAffineProjectionParameters(suffix)
      guard values.count == 3 else { throw H3CheckpointError.invalid("Diagnostic packed projection factor count differs.") }
      for (factorIndex, value) in values.enumerated() {
        factors[suffix + "." + ["packed", "scales", "biases"][factorIndex]] =
          ["shape": value.shape, "dtype": String(describing: value.dtype), "bytes": value.nbytes]
        bytes += value.nbytes
      }
    }
    guard hashes.count == 4, factors.count == 17, bytes == owner.storageBytes else {
      throw H3CheckpointError.invalid("Diagnostic did not inspect the complete resident owner.")
    }
    return (hashes, factors, bytes)
  }

  func testInstalledFastAllFiftyPagesSingleMode() throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_H3_FAST_ALL_PAGES_FIXTURE"] else {
      throw XCTSkip("Opt-in all-page preparation diagnostic; normal tests never load weights.")
    }
    let startedUTC = ISO8601DateFormatter().string(from: Date())
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: manifest)))
    let mode = try Self.admit(mode: fixture.mode, indices: fixture.blockIndices,
      cache: fixture.cacheLimitBytes, hashes: fixture.hashSmallNorms)
    let output = URL(fileURLWithPath: fixture.output)
    guard !FileManager.default.fileExists(atPath: output.path),
      FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path),
      fixture.metalSHA256 == Self.productMetal,
      let compiled = Bundle(for: Self.self).executableURL,
      compiled.resolvingSymlinksInPath() == URL(fileURLWithPath: fixture.testExecutable).resolvingSymlinksInPath(),
      try digest(compiled.resolvingSymlinksInPath()) == fixture.testExecutableSHA256 else {
      throw H3CheckpointError.invalid("Diagnostic output, product library, or actual test executable is invalid.")
    }
    func verifyPins() throws {
      for pin in fixture.sourcePins {
        let url = URL(fileURLWithPath: pin.path)
        _ = try identity(url)
        guard try digest(url) == pin.sha256 else { throw H3CheckpointError.invalid("Diagnostic source changed: \(pin.path)") }
      }
      for pin in fixture.checkpointPins {
        let url = URL(fileURLWithPath: pin.path)
        guard try identity(url) == pin.identity, try headerDigest(url) == pin.headerSHA256 else {
          throw H3CheckpointError.invalid("Diagnostic checkpoint header changed: \(pin.path)")
        }
      }
    }
    guard Set(fixture.sourcePins.map { URL(fileURLWithPath: $0.path).lastPathComponent })
      .isSuperset(of: ["H3PreparedFastAllPagesTests.swift", "H3PreparedBlock.swift", "H3NativePage.swift", "H3QwenQ8Projection.swift"]),
      fixture.checkpointPins.count == 51 else {
      throw H3CheckpointError.invalid("Diagnostic requires source pins and all 51 checkpoint headers.")
    }
    try verifyPins()
    let checkpoint = URL(fileURLWithPath: fixture.checkpoint)
    let layout = try H3CheckpointLayout(url: checkpoint)
    guard layout.fastVariant == .vsaV1 else { throw H3CheckpointError.invalid("Diagnostic is restricted to the qualified FastVSA checkpoint.") }
    let actualPages = try (0..<50).map { try H3CheckpointSource.fileURL(checkpoint, block: $0).standardizedFileURL }
    let pinnedPages = Set(fixture.checkpointPins.map { URL(fileURLWithPath: $0.path).standardizedFileURL })
    guard Set(actualPages).isSubset(of: pinnedPages), pinnedPages.count == 51 else {
      throw H3CheckpointError.invalid("Diagnostic page pins differ from the admitted checkpoint.")
    }
    let metal = URL(fileURLWithPath: fixture.metalLibrary)
    _ = try identity(metal)
    guard try digest(metal) == Self.productMetal else { throw H3CheckpointError.invalid("Diagnostic Metal library changed.") }
    // Explicitly select the product library before any MLX device or arrays.
    GPU.metallib = metal
    guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("Apple GPU required.") }
    Stream.gpu.synchronize(); Memory.clearCache()
    let baseline = Memory.activeMemory
    let previousLimit = Memory.cacheLimit
    var records: [[String: Any]] = []
    var status = "failed"
    var failure: String?
    var report: [String: Any] = [:]
    do {
      try { () throws -> Void in
        Memory.cacheLimit = Self.cacheBytes
        defer {
          Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit
        }
        for index in 0..<50 {
          try Task.checkCancellation()
          try autoreleasepool {
            Memory.peakMemory = Memory.activeMemory
            let start = CFAbsoluteTimeGetCurrent()
            let owner = try H3PreparedBlock(checkpointURL: checkpoint, index: index,
              projectionMode: .weightDecoded, useMPP: false,
              groupedFastQ8Loading: mode == .grouped)
            defer { owner.close() }
            let preparation = CFAbsoluteTimeGetCurrent() - start
            let peakBeforeHash = Memory.peakMemory
            let footprintBeforeHash = try physical()
            guard owner.usesGroupedFastQ8Loading == (mode == .grouped),
              !owner.usesGroupedBF16Loading, owner.storageBytes == 486_528_512 else {
              throw H3CheckpointError.invalid("Diagnostic prepared-owner policy or storage differs.")
            }
            let hashStart = CFAbsoluteTimeGetCurrent()
            let inspection = try inspect(owner, prefix: layout.prefix + "blocks.\(index).")
            let hashSeconds = CFAbsoluteTimeGetCurrent() - hashStart
            let releaseStart = CFAbsoluteTimeGetCurrent()
            owner.close(); Stream.gpu.synchronize(); Memory.clearCache()
            let releaseSeconds = CFAbsoluteTimeGetCurrent() - releaseStart
            guard owner.isClosed, owner.storageBytes == 0, Memory.activeMemory == baseline,
              Memory.cacheMemory == 0, Memory.cacheLimit == Self.cacheBytes else {
              throw H3CheckpointError.invalid("Diagnostic owner did not release to its original memory baseline.")
            }
            records.append(["blockIndex": index, "preparationSeconds": preparation,
              "untimedFactorHashSeconds": hashSeconds, "releaseSeconds": releaseSeconds,
              "storageBytesBeforeHash": inspection.bytes, "peakMLXBytesBeforeHash": peakBeforeHash,
              "physicalBeforeHash": footprintBeforeHash, "smallNormHashes": inspection.hashes, "admittedFactors": inspection.factors,
              "remainingActiveMLXBytes": Memory.activeMemory, "remainingCacheMLXBytes": Memory.cacheMemory])
          }
        }
        try verifyPins()
      }()
      status = "complete"
    } catch { failure = String(describing: error) }
    report = ["format": "weetodd-fast-all-pages-load-diagnostic-v1", "status": status,
      "mode": mode.rawValue, "records": records, "blockCount": records.count,
      "startedUTC": startedUTC, "finishedUTC": ISO8601DateFormatter().string(from: Date()), "PID": getpid(),
      "sumPreparationSeconds": records.reduce(0.0) { $0 + ($1["preparationSeconds"] as? Double ?? 0) },
      "sumReleaseSeconds": records.reduce(0.0) { $0 + ($1["releaseSeconds"] as? Double ?? 0) },
      "baselineActiveMLXBytes": baseline, "remainingActiveMLXBytes": Memory.activeMemory,
      "remainingCacheMLXBytes": Memory.cacheMemory, "previousCacheLimitBytes": previousLimit,
      "restoredCacheLimitBytes": Memory.cacheLimit, "physicalAfter": try physical(),
      "testExecutableSHA256": fixture.testExecutableSHA256, "metalSHA256": fixture.metalSHA256,
      "fixture": manifest, "fixtureSHA256": try digest(URL(fileURLWithPath: manifest)),
      "scope": "One declared loading mode in one fresh process; one resident owner; four small norm host hashes and factor metadata excluded from load timers but interleaved between pages. No projection, attention, transformer, or sampler execution. OS file-cache state is not controlled."]
    if let failure { report["error"] = failure }
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: output, options: .withoutOverwriting)
    guard status == "complete", Memory.activeMemory == baseline,
      Memory.cacheMemory == 0, Memory.cacheLimit == previousLimit else {
      throw H3CheckpointError.invalid("Fast all-page preparation failed; inspect the retained diagnostic receipt.")
    }
  }
}
