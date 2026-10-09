import Accelerate
import CryptoKit
import Darwin
import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

/// Numerical falsifier only. No production loader, prefetch queue or default changes.
final class H3ComfyCPUDecodeProbeTests: XCTestCase {
  private struct Fixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    struct ModelPin: Decodable { let path: String; let headerSHA256: String; let identity: [Int64] }
    let checkpoint: String
    let checkpointPin: ModelPin
    let sourcePins: [Pin]
    let metalLibrary: String
    let metalSHA256: String
    let testExecutable: String
    let testExecutableSHA256: String
    let output: String
  }
  private let productMetal = "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9"

  private static func roundBF16(_ value: Float) throws -> UInt16 {
    guard value.isFinite else { throw H3CheckpointError.invalid("CPU decode produced nonfinite F32.") }
    let bits = value.bitPattern
    let rounded = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    guard rounded & 0x7f80 != 0x7f80 else {
      throw H3CheckpointError.invalid("CPU BF16 rounding overflowed.")
    }
    return rounded
  }

  func testCPUFiniteRoundToNearestEvenAtPositiveNegativeTiesAndSubnormals() throws {
    let cases: [(UInt32, UInt16)] = [
      (0x3f808000, 0x3f80), (0x3f818000, 0x3f82),
      (0xbf808000, 0xbf80), (0xbf818000, 0xbf82),
      (0x00008000, 0), (0x00018000, 2), (0x80000000, 0x8000),
      (0x3f807fff, 0x3f80), (0x3f808001, 0x3f81),
    ]
    for (source, expected) in cases { XCTAssertEqual(try Self.roundBF16(Float(bitPattern: source)), expected) }
    for value: Float in [.infinity, -.infinity, .nan, Float.greatestFiniteMagnitude] {
      XCTAssertThrowsError(try Self.roundBF16(value))
    }
  }

  private static func basis256() -> [Float] {
    var basis = [Float](repeating: 0, count: 256 * 256)
    for row in 0..<256 {
      for column in 0..<256 {
        var sign: Float = 1
        var place = 1
        while place < 256 {
          if (row / place) % 4 + (column / place) % 4 == 3 { sign = -sign }
          place *= 4
        }
        basis[row * 256 + column] = sign / 16
      }
    }
    return basis
  }

  func testCPURegularBasisIsNormalizedAndNotSylvester() {
    let basis = Self.basis256()
    XCTAssertEqual(basis[0], 1 / 16)
    XCTAssertEqual(basis[3], -1 / 16)
    XCTAssertEqual(basis[255], 1 / 16)
    for column in [0, 1, 19, 255] {
      let dot = (0..<256).reduce(Float(0)) { $0 + basis[$1] * basis[$1 * 256 + column] }
      XCTAssertEqual(dot, column == 0 ? 1 : 0)
    }
  }

  private func digest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func identity(_ url: URL) throws -> [Int64] {
    var s = stat()
    guard url.path.withCString({ Darwin.lstat($0, &s) }) == 0,
      s.st_mode & S_IFMT == S_IFREG,
      FileManager.default.isReadableFile(atPath: url.path) else {
      throw H3CheckpointError.invalid("CPU decode fixture requires readable regular files.")
    }
    return [Int64(s.st_dev), Int64(s.st_ino), s.st_size,
      Int64(s.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(s.st_mtimespec.tv_nsec),
      Int64(s.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(s.st_ctimespec.tv_nsec)]
  }
  private func headerDigest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let length = try handle.read(upToCount: 8), length.count == 8 else {
      throw H3CheckpointError.invalid("Missing safetensors header.")
    }
    let size = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard size >= 2, size <= 16 * 1024 * 1024,
      let header = try handle.read(upToCount: Int(size)), header.count == Int(size) else {
      throw H3CheckpointError.invalid("Invalid safetensors header.")
    }
    return SHA256.hash(data: header).map { String(format: "%02x", $0) }.joined()
  }
  private func physical() throws -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    let peak = UInt64(max(0, info.ledger_phys_footprint_peak))
    guard status == KERN_SUCCESS, info.phys_footprint > 0, peak >= info.phys_footprint else {
      throw H3CheckpointError.invalid("Invalid CPU decode physical query.")
    }
    return (info.phys_footprint, peak)
  }

  private func admit(file: SafeTensorFile, name: String, rows: Int, columns: Int) throws -> [Float] {
    let stem = String(name.dropLast(".weight".count))
    guard let weight = file.tensors[name], weight.dtype == "I8",
      weight.shape == [UInt64(rows), UInt64(columns)], columns.isMultiple(of: 256),
      let scale = file.tensors[stem + ".weight_scale"], scale.dtype == "F32",
      scale.shape == [UInt64(rows), 1],
      let marker = file.tensors[stem + ".comfy_quant"], marker.dtype == "U8",
      marker.byteCount > 0, marker.byteCount <= 4096 else {
      throw H3CheckpointError.invalid("CPU decode requires exact I8/F32/group256 shapes.")
    }
    let data: Data = try file.withTensorBytes(named: stem + ".comfy_quant",
      range: 0..<marker.byteCount, access: .buffered) { Data($0) }
    guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      value["format"] as? String == "int8_tensorwise", value["convrot"] as? Bool == true,
      (value["convrot_groupsize"] as? Int ?? 256) == 256,
      Set(value.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]) else {
      throw H3CheckpointError.invalid("CPU decode requires regular ConvRot group256.")
    }
    let scales = try file.readFloat32(named: stem + ".weight_scale", access: .buffered)
    guard scales.count == rows, scales.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("CPU decode requires finite positive row scales.")
    }
    return scales
  }

  /// Independent host-only version of the existing NNC algorithm, retaining BF16.
  /// Only 1024 source rows enter F32 scratch; no MLX/NNC call or second GPU owner.
  private func decodeCPU(file: SafeTensorFile, name: String, rows: Int, columns: Int,
    scales: [Float], basis: [Float]) throws -> [UInt16] {
    var words = [UInt16](repeating: 0, count: rows * columns)
    for start in stride(from: 0, to: rows, by: 1024) {
      try Task.checkCancellation()
      let stop = min(rows, start + 1024), elements = (stop - start) * columns
      var scaled = [Float](repeating: 0, count: elements)
      try file.withTensorBytes(named: name,
        range: UInt64(start * columns)..<UInt64(stop * columns), access: .mapped) { raw in
        let q = raw.bindMemory(to: Int8.self)
        for i in 0..<elements { scaled[i] = Float(q[i]) * scales[start + i / columns] }
      }
      var restored = [Float](repeating: 0, count: elements)
      scaled.withUnsafeBufferPointer { source in
        basis.withUnsafeBufferPointer { rotation in
          restored.withUnsafeMutableBufferPointer { output in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              Int32(elements / 256), 256, 256, 1, source.baseAddress!, 256,
              rotation.baseAddress!, 256, 0, output.baseAddress!, 256)
          }
        }
      }
      for i in 0..<elements { words[start * columns + i] = try Self.roundBF16(restored[i]) }
      try Task.checkCancellation()
    }
    try file.checkUnchanged()
    return words
  }

  private func compare(file: SafeTensorFile, url: URL, name: String,
    rows: Int, columns: Int, scales: [Float], basis: [Float]) throws -> [String: Any] {
    try autoreleasepool {
      Stream.gpu.synchronize(); Memory.clearCache(); Memory.peakMemory = Memory.activeMemory
      let cpuStart = ProcessInfo.processInfo.systemUptime
      let cpu = try decodeCPU(file: file, name: name, rows: rows, columns: columns,
        scales: scales, basis: basis)
      let cpuSeconds = ProcessInfo.processInfo.systemUptime - cpuStart
      let cpuPhysical = try physical()
      let gpuStart = ProcessInfo.processInfo.systemUptime
      let gpu = try H3ComfyDecodedProjection.load(file: file, checkpointURL: url,
        name: name, rows: rows, columns: columns, reorderQKV: false, rowWindow: 16384)
      let gpuSeconds = ProcessInfo.processInfo.systemUptime - gpuStart
      let peakBeforeHash = Memory.peakMemory
      let physicalBeforeHash = try physical()
      var cpuHash = SHA256(), gpuHash = SHA256(), mismatches = 0
      var maximumError: Double = 0, allFinite = true
      var firstDifferences: [[String: Any]] = []
      let validationStart = ProcessInfo.processInfo.systemUptime
      for start in stride(from: 0, to: rows, by: 1024) {
        let stop = min(rows, start + 1024), offset = start * columns
        try autoreleasepool {
          let actual = gpu[start..<stop, 0..<columns].view(dtype: .uint16).asArray(UInt16.self)
          let reference = Array(cpu[offset..<(stop * columns)])
          cpuHash.update(data: reference.withUnsafeBytes { Data($0) })
          gpuHash.update(data: actual.withUnsafeBytes { Data($0) })
          for i in actual.indices {
            let a = actual[i], b = reference[i]
            allFinite = allFinite && a & 0x7f80 != 0x7f80 && b & 0x7f80 != 0x7f80
            let error = abs(Double(Float(bitPattern: UInt32(a) << 16))
              - Double(Float(bitPattern: UInt32(b) << 16)))
            if error.isFinite { maximumError = max(maximumError, error) }
            if a != b {
              mismatches += 1
              if firstDifferences.count < 12 {
                firstDifferences.append(["element": offset + i, "CPUWord": b, "GPUWord": a])
              }
            }
          }
        }
      }
      try file.checkUnchanged(at: url)
      return ["name": name, "shape": [rows, columns], "elements": rows * columns,
        "CPUDecodeSeconds": cpuSeconds, "GPUCurrentDecodeSeconds": gpuSeconds,
        "untimedFullWordComparisonSeconds": ProcessInfo.processInfo.systemUptime - validationStart,
        "CPUWordSHA256": cpuHash.finalize().map { String(format: "%02x", $0) }.joined(),
        "GPUWordSHA256": gpuHash.finalize().map { String(format: "%02x", $0) }.joined(),
        "differentBF16Words": mismatches, "maximumAbsoluteBF16Error": maximumError,
        "allFinite": allFinite, "firstDifferences": firstDifferences,
        "peakMLXBytesBeforeHostHash": peakBeforeHash,
        "physicalAfterCPUDecodeBytes": cpuPhysical.current,
        "physicalBeforeHostHashBytes": physicalBeforeHash.current,
        "processLifetimePhysicalPeakBeforeHostHashBytes": physicalBeforeHash.peak,
        "CPUWordStorageBytes": cpu.count * 2,
        "maximumCPURowCount": min(1024, rows), "reorderQKV": false]
    }
  }

  private func synthetic() throws -> URL {
    let rows = 1025, columns = 256
    let bitPatterns: [UInt32] = [0x3f800001, 0x3f807fff, 0x3f808000, 0x3f808001,
      0x33ac4321, 0x00010001, 0x70000001]
    let q = (0..<(rows * columns)).map { UInt8(truncatingIfNeeded: ($0 * 31 + $0 / columns * 19) % 256) }
    var payload = Data(q)
    let scaleStart = payload.count
    payload.append((0..<rows).map { bitPatterns[$0 % bitPatterns.count].littleEndian }
      .withUnsafeBytes { Data($0) })
    let markerStart = payload.count
    let marker = try JSONSerialization.data(withJSONObject: ["format": "int8_tensorwise",
      "convrot": true, "convrot_groupsize": 256])
    payload.append(marker)
    let header: [String: Any] = [
      "probe.weight": ["dtype": "I8", "shape": [rows, columns], "data_offsets": [0, scaleStart]],
      "probe.weight_scale": ["dtype": "F32", "shape": [rows, 1], "data_offsets": [scaleStart, markerStart]],
      "probe.comfy_quant": ["dtype": "U8", "shape": [marker.count], "data_offsets": [markerStart, payload.count]]]
    let encoded = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(encoded.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }; bytes.append(encoded); bytes.append(payload)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    try bytes.write(to: url); return url
  }

  func testInstalledFourComfyINT8CPUDecodesAgainstCurrentGPUCompleteBF16Words() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_CPU_CONVROT_FIXTURE"] else {
      throw XCTSkip("Select the pinned CPU ConvRot numerical falsifier fixture explicitly.")
    }
    let fixtureURL = URL(fileURLWithPath: path)
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
    guard !FileManager.default.fileExists(atPath: fixture.output),
      fixture.metalSHA256 == productMetal,
      try digest(URL(fileURLWithPath: fixture.metalLibrary)) == fixture.metalSHA256,
      try digest(URL(fileURLWithPath: fixture.testExecutable)) == fixture.testExecutableSHA256,
      let invoked = Bundle(for: Self.self).executableURL,
      invoked.resolvingSymlinksInPath().path
        == URL(fileURLWithPath: fixture.testExecutable).resolvingSymlinksInPath().path else {
      throw H3CheckpointError.invalid("CPU ConvRot fixture/library/test binding differs.")
    }
    _ = try identity(URL(fileURLWithPath: fixture.metalLibrary))
    _ = try identity(URL(fileURLWithPath: fixture.testExecutable))
    for pin in fixture.sourcePins {
      guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else {
        throw H3CheckpointError.invalid("CPU ConvRot source changed before payload.")
      }
    }
    let required = ["H3ComfyCPUDecodeProbeTests.swift", "H3ComfyDecodedProjection.swift",
      "H3PreparedBlock.swift", "H3CheckpointLayout.swift", "SafeTensorFile.swift", "Package.swift", "Package.resolved"]
    guard required.allSatisfy({ suffix in fixture.sourcePins.contains { $0.path.hasSuffix("/" + suffix) } }),
      fixture.checkpointPin.path == fixture.checkpoint,
      try identity(URL(fileURLWithPath: fixture.checkpoint)) == fixture.checkpointPin.identity,
      try headerDigest(URL(fileURLWithPath: fixture.checkpoint)) == fixture.checkpointPin.headerSHA256 else {
      throw H3CheckpointError.invalid("CPU ConvRot required source/header/stat binding differs.")
    }
    let url = URL(fileURLWithPath: fixture.checkpoint), file = try SafeTensorFile(url: url)
    let layout = try H3CheckpointLayout(url: url)
    guard layout.fastVariant == nil, layout.curveRank == nil, layout.quantizedProjections == 250 else {
      throw H3CheckpointError.invalid("CPU ConvRot probe requires full-width ordinary Comfy INT8.")
    }
    let definitions = [("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168),
      ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336)]
    let admitted = try definitions.map { suffix, rows, columns in
      try admit(file: file, name: layout.prefix + "blocks.0." + suffix + ".weight", rows: rows, columns: columns)
    }
    try file.checkUnchanged(at: url)
    GPU.metallib = URL(fileURLWithPath: fixture.metalLibrary)
    Stream.gpu.synchronize(); Memory.clearCache()
    let baseline = Memory.activeMemory, previousCacheLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousCacheLimit }
    let basis = Self.basis256()
    var records: [[String: Any]] = []
    for (index, definition) in definitions.enumerated() {
      let (suffix, rows, columns) = definition
      records.append(try compare(file: file, url: url,
        name: layout.prefix + "blocks.0." + suffix + ".weight", rows: rows, columns: columns,
        scales: admitted[index], basis: basis))
      Stream.gpu.synchronize(); Memory.clearCache()
      XCTAssertEqual(Memory.activeMemory, baseline)
    }
    let syntheticURL = try synthetic()
    defer { try? FileManager.default.removeItem(at: syntheticURL) }
    let syntheticRecord = try autoreleasepool { () -> [String: Any] in
      let syntheticFile = try SafeTensorFile(url: syntheticURL)
      let scales = try admit(file: syntheticFile, name: "probe.weight", rows: 1025, columns: 256)
      return try compare(file: syntheticFile, url: syntheticURL, name: "probe.weight",
        rows: 1025, columns: 256, scales: scales, basis: basis)
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    let remaining = Memory.activeMemory, remainingCache = Memory.cacheMemory, finalPhysical = try physical()
    Memory.cacheLimit = previousCacheLimit
    let passed = (records + [syntheticRecord]).allSatisfy {
      $0["differentBF16Words"] as? Int == 0 && $0["allFinite"] as? Bool == true
    }
    for pin in fixture.sourcePins {
      guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else {
        throw H3CheckpointError.invalid("CPU ConvRot source changed after comparison.")
      }
    }
    try file.checkUnchanged(at: url)
    let report: [String: Any] = ["format": "weetodd-comfy-cpu-convrot-falsifier-v1",
      "numericalPassed": passed, "productionQualified": false, "prefetchImplemented": false,
      "checkpoint": fixture.checkpoint, "blockIndex": 0, "group": 256,
      "CPUDecodeRowWindow": 1024, "GPUDecodeRowWindow": 16384, "records": records,
      "synthetic": syntheticRecord,
      "syntheticScaleF32Bits": [0x3f800001, 0x3f807fff, 0x3f808000, 0x3f808001, 0x33ac4321, 0x00010001, 0x70000001],
      "oneFutureCoreBF16LogicalBytes": 770_703_360,
      "maximumTwoF32RowScratchBytes": 117_440_512,
      "maximumMappedI8RowSpanBytes": 14_680_064,
      "CPUFutureAndScratchAnalyticBytes": 902_823_936,
      "CPUAllocationScope": "Analytic payload/scratch only; not a physical cap, Accelerate allocator workspace excluded. Probe holds one CPU projection plus current GPU projection, not a whole prefetched owner.",
      "physicalScope": "Shared test-process lifetime; includes prior host comparison and CPU arrays. No per-mode/reset or whole-job memory claim.",
      "timingScope": "Sequential isolated CPU decode and current GPU load including decode; full word comparison excluded. No overlap or speed qualification.",
      "activeBaselineMLXBytes": baseline, "remainingActiveMLXBytes": remaining,
      "remainingCacheMLXBytes": remainingCache,
      "previousCacheLimitBytes": previousCacheLimit, "restoredCacheLimitBytes": Memory.cacheLimit,
      "finalPhysicalFootprintBytes": finalPhysical.current,
      "processLifetimePeakPhysicalBytes": finalPhysical.peak,
      "fixtureSHA256": try digest(fixtureURL), "metalSHA256": fixture.metalSHA256,
      "testExecutableSHA256": fixture.testExecutableSHA256]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    XCTAssertEqual(remaining, baseline); XCTAssertEqual(remainingCache, 0)
    XCTAssertEqual(Memory.cacheLimit, previousCacheLimit)
    XCTAssertTrue(passed, "Any CPU/GPU word mismatch rejects exact runtime promotion; evidence retained.")
  }
}

/// Same Float32 SGEMM arithmetic, with measured host conversion phases only.
final class H3ComfyCPUDecodePhaseTests: XCTestCase {
  private struct Fixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    struct ModelPin: Decodable { let path: String; let headerSHA256: String; let identity: [Int64] }
    let checkpoint: String; let checkpointPin: ModelPin; let sourcePins: [Pin]
    let metalLibrary: String; let metalSHA256: String
    let testExecutable: String; let testExecutableSHA256: String; let output: String
  }
  private struct Phases {
    var mapScale: Double = 0, sgemm: Double = 0, roundStore: Double = 0
    var total: Double = 0
    var json: [String: Double] { ["mapScaleSeconds": mapScale, "SGEMMSeconds": sgemm,
      "roundStoreSeconds": roundStore, "totalCPUDecodeSeconds": total] }
  }
  private static func admitted(_ scales: [Float]) throws {
    // This probe deliberately excludes subnormal and extreme scales before GPU work.
    guard !scales.isEmpty, scales.allSatisfy({ $0.isFinite && $0 >= 1e-20 && $0 <= 1e20 }) else {
      throw H3CheckpointError.invalid("CPU phase probe requires normal row scales in [1e-20, 1e20].")
    }
  }
  private static func scalarRound(_ value: Float) throws -> UInt16 {
    guard value.isFinite else { throw H3CheckpointError.invalid("CPU decode produced nonfinite F32.") }
    let bits = value.bitPattern
    let rounded = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    guard rounded & 0x7f80 != 0x7f80 else { throw H3CheckpointError.invalid("CPU BF16 rounding overflowed.") }
    return rounded
  }
  private static func vectorRound(_ source: UnsafeBufferPointer<Float>,
    _ output: UnsafeMutableBufferPointer<UInt16>) {
    let count = source.count, shift = SIMD8<UInt32>(repeating: 16)
    let one = SIMD8<UInt32>(repeating: 1), bias = SIMD8<UInt32>(repeating: 0x7fff)
    var i = 0
    while i + 8 <= count {
      let bits = UnsafeRawPointer(source.baseAddress!).advanced(by: i * 4)
        .loadUnaligned(as: SIMD8<UInt32>.self)
      let rounded = (bits &+ bias &+ ((bits &>> shift) & one)) &>> shift
      for lane in 0..<8 { output[i + lane] = UInt16(truncatingIfNeeded: rounded[lane]) }
      i += 8
    }
    while i < count {
      let bits = source[i].bitPattern
      output[i] = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
      i += 1
    }
  }
  func testCPUPhaseAdmissionRejectsSubnormalAndUnsafeScales() throws {
    try Self.admitted([0.00023789293, 0.0098732775])
    for value: Float in [Float(bitPattern: 0x00010001), 0, -1, .nan, .infinity, 1e-21, 1e21] {
      XCTAssertThrowsError(try Self.admitted([value]))
    }
  }
  func testCPUVectorRoundMatchesScalarTiesSignsAndTail() throws {
    let bits: [UInt32] = [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
      0x00008000, 0x00018000, 0x80000000, 0x3f807fff, 0x3f808001,
      0x3b866cda, 0xbb866cda, 0x00000000, 0x7f000001, 0xff000001, 0x00800000,
      0x80800000, 0x3c000001]
    let values = bits.map(Float.init(bitPattern:))
    var actual = [UInt16](repeating: 0, count: values.count)
    values.withUnsafeBufferPointer { source in actual.withUnsafeMutableBufferPointer { Self.vectorRound(source, $0) } }
    XCTAssertEqual(actual, try values.map(Self.scalarRound))
  }
  private func digest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func identity(_ url: URL) throws -> [Int64] {
    var s = stat()
    guard url.path.withCString({ Darwin.lstat($0, &s) }) == 0,
      s.st_mode & S_IFMT == S_IFREG, FileManager.default.isReadableFile(atPath: url.path) else {
      throw H3CheckpointError.invalid("CPU phase fixture requires readable regular files.")
    }
    return [Int64(s.st_dev), Int64(s.st_ino), s.st_size,
      Int64(s.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(s.st_mtimespec.tv_nsec),
      Int64(s.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(s.st_ctimespec.tv_nsec)]
  }
  private func headerDigest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    guard let length = try handle.read(upToCount: 8), length.count == 8 else { throw H3CheckpointError.invalid("Missing header.") }
    let size = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard size >= 2, size <= 16 * 1024 * 1024,
      let header = try handle.read(upToCount: Int(size)), header.count == Int(size) else { throw H3CheckpointError.invalid("Invalid header.") }
    return SHA256.hash(data: header).map { String(format: "%02x", $0) }.joined()
  }
  private func physical() throws -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    let peak = UInt64(max(0, info.ledger_phys_footprint_peak))
    guard status == KERN_SUCCESS, info.phys_footprint > 0, peak >= info.phys_footprint else { throw H3CheckpointError.invalid("Invalid physical query.") }
    return (info.phys_footprint, peak)
  }
  private static func basis() -> [Float] {
    var result = [Float](repeating: 0, count: 65536)
    for row in 0..<256 { for column in 0..<256 {
      var sign: Float = 1, place = 1
      while place < 256 { if (row / place) % 4 + (column / place) % 4 == 3 { sign = -sign }; place *= 4 }
      result[row * 256 + column] = sign / 16
    } }
    return result
  }
  private func decode(file: SafeTensorFile, name: String, rows: Int, columns: Int,
    scales: [Float], basis: [Float], vectorized: Bool) throws -> (words: [UInt16], phases: Phases) {
    let totalStart = ProcessInfo.processInfo.systemUptime
    var words = [UInt16](repeating: 0, count: rows * columns), phases = Phases()
    for start in stride(from: 0, to: rows, by: 1024) {
      try Task.checkCancellation()
      let stop = min(rows, start + 1024), elements = (stop - start) * columns
      var phaseStart = ProcessInfo.processInfo.systemUptime
      var scaled = [Float](repeating: 0, count: elements)
      try file.withTensorBytes(named: name, range: UInt64(start * columns)..<UInt64(stop * columns), access: .mapped) { raw in
        let q = raw.bindMemory(to: Int8.self)
        if vectorized {
          scaled.withUnsafeMutableBufferPointer { output in
            for row in 0..<(stop - start) {
              let offset = row * columns
              vDSP_vflt8(q.baseAddress!.advanced(by: offset), 1,
                output.baseAddress!.advanced(by: offset), 1, vDSP_Length(columns))
              var scale = scales[start + row]
              vDSP_vsmul(output.baseAddress!.advanced(by: offset), 1, &scale,
                output.baseAddress!.advanced(by: offset), 1, vDSP_Length(columns))
            }
          }
        } else {
          for i in 0..<elements { scaled[i] = Float(q[i]) * scales[start + i / columns] }
        }
      }
      phases.mapScale += ProcessInfo.processInfo.systemUptime - phaseStart
      phaseStart = ProcessInfo.processInfo.systemUptime
      var restored = [Float](repeating: 0, count: elements)
      scaled.withUnsafeBufferPointer { source in basis.withUnsafeBufferPointer { rotation in restored.withUnsafeMutableBufferPointer { output in
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(elements / 256), 256, 256,
          1, source.baseAddress!, 256, rotation.baseAddress!, 256, 0, output.baseAddress!, 256)
      } } }
      phases.sgemm += ProcessInfo.processInfo.systemUptime - phaseStart
      phaseStart = ProcessInfo.processInfo.systemUptime
      if vectorized {
        restored.withUnsafeBufferPointer { source in words.withUnsafeMutableBufferPointer { output in
          Self.vectorRound(source, UnsafeMutableBufferPointer(start: output.baseAddress!.advanced(by: start * columns), count: elements))
        } }
      } else {
        for i in 0..<elements { words[start * columns + i] = try Self.scalarRound(restored[i]) }
      }
      phases.roundStore += ProcessInfo.processInfo.systemUptime - phaseStart
      try Task.checkCancellation()
    }
    try file.checkUnchanged(); phases.total = ProcessInfo.processInfo.systemUptime - totalStart
    return (words, phases)
  }
  func testInstalledNormalComfyINT8SameSGEMMCPUPhases() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_CPU_CONVROT_FIXTURE"] else { throw XCTSkip("Select a fresh pinned CPU phase fixture explicitly.") }
    let fixtureURL = URL(fileURLWithPath: path)
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
    let productMetal = "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9"
    let url = URL(fileURLWithPath: fixture.checkpoint)
    guard !FileManager.default.fileExists(atPath: fixture.output), fixture.metalSHA256 == productMetal,
      try digest(URL(fileURLWithPath: fixture.metalLibrary)) == productMetal,
      try digest(URL(fileURLWithPath: fixture.testExecutable)) == fixture.testExecutableSHA256,
      Bundle(for: Self.self).executableURL?.resolvingSymlinksInPath().path == URL(fileURLWithPath: fixture.testExecutable).resolvingSymlinksInPath().path,
      fixture.checkpointPin.path == fixture.checkpoint, try identity(url) == fixture.checkpointPin.identity,
      try headerDigest(url) == fixture.checkpointPin.headerSHA256 else { throw H3CheckpointError.invalid("CPU phase fixture binding differs.") }
    _ = try identity(URL(fileURLWithPath: fixture.metalLibrary)); _ = try identity(URL(fileURLWithPath: fixture.testExecutable))
    let required = ["H3ComfyCPUDecodeProbeTests.swift", "H3ComfyDecodedProjection.swift", "H3CheckpointLayout.swift", "SafeTensorFile.swift", "Package.swift", "Package.resolved"]
    guard required.allSatisfy({ suffix in fixture.sourcePins.contains { $0.path.hasSuffix("/" + suffix) } }) else { throw H3CheckpointError.invalid("Required source pins missing.") }
    for pin in fixture.sourcePins { guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else { throw H3CheckpointError.invalid("CPU phase source changed.") } }
    let file = try SafeTensorFile(url: url), layout = try H3CheckpointLayout(url: url)
    guard layout.fastVariant == nil, layout.curveRank == nil, layout.quantizedProjections == 250 else { throw H3CheckpointError.invalid("Requires ordinary full-width Comfy INT8.") }
    let definitions = [("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168), ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336)]
    var allScales: [[Float]] = []
    for (suffix, rows, columns) in definitions {
      let stem = layout.prefix + "blocks.0." + suffix
      guard let w = file.tensors[stem + ".weight"], w.dtype == "I8", w.shape == [UInt64(rows), UInt64(columns)],
        let s = file.tensors[stem + ".weight_scale"], s.dtype == "F32", s.shape == [UInt64(rows), 1],
        let m = file.tensors[stem + ".comfy_quant"], m.dtype == "U8", m.byteCount > 0, m.byteCount <= 4096 else { throw H3CheckpointError.invalid("Invalid core descriptor.") }
      let marker: Data = try file.withTensorBytes(named: stem + ".comfy_quant", range: 0..<m.byteCount, access: .buffered) { Data($0) }
      guard let info = try JSONSerialization.jsonObject(with: marker) as? [String: Any], info["format"] as? String == "int8_tensorwise", info["convrot"] as? Bool == true,
        (info["convrot_groupsize"] as? Int ?? 256) == 256, Set(info.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]) else { throw H3CheckpointError.invalid("Invalid group256 marker.") }
      let scales = try file.readFloat32(named: stem + ".weight_scale", access: .buffered)
      guard scales.count == rows else { throw H3CheckpointError.invalid("Invalid scales.") }
      try Self.admitted(scales); allScales.append(scales)
    }
    try file.checkUnchanged(at: url)
    GPU.metallib = URL(fileURLWithPath: fixture.metalLibrary)
    Stream.gpu.synchronize(); Memory.clearCache()
    let baseline = Memory.activeMemory, previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    let basis = Self.basis(); var records: [[String: Any]] = []
    for (index, definition) in definitions.enumerated() {
      let (suffix, rows, columns) = definition, name = layout.prefix + "blocks.0." + suffix + ".weight"
      records.append(try autoreleasepool {
        let serial = try decode(file: file, name: name, rows: rows, columns: columns, scales: allScales[index], basis: basis, vectorized: false)
        let candidate = try decode(file: file, name: name, rows: rows, columns: columns, scales: allScales[index], basis: basis, vectorized: true)
        let cpuPhysical = try physical()
        Memory.peakMemory = Memory.activeMemory
        let gpuStart = ProcessInfo.processInfo.systemUptime
        let gpu = try H3ComfyDecodedProjection.load(file: file, checkpointURL: url, name: name,
          rows: rows, columns: columns, reorderQKV: false, rowWindow: 16384)
        let gpuSeconds = ProcessInfo.processInfo.systemUptime - gpuStart, peak = Memory.peakMemory
        let beforeHash = try physical()
        var originalHash = SHA256(), candidateHash = SHA256(), gpuHash = SHA256()
        var differentOriginal = 0, differentGPU = 0, allFinite = true
        for start in stride(from: 0, to: rows, by: 1024) {
          let stop = min(rows, start + 1024), offset = start * columns
          try autoreleasepool {
            let actual = gpu[start..<stop, 0..<columns].view(dtype: .uint16).asArray(UInt16.self)
            let original = Array(serial.words[offset..<(stop * columns)])
            let transformed = Array(candidate.words[offset..<(stop * columns)])
            originalHash.update(data: original.withUnsafeBytes { Data($0) }); candidateHash.update(data: transformed.withUnsafeBytes { Data($0) }); gpuHash.update(data: actual.withUnsafeBytes { Data($0) })
            for i in actual.indices {
              differentOriginal += original[i] == transformed[i] ? 0 : 1
              differentGPU += actual[i] == transformed[i] ? 0 : 1
              allFinite = allFinite && actual[i] & 0x7f80 != 0x7f80 && transformed[i] & 0x7f80 != 0x7f80 && original[i] & 0x7f80 != 0x7f80
            }
          }
        }
        return ["name": name, "shape": [rows, columns], "elements": rows * columns,
          "baselineCPUPhases": serial.phases.json, "candidateCPUPhases": candidate.phases.json,
          "baselineCPUWordSHA256": originalHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "candidateCPUWordSHA256": candidateHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "GPUWordSHA256": gpuHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "differentBaselineWords": differentOriginal, "differentGPUWords": differentGPU,
          "allFinite": allFinite, "GPUDecodeSeconds": gpuSeconds, "peakMLXBytesBeforeHostHash": peak,
          "physicalAfterBothCPUDecodesBytes": cpuPhysical.current, "physicalBeforeHostHashBytes": beforeHash.current,
          "processLifetimePhysicalPeakBeforeHostHashBytes": beforeHash.peak, "reorderQKV": false] as [String: Any]
      })
      Stream.gpu.synchronize(); Memory.clearCache(); XCTAssertEqual(Memory.activeMemory, baseline)
    }
    try file.checkUnchanged(at: url)
    for pin in fixture.sourcePins { guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else { throw H3CheckpointError.invalid("CPU phase source changed after work.") } }
    let remaining = Memory.activeMemory, cache = Memory.cacheMemory
    Memory.cacheLimit = previousLimit
    let passed = records.allSatisfy { $0["differentBaselineWords"] as? Int == 0 && $0["differentGPUWords"] as? Int == 0 && $0["allFinite"] as? Bool == true }
    let candidateSeconds = records.reduce(0.0) { sum, record in sum + ((record["candidateCPUPhases"] as? [String: Double])?["totalCPUDecodeSeconds"] ?? 0) }
    let report: [String: Any] = ["format": "weetodd-comfy-cpu-sgemm-phases-v2", "numericalPassed": passed,
      "productionQualified": false, "prefetchImplemented": false, "records": records,
      "candidateWholeCoreDecodeSeconds": candidateSeconds, "candidateFitsFourSecondComputeWindow": candidateSeconds < 4,
      "scope": "One sequential baseline/candidate observation per matrix, same F32 SGEMM; no warmed speed qualification or overlap. Physical retains both CPU forms plus GPU and includes earlier matrices.",
      "supportedScaleRange": [1e-20, 1e20], "subnormalScalesUnsupported": true,
      "oneFutureCoreBF16LogicalBytes": 770_703_360, "maximumTwoF32RowScratchBytes": 117_440_512,
      "activeBaselineMLXBytes": baseline, "remainingActiveMLXBytes": remaining, "remainingCacheMLXBytes": cache,
      "previousCacheLimitBytes": previousLimit, "restoredCacheLimitBytes": Memory.cacheLimit,
      "fixtureSHA256": try digest(fixtureURL), "metalSHA256": productMetal, "testExecutableSHA256": fixture.testExecutableSHA256]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    XCTAssertTrue(passed); XCTAssertEqual(remaining, baseline); XCTAssertEqual(cache, 0)
    XCTAssertEqual(Memory.cacheLimit, previousLimit)
  }
}

/// Same Float32 SGEMM arithmetic, with measured host conversion phases only.
final class H3ComfyCPUBufferedDecodePhaseTests: XCTestCase {
  private struct Fixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    struct ModelPin: Decodable { let path: String; let headerSHA256: String; let identity: [Int64] }
    let checkpoint: String; let checkpointPin: ModelPin; let sourcePins: [Pin]
    let metalLibrary: String; let metalSHA256: String
    let testExecutable: String; let testExecutableSHA256: String; let output: String
  }
  private struct Phases {
    var read: Double = 0, scale: Double = 0, sgemm: Double = 0, roundStore: Double = 0
    var total: Double = 0
    var json: [String: Double] { ["readSeconds": read, "scaleSeconds": scale, "SGEMMSeconds": sgemm,
      "roundStoreSeconds": roundStore, "totalCPUDecodeSeconds": total] }
  }
  private static func admitted(_ scales: [Float]) throws {
    // This probe deliberately excludes subnormal and extreme scales before GPU work.
    guard !scales.isEmpty, scales.allSatisfy({ $0.isFinite && $0 >= 1e-20 && $0 <= 1e20 }) else {
      throw H3CheckpointError.invalid("CPU phase probe requires normal row scales in [1e-20, 1e20].")
    }
  }
  private static func scalarRound(_ value: Float) throws -> UInt16 {
    guard value.isFinite else { throw H3CheckpointError.invalid("CPU decode produced nonfinite F32.") }
    let bits = value.bitPattern
    let rounded = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    guard rounded & 0x7f80 != 0x7f80 else { throw H3CheckpointError.invalid("CPU BF16 rounding overflowed.") }
    return rounded
  }
  private static func vectorRound(_ source: UnsafeBufferPointer<Float>,
    _ output: UnsafeMutableBufferPointer<UInt16>) {
    let count = source.count, shift = SIMD8<UInt32>(repeating: 16)
    let one = SIMD8<UInt32>(repeating: 1), bias = SIMD8<UInt32>(repeating: 0x7fff)
    var i = 0
    while i + 8 <= count {
      let bits = UnsafeRawPointer(source.baseAddress!).advanced(by: i * 4)
        .loadUnaligned(as: SIMD8<UInt32>.self)
      let rounded = (bits &+ bias &+ ((bits &>> shift) & one)) &>> shift
      for lane in 0..<8 { output[i + lane] = UInt16(truncatingIfNeeded: rounded[lane]) }
      i += 8
    }
    while i < count {
      let bits = source[i].bitPattern
      output[i] = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
      i += 1
    }
  }
  func testCPUPhaseAdmissionRejectsSubnormalAndUnsafeScales() throws {
    try Self.admitted([0.00023789293, 0.0098732775])
    for value: Float in [Float(bitPattern: 0x00010001), 0, -1, .nan, .infinity, 1e-21, 1e21] {
      XCTAssertThrowsError(try Self.admitted([value]))
    }
  }
  func testCPUVectorRoundMatchesScalarTiesSignsAndTail() throws {
    let bits: [UInt32] = [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
      0x00008000, 0x00018000, 0x80000000, 0x3f807fff, 0x3f808001,
      0x3b866cda, 0xbb866cda, 0x00000000, 0x7f000001, 0xff000001, 0x00800000,
      0x80800000, 0x3c000001]
    let values = bits.map(Float.init(bitPattern:))
    var actual = [UInt16](repeating: 0, count: values.count)
    values.withUnsafeBufferPointer { source in actual.withUnsafeMutableBufferPointer { Self.vectorRound(source, $0) } }
    XCTAssertEqual(actual, try values.map(Self.scalarRound))
  }
  private func digest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func identity(_ url: URL) throws -> [Int64] {
    var s = stat()
    guard url.path.withCString({ Darwin.lstat($0, &s) }) == 0,
      s.st_mode & S_IFMT == S_IFREG, FileManager.default.isReadableFile(atPath: url.path) else {
      throw H3CheckpointError.invalid("CPU phase fixture requires readable regular files.")
    }
    return [Int64(s.st_dev), Int64(s.st_ino), s.st_size,
      Int64(s.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(s.st_mtimespec.tv_nsec),
      Int64(s.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(s.st_ctimespec.tv_nsec)]
  }
  private func headerDigest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    guard let length = try handle.read(upToCount: 8), length.count == 8 else { throw H3CheckpointError.invalid("Missing header.") }
    let size = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard size >= 2, size <= 16 * 1024 * 1024,
      let header = try handle.read(upToCount: Int(size)), header.count == Int(size) else { throw H3CheckpointError.invalid("Invalid header.") }
    return SHA256.hash(data: header).map { String(format: "%02x", $0) }.joined()
  }
  private func physical() throws -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    let peak = UInt64(max(0, info.ledger_phys_footprint_peak))
    guard status == KERN_SUCCESS, info.phys_footprint > 0, peak >= info.phys_footprint else { throw H3CheckpointError.invalid("Invalid physical query.") }
    return (info.phys_footprint, peak)
  }
  private static func basis() -> [Float] {
    var result = [Float](repeating: 0, count: 65536)
    for row in 0..<256 { for column in 0..<256 {
      var sign: Float = 1, place = 1
      while place < 256 { if (row / place) % 4 + (column / place) % 4 == 3 { sign = -sign }; place *= 4 }
      result[row * 256 + column] = sign / 16
    } }
    return result
  }
  private static func rowWindow(columns: Int, buffered: Bool) throws -> Int {
    guard [5376, 7168, 14336].contains(columns) else { throw H3CheckpointError.invalid("Unsupported CPU probe column count.") }
    return buffered ? (columns <= 7168 ? 512 : 256) : 1024
  }
  func testCPUBufferedWindowsFitFourMiB() throws {
    for columns in [5376, 7168, 14336] {
      let rows = try Self.rowWindow(columns: columns, buffered: true)
      XCTAssertLessThanOrEqual(rows * columns, 4 * 1024 * 1024)
      XCTAssertEqual(rows, columns <= 7168 ? 512 : 256)
      XCTAssertEqual(try Self.rowWindow(columns: columns, buffered: false), 1024)
    }
    for columns in [0, 8192, 28672] { XCTAssertThrowsError(try Self.rowWindow(columns: columns, buffered: true)) }
  }
  private func decode(file: SafeTensorFile, name: String, rows: Int, columns: Int,
    scales: [Float], basis: [Float], buffered: Bool) throws -> (words: [UInt16], phases: Phases) {
    let window = try Self.rowWindow(columns: columns, buffered: buffered)
    let totalStart = ProcessInfo.processInfo.systemUptime
    var words = [UInt16](repeating: 0, count: rows * columns), phases = Phases()
    for start in stride(from: 0, to: rows, by: window) {
      try Task.checkCancellation()
      let stop = min(rows, start + window), elements = (stop - start) * columns
      var scaled = [Float](repeating: 0, count: elements)
      let readStart = ProcessInfo.processInfo.systemUptime
      try file.withTensorBytes(named: name, range: UInt64(start * columns)..<UInt64(stop * columns),
        access: buffered ? .buffered : .mapped) { raw in
        // Mapped page faults can occur in scale, rather than in this entry interval.
        phases.read += ProcessInfo.processInfo.systemUptime - readStart
        let scaleStart = ProcessInfo.processInfo.systemUptime
        let q = raw.bindMemory(to: Int8.self)
        scaled.withUnsafeMutableBufferPointer { output in
          for row in 0..<(stop - start) {
            let offset = row * columns
            vDSP_vflt8(q.baseAddress!.advanced(by: offset), 1,
              output.baseAddress!.advanced(by: offset), 1, vDSP_Length(columns))
            var scale = scales[start + row]
            vDSP_vsmul(output.baseAddress!.advanced(by: offset), 1, &scale,
              output.baseAddress!.advanced(by: offset), 1, vDSP_Length(columns))
          }
        }
        phases.scale += ProcessInfo.processInfo.systemUptime - scaleStart
      }
      var phaseStart = ProcessInfo.processInfo.systemUptime
      var restored = [Float](repeating: 0, count: elements)
      scaled.withUnsafeBufferPointer { source in basis.withUnsafeBufferPointer { rotation in restored.withUnsafeMutableBufferPointer { output in
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(elements / 256), 256, 256,
          1, source.baseAddress!, 256, rotation.baseAddress!, 256, 0, output.baseAddress!, 256)
      } } }
      phases.sgemm += ProcessInfo.processInfo.systemUptime - phaseStart
      phaseStart = ProcessInfo.processInfo.systemUptime
      restored.withUnsafeBufferPointer { source in words.withUnsafeMutableBufferPointer { output in
        Self.vectorRound(source, UnsafeMutableBufferPointer(start: output.baseAddress!.advanced(by: start * columns), count: elements))
      } }
      phases.roundStore += ProcessInfo.processInfo.systemUptime - phaseStart
      try Task.checkCancellation()
    }
    try file.checkUnchanged(); phases.total = ProcessInfo.processInfo.systemUptime - totalStart
    return (words, phases)
  }
  func testInstalledNormalComfyINT8BufferedSameSGEMMCPUPhases() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_CPU_CONVROT_FIXTURE"] else { throw XCTSkip("Select a fresh pinned CPU phase fixture explicitly.") }
    let fixtureURL = URL(fileURLWithPath: path)
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
    let productMetal = "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9"
    let url = URL(fileURLWithPath: fixture.checkpoint)
    guard !FileManager.default.fileExists(atPath: fixture.output), fixture.metalSHA256 == productMetal,
      try digest(URL(fileURLWithPath: fixture.metalLibrary)) == productMetal,
      try digest(URL(fileURLWithPath: fixture.testExecutable)) == fixture.testExecutableSHA256,
      Bundle(for: Self.self).executableURL?.resolvingSymlinksInPath().path == URL(fileURLWithPath: fixture.testExecutable).resolvingSymlinksInPath().path,
      fixture.checkpointPin.path == fixture.checkpoint, try identity(url) == fixture.checkpointPin.identity,
      try headerDigest(url) == fixture.checkpointPin.headerSHA256 else { throw H3CheckpointError.invalid("CPU phase fixture binding differs.") }
    _ = try identity(URL(fileURLWithPath: fixture.metalLibrary)); _ = try identity(URL(fileURLWithPath: fixture.testExecutable))
    let required = ["H3ComfyCPUDecodeProbeTests.swift", "H3ComfyDecodedProjection.swift", "H3CheckpointLayout.swift", "SafeTensorFile.swift", "Package.swift", "Package.resolved"]
    guard required.allSatisfy({ suffix in fixture.sourcePins.contains { $0.path.hasSuffix("/" + suffix) } }) else { throw H3CheckpointError.invalid("Required source pins missing.") }
    for pin in fixture.sourcePins { guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else { throw H3CheckpointError.invalid("CPU phase source changed.") } }
    let file = try SafeTensorFile(url: url), layout = try H3CheckpointLayout(url: url)
    guard layout.fastVariant == nil, layout.curveRank == nil, layout.quantizedProjections == 250 else { throw H3CheckpointError.invalid("Requires ordinary full-width Comfy INT8.") }
    let definitions = [("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168), ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336)]
    var allScales: [[Float]] = []
    for (suffix, rows, columns) in definitions {
      let stem = layout.prefix + "blocks.0." + suffix
      guard let w = file.tensors[stem + ".weight"], w.dtype == "I8", w.shape == [UInt64(rows), UInt64(columns)],
        let s = file.tensors[stem + ".weight_scale"], s.dtype == "F32", s.shape == [UInt64(rows), 1],
        let m = file.tensors[stem + ".comfy_quant"], m.dtype == "U8", m.byteCount > 0, m.byteCount <= 4096 else { throw H3CheckpointError.invalid("Invalid core descriptor.") }
      let marker: Data = try file.withTensorBytes(named: stem + ".comfy_quant", range: 0..<m.byteCount, access: .buffered) { Data($0) }
      guard let info = try JSONSerialization.jsonObject(with: marker) as? [String: Any], info["format"] as? String == "int8_tensorwise", info["convrot"] as? Bool == true,
        (info["convrot_groupsize"] as? Int ?? 256) == 256, Set(info.keys).isSubset(of: ["format", "convrot", "convrot_groupsize"]) else { throw H3CheckpointError.invalid("Invalid group256 marker.") }
      let scales = try file.readFloat32(named: stem + ".weight_scale", access: .buffered)
      guard scales.count == rows else { throw H3CheckpointError.invalid("Invalid scales.") }
      try Self.admitted(scales); allScales.append(scales)
    }
    try file.checkUnchanged(at: url)
    GPU.metallib = URL(fileURLWithPath: fixture.metalLibrary)
    Stream.gpu.synchronize(); Memory.clearCache()
    let baseline = Memory.activeMemory, previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    let basis = Self.basis(); var records: [[String: Any]] = []
    for (index, definition) in definitions.enumerated() {
      let (suffix, rows, columns) = definition, name = layout.prefix + "blocks.0." + suffix + ".weight"
      records.append(try autoreleasepool {
        let serial = try decode(file: file, name: name, rows: rows, columns: columns, scales: allScales[index], basis: basis, buffered: false)
        let candidate = try decode(file: file, name: name, rows: rows, columns: columns, scales: allScales[index], basis: basis, buffered: true)
        let cpuPhysical = try physical()
        Memory.peakMemory = Memory.activeMemory
        let gpuStart = ProcessInfo.processInfo.systemUptime
        let gpu = try H3ComfyDecodedProjection.load(file: file, checkpointURL: url, name: name,
          rows: rows, columns: columns, reorderQKV: false, rowWindow: 16384)
        let gpuSeconds = ProcessInfo.processInfo.systemUptime - gpuStart, peak = Memory.peakMemory
        let beforeHash = try physical()
        var originalHash = SHA256(), candidateHash = SHA256(), gpuHash = SHA256()
        var differentOriginal = 0, differentGPU = 0, allFinite = true
        for start in stride(from: 0, to: rows, by: 1024) {
          let stop = min(rows, start + 1024), offset = start * columns
          try autoreleasepool {
            let actual = gpu[start..<stop, 0..<columns].view(dtype: .uint16).asArray(UInt16.self)
            let original = Array(serial.words[offset..<(stop * columns)])
            let transformed = Array(candidate.words[offset..<(stop * columns)])
            originalHash.update(data: original.withUnsafeBytes { Data($0) }); candidateHash.update(data: transformed.withUnsafeBytes { Data($0) }); gpuHash.update(data: actual.withUnsafeBytes { Data($0) })
            for i in actual.indices {
              differentOriginal += original[i] == transformed[i] ? 0 : 1
              differentGPU += actual[i] == transformed[i] ? 0 : 1
              allFinite = allFinite && actual[i] & 0x7f80 != 0x7f80 && transformed[i] & 0x7f80 != 0x7f80 && original[i] & 0x7f80 != 0x7f80
            }
          }
        }
        return ["name": name, "shape": [rows, columns], "elements": rows * columns,
          "baselineCPUPhases": serial.phases.json, "candidateCPUPhases": candidate.phases.json,
          "baselineCPUWordSHA256": originalHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "candidateCPUWordSHA256": candidateHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "GPUWordSHA256": gpuHash.finalize().map { String(format: "%02x", $0) }.joined(),
          "differentBaselineWords": differentOriginal, "differentGPUWords": differentGPU,
          "allFinite": allFinite, "GPUDecodeSeconds": gpuSeconds, "peakMLXBytesBeforeHostHash": peak,
          "physicalAfterBothCPUDecodesBytes": cpuPhysical.current, "physicalBeforeHostHashBytes": beforeHash.current,
          "processLifetimePhysicalPeakBeforeHostHashBytes": beforeHash.peak, "reorderQKV": false, "baselineCPUReadMode": "mapped", "candidateCPUReadMode": "buffered",
          "baselineCPURowWindow": 1024, "candidateCPURowWindow": try Self.rowWindow(columns: columns, buffered: true),
          "maximumCandidateReadBytes": (try Self.rowWindow(columns: columns, buffered: true)) * columns,
          "candidateTwoF32ScratchBytes": (try Self.rowWindow(columns: columns, buffered: true)) * columns * 8,
          "candidateCPUWordStorageBytes": rows * columns * 2] as [String: Any]
      })
      Stream.gpu.synchronize(); Memory.clearCache(); XCTAssertEqual(Memory.activeMemory, baseline)
    }
    try file.checkUnchanged(at: url)
    for pin in fixture.sourcePins { guard try digest(URL(fileURLWithPath: pin.path)) == pin.sha256 else { throw H3CheckpointError.invalid("CPU phase source changed after work.") } }
    let remaining = Memory.activeMemory, cache = Memory.cacheMemory
    Memory.cacheLimit = previousLimit
    let passed = records.allSatisfy { $0["differentBaselineWords"] as? Int == 0 && $0["differentGPUWords"] as? Int == 0 && $0["allFinite"] as? Bool == true }
    let candidateSeconds = records.reduce(0.0) { sum, record in sum + ((record["candidateCPUPhases"] as? [String: Double])?["totalCPUDecodeSeconds"] ?? 0) }
    let report: [String: Any] = ["format": "weetodd-comfy-cpu-buffered-sgemm-phases-v3", "numericalPassed": passed,
      "productionQualified": false, "prefetchImplemented": false, "records": records,
      "candidateWholeCoreDecodeSeconds": candidateSeconds, "candidateFitsFourSecondComputeWindow": candidateSeconds < 4,
      "scope": "One sequential mapped/buffered vDSP observation per matrix, identical F32 SGEMM. Read ends at closure entry; mapped faults can occur in scale; scratch allocation and terminal checks are in total remainder. No warmed speed qualification or overlap. Physical retains both CPU forms plus GPU and includes earlier matrices.",
      "supportedScaleRange": [1e-20, 1e20], "subnormalScalesUnsupported": true,
      "oneFutureCoreBF16LogicalBytes": 770_703_360, "maximumTwoF32RowScratchBytes": 117_440_512,
      "maximumCandidateTwoF32RowScratchBytes": 29_360_128, "maximumCandidateBufferedReadBytes": 3_670_016,
      "activeBaselineMLXBytes": baseline, "remainingActiveMLXBytes": remaining, "remainingCacheMLXBytes": cache,
      "previousCacheLimitBytes": previousLimit, "restoredCacheLimitBytes": Memory.cacheLimit,
      "fixtureSHA256": try digest(fixtureURL), "metalSHA256": productMetal, "testExecutableSHA256": fixture.testExecutableSHA256]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: fixture.output), options: .withoutOverwriting)
    XCTAssertTrue(passed); XCTAssertEqual(remaining, baseline); XCTAssertEqual(cache, 0)
    XCTAssertEqual(Memory.cacheLimit, previousLimit)
  }
}
