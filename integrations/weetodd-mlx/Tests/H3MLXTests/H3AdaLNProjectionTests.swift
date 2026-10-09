import Foundation
import CryptoKit
import Darwin
import MLX
import XCTest
@testable import H3MLX

final class H3AdaLNProjectionTests: XCTestCase {
  func testDeferredFastFactorAdmissionRejectsEveryMissingOrInvalidFactor() throws {
    let stem = "blocks.0.adaln_proj.linear"
    let admitted = [stem + ".weight":H3TensorInfo(dtype:"U32",shape:[96768,672]),
      stem + ".scales":H3TensorInfo(dtype:"BF16",shape:[96768,42]),
      stem + ".biases":H3TensorInfo(dtype:"BF16",shape:[96768,42]),
      stem + ".bias":H3TensorInfo(dtype:"BF16",shape:[96768])]
    XCTAssertNoThrow(try H3AdaLNProjection.validateFastFactors(tensors:admitted,weightName:stem + ".weight"))
    for key in admitted.keys {
      var changed = admitted; changed.removeValue(forKey:key)
      XCTAssertThrowsError(try H3AdaLNProjection.validateFastFactors(tensors:changed,weightName:stem + ".weight"))
      changed = admitted; changed[key] = H3TensorInfo(dtype:"F32",shape:admitted[key]!.shape)
      XCTAssertThrowsError(try H3AdaLNProjection.validateFastFactors(tensors:changed,weightName:stem + ".weight"))
      changed = admitted; changed[key] = H3TensorInfo(dtype:admitted[key]!.dtype,shape:[1])
      XCTAssertThrowsError(try H3AdaLNProjection.validateFastFactors(tensors:changed,weightName:stem + ".weight"))
    }
  }

  func testDeferredFastCancellationPrecedesCheckpointAccess() async throws {
    let task = Task { () throws -> Bool in
      try Device.withDefaultDevice(.cpu) {
        let time = MLXArray.zeros([1,2688],dtype:.float32)
        withUnsafeCurrentTask { $0?.cancel() }
        do {
          _ = try H3AdaLNProjection.evaluate(checkpointURL:URL(fileURLWithPath:"/nonexistent/h3-adaln"),
            blockIndex:0,timeEmbeddings:time,lora:nil,deferredFastLoading:true)
          return false
        } catch is CancellationError { return true }
      }
    }
    let cancelled = try await task.value
    XCTAssertTrue(cancelled)
  }

  func testInstalledFastAll50TablesMatchSerialBitsAndMeasurePreparation() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_FAST_ADALN_TEST"] == "1",
      let checkpointPath = env["WEETODD_H3_TEST_CHECKPOINT"],
      let outputPath = env["WEETODD_H3_FAST_ADALN_OUTPUT"],
      let metalPath = env["WEETODD_H3_FAST_ADALN_METAL"],
      let metalSHA = env["WEETODD_H3_FAST_ADALN_METAL_SHA256"],
      let sourceRoot = env["WEETODD_H3_FAST_ADALN_SOURCE_ROOT"] else {
      throw XCTSkip("Opt-in actual FastQ8 all50 byte parity, production-library identity and preparation timing.")
    }
    func sha(_ data:Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    func pin(_ url:URL) throws -> [String:Any] {
      var s = stat()
      guard lstat(url.path,&s) == 0, s.st_mode & S_IFMT == S_IFREG else {
        throw H3CheckpointError.invalid("Missing or nonregular Fast AdaLN qualification input.")
      }
      return ["path":url.path,"bytes":Int(s.st_size),"device":Int(s.st_dev),"inode":UInt64(s.st_ino),
        "mtimeSeconds":Int(s.st_mtimespec.tv_sec),"mtimeNanoseconds":Int(s.st_mtimespec.tv_nsec),
        "ctimeSeconds":Int(s.st_ctimespec.tv_sec),"ctimeNanoseconds":Int(s.st_ctimespec.tv_nsec)]
    }
    guard !FileManager.default.fileExists(atPath:outputPath), metalSHA.count == 64 else {
      throw H3CheckpointError.invalid("Fast AdaLN receipt must be fresh and product library explicitly pinned.")
    }
    let requestedOrder = env["WEETODD_H3_FAST_ADALN_ORDER"] ?? "serial-first"
    guard ["serial-first", "deferred-first"].contains(requestedOrder) else {
      throw H3CheckpointError.invalid("Fast AdaLN qualification order must be serial-first or deferred-first.")
    }
    let trialOrder = requestedOrder == "serial-first" ? [false,true,true,false] : [true,false,false,true]
    let metal = URL(fileURLWithPath:metalPath)
    _ = try pin(metal)
    guard sha(try Data(contentsOf:metal)) == metalSHA else {
      throw H3CheckpointError.invalid("Changed Fast AdaLN product Metal library.")
    }
    GPU.metallib = metal
    let checkpoint = URL(fileURLWithPath:checkpointPath)
    let layout = try H3CheckpointLayout(url:checkpoint)
    guard layout.fastVariant != nil else { throw XCTSkip("Requires actual admitted FastH3 Q8 pages.") }
    var identities: [[String:Any]] = []
    for name in ["config.json","quant_config.json","paged_manifest.json"] {
      let url = checkpoint.appendingPathComponent(name)
      var identity = try pin(url); identity["sha256"] = sha(try Data(contentsOf:url)); identities.append(identity)
    }
    for index in 0..<50 { identities.append(try pin(H3CheckpointSource.fileURL(checkpoint,block:index))) }
    let video = try H3Schedule(requestedSteps:5,shift:12)
    let audio = try H3Schedule(requestedSteps:5,shift:3)
    let table = Set(video.timesteps + audio.timesteps).sorted()
    let time = try H3TimeEmbedding.evaluate(checkpointURL:checkpoint,timesteps:MLXArray(table))
    let timeBytes = time.asType(.float32).asArray(Float.self).withUnsafeBytes { Data($0) }
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    Stream.gpu.synchronize(); Memory.clearCache()
    let activeBefore = Memory.activeMemory
    var expected: [String]?
    var runs: [[String:Any]] = []
    // Reverse the middle pair to distinguish load-order effects from a toggle.
    for deferred in trialOrder {
      Memory.clearCache(); Memory.peakMemory = Memory.activeMemory
      let measured = try autoreleasepool { () throws -> ([String:Any],[String]) in
        var tables: [MLXArray] = [], blockSeconds: [Double] = []
        for index in 0..<50 {
          let began = CFAbsoluteTimeGetCurrent()
          tables.append(try H3AdaLNProjection.evaluate(checkpointURL:checkpoint,blockIndex:index,
            timeEmbeddings:time,lora:nil,deferredFastLoading:deferred))
          blockSeconds.append(CFAbsoluteTimeGetCurrent()-began)
        }
        let peak = Memory.peakMemory
        let hashes = tables.map { value in
          XCTAssertEqual(value.shape,[table.count,96768]); XCTAssertEqual(value.dtype,.bfloat16)
          return value.view(dtype:.uint16).asArray(UInt16.self).withUnsafeBytes { sha(Data($0)) }
        }
        return (["deferredNativeLoading":deferred,"all50Seconds":blockSeconds.reduce(0,+),
          "blockSeconds":blockSeconds,"first25Seconds":blockSeconds.prefix(25).reduce(0,+),
          "last25Seconds":blockSeconds.suffix(25).reduce(0,+),"peakMLXBytesBeforeHostHashes":peak,
          "tableBF16SHA256":hashes],hashes)
      }
      if let expected { XCTAssertEqual(measured.1,expected,"Every BF16 word in all50 tables must match serial loading.") }
      else { expected = measured.1 }
      Stream.gpu.synchronize(); Memory.clearCache()
      XCTAssertEqual(Memory.activeMemory,activeBefore,"Returned tables and all transient page factors must retire.")
      var report = measured.0
      report["remainingActiveMLXBytes"] = Memory.activeMemory
      report["cacheBytesAfterRun"] = Memory.cacheMemory
      runs.append(report)
      try H3CheckpointSource.checkUnchanged(checkpoint)
    }
    var sources: [String:String] = [:]
    for relative in ["Sources/H3MLX/H3AdaLNProjection.swift","Sources/H3MLX/H3NativePage.swift",
      "Sources/H3MLX/H3QwenQ8Projection.swift","Tests/H3MLXTests/H3AdaLNProjectionTests.swift"] {
      sources[relative] = sha(try Data(contentsOf:URL(fileURLWithPath:sourceRoot).appendingPathComponent(relative)))
    }
    let binary = Bundle(for:Self.self).executableURL
    Memory.cacheLimit = previousLimit
    Stream.gpu.synchronize(); Memory.clearCache()
    let receipt: [String:Any] = ["scope":"All50 actual FastQ8 AdaLN tables; canonical four-evaluation timetable; load+projection timing excludes host hashing/time embedding. Fresh-process whole-output qualification remains separate.",
      "requestedOrder":requestedOrder,"activeBaselineMLXBytes":activeBefore,
      "previousCacheLimit":previousLimit,"restoredCacheLimit":Memory.cacheLimit,
      "terminalCacheBytes":Memory.cacheMemory,
      "checkpointIdentities":identities,"fastVariant":layout.fastVariant!.rawValue,
      "metalPath":metalPath,"metalSHA256":metalSHA,"testBinaryPath":binary?.path ?? "unknown",
      "testBinarySHA256":try binary.map { sha(try Data(contentsOf:$0)) } ?? "unknown",
      "sourceSHA256":sources,"timetable":table,"timeFloat32SHA256":sha(timeBytes),"runs":runs]
    try JSONSerialization.data(withJSONObject:receipt,options:[.prettyPrinted,.sortedKeys])
      .write(to:URL(fileURLWithPath:outputPath),options:.withoutOverwriting)
  }

  func testExperimentalLargerWindowMatchesInstalledAdaLN() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_WINDOW_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let timeFixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"] else {
      throw XCTSkip("Enable the installed larger-window AdaLN qualification explicitly.")
    }
    let time = try Data(contentsOf: URL(fileURLWithPath: timeFixture)
      .appendingPathComponent("output.f32")).withUnsafeBytes {
        MLXArray($0, [3, 2688], type: Float.self)
      }
    let expected = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16")).withUnsafeBytes {
        MLXArray($0, [3, 96768], type: UInt16.self).view(dtype: .bfloat16)
      }
    if ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_PROFILE"] == "1" {
      let checkpointURL = URL(fileURLWithPath: checkpoint)
      let coldStarted = Date()
      _ = try H3CheckpointLayout(url: checkpointURL)
      let coldSeconds = Date().timeIntervalSince(coldStarted)
      let warmStarted = Date()
      _ = try H3CheckpointLayout(url: checkpointURL)
      print("H3 layout first=\(coldSeconds) cached=\(Date().timeIntervalSince(warmStarted))")
    }
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let started = Date()
    let actual = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: time, rowWindow: 16384)
    let elapsed = Date().timeIntervalSince(started)
    let error = abs(actual.asType(.float32) - expected.asType(.float32))
    let peak = max(error).item(Float.self)
    let average = mean(error).item(Float.self)
    print("H3 AdaLN 16384-row window seconds=\(elapsed) max=\(peak) "
      + "mean=\(average) peak_mlx=\(Memory.peakMemory)")
    XCTAssertEqual(peak, 0)
  }

  func testInstalledLayoutCacheRejectsReplacedCheckpointPath() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_LAYOUT_CACHE_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Enable the installed H3 layout cache check explicitly.")
    }
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-h3-layout-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let link = directory.appendingPathComponent("checkpoint.safetensors")
    try FileManager.default.createSymbolicLink(at: link,
      withDestinationURL: URL(fileURLWithPath: checkpoint))
    let first = try H3CheckpointLayout(url: link)
    let second = try H3CheckpointLayout(url: link)
    XCTAssertEqual(first.quantizedProjections, second.quantizedProjections)
    let invalid = directory.appendingPathComponent("invalid.safetensors")
    try Data([0]).write(to: invalid)
    try FileManager.default.removeItem(at: link)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: invalid)
    XCTAssertThrowsError(try H3CheckpointLayout(url: link))
  }
  func testInstalledBlockZeroMatchesStreamedReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let timeFixture = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let adaFixture = ProcessInfo.processInfo.environment["WEETODD_H3_ADALN_ORACLE"] else {
      throw XCTSkip("Set the installed H3 checkpoint, time and AdaLN oracle paths.")
    }
    let timeData = try Data(contentsOf: URL(fileURLWithPath: timeFixture)
      .appendingPathComponent("output.f32"))
    let time = timeData.withUnsafeBytes { MLXArray($0, [3, 2688], type: Float.self) }
    let expectedData = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0.u16"))
    let expected = expectedData.withUnsafeBytes {
      MLXArray($0, [3, 96768], type: UInt16.self).view(dtype: .bfloat16)
    }
    let actual = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: time)
    XCTAssertEqual(actual.shape, [3, 96768])
    XCTAssertEqual(actual.dtype, .bfloat16)
    XCTAssertEqual(max(abs(actual.asType(.float32) - expected.asType(.float32)))
      .item(Float.self), 0)
    let repeatedTime = concatenated(Array(repeating: time, count: 7), axis: 0)[0..<20, 0..<2688]
    let repeatedData = try Data(contentsOf: URL(fileURLWithPath: adaFixture)
      .appendingPathComponent("block0-20.u16"))
    let repeatedExpected = repeatedData.withUnsafeBytes {
      MLXArray($0, [20, 96768], type: UInt16.self).view(dtype: .bfloat16)
    }
    let repeated = try H3AdaLNProjection.evaluate(
      checkpointURL: URL(fileURLWithPath: checkpoint), blockIndex: 0,
      timeEmbeddings: repeatedTime)
    XCTAssertEqual(repeated.shape, [20, 96768])
    // The 16+4 bounded GEMM and one 20-row GEMM select different Metal kernels;
    // the drift is below one BF16 quantum at the tested projection scale.
    XCTAssertLessThan(max(abs(repeated.asType(.float32)
      - repeatedExpected.asType(.float32))).item(Float.self), 0.005)
  }
}
