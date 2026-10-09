import Foundation
import CryptoKit
import Darwin
import MLX
import TensorIO
import XCTest
@testable import H3MLX

/// Opt-in full saved-latent decoder/media qualification. No sampling or audio inference.
final class H3VideoPrecisionFullQualificationTests: XCTestCase {
  private struct Pin: Decodable { let path: String; let sha256: String }
  private struct Fixture: Decodable {
    let checkpoint: String; let checkpointIdentity: [Int64]; let checkpointHeaderSHA256: String
    let latentPath: String; let latentSHA256: String
    let metalLibrary: String; let metalSHA256: String
    let testExecutable: String; let testExecutableSHA256: String
    let sourcePins: [Pin]; let outputDirectory: String
  }
  private func sha(_ data: Data) -> String {
    SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
  }
  private func fileSHA(_ url: URL) throws -> String {
    let f = try FileHandle(forReadingFrom:url); defer { try? f.close() }
    var h = SHA256()
    while let bytes = try f.read(upToCount:4 * 1024 * 1024), !bytes.isEmpty { h.update(data:bytes) }
    return h.finalize().map { String(format:"%02x",$0) }.joined()
  }
  private func identity(_ url: URL) throws -> [Int64] {
    var s = stat()
    guard url.path.withCString({ Darwin.lstat($0,&s) }) == 0,
      s.st_mode & S_IFMT == S_IFREG,FileManager.default.isReadableFile(atPath:url.path) else {
      throw H3CheckpointError.invalid("Full decoder qualification requires readable regular files.")
    }
    return [Int64(s.st_dev),Int64(s.st_ino),s.st_size,
      Int64(s.st_mtimespec.tv_sec)*1_000_000_000+Int64(s.st_mtimespec.tv_nsec),
      Int64(s.st_ctimespec.tv_sec)*1_000_000_000+Int64(s.st_ctimespec.tv_nsec)]
  }
  private func headerSHA(_ url: URL) throws -> String {
    let f = try FileHandle(forReadingFrom:url); defer { try? f.close() }
    guard let prefix = try f.read(upToCount:8),prefix.count == 8 else {
      throw H3CheckpointError.invalid("Missing full decoder checkpoint header.")
    }
    let n = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset*8) }
    guard n >= 2,n <= 64 * 1024 * 1024,
      let header = try f.read(upToCount:Int(n)),header.count == Int(n) else {
      throw H3CheckpointError.invalid("Invalid full decoder checkpoint header.")
    }
    return sha(header)
  }
  private func physical() -> [String:UInt64] {
    var info = task_vm_info_data_t()
    var n = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to:&info) {
      $0.withMemoryRebound(to:integer_t.self,capacity:Int(n)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&n)
      }
    }
    guard status == KERN_SUCCESS else { return [:] }
    return ["currentPhysicalFootprintBytes":info.phys_footprint,
      "sharedProcessLifetimePeakPhysicalFootprintBytes":UInt64(max(0,info.ledger_phys_footprint_peak))]
  }

  func testInstalledFullRefSavedLatentFP32VersusFP16StreamingQualification() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_VIDEO_PRECISION_FULL_FIXTURE"] else {
      throw XCTSkip("Set new pinned full Ref saved-latent precision fixture; default loads no weights.")
    }
    let manifest = URL(fileURLWithPath:path),bytes = try Data(contentsOf:manifest)
    let fixture = try JSONDecoder().decode(Fixture.self,from:bytes)
    let output = URL(fileURLWithPath:fixture.outputDirectory)
    let checkpoint = URL(fileURLWithPath:fixture.checkpoint)
    let rawURL = URL(fileURLWithPath:fixture.latentPath)
    let metal = URL(fileURLWithPath:fixture.metalLibrary)
    guard fixture.latentSHA256 == "2d1759aac0751eaafd9bcb802d9cd1b3ebd35448b59d4be40c5f5df5719221f4",
      fixture.metalSHA256 == "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9",
      !FileManager.default.fileExists(atPath:output.path),
      FileManager.default.fileExists(atPath:output.deletingLastPathComponent().path),
      let executable = Bundle(for:Self.self).executableURL?.resolvingSymlinksInPath(),
      executable == URL(fileURLWithPath:fixture.testExecutable).resolvingSymlinksInPath(),
      Set(fixture.sourcePins.map { URL(fileURLWithPath:$0.path).lastPathComponent })
        .isSuperset(of:["H3VideoDecodePrecision.swift","H3VideoVAEDecodeSession.swift",
          "H3VideoVAEBlock.swift","H3VideoVAETileDecoder.swift","H3VideoVAEDecoder.swift",
          "H3VideoPrecisionFullQualificationTests.swift"]) else {
      throw H3CheckpointError.invalid("Full precision qualification pins/output differ.")
    }
    func verify() throws {
      try Task.checkCancellation()
      guard try identity(checkpoint) == fixture.checkpointIdentity,
        try headerSHA(checkpoint) == fixture.checkpointHeaderSHA256,
        (1..<(32 * 1024 * 1024)).contains(try identity(rawURL)[2]),
        try fileSHA(rawURL) == fixture.latentSHA256,
        try fileSHA(metal) == fixture.metalSHA256,
        try fileSHA(executable) == fixture.testExecutableSHA256,
        try Data(contentsOf:manifest) == bytes else {
        throw H3CheckpointError.invalid("Full precision fixture changed.")
      }
      for pin in fixture.sourcePins {
        guard try fileSHA(URL(fileURLWithPath:pin.path)) == pin.sha256 else {
          throw H3CheckpointError.invalid("Full precision source changed: \(pin.path)")
        }
      }
    }
    try verify(); GPU.metallib = metal
    let rawFile = try SafeTensorFile(url:rawURL)
    guard rawFile.tensors["video_latents"]?.dtype == "F32",
      rawFile.tensors["video_latents"]?.shape == [1,24,37,48,86] else {
      throw H3CheckpointError.invalid("Full precision input must be actual unresized1376x768 Ref latent.")
    }
    let input: MLXArray = try autoreleasepool {
      let raw = try rawFile.withTensorBytes(named:"video_latents") {
        MLXArray($0,[1,24,37,48,86],type:Float.self)
      }
      let layout = try H3VideoVAELayout(url:checkpoint)
      let rows = raw.transposed(0,2,3,4,1).reshaped([1,37,24,2,43,2,24])
        .transposed(0,1,2,4,6,3,5).reshaped([1,37*24*43,96])
      let result = contiguous(try H3LatentCodec.videoDecoderInput(rows:rows,
        latentFrames:37,latentHeight:48,latentWidth:86,
        mean:layout.latentsMean,standardDeviation:layout.latentsStandardDeviation))
      eval(result); return result
    }
    let inputSHA = input.asArray(Float.self).withUnsafeBytes { sha(Data($0)) }
    Stream.gpu.synchronize(); Memory.clearCache()
    let baseline = Memory.activeMemory,cacheLimit = Memory.cacheLimit
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = cacheLimit }
    try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
    var reports: [[String:Any]] = []
    var failure: Error?
    for precision: H3VideoDecodePrecision in [.float32,.float16] {
      let rawOutput = output.appendingPathComponent(precision.rawValue + ".rgb24")
      guard FileManager.default.createFile(atPath:rawOutput.path,contents:nil) else {
        throw H3CheckpointError.invalid("Cannot create full precision RGB output.")
      }
      let writer = try FileHandle(forWritingTo:rawOutput)
      var frames = 0,hostSeconds = 0.0,decodePeak = baseline
      var floatHasher = SHA256(),rgbHasher = SHA256()
      var chunkReports: [[String:Any]] = []
      var closed: H3VideoVAEDecodeSession.Statistics?
      var passError: Error?
      let beforePhysical = physical()
      let started = ProcessInfo.processInfo.systemUptime
      do {
        try autoreleasepool {
          Stream.gpu.synchronize(); Memory.clearCache(); Memory.peakMemory = baseline
          try H3VideoVAEDecoder.decodeChunks(checkpointURL:checkpoint,latent:input,
            retainWeights:true,memoryMode:.lowMemoryBF16,precision:precision,spatialBatchSize:1,
            allocationCacheLimitBytes:128 * 1024 * 1024,groupedStageLoading:true,
            onSessionClosed:{ closed = $0 }) { chunk in
            try Task.checkCancellation()
            // Include pending blend GPU work in decoder time/peak, then exclude
            // only host validation, RGB conversion and disk writes from the observation.
            eval(chunk); Stream.gpu.synchronize()
            decodePeak = max(decodePeak,Memory.peakMemory)
            let hostStarted = ProcessInfo.processInfo.systemUptime
            try autoreleasepool {
              guard chunk.dtype == .float32,
                chunk.shape == [1,chunk.shape[1],768,1376,3] else {
                throw H3CheckpointError.invalid("Full precision published chunk shape/dtype differs.")
              }
              let values = chunk.asArray(Float.self)
              guard values.allSatisfy(\.isFinite) else {
                throw H3CheckpointError.invalid("Full precision published nonfinite pixels.")
              }
              let floatSHA = values.withUnsafeBytes { buffer -> String in
                let data = Data(bytesNoCopy:UnsafeMutableRawPointer(mutating:buffer.baseAddress!),
                  count:buffer.count,deallocator:.none);floatHasher.update(data:data);return sha(data)
              }
              let rgb = try H3LatentCodec.videoPixelsRGB8(chunk).asArray(UInt8.self)
              let rgbSHA = try rgb.withUnsafeBytes { buffer -> String in
                let data = Data(bytesNoCopy:UnsafeMutableRawPointer(mutating:buffer.baseAddress!),
                  count:buffer.count,deallocator:.none);rgbHasher.update(data:data);try writer.write(contentsOf:data)
                return sha(data)
              }
              chunkReports.append(["frames":chunk.shape[1],"shape":chunk.shape,
                "float32Finite":true,"float32SHA256":floatSHA,"rgb8SHA256":rgbSHA])
              frames += chunk.shape[1]
            }
            hostSeconds += ProcessInfo.processInfo.systemUptime-hostStarted
            Memory.peakMemory = Memory.activeMemory
          }
          Stream.gpu.synchronize();decodePeak = max(decodePeak,Memory.peakMemory)
        }
      } catch { passError = error }
      let elapsed = ProcessInfo.processInfo.systemUptime-started
      try writer.close()
      Stream.gpu.synchronize();Memory.clearCache()
      let afterActive = Memory.activeMemory
      if let stats = closed {
        XCTAssertTrue(stats.closed);XCTAssertEqual(stats.remainingResidentBytes,0)
        XCTAssertEqual(stats.computePrecision,precision.rawValue)
        XCTAssertEqual(stats.projectionLoads,144);XCTAssertEqual(stats.tensorLoads,297)
        XCTAssertEqual(stats.groupedPreparationGroups,37)
        XCTAssertEqual(stats.maximumResidentBytes,2_582_138_032+stats.maximumGridCacheBytes)
      }
      XCTAssertEqual(afterActive,baseline,"Success/failure/cancel must release decoder graph/owner.")
      XCTAssertEqual(Memory.cacheMemory,0);XCTAssertEqual(Memory.cacheLimit,cacheLimit)
      reports.append(["computePrecision":precision.rawValue,"precisionPolicy":precision.diagnostics,
        "memoryMode":"low_memory_bf16","spatialBatchSize":1,"groupedStageLoading":true,
        "frames":frames,"chunkReports":chunkReports,"float32Finite":passError == nil,
        "float32SHA256":floatHasher.finalize().map { String(format:"%02x",$0) }.joined(),
        "rgb8SHA256":rgbHasher.finalize().map { String(format:"%02x",$0) }.joined(),
        "rgb24File":rawOutput.path,"rgb24Bytes":frames*768*1376*3,
        "wallSeconds":elapsed,"hostValidationAndWriteSeconds":hostSeconds,
        "decoderSecondsExcludingHostValidationAndWrites":elapsed-hostSeconds,
        "mlxDecoderPeakBytes":decodePeak,"mlxActiveAfterBytes":afterActive,
        "cacheBytesAfter":Memory.cacheMemory,"cacheLimitRestored":Memory.cacheLimit == cacheLimit,
        "sessionClosed":closed?.closed ?? false,"remainingResidentBytes":closed?.remainingResidentBytes ?? 0,
        "maximumResidentBytes":closed?.maximumResidentBytes ?? 0,
        "physicalBefore":beforePhysical,"physicalAfter":physical(),
        "status":passError == nil ? "success" : "failed_or_cancelled",
        "error":passError.map { String(describing:$0) } ?? ""])
      if let error = passError { failure = error;break }
      XCTAssertEqual(frames,124)
      XCTAssertEqual(try identity(rawOutput)[2],Int64(124*768*1376*3))
      try verify();try rawFile.checkUnchanged(at:rawURL)
    }
    let receipt: [String:Any] = ["format":"weetodd-full-saved-latent-precision-qualification-v1",
      "status":failure == nil ? "success" : "failed_or_cancelled",
      "fixtureSHA256":sha(bytes),"latentSHA256":fixture.latentSHA256,
      "decoderInputFloat32SHA256":inputSHA,"rawLatentShape":[1,24,37,48,86],
      "width":1376,"height":768,"frames":124,"fps":24,"memoryMode":"low_memory_bf16",
      "order":"FP32 then FP16 once each","warmupExecuted":false,
      "samplingExecuted":false,"audioInferenceExecuted":false,"qualityApproval":false,
      "speedScope":"two serial decoder observations; not warm ABBA or whole-job qualification",
      "memoryScope":"streamed MLX outputs; shared process physical lifetime peak is not per-precision independent",
      "timingScope":"decode/drains included; callback host validation/RGB conversion/disk writes subtracted",
      "mlxInputBaselineBytes":baseline,"passes":reports]
    try JSONSerialization.data(withJSONObject:receipt,options:[.prettyPrinted,.sortedKeys])
      .write(to:output.appendingPathComponent("full-decoder-result.json"),options:.atomic)
    if let failure { throw failure }
  }
}
