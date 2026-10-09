import Foundation
import CryptoKit
import Darwin
import MLXRandom
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3PreparedBlockTests: XCTestCase {
  private let missing = URL(fileURLWithPath: "/nonexistent/weetodd-prepared-block.safetensors")

  func testInvalidIndexIsRejectedBeforeOpeningCheckpoint() {
    for index in [-1, 50] {
      XCTAssertThrowsError(try H3PreparedBlock(checkpointURL: missing, index: index,
        projectionMode: .weightDecoded)) {
        XCTAssertEqual($0 as? H3CheckpointError,
          .invalid("Invalid prepared H3 block index."))
      }
    }
  }

  func testActivationRotatedIsRejectedBeforeOpeningCheckpoint() {
    XCTAssertThrowsError(try H3PreparedBlock(checkpointURL: missing, index: 0,
      projectionMode: .activationRotated)) {
      XCTAssertEqual($0 as? H3CheckpointError,
        .invalid("Prepared H3 blocks require weight-decoded projections."))
    }
  }

  func testUnsupportedDecodeWindowIsRejectedBeforeOpeningCheckpoint() {
    XCTAssertThrowsError(try H3PreparedBlock(checkpointURL: missing, index: 0,
      projectionMode: .weightDecoded, rowWindow: 0)) {
      XCTAssertEqual($0 as? H3CheckpointError,
        .invalid("Prepared H3 decode row window is unsupported."))
    }
  }

  func testMissingCheckpointCannotCreatePreparedOwner() {
    XCTAssertThrowsError(try H3PreparedBlock(checkpointURL: missing, index: 0,
      projectionMode: .weightDecoded))
  }

  func testGroupedLoadingCancellationPrecedesCheckpointAccess() async throws {
    let unavailableCheckpoint = missing
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try H3PreparedBlock(checkpointURL:unavailableCheckpoint,index:0,
          projectionMode:.weightDecoded,groupedBF16Loading:true)
        return false
      } catch is CancellationError { return true }
      catch { return false }
    }
    let wasCancelled = await task.value
    XCTAssertTrue(wasCancelled)
  }

  func testGroupedFastAdmissionRejectsMissingGateAndMalformedCompanionsWithoutArrays() throws {
    let prefix = "blocks.0."
    var tensors: [String:H3TensorInfo] = [:]
    for (suffix, width) in [("norm1.weight",5376),("norm2.weight",5376),
      ("attn.q_norm.weight",128),("attn.k_norm.weight",128)] {
      tensors[prefix + suffix] = H3TensorInfo(dtype:"BF16",shape:[UInt64(width)])
    }
    for (suffix, rows, columns) in [("attn.qkv_proj",21504,5376),
      ("attn.out_proj",5376,7168),("mlp.fc1",28672,5376),("mlp.fc2",5376,14336)] {
      tensors[prefix + suffix + ".weight"] = H3TensorInfo(dtype:"U32",
        shape:[UInt64(rows),UInt64(columns / 4)])
      for companion in ["scales","biases"] {
        tensors[prefix + suffix + "." + companion] = H3TensorInfo(dtype:"BF16",
          shape:[UInt64(rows),UInt64(columns / 64)])
      }
    }
    XCTAssertNoThrow(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
      variant:.denseV1,tensor:{ tensors[$0] }))
    XCTAssertThrowsError(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
      variant:.vsaV1,tensor:{ tensors[$0] }))
    tensors[prefix + "attn.gate_compress.weight"] = H3TensorInfo(dtype:"BF16",shape:[7168,5376])
    XCTAssertNoThrow(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
      variant:.vsaV1,tensor:{ tensors[$0] }))
    let valid = tensors
    for name in valid.keys {
      var missing = valid; missing.removeValue(forKey:name)
      XCTAssertThrowsError(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
        variant:.vsaV1,tensor:{ missing[$0] }),name)
    }
    tensors[prefix + "mlp.fc2.biases"] = H3TensorInfo(dtype:"F16",shape:[5376,224])
    XCTAssertThrowsError(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
      variant:.vsaV1,tensor:{ tensors[$0] }))
    tensors = valid
    tensors[prefix + "mlp.fc1.scales"] = H3TensorInfo(dtype:"BF16",shape:[28672,85])
    XCTAssertThrowsError(try H3PreparedBlock.validateGroupedFastAdmission(prefix:prefix,
      variant:.vsaV1,tensor:{ tensors[$0] }))
  }

  func testGroupedFastCancellationPrecedesCheckpointAccess() async throws {
    let unavailableCheckpoint = missing
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try H3PreparedBlock(checkpointURL:unavailableCheckpoint,index:0,
          projectionMode:.weightDecoded,groupedFastQ8Loading:true)
        return false
      } catch is CancellationError { return true }
      catch { return false }
    }
    let wasCancelled = await task.value
    XCTAssertTrue(wasCancelled)
  }

  private struct FastLoadingFixture: Decodable {
    struct Pin: Decodable { let path: String; let sha256: String }
    struct CheckpointPin: Decodable { let path: String; let headerSHA256: String; let identity: [Int64] }
    let checkpoint: String
    let metalLibrary: String
    let metalSHA256: String
    let testExecutable: String
    let testExecutableSHA256: String
    let sourcePins: [Pin]
    let checkpointPins: [CheckpointPin]
    let output: String
    let expectedOutputFloat32SHA256: String
    let minimumImprovementSeconds: Double?
    let maximumPeakMLXBytes: Int?
    let maximumPhysicalPeakBytes: UInt64?
  }

  private func fastFileDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom:url)
    defer { try? file.close() }
    var hasher = SHA256()
    while let data = try file.read(upToCount:4 * 1024 * 1024), !data.isEmpty { hasher.update(data:data) }
    return hasher.finalize().map { String(format:"%02x",$0) }.joined()
  }

  private func fastFileIdentity(_ url: URL) throws -> [Int64] {
    var value = stat()
    guard url.path.withCString({ Darwin.lstat($0,&value) }) == 0,
      value.st_mode & S_IFMT == S_IFREG,
      FileManager.default.isReadableFile(atPath:url.path) else {
      throw H3CheckpointError.invalid("Fast loading fixture requires readable regular files.")
    }
    return [Int64(value.st_dev),Int64(value.st_ino),value.st_size,
      Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec),
      Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)]
  }

  private func fastHeaderDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom:url)
    defer { try? file.close() }
    guard let prefix = try file.read(upToCount:8), prefix.count == 8 else {
      throw H3CheckpointError.invalid("Missing Fast loading fixture tensor header.")
    }
    let count = prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << ($1.offset * 8)) }
    guard count >= 2, count <= 64 * 1024 * 1024,
      let header = try file.read(upToCount:Int(count)), header.count == Int(count) else {
      throw H3CheckpointError.invalid("Invalid Fast loading fixture tensor header.")
    }
    return SHA256.hash(data:header).map { String(format:"%02x",$0) }.joined()
  }

  private func fastPhysicalMemory() throws -> (current: UInt64, lifetimePeak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to:&info) {
      $0.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
      }
    }
    guard result == KERN_SUCCESS else { throw H3CheckpointError.invalid("Cannot measure Fast loading physical footprint.") }
    return (info.phys_footprint,UInt64(max(0,info.ledger_phys_footprint_peak)))
  }

  func testInstalledGroupedFastQ8OwnersAndTwoBlockABBA() throws {
    guard let manifest = ProcessInfo.processInfo.environment["WEETODD_H3_GROUPED_FAST_FIXTURE"] else {
      throw XCTSkip("Set a pinned installed FastQ8 loading fixture; default does not load weights.")
    }
    let fixture = try JSONDecoder().decode(FastLoadingFixture.self,
      from:Data(contentsOf:URL(fileURLWithPath:manifest)))
    let outputURL = URL(fileURLWithPath:fixture.output)
    guard !FileManager.default.fileExists(atPath:outputURL.path),
      FileManager.default.fileExists(atPath:outputURL.deletingLastPathComponent().path),
      fixture.expectedOutputFloat32SHA256.count == 64, !fixture.sourcePins.isEmpty else {
      throw H3CheckpointError.invalid("Fast loading fixture output must be new and all pins declared.")
    }
    guard let compiledExecutable = Bundle(for:Self.self).executableURL else {
      throw H3CheckpointError.invalid("Missing compiled Fast loading qualification test bundle.")
    }
    let executable = compiledExecutable.resolvingSymlinksInPath()
    let expectedExecutable = URL(fileURLWithPath:fixture.testExecutable).resolvingSymlinksInPath()
    guard executable == expectedExecutable,
      try fastFileDigest(executable) == fixture.testExecutableSHA256 else {
      throw H3CheckpointError.invalid("Fast loading fixture test executable changed.")
    }
    func verifyPins() throws {
      for pin in fixture.sourcePins {
        let url = URL(fileURLWithPath:pin.path)
        _ = try fastFileIdentity(url)
        guard try fastFileDigest(url) == pin.sha256 else {
          throw H3CheckpointError.invalid("Fast loading fixture source changed: \(pin.path)")
        }
      }
      for pin in fixture.checkpointPins {
        let url = URL(fileURLWithPath:pin.path)
        guard try fastFileIdentity(url) == pin.identity,
          try fastHeaderDigest(url) == pin.headerSHA256 else {
          throw H3CheckpointError.invalid("Fast loading fixture checkpoint changed: \(pin.path)")
        }
      }
    }
    try verifyPins()
    let checkpoint = URL(fileURLWithPath:fixture.checkpoint)
    let pageURLs = try (0..<2).map { try H3CheckpointSource.fileURL(checkpoint,block:$0) }
    guard Set(fixture.checkpointPins.map { URL(fileURLWithPath:$0.path).standardizedFileURL }) == Set(pageURLs),
      Set(fixture.sourcePins.map { URL(fileURLWithPath:$0.path).lastPathComponent })
        .isSuperset(of:["H3PreparedBlock.swift","H3PreparedBlockTests.swift"]) else {
      throw H3CheckpointError.invalid("Fast loading fixture must pin both pages and both changed sources.")
    }
    let metal = URL(fileURLWithPath:fixture.metalLibrary)
    _ = try fastFileIdentity(metal)
    guard try fastFileDigest(metal) == fixture.metalSHA256 else {
      throw H3CheckpointError.invalid("Fast loading fixture Metal library changed.")
    }
    GPU.metallib = metal
    let layout = try H3CheckpointLayout(url:checkpoint)
    guard let variant = layout.fastVariant else {
      throw H3CheckpointError.invalid("Fast loading fixture requires the released packed FastH3 checkpoint.")
    }
    let tiles = variant == .vsaV1
      ? try H3FastTiles(prefixSegments:[171,414],videoGrid:[37,12,21]) : nil
    let rows = tiles?.rows ?? 13315
    let x = MLXRandom.normal([1,rows,5376],key:MLXRandom.key(901)).asType(.bfloat16)
    let mod = (MLXRandom.normal([1,96768],key:MLXRandom.key(902)) * Float(0.05)).asType(.bfloat16)
    let indices = MLXArray((0..<rows).map { Int32($0 < 171 ? 0 : ($0 < 585 ? 1 : 2)) })
    let positions = MLXArray((0..<rows).flatMap { r -> [Float] in
      r < 585 ? [Float(r),0,0] : [Float((r-585)/252),Float(((r-585)/21)%12),Float((r-585)%21)]
    },[rows,3])
    let angles = try H3TransformerBlock.prepareRotaryAngles(checkpointURL:checkpoint,positions:positions)
    eval([x,mod,indices,positions])
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    Stream.gpu.synchronize(); Memory.clearCache()
    let residentBefore = Memory.activeMemory
    var expectedParameters: [Int:[String:String]] = [:]
    var records: [[String:Any]] = []
    var warmSeconds: [Bool:[Double]] = [false:[],true:[]]
    func wordsDigest(_ value: MLXArray) -> String {
      var hasher = SHA256()
      // Slice by actual row count so even the largest packed factor uses only
      // bounded temporary host arrays during the untimed complete byte check.
      for start in stride(from:0,to:value.shape[0],by:256) {
        let part = value[start..<min(start + 256,value.shape[0])]
        if value.dtype == .uint32 {
          let words = part.asArray(UInt32.self)
          words.withUnsafeBytes { hasher.update(data:Data($0)) }
        } else {
          let words = part.view(dtype:.uint16).asArray(UInt16.self)
          words.withUnsafeBytes { hasher.update(data:Data($0)) }
        }
      }
      return hasher.finalize().map { String(format:"%02x",$0) }.joined()
    }
    func inspect(_ owner: H3PreparedBlock) throws -> [String:String] {
      var hashes: [String:String] = [:]
      let prefix = layout.prefix + "blocks.\(owner.index)."
      var norms = [("norm1.weight",[5376]),("norm2.weight",[5376]),
        ("attn.q_norm.weight",[128]),("attn.k_norm.weight",[128])]
      if variant == .vsaV1 { norms.append(("attn.gate_compress.weight",[7168,5376])) }
      for (suffix,shape) in norms { hashes[suffix] = wordsDigest(try owner.read(prefix + suffix,shape:shape)) }
      for (suffix,r,c) in [("attn.qkv_proj",21504,5376),
        ("attn.out_proj",5376,7168),("mlp.fc1",28672,5376),("mlp.fc2",5376,14336)] {
        for (i,value) in try owner.readAffineProjectionParameters(suffix).enumerated() {
          hashes[suffix + "." + ["packed","scales","biases"][i]] = wordsDigest(value)
        }
        let input = MLXArray((0..<c).map { Float(($0 % 31)-15)/31 },[1,1,c]).asType(.bfloat16)
        hashes[suffix + ".output"] = wordsDigest(try owner.project(suffix,
          activation:input,rows:r,columns:c,qkv:suffix == "attn.qkv_proj"))
      }
      return hashes
    }
    func trial(grouped: Bool, warmup: Bool) throws -> [String:Any] {
      Stream.gpu.synchronize(); Memory.clearCache(); Memory.peakMemory = Memory.activeMemory
      let start = CFAbsoluteTimeGetCurrent()
      var owners: [H3PreparedBlock] = []
      defer { owners.forEach { $0.close() } }
      for index in 0..<2 {
        owners.append(try H3PreparedBlock(checkpointURL:checkpoint,index:index,
          projectionMode:.weightDecoded,useMPP:false,groupedFastQ8Loading:grouped))
      }
      let preparation = CFAbsoluteTimeGetCurrent() - start
      let preparationPeak = Memory.peakMemory
      let storage = owners.reduce(0) { $0 + $1.storageBytes }
      XCTAssertTrue(owners.allSatisfy { $0.usesGroupedFastQ8Loading == grouped && !$0.usesGroupedBF16Loading })
      var validationSeconds = 0.0
      if warmup {
        let validationStart = CFAbsoluteTimeGetCurrent()
        for owner in owners {
          let hashes = try inspect(owner)
          if grouped { XCTAssertEqual(hashes,expectedParameters[owner.index]) }
          else { expectedParameters[owner.index] = hashes }
        }
        validationSeconds = CFAbsoluteTimeGetCurrent() - validationStart
        Stream.gpu.synchronize()
        Memory.peakMemory = Memory.activeMemory
      }
      var result = x
      for index in 0..<2 {
        result = try H3TransformerBlock.evaluate(checkpointURL:checkpoint,index:index,
          input:result,modulation:mod,modulationIndices:indices,positions:positions,
          rotaryAngles:angles,fastTiles:tiles,preparedWeights:owners[index],preparedFeedRowChunk:8192,
          observe: { _, _ in })
      }
      eval(result); Stream.gpu.synchronize()
      owners.forEach { $0.close() }
      let seconds = CFAbsoluteTimeGetCurrent() - start - validationSeconds
      let peak = max(preparationPeak,Memory.peakMemory)
      let physical = try fastPhysicalMemory()
      // Full Float32 output bytes are checked after the complete-pair timer.
      let values = result.asArray(Float.self)
      let digest = values.withUnsafeBytes { SHA256.hash(data:Data($0)).map { String(format:"%02x",$0) }.joined() }
      XCTAssertEqual(digest,fixture.expectedOutputFloat32SHA256)
      for owner in owners {
        XCTAssertTrue(owner.isClosed); XCTAssertEqual(owner.storageBytes,0)
        XCTAssertThrowsError(try owner.readAffineProjectionParameters("attn.qkv_proj"))
        XCTAssertThrowsError(try owner.read(layout.prefix + "blocks.\(owner.index).norm1.weight",shape:[5376]))
        XCTAssertThrowsError(try owner.project("attn.qkv_proj",activation:x,rows:21504,columns:5376,qkv:true))
        owner.close()
      }
      if !warmup, let maximum = fixture.maximumPeakMLXBytes { XCTAssertLessThanOrEqual(peak,maximum) }
      if !warmup { warmSeconds[grouped,default:[]].append(seconds) }
      return ["groupedFastQ8Loading":grouped,"warmup":warmup,"preparationSeconds":preparation,
        "seconds":seconds,"untimedValidationSeconds":validationSeconds,"peakMLXBytes":peak,
        "preparationPeakMLXBytesBeforeValidation":preparationPeak,
        "currentPhysicalFootprintBytesBeforeOutputHash":physical.current,
        "processLifetimePeakPhysicalFootprintBytes":physical.lifetimePeak,
        "storageBytesBeforeExecution":storage,"outputFloat32SHA256":digest]
    }
    // Retain both first passes. Their byte validation is untimed and their
    // combined physical high-water mark is explicitly not a per-mode reset.
    for grouped in [false,true] {
      records.append(try autoreleasepool { try trial(grouped:grouped,warmup:true) })
      Stream.gpu.synchronize(); Memory.clearCache()
      XCTAssertEqual(Memory.activeMemory,residentBefore)
    }
    for grouped in [false,true,true,false,false,true,true,false] {
      records.append(try autoreleasepool { try trial(grouped:grouped,warmup:false) })
      Stream.gpu.synchronize(); Memory.clearCache()
      XCTAssertEqual(Memory.activeMemory,residentBefore)
      XCTAssertEqual(Memory.cacheMemory,0)
    }
    func median(_ values: [Double]) -> Double {
      let sorted = values.sorted(); return (sorted[1] + sorted[2]) / 2
    }
    let serial = median(warmSeconds[false]!), grouped = median(warmSeconds[true]!)
    let physical = try fastPhysicalMemory()
    if let minimum = fixture.minimumImprovementSeconds { XCTAssertGreaterThanOrEqual(serial-grouped,minimum) }
    if let maximum = fixture.maximumPhysicalPeakBytes { XCTAssertLessThanOrEqual(physical.lifetimePeak,maximum) }
    try verifyPins()
    let report: [String:Any] = ["format":"weetodd-grouped-fast-q8-qualification-v1",
      "variant":variant.rawValue,"inputRows":rows,"windowSize":2,"MPP":false,"adapter":false,
      "inputRandomKey":901,"modulationRandomKey":902,"modulationScale":0.05,
      "prefixSegments":[171,414],"videoGrid":[37,12,21],"feedRowChunk":8192,
      "records":records,"parameterHashes":Dictionary(uniqueKeysWithValues:expectedParameters.map { (String($0.key),$0.value) }),
      "warmSerialMedianSeconds":serial,"warmGroupedMedianSeconds":grouped,
      "absoluteWarmImprovementSeconds":serial-grouped,
      "remainingActiveMLXBytes":Memory.activeMemory,"inputResidentMLXBytes":residentBefore,
      "remainingOwnerActiveMLXBytes":Memory.activeMemory-residentBefore,
      "remainingCacheMLXBytes":Memory.cacheMemory,
      "processLifetimePeakPhysicalFootprintBytes":physical.lifetimePeak,
      "physicalScope":"Shared test-process lifetime high-water mark includes untimed exact byte validation; cannot reset or attribute independently to each ABBA mode.",
      "fixture":manifest,"fixtureSHA256":try fastFileDigest(URL(fileURLWithPath:manifest)),
      "testExecutableSHA256":fixture.testExecutableSHA256,"metalSHA256":fixture.metalSHA256,
      "scope":"Two actual Fast owners and two complete blocks; no sampler or model generation. Production groupedFastQ8 default remains disabled."]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
      .write(to:outputURL,options:.withoutOverwriting)
  }

  private func installed() throws -> URL {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_PREPARED_BLOCK_TESTS"] == "1",
      let checkpoint = environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed H3 prepared-block projection and ownership qualification.")
    }
    return URL(fileURLWithPath: checkpoint)
  }

  func testInstalledGroupedBF16FirstTwoOwnersMatchSerialBytesAndProjections() throws {
    let checkpoint = try installed()
    let layout = try H3CheckpointLayout(url: checkpoint)
    guard layout.curveRank != nil, layout.fastVariant == nil else {
      throw XCTSkip("Grouped loading qualification requires the ordinary pruned BF16 checkpoint.")
    }
    let env = ProcessInfo.processInfo.environment
    let output = env["WEETODD_H3_GROUPED_BF16_OUTPUT"].map { URL(fileURLWithPath:$0) }
    if let output, FileManager.default.fileExists(atPath:output.path) {
      throw H3CheckpointError.invalid("Grouped loading receipt must be fresh.")
    }
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    func digest(_ value: MLXArray) -> String {
      let words = value.view(dtype:.uint16).asArray(UInt16.self)
      return words.withUnsafeBytes { bytes in
        var hasher = SHA256()
        // Bound temporary host copies while hashing actual BF16 payload bytes.
        for start in stride(from:0,to:bytes.count,by:4 * 1024 * 1024) {
          hasher.update(data:Data(bytes[start..<min(start + 4 * 1024 * 1024,bytes.count)]))
        }
        return hasher.finalize().map { String(format:"%02x",$0) }.joined()
      }
    }
    var expected: [Int:[String:String]] = [:]
    var reports: [[String:Any]] = []
    for grouped in [false,true] {
      Stream.gpu.synchronize(); Memory.clearCache()
      let activeBefore = Memory.activeMemory
      Memory.peakMemory = activeBefore
      var loadSeconds: [Double] = []
      for iteration in 0..<4 {
        var twoOwnerSeconds = 0.0
        for index in 0..<2 {
          try Task.checkCancellation()
          let start = CFAbsoluteTimeGetCurrent()
          let owner = try H3PreparedBlock(checkpointURL:checkpoint,index:index,
            projectionMode:.weightDecoded,groupedBF16Loading:grouped)
          twoOwnerSeconds += CFAbsoluteTimeGetCurrent() - start
          defer { owner.close() }
          XCTAssertEqual(owner.usesGroupedBF16Loading,grouped)
          if iteration == 0 {
            var hashes: [String:String] = [:]
            func inspect() throws {
              for (suffix,shape) in [("norm1.weight",[5376]),("norm2.weight",[5376]),
                ("attn.q_norm.weight",[128]),("attn.k_norm.weight",[128])] {
                hashes[suffix] = digest(try owner.read(layout.prefix + "blocks.\(index)." + suffix,
                  shape:shape))
              }
              for (suffix,rows,columns) in [("attn.qkv_proj",21504,5376),
                ("attn.out_proj",5376,7168),("mlp.fc1",28672,5376),("mlp.fc2",5376,14336)] {
                hashes[suffix + ".weight"] = digest(try owner.readDenseProjectionWeight(suffix))
                let input = MLXArray((0..<columns).map { Float(($0 % 31)-15)/31 },
                  [1,1,columns]).asType(.bfloat16)
                hashes[suffix + ".output"] = digest(try owner.project(suffix,
                  activation:input,rows:rows,columns:columns,qkv:suffix == "attn.qkv_proj"))
              }
            }
            try inspect()
            if grouped { XCTAssertEqual(hashes,expected[index]) }
            else { expected[index] = hashes }
            reports.append(["grouped":grouped,"index":index,"hashes":hashes])
          }
          owner.close(); Memory.clearCache()
          XCTAssertTrue(owner.isClosed)
          XCTAssertEqual(owner.storageBytes,0)
          XCTAssertEqual(Memory.activeMemory,activeBefore,"Closed retained owner must release actual arrays.")
          XCTAssertThrowsError(try owner.readDenseProjectionWeight("attn.qkv_proj"))
        }
        loadSeconds.append(twoOwnerSeconds)
      }
      reports.append(["grouped":grouped,"twoOwnerLoadSeconds":loadSeconds,
        "warmMedianTwoOwnerLoadSeconds":loadSeconds.dropFirst().sorted()[1],
        "peakMLXBytes":Memory.peakMemory,"remainingActiveMLXBytes":Memory.activeMemory,
        "scope":"Serial owners0/1, exact BF16 matrices/norms/projections; hash host validation excluded from load timing."])
    }
    if let output {
      try JSONSerialization.data(withJSONObject:reports,options:[.prettyPrinted,.sortedKeys])
        .write(to:output,options:.withoutOverwriting)
    }
  }

  func testInstalledPreparedProjectionsMatchEagerCheckpointMath() throws {
    let checkpoint = try installed()
    let owner = try H3PreparedBlock(checkpointURL: checkpoint, index: 0,
      projectionMode: .weightDecoded)
    defer { owner.close() }
    XCTAssertEqual(owner.index, 0)
    XCTAssertEqual(owner.checkpointURL, checkpoint)
    XCTAssertFalse(owner.isClosed)
    XCTAssertGreaterThan(owner.storageBytes, 0)
    for (suffix, rows, columns) in [
      ("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168),
      ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336),
    ] {
      let qkv = suffix == "attn.qkv_proj"
      let input = MLXArray((0..<columns).map { Float(($0 % 31) - 15) / 31 },
        [1, 1, columns]).asType(.bfloat16)
      let actual = try owner.project(suffix, activation: input,
        rows: rows, columns: columns, qkv: qkv)
      eval(actual)
      let expected: MLXArray
      let name = owner.layout.prefix + "blocks.0." + suffix + ".weight"
      if owner.file.tensors[name]?.dtype == "U32" {
        expected = try H3QwenQ8Projection(file: owner.file, name: name).project(input)
      } else if owner.layout.curveRank != nil {
        let weight = try owner.file.withTensorBytes(named: name) {
          MLXArray($0, [rows, columns], type: UInt16.self).view(dtype: .bfloat16)
        }
        expected = matmul(input, H3QKVRowOrder.forHeadMajorAttention(weight,
          heads: 56, headWidth: 128, groupedSource: false).T)
      } else {
        let weight = try H3ComfyDecodedProjection.load(file: owner.file,
          checkpointURL: owner.tensorURL, name: name, rows: rows, columns: columns,
          reorderQKV: qkv, rowWindow: 16384)
        expected = matmul(input, weight.T)
      }
      XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), suffix)
    }
    for (suffix, shape) in [
      ("norm1.weight", [5376]), ("norm2.weight", [5376]),
      ("attn.q_norm.weight", [128]), ("attn.k_norm.weight", [128]),
    ] {
      let name = owner.layout.prefix + "blocks.0." + suffix
      let first = try owner.read(name, shape: shape, dtype: "BF16")
      XCTAssertTrue(first === (try owner.read(name, shape: shape, dtype: "BF16")),
        "Prepared norm reads must return their stable owned array.")
      let expected = try owner.file.withTensorBytes(named: name) {
        MLXArray($0, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
      XCTAssertEqual(first.asArray(Float.self), expected.asArray(Float.self), suffix)
    }
    if owner.layout.fastVariant == .vsaV1 {
      let name = owner.layout.prefix + "blocks.0.attn.gate_compress.weight"
      let first = try owner.read(name, shape: [7168, 5376], dtype: "BF16")
      XCTAssertTrue(first === (try owner.read(name, shape: [7168, 5376], dtype: "BF16")))
    }
    try owner.checkUnchanged()
  }

  func testInstalledClosedOwnerRejectsReadsAndProjections() throws {
    let checkpoint = try installed()
    let input = MLXArray.zeros([1, 1, 5376], dtype: .bfloat16)
    let owner = try H3PreparedBlock(checkpointURL: checkpoint, index: 0,
      projectionMode: .weightDecoded)
    owner.close()
    XCTAssertTrue(owner.isClosed)
    XCTAssertEqual(owner.storageBytes, 0)
    XCTAssertThrowsError(try owner.read(owner.layout.prefix + "blocks.0.norm1.weight",
      shape: [5376], dtype: "BF16"))
    XCTAssertThrowsError(try owner.project("attn.qkv_proj", activation: input,
      rows: 21504, columns: 5376, qkv: true))
    XCTAssertThrowsError(try owner.checkUnchanged())
    owner.close()
    XCTAssertEqual(owner.storageBytes, 0)
  }

  func testInstalledCloseReleasesActualArraysWhileOwnerIsRetained() throws {
    let checkpoint = try installed()
    Stream.gpu.synchronize()
    Memory.clearCache()
    let before = Memory.activeMemory
    let owner = try H3PreparedBlock(checkpointURL: checkpoint, index: 0,
      projectionMode: .weightDecoded)
    XCTAssertGreaterThan(Memory.activeMemory, before)
    owner.close()
    Memory.clearCache()
    XCTAssertTrue(owner.isClosed)
    XCTAssertEqual(owner.storageBytes, 0)
    XCTAssertEqual(Memory.activeMemory, before,
      "Closing a retained owner must drop its actual projection, norm and gate arrays.")
  }
  func testInstalledTwoBlockWindowMatchesEagerOutputAndMeasuresPreparation() throws {
    let env=ProcessInfo.processInfo.environment
    guard let reportPath=env["WEETODD_H3_PREPARED_WINDOW_OUTPUT"],
      let checkpointPath=env["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Opt-in exact two-block prepared-window performance qualification.")
    }
    guard !FileManager.default.fileExists(atPath:reportPath) else {
      throw H3CheckpointError.invalid("Prepared-window report must be fresh.")
    }
    let checkpoint=URL(fileURLWithPath:checkpointPath)
    let layout=try H3CheckpointLayout(url:checkpoint)
    let tiles=layout.fastVariant == .vsaV1
      ? try H3FastTiles(prefixSegments:[171,414],videoGrid:[37,12,21]):nil
    guard let rows=tiles?.rows ?? Int(env["WEETODD_H3_PREPARED_WINDOW_ROWS"] ?? "13315"),
      (1...40_000).contains(rows) else {
      throw H3CheckpointError.invalid("Prepared-window fixture rows are invalid.")
    }
    let x=MLXRandom.normal([1,rows,5376],key:MLXRandom.key(901)).asType(.bfloat16)
    let mod=(MLXRandom.normal([1,96768],key:MLXRandom.key(902))*Float(0.05)).asType(.bfloat16)
    let indices=MLXArray((0..<rows).map { Int32($0<171 ? 0 : ($0<585 ? 1:2)) })
    let positions=MLXArray((0..<rows).flatMap { r -> [Float] in
      r<585 ? [Float(r),0,0]:[Float((r-585)/252),Float(((r-585)/21)%12),Float((r-585)%21)]
    },[rows,3])
    let angles=try H3TransformerBlock.prepareRotaryAngles(checkpointURL:checkpoint,positions:positions)
    let adapter: H3LoRAFile? = try env["WEETODD_H3_TEST_TURBO_LORA"].map {
      try H3LoRAFile(url:URL(fileURLWithPath:$0),strength:1)
    }
    eval([x,mod,indices,positions])
    let previousLimit=Memory.cacheLimit
    Memory.cacheLimit=128*1024*1024
    defer { Stream.gpu.synchronize();Memory.clearCache();Memory.cacheLimit=previousLimit }
    let windowSize=Int(env["WEETODD_H3_PREPARED_WINDOW_SIZE"] ?? "2") ?? 2
    guard [1,2].contains(windowSize) else { throw H3CheckpointError.invalid("Invalid fixture window size.") }
    let useMPP=env["WEETODD_H3_PREPARED_MPP"] == "1"
    let feedRowChunk = Int(env["WEETODD_H3_PREPARED_FEED_ROW_CHUNK"] ?? "8192") ?? 8192
    guard [8192,16_384].contains(feedRowChunk) else {
      throw H3CheckpointError.invalid("Unsupported prepared fixture feed-forward row chunk.")
    }
    let diagnostic=env["WEETODD_H3_PREPARED_WINDOW_DIAGNOSTIC"] == "1"
    var referenceHash:String?
    var reports:[[String:Any]]=[]
    for prepared in [false,true] {
      var seconds:[Double]=[], preparation:[Double]=[]
      var lastHash=""
      var boundaries:[[String:Any]]=[]
      var diagnosticPeak=0
      var boundaryStart=CFAbsoluteTimeGetCurrent()
      Memory.clearCache();Memory.peakMemory=Memory.activeMemory
      for _ in 0..<(diagnostic ? 1:4) {
        let start=CFAbsoluteTimeGetCurrent()
        var owners:[H3PreparedBlock]=[]
        defer { owners.forEach { $0.close() } }
        var preparationSeconds=0.0
        if prepared && windowSize == 2 {
          let readyStart=CFAbsoluteTimeGetCurrent()
          for index in 0..<2 {
            owners.append(try H3PreparedBlock(checkpointURL:checkpoint,index:index,
              projectionMode:.weightDecoded,useMPP:useMPP,
              groupedBF16Loading:env["WEETODD_H3_PREPARED_GROUPED_BF16"] != "0"))
          }
          preparationSeconds += CFAbsoluteTimeGetCurrent()-readyStart
        }
        var output=x
        for index in 0..<2 {
          if prepared && windowSize == 1 {
            let readyStart=CFAbsoluteTimeGetCurrent()
            owners=[try H3PreparedBlock(checkpointURL:checkpoint,index:index,projectionMode:.weightDecoded,useMPP:useMPP,
              groupedBF16Loading:env["WEETODD_H3_PREPARED_GROUPED_BF16"] != "0")]
            preparationSeconds += CFAbsoluteTimeGetCurrent()-readyStart
          }
          output=try H3TransformerBlock.evaluate(checkpointURL:checkpoint,index:index,
            input:output,modulation:mod,modulationIndices:indices,positions:positions,
            lora:adapter,rotaryAngles:angles,fastTiles:tiles,
            preparedWeights:prepared ? owners[windowSize == 1 ? 0:index]:nil,
            preparedFeedRowChunk:feedRowChunk,observe:{ name,value in
              if diagnostic {
                eval(value);Stream.gpu.synchronize()
                let now=CFAbsoluteTimeGetCurrent()
                diagnosticPeak=max(diagnosticPeak,Memory.peakMemory)
                boundaries.append(["block":index,"boundary":name,"seconds":now-boundaryStart,
                  "activeMLXBytes":Memory.activeMemory,"peakMLXBytes":Memory.peakMemory])
                Memory.peakMemory=Memory.activeMemory;boundaryStart=now
              }
            })
          if prepared && windowSize == 1 { eval(output);owners.forEach { $0.close() };owners=[] }
        }
        eval(output);Stream.gpu.synchronize()
        owners.forEach { $0.close() };owners=[]
        seconds.append(CFAbsoluteTimeGetCurrent()-start)
        preparation.append(preparationSeconds)
        let values=output.asArray(Float.self)
        lastHash=values.withUnsafeBytes { SHA256.hash(data:Data($0)).map { String(format:"%02x",$0) }.joined() }
        if let frozen=env["WEETODD_H3_PREPARED_WINDOW_EXPECTED_SHA256"] { XCTAssertEqual(lastHash,frozen) }
        if let referenceHash { XCTAssertEqual(lastHash,referenceHash) }
        else { referenceHash=lastHash }
      }
      let median=diagnostic ? seconds[0]:seconds.dropFirst().sorted()[1]
      let peak=max(Memory.peakMemory,diagnosticPeak)
      if prepared && !diagnostic {
        if let ratio=env["WEETODD_H3_PREPARED_WINDOW_MAX_RATIO"].flatMap(Double.init),
          let eager=reports.first?["warmMedianSeconds"] as? Double {
          XCTAssertLessThanOrEqual(median,eager*ratio)
        }
        if let limit=env["WEETODD_H3_PREPARED_WINDOW_MAX_SECONDS"].flatMap(Double.init) {
          XCTAssertLessThanOrEqual(median,limit)
        }
        if let limit=env["WEETODD_H3_PREPARED_WINDOW_MAX_BYTES"].flatMap(Int.init) {
          XCTAssertLessThanOrEqual(peak,limit)
        }
      }
      let verification = H3MPPProjection.verificationStatus(scope: checkpoint.standardizedFileURL.path)
      reports.append(["verifiedMPPSignatures": verification.verified, "fallbackMPPSignatures": verification.fallback,
        "prepared":prepared,"groupedBF16Loading":env["WEETODD_H3_PREPARED_GROUPED_BF16"] != "0", "inputRows":rows,"seconds":seconds,
        "warmMedianSeconds":median,"preparationSeconds":preparation,
        "peakMLXBytes":peak,"outputFloat32SHA256":lastHash,
        "adapter":adapter != nil,"mpp":prepared && useMPP && layout.fastVariant == nil,
        "feedRowChunk":feedRowChunk,"windowSize":windowSize,"diagnosticOnly":diagnostic,"boundaries":boundaries])
    }
    try JSONSerialization.data(withJSONObject:reports,options:[.prettyPrinted,.sortedKeys])
      .write(to:URL(fileURLWithPath:reportPath))
  }

}
