import CryptoKit
import Foundation
import XCTest
@testable import TensorIO

// These fixtures describe tensors directly; no torch/Python runtime creates or reads them.
private enum TorchFixture {
  static func le<T:FixedWidthInteger>(_ value:T)->Data { var v=value.littleEndian;return withUnsafeBytes(of:&v) { Data($0) } }
  static func text(_ s:String)->Data { Data(s.utf8) }
  static func string(_ s:String)->Data { Data([0x58])+le(UInt32(s.utf8.count))+text(s) }
  static func integer(_ n:Int32)->Data { Data([0x4a])+le(n) }
  static func tuple(_ values:[Int32])->Data { Data([0x28])+values.reduce(Data()) { $0+integer($1) }+Data([0x74]) }
  static func tensor(offset:Int32=0,shape:[Int32]=[2,2],strides:[Int32]=[2,1],bf16:Bool=false)->Data {
    var data=text("ctorch._utils\n_rebuild_tensor_v2\n(")+Data([0x28])+string("storage")
    data += text(bf16 ? "ctorch\nBFloat16Storage\n" : "ctorch\nFloatStorage\n")
    data += string("0")+string("cpu")+integer(8)+Data([0x74,0x51])
    data += integer(offset)+tuple(shape)+tuple(strides)+Data([0x89])
    return data+text("ccollections\nOrderedDict\n)R")+Data([0x74,0x52])

  }
  static func pickle(_ tensors:[(String,Data)])->Data {
    Data([0x80,2])+text("ccollections\nOrderedDict\n)R(")+tensors.reduce(Data()) { $0+string($1.0)+$1.1 }+Data([0x75,0x2e])
  }
  static func zip(pickle:Data,storage:Data=Data((0..<32).map(UInt8.init)),big:Bool=false,zip64:Bool=false,
    compressed:Bool=false,traversal:Bool=false,overlap:Bool=false,root:String="step-step=54000_lora")->Data {
    let entries=[("data.pkl",pickle),("byteorder",text(big ? "big":"little")),("data/0",storage),("version",text("3\n")),(".data/serialization_id",text("0"))]
    var result=Data(),central=Data()
    for (index,item) in entries.enumerated() {
      let name=text(root+"/"+(traversal && index==2 ? "../data/0":item.0))
      let offset=UInt32(result.count),size=UInt32(item.1.count),method:UInt16=compressed && index==2 ? 8:0
      result += le(UInt32(0x04034b50))+le(UInt16(20))+le(UInt16(0))+le(method)
      result += le(UInt32(0))+le(UInt32(0))+le(size)+le(size)
      result += le(UInt16(name.count))+le(UInt16(0))+name+item.1
      let extra=zip64 ? le(UInt16(1))+le(UInt16(24))+le(UInt64(size))+le(UInt64(size))+le(UInt64(offset)):Data()
      central += le(UInt32(0x02014b50))+le(UInt16(45))+le(UInt16(45))+le(UInt16(0))+le(method)
      central += le(UInt32(0))+le(UInt32(0))+le(zip64 ? UInt32.max:size)+le(zip64 ? UInt32.max:size)
      central += le(UInt16(name.count))+le(UInt16(extra.count))+le(UInt16(0))+le(UInt16(0))+le(UInt16(0))
      central += le(UInt32(0))+le(overlap && index==2 ? UInt32(0):(zip64 ? UInt32.max:offset))+name+extra

    }
    let start=UInt64(result.count);result += central
    if zip64 {
      let z=UInt64(result.count)
      result += le(UInt32(0x06064b50))+le(UInt64(44))+le(UInt16(45))+le(UInt16(45))
      result += le(UInt32(0))+le(UInt32(0))+le(UInt64(entries.count))+le(UInt64(entries.count))
      result += le(UInt64(central.count))+le(start)
      result += le(UInt32(0x07064b50))+le(UInt32(0))+le(z)+le(UInt32(1))
    }
    result += le(UInt32(0x06054b50))+le(UInt16(0))+le(UInt16(0))
    result += le(zip64 ? UInt16.max:UInt16(entries.count))+le(zip64 ? UInt16.max:UInt16(entries.count))
    result += le(zip64 ? UInt32.max:UInt32(central.count))+le(zip64 ? UInt32.max:UInt32(start))+le(UInt16(0))
    return result
  }
  static func hash(_ data:Data)->String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
}

final class TorchTensorArchiveTests:XCTestCase {
  private func temporary(_ body:(URL)throws->Void)throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) };try body(root)
  }
  private func open(_ data:Data,at root:URL,profile:TorchTensorImportProfile = .tinyDecoder)throws->TorchTensorArchive {
    let url=root.appendingPathComponent("source.ckpt");try data.write(to:url)
    let fixtureProfile=TorchTensorImportProfile(id:profile.id,archiveRoot:profile.archiveRoot,sourceSHA256:TorchFixture.hash(data),dtype:profile.dtype,maximumTensors:profile.maximumTensors,maximumStorages:profile.maximumStorages,rootTensor:profile.rootTensor,allowsMetadata:profile.allowsMetadata)
    return try TorchTensorArchive.open(url:url,expectedSHA256:TorchFixture.hash(data),profile:fixtureProfile)
  }
  func testSharedStorageNonzeroOffsetAndStridedViewsAreLogicalBytes()throws {
    try temporary { root in
      let d=TorchFixture.zip(pickle:TorchFixture.pickle([("contiguous",TorchFixture.tensor(offset:1)),("transpose",TorchFixture.tensor(offset:1,strides:[1,2]))]))
      let archive=try open(d,at:root)
      XCTAssertEqual(archive.tensors.count,2)
      XCTAssertEqual(archive.tensors["transpose"]?.shape,[2,2])
      XCTAssertEqual(archive.tensors["transpose"]?.strides,[1,2])
      XCTAssertEqual(archive.tensors["transpose"]?.storageOffset,1)
      XCTAssertEqual(archive.tensors["transpose"]?.storageByteRange,archive.tensors["contiguous"]?.storageByteRange)
      XCTAssertEqual(try archive.withTensorBytes(named:"contiguous") { Array($0) },Array(4..<20))
      XCTAssertEqual(try archive.withTensorBytes(named:"transpose") { Array($0) },Array(4..<8)+Array(12..<16)+Array(8..<12)+Array(16..<20))
      XCTAssertThrowsError(try archive.withTensorBytes(named:"contiguous",range:0..<17) { _ in })
    }
  }
  func testZIP64AndBigEndianF32AreNormalizedToLittleEndian()throws {
    try temporary { root in
      let archive=try open(TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor(shape:[2],strides:[1]))]),big:true,zip64:true),at:root)
      XCTAssertEqual(try archive.withTensorBytes(named:"x") { Array($0) },[3,2,1,0,7,6,5,4])
    }
  }
  func testBF16SingleTensorProfileAndDeterministicAtomicImport()throws {
    try temporary { root in
      // Positive prompt has a tensor at the root instead of a state dictionary.
      let pickle=Data([0x80,2])+TorchFixture.tensor(shape:[2],strides:[1],bf16:true)+Data([0x2e])
      let zip=TorchFixture.zip(pickle:pickle,storage:Data(0..<16),big:true,root:"posi_prompt")
      let archive=try open(zip,at:root,profile:.positivePrompt)
      XCTAssertEqual(archive.tensors["tensor"]?.dtype,"BF16")
      XCTAssertEqual(try archive.withTensorBytes(named:"tensor") { Array($0) },[1,0,3,2])
      let a=root.appendingPathComponent("a.safetensors"),b=root.appendingPathComponent("b.safetensors")
      XCTAssertEqual(try archive.importSafetensors(to:a),try archive.importSafetensors(to:b))
      XCTAssertEqual(try Data(contentsOf:a),try Data(contentsOf:b))
      XCTAssertEqual(try SafeTensorFile(url:a).withTensorBytes(named:"tensor") { Array($0) },[1,0,3,2])
      XCTAssertThrowsError(try archive.importSafetensors(to:a))
      XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path).sorted(),["a.safetensors","b.safetensors","source.ckpt"])
    }
  }
  func testRejectsUnsafePickleAndMalformedDescriptors()throws {
    try temporary { root in
      let bad=[Data([0x80,2])+TorchFixture.text("cos\nsystem\n)R."),Data([0x80,2,0x4e,0x2e]),
        Data([0x80,2])+TorchFixture.text("ccollections\nOrderedDict\n)R}b."),
        TorchFixture.pickle([("x",TorchFixture.tensor(offset:7))]),
        TorchFixture.pickle([("x",TorchFixture.tensor(shape:[Int32.max,Int32.max,Int32.max],strides:[1,1,1]))]),
        TorchFixture.pickle([("x",TorchFixture.tensor(shape:[2],strides:[-1]))]),
        TorchFixture.pickle([("x",TorchFixture.tensor()),("x",TorchFixture.tensor())])]
      for pickle in bad { XCTAssertThrowsError(try open(TorchFixture.zip(pickle:pickle),at:root)) }
    }
  }
  func testRejectsCompressedTraversalOverlappingAndTruncatedZIP()throws {
    try temporary { root in
      let p=TorchFixture.pickle([("x",TorchFixture.tensor())])
      for data in [TorchFixture.zip(pickle:p,compressed:true),TorchFixture.zip(pickle:p,traversal:true),TorchFixture.zip(pickle:p,overlap:true),TorchFixture.zip(pickle:p).dropLast()] {
        XCTAssertThrowsError(try open(Data(data),at:root))
      }
    }
  }
  func testHashAndSourceIdentityRejectChangedSourceBeforePublishing()throws {
    try temporary { root in
      let archive=try open(TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor())])),at:root)
      let source=root.appendingPathComponent("source.ckpt")
      XCTAssertThrowsError(try TorchTensorArchive.open(url:source,expectedSHA256:String(repeating:"0",count:64),profile:.tinyDecoder))
      try Data([0]).write(to:source)
      XCTAssertThrowsError(try archive.withTensorBytes(named:"x") { _ in })
      let output=root.appendingPathComponent("out.safetensors")
      XCTAssertThrowsError(try archive.importSafetensors(to:output))
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
    }
  }
}

extension TorchTensorArchiveTests {
  func testRejectsLogicalPayloadAboveSignedAddressSpaceEvenWhenStorageIsBroadcast()throws {
    try temporary { root in
      let pickle=TorchFixture.pickle([("x",TorchFixture.tensor(shape:[Int32.max,Int32.max],strides:[0,0]))])
      XCTAssertThrowsError(try open(TorchFixture.zip(pickle:pickle),at:root))
    }
  }
  func testUnalignedLogicalSliceAndSourceReplacementAreChecked()throws {
    try temporary { root in
      let archive=try open(TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor(offset:1,strides:[1,2]))]),big:true),at:root)
      XCTAssertEqual(try archive.withTensorBytes(named:"x",range:1..<7) { Array($0) },[6,5,4,15,14,13])
      let replacement=root.appendingPathComponent("replacement")
      try Data([0]).write(to:replacement)
      _ = try FileManager.default.replaceItemAt(root.appendingPathComponent("source.ckpt"),withItemAt:replacement)
      XCTAssertThrowsError(try archive.checkUnchanged())
    }
  }
  func testConversionIdentityInvalidatesOnImporterVersionAndProfile() {
    let profile=TorchTensorImportProfile.positivePrompt
    let key=TorchTensorArchive.conversionKey(sourceSHA256:profile.sourceSHA256,profile:profile)
    XCTAssertNotEqual(key,TorchTensorArchive.conversionKey(sourceSHA256:profile.sourceSHA256,profile:profile,importerVersion:"next"))
    XCTAssertNotEqual(key,TorchTensorArchive.conversionKey(sourceSHA256:profile.sourceSHA256,profile:.tinyDecoder))
  }
  func testInstalledPinnedArchivesHaveExactDescriptorsAndOriginalStorageBytes()throws {
    guard let configured=ProcessInfo.processInfo.environment["WEETODD_FLASHVSR_SOURCE_ROOT"] else {
      throw XCTSkip("Set WEETODD_FLASHVSR_SOURCE_ROOT for the separate installed-weight qualification.")
    }
    let sourceRoot=URL(fileURLWithPath:configured,isDirectory:true)
    let cases:[(String,TorchTensorImportProfile,Int,String,String,String)]=[
      ("TCDecoder.ckpt",.tinyDecoder,66,"42a274f0ef86d3dd80017b296b29a1a24d8f523cd3b27aaf15c0407b2e604a9b","decoder.1.bias","a328c2bd899a81bd8183a3bddbfc89bd"),
      ("Wan2.1_VAE.pth",.wanVAE,194,"32f77cca5507e543ef0f0bee9478711174b912314c6f4a4bd6dc7d2d36e2a245","conv1.bias","ff6f39bd8d2254bdd7430abe225efa3d"),
      ("posi_prompt.pth",.positivePrompt,1,"db22195fae7ee8b496206856faf24a44df8d43f87433be8a66e19b30efeb7b7b","tensor","ee3a6c3cd13bd83d6b3b003d833cf53c")]
    for (name,profile,count,metadataHash,sampleName,sampleHex) in cases {
      let source=sourceRoot.appendingPathComponent(name)
      let archive=try TorchTensorArchive.open(url:source,expectedSHA256:profile.sourceSHA256,profile:profile)
      XCTAssertEqual(archive.sourceSHA256,profile.sourceSHA256)
      XCTAssertEqual(archive.tensors.count,count)
      // This hash covers every key, dtype, dimension, storage key, offset and
      // stride, independently decoded with read-only stdlib pickletools.
      let lines=archive.tensors.keys.sorted().map { name in
        let t=archive.tensors[name]!
        return name+"|"+t.dtype+"|"+t.shape.map(String.init).joined(separator:",")+"|"+t.storageKey+"|"+String(t.storageOffset)+"|"+t.strides.map(String.init).joined(separator:",")+"\n"
      }.joined()
      XCTAssertEqual(TorchFixture.hash(Data(lines.utf8)),metadataHash)
      let bytes=try archive.withTensorBytes(named:sampleName,range:0..<16) { Data($0) }
      XCTAssertEqual(bytes.map { String(format:"%02x",$0) }.joined(),sampleHex)
      let t=archive.tensors[sampleName]!,width:UInt64=t.dtype=="F32" ? 4:2
      let handle=try FileHandle(forReadingFrom:source);defer { try? handle.close() }
      try handle.seek(toOffset:t.storageByteRange.lowerBound+t.storageOffset*width)
      XCTAssertEqual(try handle.read(upToCount:16),bytes)
      for name in archive.tensors.keys.sorted() {
        let t=archive.tensors[name]!,end=min(t.byteCount,128)
        XCTAssertEqual(try archive.withTensorBytes(named:name,range:0..<end) { $0.count },Int(end))
      }
      if profile == .positivePrompt { XCTAssertEqual(archive.tensors["tensor"]?.shape,[1,512,4096]) }
      print("INSTALLED TORCH PROOF \(name) sha256=\(archive.sourceSHA256) tensors=\(count) metadata=\(metadataHash)")
    }
  }
}

extension TorchTensorArchiveTests {
  @MainActor func testCancellationRemovesPartiallyWrittenImportAndNeverPublishes()async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    // A legal broadcast view makes a 2 MiB logical tensor from eight stored
    // elements; gathering keeps a partial writer alive until cancellation.
    let data=TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor(shape:[524288],strides:[0]))]))
    let source=root.appendingPathComponent("source.ckpt"),output=root.appendingPathComponent("out.safetensors")
    try data.write(to:source)
    let hash=TorchFixture.hash(data)
    let profile=TorchTensorImportProfile(id:"fixture",archiveRoot:"step-step=54000_lora",sourceSHA256:hash,
      dtype:"F32",maximumTensors:66,maximumStorages:1,rootTensor:false,allowsMetadata:false)
    let operation=Task.detached {
      let archive=try TorchTensorArchive.open(url:source,expectedSHA256:hash,profile:profile)
      return try archive.importSafetensors(to:output)
    }
    var sawPartial=false
    for _ in 0..<2000 {
      if try FileManager.default.contentsOfDirectory(atPath:root.path).contains(where:{ $0.hasSuffix(".partial") }) {
        sawPartial=true;break
      }
      try await Task.sleep(for:.milliseconds(1))
    }
    operation.cancel()
    do { _ = try await operation.value;XCTFail("A cancelled import must not publish a tensor page.") }
    catch { XCTAssertTrue(error is CancellationError,"Unexpected cancellation error: \(error)") }
    XCTAssertTrue(sawPartial,"The test must cancel an active writer, not an unopened source.")
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source.ckpt"])
  }
  func testOutputSymlinkSurvivesFailedAtomicPublishWithoutPartialFiles()throws {
    try temporary { root in
      let archive=try open(TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor())])),at:root)
      let existing=root.appendingPathComponent("existing"),link=root.appendingPathComponent("link.safetensors")
      try Data([9,8]).write(to:existing)
      try FileManager.default.createSymbolicLink(at:link,withDestinationURL:existing)
      XCTAssertThrowsError(try archive.importSafetensors(to:link))
      XCTAssertEqual(try Data(contentsOf:existing),Data([9,8]))
      XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath:link.path),existing.path)
      XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:root.path).contains { $0.hasSuffix(".partial") })
    }
  }
  func testPinnedProfileCannotBeUsedWithArbitrarySameGrammarArchive()throws {
    try temporary { root in
      let data=TorchFixture.zip(pickle:TorchFixture.pickle([("x",TorchFixture.tensor())]))
      let source=root.appendingPathComponent("source.ckpt");try data.write(to:source)
      XCTAssertThrowsError(try TorchTensorArchive.open(url:source,expectedSHA256:TorchFixture.hash(data),profile:.tinyDecoder))
      XCTAssertThrowsError(try TorchTensorArchive.open(url:source,expectedSHA256:TorchTensorImportProfile.tinyDecoder.sourceSHA256,profile:.tinyDecoder))
    }
  }
}
