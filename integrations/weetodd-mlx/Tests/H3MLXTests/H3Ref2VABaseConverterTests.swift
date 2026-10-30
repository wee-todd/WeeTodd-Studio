import Foundation
import CryptoKit
import TensorIO
import XCTest
@testable import H3MLX

final class H3Ref2VABaseConverterTests:XCTestCase {
  func testBF16RoundToNearestEvenAndSignedSpecialValues() {
    let inputs:[UInt32]=[0x3f800000,0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00000000,0x80000000,0x7f800000,0xff800000]
    XCTAssertEqual(inputs.map(H3Ref2VABaseConverter.bfloat16Bits),[0x3f80,0x3f80,0x3f82,0xbf80,0xbf82,0,0x8000,0x7f80,0xff80])
    XCTAssertEqual(H3Ref2VABaseConverter.bfloat16Bits(0x7f800001)&0x7fc0,0x7fc0)
  }
  func testExactlySixNamedMatricesAreCastAndAllNamesGainLegacyNamespace() throws {
    let result=try H3Ref2VABaseConverter.preparedTensor(name:"video_patch_proj.weight",tensor:.init(dtype:"F32",shape:[5376,96]))
    XCTAssertEqual(result.name,"model.diffusion_model.video_patch_proj.weight")
    XCTAssertEqual(result.info.dtype,"BF16")
    let untouched=try H3Ref2VABaseConverter.preparedTensor(name:"blocks.0.attn.qkv_proj.weight",tensor:.init(dtype:"I8",shape:[21504,5376]))
    XCTAssertEqual(untouched.info.dtype,"I8")
    XCTAssertEqual(H3Ref2VABaseConverter.castShapes.count,6)
    XCTAssertThrowsError(try H3Ref2VABaseConverter.preparedTensor(name:"model.diffusion_model.video_patch_proj.weight",tensor:.init(dtype:"F32",shape:[5376,96])))
    XCTAssertThrowsError(try H3Ref2VABaseConverter.preparedTensor(name:"video_patch_proj.weight",tensor:.init(dtype:"BF16",shape:[5376,96])))
    XCTAssertThrowsError(try H3Ref2VABaseConverter.preparedTensor(name:"video_patch_proj.weight",tensor:.init(dtype:"F32",shape:[96,5376])))
  }
  func testRejectsUnpinnedSourceBeforeCreatingOutput() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let source=root.appendingPathComponent("source.safetensors"),output=root.appendingPathComponent("prepared.safetensors")
    try Data([0,1,2,3]).write(to:source)
    XCTAssertThrowsError(try H3Ref2VABaseConverter.convert(source:source,destination:output,modelRoot:root))
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
  }
  func testInstalledConversionMatchesEveryQualifiedTensorPayload() throws {
    guard let converted=ProcessInfo.processInfo.environment["WEETODD_H3_REF2VA_CONVERTED"],
      let baseline=ProcessInfo.processInfo.environment["WEETODD_H3_REF2VA_BASELINE"] else {
      throw XCTSkip("Explicit installed converter parity requires both local model files.")
    }
    let urls=[converted,baseline].map{URL(fileURLWithPath:$0)},files=try urls.map{try SafeTensorFile(url:$0)}
    XCTAssertEqual(files[0].metadata["format"],H3Ref2VABaseConverter.version)
    XCTAssertEqual(files[0].metadata["source_sha256"],H3Ref2VABaseConverter.sourceSHA256)
    XCTAssertEqual(Set(files[0].tensors.keys),Set(files[1].tensors.keys))
    guard Set(files[0].tensors.keys) == Set(files[1].tensors.keys) else { return }
    for name in files[0].tensors.keys.sorted() {
      let first=files[0].tensors[name]!,second=files[1].tensors[name]!
      XCTAssertEqual(first.dtype,second.dtype,name);XCTAssertEqual(first.shape,second.shape,name)
      XCTAssertEqual(first.byteCount,second.byteCount,name)
      var digests=[String]()
      for file in files {
        var hash=SHA256(),offset:UInt64=0
        while offset < first.byteCount {
          let end=Swift.min(offset+4*1024*1024,first.byteCount)
          try autoreleasepool {
            try file.withTensorBytes(named:name,range:offset..<end,access:.buffered){hash.update(bufferPointer:$0)}
          }
          offset=end
        }
        digests.append(hash.finalize().map{String(format:"%02x",$0)}.joined())
      }
      XCTAssertEqual(digests[0],digests[1],name)
    }
    for (index,file) in files.enumerated(){try file.checkUnchanged(at:urls[index])}
  }
}
