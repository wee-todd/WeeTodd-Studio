import Foundation
import CryptoKit
import XCTest
@testable import TensorIO

final class SafeTensorStreamWriterTests:XCTestCase {
  private func temporary(_ body:(URL) throws -> Void) throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    try body(root.appendingPathComponent("page.safetensors"))
  }
  func testChunkedOutputHasContiguousHeaderAndIndependentFileHash() throws {
    try temporary { url in
      let writer=try SafeTensorStreamWriter(url:url,tensors:["a":try .init(dtype:"U8",shape:[3]),"b":try .init(dtype:"F32",shape:[1])],metadata:["format":"mlx"])
      try Data([1,2]).withUnsafeBytes { try writer.append(tensor:"a",bytes:$0) }
      try Data([3]).withUnsafeBytes { try writer.append(tensor:"a",bytes:$0) }
      var one=Float(1)
      try withUnsafeBytes(of:&one) { try writer.append(tensor:"b",bytes:$0) }
      let hash=try writer.finish(),file=try SafeTensorFile(url:url)
      XCTAssertEqual(hash,SHA256.hash(data:try Data(contentsOf:url)).map { String(format:"%02x",$0) }.joined())
      XCTAssertEqual(file.metadata,["format":"mlx"])
      XCTAssertEqual(file.tensors["b"]?.byteOffset,3)
      XCTAssertEqual(try file.withTensorBytes(named:"a") { Array($0) },[1,2,3])
      XCTAssertThrowsError(try writer.finish())
    }
  }
  func testExistingFileAndSymlinkAreNeverRemovedOrOverwritten() throws {
    try temporary { url in
      let original=Data([9,8,7]);try original.write(to:url)
      XCTAssertThrowsError(try SafeTensorStreamWriter(url:url,tensors:["x":try .init(dtype:"U8",shape:[1])]))
      XCTAssertEqual(try Data(contentsOf:url),original)
      let link=url.deletingLastPathComponent().appendingPathComponent("link.safetensors")
      try FileManager.default.createSymbolicLink(at:link,withDestinationURL:url)
      XCTAssertThrowsError(try SafeTensorStreamWriter(url:link,tensors:["x":try .init(dtype:"U8",shape:[1])]))
      XCTAssertEqual(try Data(contentsOf:link),original)
    }
  }
  func testIncompleteWriterAndBadTensorOrderLeaveNoUsablePage() throws {
    try temporary { url in
      try autoreleasepool {
        let writer=try SafeTensorStreamWriter(url:url,tensors:["a":try .init(dtype:"U8",shape:[2]),"b":try .init(dtype:"U8",shape:[1])])
        XCTAssertThrowsError(try Data([3]).withUnsafeBytes { try writer.append(tensor:"b",bytes:$0) })
        try Data([1]).withUnsafeBytes { try writer.append(tensor:"a",bytes:$0) }
        XCTAssertThrowsError(try writer.finish())
      }
      XCTAssertFalse(FileManager.default.fileExists(atPath:url.path))
    }
  }
  func testOverflowAndOversizedWindowRejectBeforeWrite() throws {
    XCTAssertThrowsError(try SafeTensorStreamWriter.Tensor(dtype:"BF16",shape:[UInt64.max,2]))
    XCTAssertThrowsError(try SafeTensorStreamWriter.Tensor(dtype:"unsupported",shape:[1]))
    XCTAssertThrowsError(try SafeTensorStreamWriter.validateHeader(
      tensors:["a":try .init(dtype:"U8",shape:[1])],maximumHeaderBytes:16))
    try temporary { url in
      // Header-only admission has no output side effects, including failure.
      XCTAssertThrowsError(try SafeTensorStreamWriter.validateHeader(
        tensors:[String(repeating:"x",count:1024*1024):try .init(dtype:"U8",shape:[1])]))
      XCTAssertFalse(FileManager.default.fileExists(atPath:url.path))
      try autoreleasepool {
        let writer=try SafeTensorStreamWriter(url:url,tensors:["a":try .init(dtype:"U8",shape:[5*1024*1024])])
        XCTAssertThrowsError(try Data(repeating:0,count:4*1024*1024+1).withUnsafeBytes { try writer.append(tensor:"a",bytes:$0) })
      }
      XCTAssertFalse(FileManager.default.fileExists(atPath:url.path))
    }
  }
}
