import Foundation
import XCTest
@testable import TensorIO

final class SafeTensorInt8WriterTests:XCTestCase {
  func testSignedInt8PayloadPreservesQuantizedBytesAndDescriptorOnRoundTrip()throws {
    let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:folder) }
    let url=folder.appendingPathComponent("quantized.safetensors")
    let values:[Int8]=[-128,-1,0,127]
    let writer=try SafeTensorStreamWriter(url:url,tensors:["weight":try .init(dtype:"I8",shape:[2,2])])
    try values.withUnsafeBytes { try writer.append(tensor:"weight",bytes:$0) }
    _ = try writer.finish()
    let reopened=try SafeTensorFile(url:url)
    XCTAssertEqual(reopened.tensors["weight"]?.dtype,"I8")
    XCTAssertEqual(reopened.tensors["weight"]?.shape,[2,2])
    XCTAssertEqual(reopened.tensors["weight"]?.byteCount,4)
    XCTAssertEqual(try reopened.withTensorBytes(named:"weight") { Array($0) },[128,255,0,127])
  }
}
