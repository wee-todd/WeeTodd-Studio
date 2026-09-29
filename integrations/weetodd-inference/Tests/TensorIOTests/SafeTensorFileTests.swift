import Foundation
import XCTest
@testable import TensorIO

final class SafeTensorFileTests: XCTestCase {
  func withFile(_ header: String, payload: [UInt8], _ body: (URL) throws -> Void) throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    var length = UInt64(header.utf8.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }
    data.append(contentsOf: header.utf8)
    data.append(contentsOf: payload)
    try data.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    try body(url)
  }

  func testReadsMappedTensorWithoutCopyingWholeCheckpoint() throws {
    try withFile(#"{"__metadata__":{"model_version":"2.5.0"},"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"F32","shape":[1],"data_offsets":[3,7]}}"#,
                 payload: [5, 6, 7, 0, 0, 128, 63]) { url in
      let file = try SafeTensorFile(url: url)
      XCTAssertEqual(file.metadata["model_version"], "2.5.0")
      XCTAssertEqual(file.tensors["b"]?.shape, [1])
      let bytes = try file.withTensorBytes(named: "b") { Array($0) }
      XCTAssertEqual(bytes, [0, 0, 128, 63])
      XCTAssertThrowsError(try file.withTensorBytes(named: "missing") { $0.count })
      enum Abort: Error { case stop }
      XCTAssertThrowsError(try file.withTensorBytes(named: "a") { _ in throw Abort.stop })
      XCTAssertEqual(try file.withTensorBytes(named: "a") { Array($0) }, [5, 6, 7])
    }
  }

  func testRejectsTruncatedPayloadAndWrongByteCount() throws {
    for header in [
      #"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
      #"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
    ] {
      try withFile(header, payload: [0, 0, 0, 0]) { url in
        XCTAssertThrowsError(try SafeTensorFile(url: url))
      }
    }
  }

  func testRejectsOverlapsHolesAndUnindexedTrailingBytes() throws {
    for header in [
      #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
      #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
      #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
    ] {
      try withFile(header, payload: [1, 2, 3]) { url in
        XCTAssertThrowsError(try SafeTensorFile(url: url))
      }
    }
  }

  func testRejectsOverflowUnknownDtypeAndInvalidDimensions() throws {
    for header in [
      #"{"a":{"dtype":"F64","shape":[18446744073709551615,8],"data_offsets":[0,0]}}"#,
      #"{"a":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#,
      #"{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}"#,
      #"{"a":{"dtype":"U8","shape":[true],"data_offsets":[0,1]}}"#,
      #"{"a":{"dtype":"U8","shape":[1.25],"data_offsets":[0,1]}}"#,
      #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#,
    ] {
      try withFile(header, payload: [0]) { url in
        XCTAssertThrowsError(try SafeTensorFile(url: url))
      }
    }
  }

  func testSupportsScalarsAndEmptyTensors() throws {
    try withFile(#"{"empty":{"dtype":"F32","shape":[0,9],"data_offsets":[0,0]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
                 payload: [0, 0, 128, 63]) { url in
      let file = try SafeTensorFile(url: url)
      XCTAssertEqual(try file.withTensorBytes(named: "empty") { $0.count }, 0)
      XCTAssertEqual(file.tensors["scalar"]?.byteCount, 4)
    }
  }

  func testBoundsHeaderBeforeAllocation() throws {
    try withFile(#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
                 payload: [0]) { url in
      XCTAssertThrowsError(try SafeTensorFile(url: url, maximumHeaderBytes: 16))
    }
  }
}
