import Foundation
import XCTest
import InferenceTestSupport
@testable import TensorIO

final class Q8SIMDTests: XCTestCase {
  func testSIMDIsBitIdenticalToFusedScalarAcrossGroupsRowsAndWindows() throws {
    let rows = 65539, columns = 64
    let scales = (0..<rows).map { Float($0 % 29 - 14) * 0.00312347 }
    let biases = (0..<rows).map { Float($0 % 37 - 18) * 0.000032417 }
    let packed = Data((0..<(rows * columns)).map { UInt8($0 % 256) })
    try withTensorFile(tensors: [("w.weight", [rows, 16], "U32"),
      ("w.scales", [rows, 1], "F32"), ("w.biases", [rows, 1], "F32")], payloads: [
      "w.weight": packed, "w.scales": scales.withUnsafeBytes { Data($0) },
      "w.biases": biases.withUnsafeBytes { Data($0) }]) { url in
      let q = try MLXAffineQ8(file: SafeTensorFile(url: url), weight: "w.weight", groupSize: 64)
      let actual = try q.readRows(1..<rows, decoding: .simd)
      let expected = try q.readRows(1..<rows, decoding: .scalar)
      XCTAssertEqual(actual.map(\.bitPattern), expected.map(\.bitPattern))
      XCTAssertEqual(try q.readRows(0..<0, decoding: .simd), [])
    }
  }
}
