import Foundation
import XCTest
import InferenceTestSupport
@testable import TensorIO

final class FloatDecodingTests: XCTestCase {
  @MainActor
  func testCanceledTaskDoesNotAllocateDecodedWeights() async throws {
    let rejected = try await Task {
      try withTensorFile(tensors: [("plain", [1], "F32"), ("dense.weight", [1, 16], "U32"),
        ("dense.scales", [1, 1], "F32"), ("dense.biases", [1, 1], "F32")]) { url in
        let file = try SafeTensorFile(url: url)
        let q8 = try MLXAffineQ8(file: file, weight: "dense.weight", groupSize: 64)
        withUnsafeCurrentTask { $0?.cancel() }
        var rejected = 0
        do { _ = try file.readFloat32(named: "plain") } catch is CancellationError { rejected += 1 }
        do { _ = try q8.readRows(0..<1) } catch is CancellationError { rejected += 1 }
        return rejected
      }
    }.value
    XCTAssertEqual(rejected, 2)
  }

  func testFloatFormatsAndSlicesPreserveSignedValues() throws {
    try withTensorFile(tensors: [("bf", [3], "BF16"), ("half", [3], "F16"), ("wide", [3], "F32")], payloads: [
      "bf": Data([0, 0x3f, 0, 0xc0, 0x80, 0x3f]),
      "half": Data([0, 0x38, 0, 0xc0, 0, 0x3c]),
      "wide": Data([0, 0, 0, 0x3f, 0, 0, 0, 0xc0, 0, 0, 0x80, 0x3f])]) { url in
      let file = try SafeTensorFile(url: url)
      for name in ["bf", "half", "wide"] {
        XCTAssertEqual(try file.readFloat32(named: name), [0.5, -2, 1])
        XCTAssertEqual(try file.readFloat32(named: name, elements: 1..<3), [-2, 1])
        XCTAssertThrowsError(try file.readFloat32(named: name, maximumBytes: 8))
        XCTAssertThrowsError(try file.readFloat32(named: name, elements: 2..<4))
      }
    }
  }

  func testDenseSlicesCrossFourMiBMappingBoundaryWithoutChangingBits() throws {
    let count = 2 * 1024 * 1024 + 7
    let bits: [UInt16] = [0x3f00,0xc000,0x3f80]
    let values: [Float] = [0.5,-2,1]
    let payload = (0..<count).map { bits[$0 % 3].littleEndian }.withUnsafeBytes { Data($0) }
    try withTensorFile(tensors:[("dense",[count],"BF16")],payloads:["dense":payload]) { url in
      let file = try SafeTensorFile(url:url)
      let output = try file.readFloat32(named:"dense",elements:1..<UInt64(count))
      XCTAssertEqual(output.count,count-1)
      for index in [0,65535,65536,2097151,2097152,count-2] {
        XCTAssertEqual(output[index],values[(index+1)%3],"Decoded mapping boundary \(index)")
      }
    }
  }

  func testAffineQ8UnpacksLittleEndianLanesAndRespectsRowAndGroupBoundaries() throws {
    let packed = Data((0..<256).map { UInt8($0 % 128) })
    let scales: [Float] = [0.5, 2, -1, 0.25], biases: [Float] = [-1, 10, 3, -4]
    try withTensorFile(tensors: [("dense.weight", [2, 32], "U32"),
      ("dense.scales", [2, 2], "F32"), ("dense.biases", [2, 2], "F32")], payloads: [
        "dense.weight": packed, "dense.scales": scales.withUnsafeBytes { Data($0) },
        "dense.biases": biases.withUnsafeBytes { Data($0) }]) { url in
      let matrix = try MLXAffineQ8(file: SafeTensorFile(url: url), weight: "dense.weight", groupSize: 64)
      XCTAssertEqual(matrix.shape, [2, 128])
      let rows = try matrix.readRows(1..<2)
      XCTAssertEqual(rows.count, 128)
      XCTAssertEqual(Array(rows.prefix(4)), [3, 2, 1, 0])
      XCTAssertEqual(rows[63], -60)
      XCTAssertEqual(rows[64], 12)
      XCTAssertEqual(rows[127], 27.75)
      let first = try matrix.readRows(0..<1)
      XCTAssertEqual(first[0], -1)
      XCTAssertEqual(first[63], 30.5)
      XCTAssertEqual(first[64], 138)
      XCTAssertThrowsError(try matrix.readRows(0..<2, maximumBytes: 100))
      XCTAssertThrowsError(try matrix.readRows(1..<3))
    }
  }

  func testRejectsQuantizedCompanionMismatchBeforeDecoding() throws {
    for companions: [(String, [Int], String)] in [
      [("dense.scales", [1, 1], "F32")],
      [("dense.scales", [1, 2], "F32"), ("dense.biases", [1, 2], "F32")],
      [("dense.scales", [1, 1], "F32"), ("dense.biases", [1, 1], "BF16")],
    ] {
      try withTensorFile(tensors: [("dense.weight", [1, 16], "U32")] + companions) { url in
        XCTAssertThrowsError(try MLXAffineQ8(file: SafeTensorFile(url: url), weight: "dense.weight", groupSize: 64))
      }
    }
  }

  func testQ8LargeReadPreservesGroupsAcrossMappingBoundariesAndOffsetRows() throws {
    // Cross both the former 256 KiB and new 4 MiB mapping windows, starting at row 1.
    let rows = 65539
    let packed = Data((0..<(rows * 64)).map { UInt8($0 % 256) })
    let scales = (0..<rows).map { Float($0 % 13 + 1) }
    let biases = (0..<rows).map { -Float($0 % 17) }
    try withTensorFile(tensors: [("dense.weight", [rows, 16], "U32"),
      ("dense.scales", [rows, 1], "F32"), ("dense.biases", [rows, 1], "F32")], payloads: [
        "dense.weight": packed, "dense.scales": scales.withUnsafeBytes { Data($0) },
        "dense.biases": biases.withUnsafeBytes { Data($0) }]) { url in
      let matrix = try MLXAffineQ8(file: SafeTensorFile(url: url), weight: "dense.weight", groupSize: 64)
      let actual = try matrix.readRows(1..<rows)
      for local in [0, 63, 64, 262143, 262144, 4194303, 4194304, actual.count - 1] {
        let absolute = local + 64, group = absolute / 64
        let expected = Float(absolute % 256) * Float(group % 13 + 1) - Float(group % 17)
        XCTAssertEqual(actual[local], expected, "Mapping boundary at \(local)")
      }
    }
  }
}

extension FloatDecodingTests {
  func testBufferedWeightReadsMatchMappedSlicesAcrossWindowsAndRejectMutation() throws {
    let count = 2*1024*1024+7
    let payload = (0..<count).map { UInt16($0 % 2 == 0 ? 0x3f80 : 0xc000).littleEndian }.withUnsafeBytes { Data($0) }
    try withTensorFile(tensors: [("dense",[count],"BF16")],payloads: ["dense":payload]) { url in
      let file = try SafeTensorFile(url: url)
      XCTAssertEqual(try file.readFloat32(named: "dense",elements: 1..<UInt64(count),access: .buffered),
        try file.readFloat32(named: "dense",elements: 1..<UInt64(count)))
      XCTAssertThrowsError(try file.withTensorBytes(named: "dense",range: 0..<UInt64(payload.count),access: .buffered) { _ in })
      let handle = try FileHandle(forWritingTo: url); try handle.truncate(atOffset: 16); try handle.close()
      XCTAssertThrowsError(try file.readFloat32(named: "dense",access: .buffered))
    }
  }
}
