import Foundation
import TensorIO
import XCTest
@testable import H3MLX

/// Raw-byte/identity admission only; no MLX arrays or model evaluation.
final class H3TensorPayloadTests: XCTestCase {
  private enum Failure: Error { case injected }

  private func fixture(bytes: Data) throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".safetensors")
    let header = try JSONSerialization.data(withJSONObject: ["raw": [
      "dtype": "U8", "shape": [bytes.count], "data_offsets": [0, bytes.count]]])
    var size = UInt64(header.count).littleEndian
    var data = withUnsafeBytes(of: &size) { Data($0) }
    data.append(header); data.append(bytes)
    try data.write(to: url)
    return url
  }

  private func crossingBytes() -> Data {
    var bytes = Data(repeating: 0xa5, count: H3TensorPayload.bufferedSpanBytes + 17)
    for (index, value) in [(0, 1), (4095, 7), (H3TensorPayload.bufferedSpanBytes - 1, 91),
      (H3TensorPayload.bufferedSpanBytes, 255), (bytes.count - 1, 49)] {
      bytes[index] = UInt8(value)
    }
    return bytes
  }

  func testOwnedSpansPreserveExactBytesSlicesAndExistingAPICap() throws {
    let bytes = crossingBytes(), url = try fixture(bytes: bytes)
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    XCTAssertEqual(try H3TensorPayload.readBuffered(file: file, name: "raw",
      range: 0..<UInt64(bytes.count)), bytes)
    XCTAssertEqual(try H3TensorPayload.withTensorBytes(file: file, name: "raw") { Data($0) }, bytes)
    XCTAssertEqual(try H3TensorPayload.withTensorBytes(file: file, name: "raw",
      range: UInt64(bytes.count - 18)..<UInt64(bytes.count)) { Data($0) }, bytes.suffix(18))
    XCTAssertEqual(try H3TensorPayload.readBuffered(file: file, name: "raw", range: 1..<1), Data())
    XCTAssertThrowsError(try file.withTensorBytes(named: "raw", range: 0..<UInt64(bytes.count),
      access: .buffered) { _ in })
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "missing", range: 0..<1))
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "raw", range: 0..<UInt64(bytes.count + 1)))
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "raw",
      range: 0..<(H3TensorPayload.maximumOwnedBytes + 1)))
  }

  func testLargeOptionalPayloadFallsBackToMappedWithoutOwnedCopy() throws {
    let count = H3TensorPayload.maximumOwnedBytes + 2
    let header = try JSONSerialization.data(withJSONObject: ["factor": [
      "dtype": "BF16", "shape": [count / 2], "data_offsets": [0, count]]])
    var size = UInt64(header.count).littleEndian
    var prefix = withUnsafeBytes(of: &size) { Data($0) }; prefix.append(header)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: url) }
    try prefix.write(to: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd(); try handle.write(contentsOf: Data([0x12]))
    try handle.seek(toOffset: UInt64(prefix.count) + count - 1)
    try handle.write(contentsOf: Data([0x34])); try handle.close()
    let file = try SafeTensorFile(url: url)
    let ends = try H3TensorPayload.withTensorBytes(file: file, name: "factor") {
      ($0.count, $0[0], $0[$0.count - 1])
    }
    XCTAssertEqual(ends.0, Int(count)); XCTAssertEqual(ends.1, 0x12); XCTAssertEqual(ends.2, 0x34)
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "factor", range: 0..<count))
    try file.checkUnchanged(at: url)
  }

  func testSmallNormAndFactorLimitsKeepLargerStandardFactorsSupported() throws {
    let normLimit: UInt64 = 4 * 1024 * 1024, factorLimit: UInt64 = 16 * 1024 * 1024
    XCTAssertEqual(H3TensorPayload.access(byteCount: 64, maximumBufferedBytes: normLimit), .buffered)
    XCTAssertEqual(H3TensorPayload.access(byteCount: normLimit, maximumBufferedBytes: normLimit), .buffered)
    XCTAssertEqual(H3TensorPayload.access(byteCount: normLimit + 1, maximumBufferedBytes: normLimit), .mapped)
    XCTAssertEqual(H3TensorPayload.access(byteCount: factorLimit, maximumBufferedBytes: factorLimit), .buffered)
    XCTAssertEqual(H3TensorPayload.access(byteCount: factorLimit + 2, maximumBufferedBytes: factorLimit), .mapped)
    XCTAssertEqual(H3TensorPayload.access(byteCount: H3TensorPayload.maximumOwnedBytes + 1,
      maximumBufferedBytes: .max), .mapped)
    let normBits: [UInt16] = [0x3f80, 0xc000, 0x3f00]
    let norm = normBits.map(\.littleEndian).withUnsafeBytes { Data($0) }
    let normURL = try fixture(bytes: norm)
    defer { try? FileManager.default.removeItem(at: normURL) }
    let normFile = try SafeTensorFile(url: normURL)
    XCTAssertEqual(try H3TensorPayload.withTensorBytes(file: normFile, name: "raw",
      maximumBufferedBytes: normLimit) { Data($0) }, norm)
    let biasSlice = try H3TensorPayload.withTensorBytes(file: normFile, name: "raw",
      range: 2..<6, maximumBufferedBytes: normLimit) { bytes in
      (0..<2).map { bytes.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self).littleEndian }
    }
    XCTAssertEqual(biasSlice, Array(normBits[1..<3]))

    // A sparse synthetic factor avoids allocating/copying a real LoRA. Its size
    // exceeds the qualified 16 MiB factor cap but remains a supported payload.
    let count = factorLimit + 2
    let header = try JSONSerialization.data(withJSONObject: ["factor": [
      "dtype": "BF16", "shape": [count / 2], "data_offsets": [0, count]]])
    var size = UInt64(header.count).littleEndian
    var prefix = withUnsafeBytes(of: &size) { Data($0) }; prefix.append(header)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: url) }
    try prefix.write(to: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd(); try handle.write(contentsOf: Data([0x3f]))
    try handle.seek(toOffset: UInt64(prefix.count) + count - 1)
    try handle.write(contentsOf: Data([0xc0])); try handle.close()
    let file = try SafeTensorFile(url: url)
    let ends = try H3TensorPayload.withTensorBytes(file: file, name: "factor",
      maximumBufferedBytes: factorLimit) { ($0.count, $0[0], $0[$0.count - 1]) }
    XCTAssertEqual(ends.0, Int(count)); XCTAssertEqual(ends.1, 0x3f); XCTAssertEqual(ends.2, 0xc0)
    try file.checkUnchanged(at: url)
  }

  func testFailureAndMutationBetweenSpansRejectPartialOutput() throws {
    let bytes = crossingBytes(), url = try fixture(bytes: bytes)
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    var observed = 0
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "raw",
      range: 0..<UInt64(bytes.count), afterSpan: { span in
        observed = span; throw Failure.injected
      })) { error in XCTAssertTrue(error is Failure) }
    XCTAssertEqual(observed, 1)
    XCTAssertThrowsError(try H3TensorPayload.readBuffered(file: file, name: "raw",
      range: 0..<UInt64(bytes.count), afterSpan: { span in
        if span == 1 {
          let handle = try FileHandle(forWritingTo: url)
          try handle.seekToEnd(); try handle.write(contentsOf: Data([1])); try handle.close()
        }
      }))
    XCTAssertThrowsError(try H3TensorPayload.withTensorBytes(file: file, name: "raw") { _ in })
  }

  @MainActor
  func testTaskCancellationStopsBeforeNextBufferedSpan() async throws {
    let bytes = crossingBytes(), url = try fixture(bytes: bytes)
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    let observed = await Task { () -> Int in
      var spans = 0
      do {
        _ = try H3TensorPayload.readBuffered(file: file, name: "raw",
          range: 0..<UInt64(bytes.count), afterSpan: { span in
            spans = span; withUnsafeCurrentTask { $0?.cancel() }
          })
      } catch is CancellationError { return spans } catch { return -1 }
      return -2
    }.value
    XCTAssertEqual(observed, 1)
  }
}
