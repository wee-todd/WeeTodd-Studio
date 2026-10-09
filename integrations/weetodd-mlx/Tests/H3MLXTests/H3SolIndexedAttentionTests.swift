import XCTest
import Foundation
import CryptoKit
import MLX
@testable import H3MLX

final class H3SolIndexedAttentionTests: XCTestCase {
  private func admitGPU() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_SOL_CONSUMER_TEST"] == "1" else {
      throw XCTSkip("Opt-in Sol consumer tests require an explicitly pinned product Metal library.")
    }
    guard let library = environment["WEETODD_H3_SOL_TEST_METALLIB"],
      let supplied = environment["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw H3CheckpointError.invalid("Missing explicit Sol test Metal library binding.")
    }
    let expected = supplied
    let url = URL(fileURLWithPath: library)
    guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true,
      FileManager.default.isReadableFile(atPath: url.path) else {
      throw H3CheckpointError.invalid("Sol test Metal library must be a readable regular file.")
    }
    let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    guard library.hasPrefix("/"), expected.count == 64, expected.allSatisfy({ $0.isHexDigit }), digest == expected else {
      throw H3CheckpointError.invalid("Sol test Metal library does not match the pinned product library.")
    }
    GPU.metallib = url
    guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("Sol requires the GPU device.") }
  }
  private func rounded(_ value: Float) -> Float {
    let bits = value.bitPattern
    let result = bits &+ 0x7fff &+ ((bits >> 16) & 1)
    return Float(bitPattern: result & 0xffff0000)
  }

  private func inputs(rows: Int, heads: Int) -> [[Float]] {
    (0..<3).map { kind in
      (0..<(heads * rows * 128)).map { index in
        let channel = index % 128, row = (index / 128) % rows, head = index / (rows * 128)
        return Float((channel * (kind + 3) + row * (kind + 5) + head * 7) % 29 - 14)
          / Float(kind == 2 ? 8 : 64)
      }
    }
  }

  private func means(_ values: [Float], rows: Int, heads: Int) -> [Float] {
    let blocks = (rows + 63) / 64
    return (0..<(heads * blocks * 128)).map { index in
      let channel = index % 128, block = (index / 128) % blocks, head = index / (blocks * 128)
      let start = block * 64, end = min(start + 64, rows)
      var sum = Double(0)
      for row in start..<end { sum += Double(values[(head * rows + row) * 128 + channel]) }
      return rounded(Float(sum / Double(end - start)))
    }
  }

  private func bits(rows: Int, heads: Int, selected: (Int, Int, Int) -> Bool) -> [UInt32] {
    let blocks = (rows + 63) / 64, words = (blocks + 31) / 32
    var result = [UInt32](repeating: 0, count: heads * blocks * words)
    for head in 0..<heads {
      for query in 0..<blocks {
        for key in 0..<blocks where selected(head, query, key) {
          result[(head * blocks + query) * words + key / 32] |= UInt32(1) << (key % 32)
        }
      }
    }
    return result
  }

  // Independent scalar oracle: one token per coarse block, with its true
  // multiplicity inside the same softmax as fine tokens. No Steel operations.
  private func oracle(_ values: [[Float]], rows: Int, heads: Int, routes: [UInt32],
    scale: Float) -> [Float] {
    let blocks = (rows + 63) / 64, words = (blocks + 31) / 32
    let pooledK = means(values[1], rows: rows, heads: heads)
    let pooledV = means(values[2], rows: rows, heads: heads)
    var output = [Float](repeating: 0, count: heads * rows * 128)
    for head in 0..<heads {
      for row in 0..<rows {
        var maximum = -Double.infinity, denominator = Double(0)
        var accumulation = [Double](repeating: 0, count: 128)
        for block in 0..<blocks {
          let fine = ((routes[(head * blocks + row / 64) * words + block / 32] >> (block % 32)) & 1) != 0
          let count = min(64, rows - block * 64)
          for offset in 0..<(fine ? count : 1) {
            var score = Double(0)
            let keyBase = fine ? (head * rows + block * 64 + offset) * 128 : (head * blocks + block) * 128
            for channel in 0..<128 {
              score += Double(values[0][(head * rows + row) * 128 + channel])
                * Double(fine ? values[1][keyBase + channel] : pooledK[keyBase + channel])
            }
            score = score * Double(scale) + (fine ? 0 : log(Double(count)))
            let next = max(maximum, score), correction = exp(maximum - next), weight = exp(score - next)
            denominator = denominator * correction + weight
            for channel in 0..<128 {
              let value = fine ? values[2][keyBase + channel] : pooledV[keyBase + channel]
              accumulation[channel] = accumulation[channel] * correction + weight * Double(value)
            }
            maximum = next
          }
        }
        for channel in 0..<128 {
          output[(head * rows + row) * 128 + channel] = rounded(Float(accumulation[channel] / denominator))
        }
      }
    }
    return output
  }

  private func run(_ values: [[Float]], rows: Int, heads: Int, routes: [UInt32],
    strided: Bool = false) throws -> MLXArray {
    let geometry = try H3SolGeometry(rows: rows, heads: heads, approximationRange: 0..<rows)
    let blocks = (rows + 63) / 64, words = (blocks + 31) / 32
    func array(_ values: [Float]) -> MLXArray {
      let dense = MLXArray(values, [1, heads, rows, 128]).asType(.bfloat16)
      if !strided { return dense }
      // Token-major backing produces genuine head/row-strided original views.
      let backing = contiguous(dense.transposed(0, 2, 1, 3))
      return backing.transposed(0, 2, 1, 3)
    }
    return try H3SolIndexedAttention.evaluate(query: array(values[0]), key: array(values[1]),
      value: array(values[2]), keyMeans: MLXArray(means(values[1], rows: rows, heads: heads), [1, heads, blocks, 128]).asType(.bfloat16),
      valueMeans: MLXArray(means(values[2], rows: rows, heads: heads), [1, heads, blocks, 128]).asType(.bfloat16),
      exactRouteBits: MLXArray(routes, [1, heads, blocks, words]), geometry: geometry)
  }

  func testAllFineRetainsDenseSteelBF16Bits() throws {
    try admitGPU()
    let rows = 128, heads = 2, values = inputs(rows: rows, heads: heads)
    let selected = bits(rows: rows, heads: heads) { _, _, _ in true }
    let actual = try run(values, rows: rows, heads: heads, routes: selected)
    let arrays = values.map { MLXArray($0, [1, heads, rows, 128]).asType(.bfloat16) }
    let expected = MLXFast.scaledDotProductAttention(queries: arrays[0], keys: arrays[1],
      values: arrays[2], scale: 1 / Float(128).squareRoot(), mask: nil)
    XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), expected.view(dtype: .uint16).asArray(UInt16.self))
  }

  func testMixedRaggedAndDifferentHeadRoutesAgainstDoubleOracle() throws {
    try admitGPU()
    for rows in [33, 65, 129] {
      let heads = 2, values = inputs(rows: rows, heads: heads)
      let selected = bits(rows: rows, heads: heads) { head, query, key in (head + query + key) % 2 == 0 }
      let actual = try run(values, rows: rows, heads: heads, routes: selected).asType(.float32).asArray(Float.self)
      let expected = oracle(values, rows: rows, heads: heads, routes: selected, scale: 1 / Float(128).squareRoot())
      XCTAssertEqual(actual.count, expected.count)
      XCTAssertTrue(actual.allSatisfy(\.isFinite))
      XCTAssertLessThanOrEqual(zip(actual, expected).map { pair in abs(Double(pair.0) - Double(pair.1)) }.max()!, 0.0078125)
    }
  }

  func testPreparedRoutesProtectBoundaryCrossingPrefixGroups() throws {
    try admitGPU()
    let rows = 257, heads = 2
    let values = inputs(rows: rows, heads: heads)
    let geometry = try H3SolGeometry(rows: rows, heads: heads,
      approximationRange: 65..<rows, tau: 16)
    let arrays = values.map { MLXArray($0, [1, heads, rows, 128]).asType(.bfloat16) }
    let prepared = try H3SolRouting.prepare(query: arrays[0], key: arrays[1], value: arrays[2], geometry: geometry)
    let routes = prepared.exactRouteBits.asArray(UInt32.self)
    for head in 0..<heads {
      for query in 0..<2 { XCTAssertEqual(routes[head * 5 + query], UInt32(31)) }
    }
    let actual = try H3SolIndexedAttention.evaluate(query: arrays[0], key: arrays[1], value: arrays[2], prepared: prepared)
      .asType(.float32).asArray(Float.self)
    let expected = oracle(values, rows: rows, heads: heads, routes: routes, scale: geometry.scale)
    XCTAssertTrue(actual.allSatisfy(\.isFinite))
    XCTAssertLessThanOrEqual(zip(actual, expected).map { pair in abs(Double(pair.0) - Double(pair.1)) }.max()!, 0.0078125)
  }

  func testAllCoarseUsesTailMultiplicityRatherThanUniformBlockWeight() throws {
    try admitGPU()
    let rows = 129
    let zero = [Float](repeating: 0, count: rows * 128)
    let value = (0..<(rows * 128)).map { index -> Float in
      let row = index / 128
      return row < 64 ? 2 : (row < 128 ? -1 : 10)
    }
    let actual = try run([zero, zero, value], rows: rows, heads: 1,
      routes: bits(rows: rows, heads: 1) { _, _, _ in false }).asType(.float32).asArray(Float.self)
    let expected = rounded(Float(74.0 / 129.0))
    XCTAssertTrue(actual.allSatisfy { $0.isFinite && abs($0 - expected) <= 0.00390625 })
  }

  func testFullyMaskedCoarseBlocksAndPositiveStridesRemainSafe() throws {
    try admitGPU()
    let rows = 1025, heads = 1
    let zero = [Float](repeating: 0, count: rows * 128)
    let value = (0..<(rows * 128)).map { Float(($0 / 128) % 7 - 3) / 4 }
    let routes = bits(rows: rows, heads: heads) { _, _, key in key < 16 }
    let actual = try run([zero, zero, value], rows: rows, heads: heads, routes: routes).asType(.float32).asArray(Float.self)
    let mean = rounded(Float(value.enumerated().filter { $0.offset % 128 == 0 }.reduce(Double(0)) { $0 + Double($1.element) } / Double(rows)))
    XCTAssertTrue(actual.allSatisfy { $0.isFinite && abs($0 - mean) <= 0.000244140625 })
    let small = inputs(rows: 65, heads: 3), smallRoutes = bits(rows: 65, heads: 3) { h, q, k in (h + q + k) % 2 == 0 }
    let dense = try run(small, rows: 65, heads: 3, routes: smallRoutes)
    let strided = try run(small, rows: 65, heads: 3, routes: smallRoutes, strided: true)
    XCTAssertEqual(dense.view(dtype: .uint16).asArray(UInt16.self), strided.view(dtype: .uint16).asArray(UInt16.self))
  }

  func testConstantAndDeviceRouteAddressSpacesProduceIdenticalHeadWords() throws {
    try admitGPU()
    // Pinned MLX uses constant pointers for fewer than 8 elements, device
    // pointers at 8 or more. N=65 has two route words per head.
    let rows = 65, base = inputs(rows: rows, heads: 1)
    let smallRoutes = bits(rows: rows, heads: 1) { _, query, key in (query + key) % 2 == 0 }
    XCTAssertEqual(smallRoutes.count, 2)
    let small = try run(base, rows: rows, heads: 1, routes: smallRoutes)
    let repeated = base.map { values in Array(repeating: values, count: 4).flatMap { $0 } }
    let largeRoutes = bits(rows: rows, heads: 4) { _, query, key in (query + key) % 2 == 0 }
    XCTAssertEqual(largeRoutes.count, 8)
    let large = try run(repeated, rows: rows, heads: 4, routes: largeRoutes)
    let expected = small.view(dtype: .uint16).asArray(UInt16.self)
    for head in 0..<4 {
      let actual = large[0..<1, head..<(head + 1), 0..<rows, 0..<128]
      XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), expected)
    }
  }

  func testCancellationRejectsBeforeConsumerSubmission() async throws {
    try admitGPU()
    let cancelled = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      let geometry = try H3SolGeometry(rows: 1, heads: 1, approximationRange: 0..<1)
      let q = MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16)
      do {
        _ = try H3SolIndexedAttention.evaluate(query: q, key: q, value: q,
          keyMeans: q, valueMeans: q, exactRouteBits: MLXArray([UInt32(1)], [1, 1, 1, 1]), geometry: geometry)
        return false
      } catch is CancellationError { return true }
    }
    let rejected = try await cancelled.value
    XCTAssertTrue(rejected)
  }

  func testInvalidDtypeFeatureStrideAndRouteShapeRejectBeforeConsumer() throws {
    try admitGPU()
    let geometry = try H3SolGeometry(rows: 65, heads: 1, approximationRange: 0..<65)
    let q = MLXArray.zeros([1, 1, 65, 128], dtype: .bfloat16)
    let means = MLXArray.zeros([1, 1, 2, 128], dtype: .bfloat16)
    let routes = MLXArray([UInt32(3), UInt32(3)], [1, 1, 2, 1])
    XCTAssertThrowsError(try H3SolIndexedAttention.evaluate(query: q.asType(.float32), key: q, value: q,
      keyMeans: means, valueMeans: means, exactRouteBits: routes, geometry: geometry))
    let stepped = MLXArray.zeros([1, 1, 65, 256], dtype: .bfloat16)[.ellipsis, .stride(by: 2)]
    XCTAssertThrowsError(try H3SolIndexedAttention.evaluate(query: stepped, key: q, value: q,
      keyMeans: means, valueMeans: means, exactRouteBits: routes, geometry: geometry))
    XCTAssertThrowsError(try H3SolIndexedAttention.evaluate(query: q, key: q, value: q,
      keyMeans: means, valueMeans: means, exactRouteBits: routes[0..<1, 0..<1, 0..<1, 0..<1], geometry: geometry))
    XCTAssertThrowsError(try H3SolIndexedAttention.evaluate(query: q[.ellipsis, .stride(by: -1), 0..<128], key: q, value: q,
      keyMeans: means, valueMeans: means, exactRouteBits: routes, geometry: geometry))
  }
}
