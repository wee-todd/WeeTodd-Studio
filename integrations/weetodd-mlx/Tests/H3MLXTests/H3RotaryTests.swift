import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3RotaryTests: XCTestCase {
  private func expression(_ value: MLXArray, cosine: MLXArray, sine: MLXArray,
    rotaryWidth: Int = 96) -> MLXArray {
    let first = value[.ellipsis, 0..<(rotaryWidth / 2)]
    let second = value[.ellipsis, (rotaryWidth / 2)..<rotaryWidth]
    let rotated = concatenated([-second, first], axis: -1)
    let leading = value[.ellipsis, 0..<rotaryWidth] * cosine + rotated * sine
    return concatenated([leading, value[.ellipsis, rotaryWidth..<value.shape.last!]], axis: -1)
  }

  private func assertBF16Parity(_ value: MLXArray, cosine: MLXArray, sine: MLXArray,
    file: StaticString = #filePath, line: UInt = #line) {
    let expected = expression(value, cosine: cosine, sine: sine)
    let actual = H3Rotary.apply(value, cosine: cosine, sine: sine)
    XCTAssertEqual(actual.shape, value.shape, file: file, line: line)
    XCTAssertEqual(actual.dtype, .bfloat16, file: file, line: line)
    XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self),
      expected.view(dtype: .uint16).asArray(UInt16.self), file: file, line: line)
    XCTAssertEqual(actual[.ellipsis, 96..<128].view(dtype: .uint16).asArray(UInt16.self),
      value[.ellipsis, 96..<128].view(dtype: .uint16).asArray(UInt16.self),
      "The unrotated features must retain their exact BF16 bits.", file: file, line: line)
  }

  func testSeededBF16TransposedNormOutputsMatchEachRoundedOperation() {
    for seed in [940, 941, 942] {
      let value = MLXRandom.normal([1, 11, 3, 128],
        key: MLXRandom.key(UInt64(seed))).asType(.bfloat16).transposed(0, 2, 1, 3)
      let angles = MLXRandom.normal([1, 1, 11, 96],
        key: MLXRandom.key(UInt64(seed + 10)))
      assertBF16Parity(value, cosine: MLX.cos(angles).asType(.bfloat16),
        sine: MLX.sin(angles).asType(.bfloat16))
    }
  }

  func testFeatureStridesAndStridedAnglesAvoidAssumingContiguousInputs() {
    let value = MLXRandom.normal([1, 7, 4, 256], key: MLXRandom.key(950))
      .asType(.bfloat16)[.ellipsis, .stride(by: 2)].transposed(0, 2, 1, 3)
    let angles = MLXRandom.normal([1, 1, 7, 192], key: MLXRandom.key(951))
    let cosine = MLX.cos(angles).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
    let sine = MLX.sin(angles).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
    assertBF16Parity(value, cosine: cosine, sine: sine)
  }

  func testReversedHeadRowAndFeatureViewsPreserveExactBF16Arithmetic() {
    let backing = MLXRandom.normal([1, 9, 3, 128], key: MLXRandom.key(954))
      .asType(.bfloat16).transposed(0, 2, 1, 3)
    let angles = MLXRandom.normal([1, 1, 9, 96], key: MLXRandom.key(955))
    let cosine = MLX.cos(angles).asType(.bfloat16)[0..<1, 0..<1, .stride(by: -1), .stride(by: -1)]
    let sine = MLX.sin(angles).asType(.bfloat16)[0..<1, 0..<1, .stride(by: -1), .stride(by: -1)]
    let reversed = backing[0..<1, .stride(by: -1), .stride(by: -1), 0..<128]
    assertBF16Parity(reversed, cosine: cosine, sine: sine)
    assertBF16Parity(reversed[.ellipsis, .stride(by: -1)], cosine: cosine, sine: sine)
  }

  func testBF16RoundingCancellationSubnormalAndRangeEdges() {
    let bits: [UInt16] = [
      0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
      0x3f7f, 0xbf7f, 0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x3b80, 0xbb80,
      0x7f7f, 0xff7f,
    ]
    let value = MLXArray((0..<(1 * 3 * 2 * 128)).map { bits[$0 % bits.count] },
      [1, 3, 2, 128]).view(dtype: .bfloat16).transposed(0, 2, 1, 3)
    let coefficients: [Float] = [1, -1, 0.5, -0.5, 0.99609375, -0.99609375, 0, -0.0]
    let cosine = MLXArray((0..<(3 * 96)).map { coefficients[$0 % coefficients.count] },
      [1, 1, 3, 96]).asType(.bfloat16)
    let sine = MLXArray((0..<(3 * 96)).map { coefficients[($0 + 3) % coefficients.count] },
      [1, 1, 3, 96]).asType(.bfloat16)
    assertBF16Parity(value, cosine: cosine, sine: sine)
  }

  func testFloat32FallbackMatchesUnfusedExpression() {
    let value = MLXRandom.normal([1, 5, 2, 256], key: MLXRandom.key(952))[.ellipsis, .stride(by: 2)].transposed(0, 2, 1, 3)
    let angles = MLXRandom.normal([1, 1, 5, 96], key: MLXRandom.key(953))
    let cosine = MLX.cos(angles), sine = MLX.sin(angles)
    XCTAssertEqual(H3Rotary.apply(value, cosine: cosine, sine: sine).asArray(Float.self),
      expression(value, cosine: cosine, sine: sine).asArray(Float.self))
  }

  func testSmallerEvenRotaryGeometryPreservesTheTail() {
    let value = MLXArray((0..<48).map { Float($0 - 24) / 16 }, [1, 2, 3, 8])
      .asType(.bfloat16)
    let cosine = MLXArray.ones([1, 1, 3, 4], dtype: .bfloat16) * Float(0.5)
    let sine = MLXArray.ones([1, 1, 3, 4], dtype: .bfloat16) * Float(-0.5)
    let actual = H3Rotary.apply(value, cosine: cosine, sine: sine, rotaryWidth: 4)
    let expected = expression(value, cosine: cosine, sine: sine, rotaryWidth: 4)
    XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self),
      expected.view(dtype: .uint16).asArray(UInt16.self))
  }

  func testCPUFallbackMatchesExpressionForBF16AndFloat32() {
    Device.withDefaultDevice(.cpu) {
      for dtype in [DType.bfloat16, .float32] {
        let values: [Float] = (0..<768).map { Float(($0 % 31) - 15) / Float(16) }
        let value = MLXArray(values, [1, 3, 2, 128]).asType(dtype).transposed(0, 2, 1, 3)
        let cosine = MLXArray((0..<(3 * 96)).map { Float(($0 % 7) - 3) / 4 },
          [1, 1, 3, 96]).asType(dtype)
        let sine = MLXArray((0..<(3 * 96)).map { Float(($0 % 11) - 5) / 8 },
          [1, 1, 3, 96]).asType(dtype)
        let actual = H3Rotary.apply(value, cosine: cosine, sine: sine)
        let expected = expression(value, cosine: cosine, sine: sine)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
      }
    }
  }
}
