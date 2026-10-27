import MLX
import XCTest
@testable import H3MLX

final class H3PDDTests: XCTestCase {
  private func plan() throws -> H3PDDPlan { try H3PDDPlan(contract: H3OmniContract(data: omniContractData())) }

  // A separate Float64 scalar oracle uses differences of evaluated nodes, not
  // the implementation's cancellation-free interval coefficients.
  private func sigma(_ index: Int, shift: Double) -> Double {
    let u = 0.999 * Double(32 - index) / 32
    return shift * u / (1 + (shift - 1) * u / 0.999)
  }

  func testEightFourHeadBlocksUseMaxTimeNormalizedModalityGrids() throws {
    let p = try plan()
    XCTAssertEqual(p.fineGrid.count, 33)
    for modality: H3PDDPlan.Modality in [.video, .audio] {
      let shift: Double = modality == .video ? 12 : 3
      let fine = p.fineSigmas(for: modality), coarse = p.coarseSigmas(for: modality)
      XCTAssertEqual(fine.first!, 0.999, accuracy: 1e-15)
      XCTAssertEqual(coarse.count, 9)
      XCTAssertEqual(coarse.last, 0)
      for i in 0...32 { XCTAssertEqual(fine[i], sigma(i, shift: shift), accuracy: 3e-15) }
      let weights = p.integrationWeights(for: modality)
      for step in 0..<8 {
        let b = try p.block(step: step, modality: modality)
        XCTAssertEqual(b.headRange, (step * 4)..<(step * 4 + 4))
        XCTAssertEqual(b.normalizedWeights.count, 4)
        let total = sigma(step * 4 + 4, shift: shift) - sigma(step * 4, shift: shift)
        XCTAssertLessThan(b.integrationWeight, 0)
        XCTAssertEqual(b.integrationWeight, total, accuracy: 2e-15)
        XCTAssertEqual(b.normalizedWeights.reduce(0,+), 1, accuracy: 1e-15)
        XCTAssertEqual(p.timesteps(for: modality)[step], 1 - Float(coarse[step]))
        for j in 0..<4 {
          let index = step * 4 + j
          let delta = sigma(index + 1, shift: shift) - sigma(index, shift: shift)
          XCTAssertEqual(weights[index], delta, accuracy: 2e-15)
          XCTAssertEqual(b.normalizedWeights[j], delta / total, accuracy: 1e-12)
        }
      }
    }
    XCTAssertNotEqual(p.integrationWeights(for: .video), p.integrationWeights(for: .audio))
    XCTAssertThrowsError(try p.block(step: -1, modality: .video))
    XCTAssertThrowsError(try p.block(step: 8, modality: .video))
  }

  func testLatentUpdateIsDataWardAndMatchesIndependentFineIntervalIntegral() throws {
    let p = try plan()
    for modality: H3PDDPlan.Modality in [.video, .audio] {
      let shift: Double = modality == .video ? 12 : 3
      for step in 0..<8 {
        let b = try p.block(step: step, modality: modality)
        let heads = b.headRange.map { Double($0 + 1) / 7 }
        let deltas = b.headRange.map { sigma($0 + 1, shift: shift) - sigma($0, shift: shift) }
        let mean = zip(heads, deltas).map(*).reduce(0,+) / deltas.reduce(0,+)
        let expected = 2 - zip(heads, deltas).map(*).reduce(0,+)
        let result = try p.advance(sample: MLXArray([Float(2)], [1,1]),
          mean: MLXArray([Float(mean)], [1,1]), step: step, modality: modality)
        XCTAssertGreaterThan(result.item(Float.self), 2)
        XCTAssertEqual(Double(result.item(Float.self)), expected, accuracy: 2e-6)
      }
    }
  }

  func testParameterFusionMatchesIndependentHeadThenProjectionOracleAndCasts() throws {
    let p = try plan()
    let values = (0..<192).map { Float(($0 * 17) % 31 - 15) / 8 }
    let biases = (0..<64).map { Float($0 % 13 - 6) / 4 }
    for dtype: DType in [.float32, .bfloat16] {
      let weight = MLXArray(values, [64,3]).asType(dtype)
      let bias = MLXArray(biases, [64]).asType(dtype)
      for modality: H3PDDPlan.Modality in [.video, .audio] {
        let shift: Double = modality == .video ? 12 : 3
        for step in [0,3,7] {
          let fused = try H3PDDHeadFusion.fuse(weight: weight, bias: bias, plan: p, step: step, modality: modality)
          XCTAssertEqual(fused.weight.dtype, dtype); XCTAssertEqual(fused.bias?.dtype, dtype)
          XCTAssertEqual(fused.weight.shape, [2,3]); XCTAssertEqual(fused.bias?.shape, [2])
          let output = (matmul(MLXArray([Float(1),-2,0.5], [1,3]), fused.weight.asType(.float32).T)
            + fused.bias!.asType(.float32)).asArray(Float.self)
          for channel in 0..<2 {
            var expected: Double = 0
            let start = step * 4, end = start + 4
            let total = sigma(end, shift: shift) - sigma(start, shift: shift)
            for head in start..<end {
              let row = head * 2 + channel
              let projection = Double(values[row*3]) - 2 * Double(values[row*3+1])
                + 0.5 * Double(values[row*3+2]) + Double(biases[row])
              expected += projection * (sigma(head+1, shift: shift)-sigma(head, shift: shift)) / total
            }
            XCTAssertEqual(Double(output[channel]), expected, accuracy: dtype == .float32 ? 2e-6 : 0.025)
          }
        }
      }
      let withoutBias = try H3PDDHeadFusion.fuse(weight: weight, bias: nil, plan: p, step: 1, modality: .audio)
      XCTAssertNil(withoutBias.bias)
      let updated = try p.advance(sample: MLXArray([Float(2)], [1,1]).asType(dtype),
        mean: MLXArray([Float(1)], [1,1]).asType(dtype), step: 7, modality: .audio)
      XCTAssertEqual(updated.dtype, dtype)
      XCTAssertEqual(updated.item(Float.self), dtype == .float32 ? 2.2997 : 2.296875, accuracy: 1e-6)
    }
  }

  // CPU round-to-nearest-even BF16 oracle; no production fusion or MLX cast
  // is used to form the expectation.
  private func roundBFloat(_ value: Float) -> Float {
    let bits = value.bitPattern
    return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
  }

  func testFusionRoundsParametersOnlyAfterFP32Accumulation() throws {
    let p = try plan()
    let pattern: [Float] = [1024, 1, -1024, 2]
    let values = (0..<192).map { pattern[($0 / 6) % 4] + Float($0 % 3) * 4 }
    let biases = (0..<64).map { pattern[($0 / 2) % 4] }
    for dtype: DType in [.float32, .bfloat16] {
      let weight = MLXArray(values, [64,3]).asType(dtype), bias = MLXArray(biases, [64]).asType(dtype)
      // Decode source precision independently before the Float64 oracle.
      let source = values.map { dtype == .bfloat16 ? roundBFloat($0) : $0 }
      for modality: H3PDDPlan.Modality in [.video, .audio] {
        let shift: Double = modality == .video ? 12 : 3
        for step in 0..<8 {
          let fused = try H3PDDHeadFusion.fuse(weight: weight, bias: bias, plan: p, step: step, modality: modality)
          let actual = fused.weight.asType(.float32).asArray(Float.self)
          let actualBias = fused.bias!.asType(.float32).asArray(Float.self)
          let total = sigma(step * 4 + 4, shift: shift) - sigma(step * 4, shift: shift)
          for feature in 0..<6 {
            var expected: Double = 0
            for head in (step * 4)..<(step * 4 + 4) {
              expected += Double(source[head * 6 + feature])
                * (sigma(head + 1, shift: shift) - sigma(head, shift: shift)) / total
            }
            let cast = dtype == .bfloat16 ? roundBFloat(Float(expected)) : Float(expected)
            XCTAssertEqual(actual[feature], cast, accuracy: dtype == .bfloat16 ? 0 : 0.0001)
          }
          for feature in 0..<2 {
            var expected: Double = 0
            for head in (step * 4)..<(step * 4 + 4) {
              expected += Double(biases[head * 2 + feature])
                * (sigma(head + 1, shift: shift) - sigma(head, shift: shift)) / total
            }
            let cast = dtype == .bfloat16 ? roundBFloat(Float(expected)) : Float(expected)
            XCTAssertEqual(actualBias[feature], cast, accuracy: dtype == .bfloat16 ? 0 : 0.0001)
          }
        }
      }
    }
  }

  func testBF16IntermediateX0PrecedesFP32Blend() throws {
    let p = try plan()
    let samples: [Float] = [1, -1, 0.125, 12], means: [Float] = [2.25, -3.5, 0.5, -8]
    for modality: H3PDDPlan.Modality in [.video, .audio] {
      let shift: Double = modality == .video ? 12 : 3
      for step in 0..<8 {
        let current = Float(sigma(step * 4, shift: shift))
        let next = Float(sigma(step * 4 + 4, shift: shift))
        let time = Float(1) - current
        let modelScale = roundBFloat(Float(Double(1) - Double(time)))
        let ratio = Double(next) / Double(current)
        let expected = zip(samples, means).map { sample, mean in
          let x0 = roundBFloat(sample + roundBFloat(modelScale * mean))
          return roundBFloat(Float(ratio) * sample + Float(1 - ratio) * x0)
        }
        let result = try p.advance(sample: MLXArray(samples, [4,1]).asType(.bfloat16),
          mean: MLXArray(means, [4,1]).asType(.bfloat16), step: step, modality: modality)
        XCTAssertEqual(result.asType(.float32).asArray(Float.self), expected)
      }
    }
  }

  func testMalformedProjectionAndLatentShapesFailBeforeMLXOperations() throws {
    let p = try plan()
    let valid = MLXArray.zeros([64,3])
    for bad in [MLXArray.zeros([63,3]), MLXArray.zeros([32]), MLXArray.zeros([32,2,3]),
                MLXArray.zeros([32,0]), MLXArray.zeros([64,3], dtype: .int32)] {
      XCTAssertThrowsError(try H3PDDHeadFusion.fuse(weight: bad, bias: nil, plan: p, step: 0, modality: .video))
    }
    for bias in [MLXArray.zeros([63]), MLXArray.zeros([64,1]), MLXArray.zeros([64], dtype: .int32)] {
      XCTAssertThrowsError(try H3PDDHeadFusion.fuse(weight: valid, bias: bias, plan: p, step: 0, modality: .video))
    }
    XCTAssertThrowsError(try H3PDDHeadFusion.fuse(weight: valid, bias: nil, plan: p, step: 8, modality: .video))
    for (sample, mean) in [(MLXArray.zeros([2,1]), MLXArray.zeros([1,2])),
      (MLXArray.zeros([1], dtype: .int32), MLXArray.zeros([1])),
      (MLXArray.zeros([1]), MLXArray.zeros([1], dtype: .int32)),
      (MLXArray.zeros([0]), MLXArray.zeros([0]))] {
      XCTAssertThrowsError(try p.advance(sample: sample, mean: mean, step: 0, modality: .video))
    }
    XCTAssertThrowsError(try p.advance(sample: MLXArray.zeros([1]), mean: MLXArray.zeros([1]), step: 8, modality: .video))
  }
}
