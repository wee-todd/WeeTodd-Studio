import XCTest
import MLX
@testable import ReCaVSRMLX

final class ReCaTensorMathTests: XCTestCase {
  func testPixelUnshuffleUsesChannelThenSubpixelOrder() throws {
    try Device.withDefaultDevice(.cpu) {
      let x = MLXArray((0..<12).map(Float.init), [1, 2, 2, 3])
      let packed = try ReCaLQProjection.pixelUnshuffle(x, factor: 2)
      XCTAssertEqual(packed.shape, [1, 1, 1, 12])
      XCTAssertEqual(packed.asArray(Float.self), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    }
  }
  func testCausalConvolutionRetainsStridePhaseAcrossUnevenChunks() throws {
    try Device.withDefaultDevice(.cpu) {
      let weight = MLXArray([Float(1), 2, 3, 4], [1, 4, 1, 1, 1])
      let conv = ReCaCausalConv(weight: weight, bias: MLXArray([Float(0)]), temporalStride: 2, spatialStride: 1, spatialPadding: 0)
      let x = MLXArray((0..<13).map(Float.init), [13, 1, 1, 1])
      var whole = ReCaCausalState(), chunks = ReCaCausalState()
      let expected = try conv.step(x, state: &whole)
      var parts: [MLXArray] = []
      for range in [0..<5, 5..<9, 9..<13] { parts.append(try conv.step(x[range], state: &chunks)) }
      let actual = concatenated(parts, axis: 0)
      XCTAssertEqual(expected.asArray(Float.self), [0, 11, 30, 50, 70, 90, 110])
      XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
      XCTAssertEqual(chunks.seenFrames, 13)
      XCTAssertEqual(chunks.history?.shape, [3, 1, 1, 1])
    }
  }
  func testRMSNormalizationUsesEpsilonAndRestoresSourceDtype() throws {
    Device.withDefaultDevice(.cpu) {
      let x = MLXArray([Float(3), 4], [1, 2])
      let y = ReCaMath.rms(x, weight: MLXArray([Float(2), 1]), epsilon: 1e-6)
      let values = y.asArray(Float.self)
      XCTAssertEqual(values[0], 6 / sqrt(12.5 + 1e-6), accuracy: 1e-6)
      XCTAssertEqual(values[1], 4 / sqrt(12.5 + 1e-6), accuracy: 1e-6)
    }
  }
  func testWindowAttentionMatchesIndependentVisibleKeyMean() throws {
    try Device.withDefaultDevice(.cpu) {
      let height = 5, width = 7, span = height * width
      let plan = ReCaAttentionPlan(height: height, width: width, windowHeight: 4, windowWidth: 4, slidesAtEdge: true)
      let q = MLXArray.zeros([span * 2, 2, 4])
      let k = MLXArray.zeros([span * 3, 1, 4])
      let v = broadcast(MLXArray((0..<(span * 3)).map(Float.init), [span * 3, 1, 1]), to: [span * 3, 1, 4])
      let result = try plan.apply(query: q, key: k, value: v)[0..., 1, 0].asArray(Float.self)
      // Top-left has a shifted 4×4 spatial window in each of the three planes.
      XCTAssertEqual(result[0], 47, accuracy: 1e-4)
      XCTAssertEqual(result[span], 47, accuracy: 1e-4)
      // Bottom-right shifts to rows 1...4 and columns 3...6.
      XCTAssertEqual(result[span - 1], 57, accuracy: 1e-4)
    }
  }
  func testRotaryKeepsTemporalRotationOutOfPersistentSpatialKeys() throws {
    Device.withDefaultDevice(.cpu) {
      let x = MLXArray([Float(1), 0, 1, 0, 1, 0], [1, 1, 6])
      let spatial = ReCaRotary(height: 1, width: 1, times: [0], axes: [2, 2, 2], spatial: true, temporal: false)
      let temporal = ReCaRotary(height: 1, width: 1, times: [1], axes: [2, 2, 2], spatial: false, temporal: true)
      let cached = spatial.apply(x)
      XCTAssertEqual(cached.asArray(Float.self), [1, 0, 1, 0, 1, 0])
      let rotated = temporal.apply(cached).asArray(Float.self)
      XCTAssertEqual(rotated[0], cos(Float(1)), accuracy: 1e-6)
      XCTAssertEqual(rotated[1], sin(Float(1)), accuracy: 1e-6)
      XCTAssertEqual(Array(rotated[2...]), [1, 0, 1, 0])
      XCTAssertEqual(cached.asArray(Float.self), [1, 0, 1, 0, 1, 0])
    }
  }
  func testNonzeroWindowGQAMatchesIndependentDoublePrecisionOracle() throws {
    for device in [Device.cpu, Device.gpu] { try Device.withDefaultDevice(device) {
      let h = 5, w = 9, span = h * w, qHeads = 4, kvHeads = 2, dim = 4
      let q = (0..<(span * 2 * qHeads * dim)).map { Float(sin(Double($0) * 0.37) * 0.8) }
      let k = (0..<(span * 3 * kvHeads * dim)).map { Float(cos(Double($0) * 0.23) * 0.9) }
      let v = (0..<k.count).map { Float(sin(Double($0) * 0.19)) }
      for shifted in [true, false] {
        let actual = try ReCaAttentionPlan(height: h, width: w, windowHeight: 4, windowWidth: 6,
          slidesAtEdge: shifted).apply(query: MLXArray(q, [span * 2, qHeads, dim]),
            key: MLXArray(k, [span * 3, kvHeads, dim]), value: MLXArray(v, [span * 3, kvHeads, dim])).asArray(Float.self)
        for token in 0..<(span * 2) { for head in 0..<qHeads {
          let y = token % span / w, x = token % w
          let y0 = shifted ? min(max(0, y - 2), h - 4) : max(0, y - 2)
          let x0 = shifted ? min(max(0, x - 3), w - 6) : max(0, x - 3)
          let y1 = shifted ? y0 + 4 : min(h, y + 2)
          let x1 = shifted ? x0 + 6 : min(w, x + 3)
          var logits: [Double] = [], indices: [Int] = []
          for t in 0..<3 { for row in y0..<y1 { for col in x0..<x1 {
            let index = ((t * span + row * w + col) * kvHeads + head / 2) * dim
            indices.append(index)
            logits.append((0..<dim).reduce(0) { $0 + Double(q[(token * qHeads + head) * dim + $1]) * Double(k[index + $1]) } / 2)
          } } }
          let largest = logits.max()!, scores = logits.map { exp($0 - largest) }, total = scores.reduce(0, +)
          for d in 0..<dim {
            let expected = zip(scores, indices).reduce(0) { $0 + $1.0 * Double(v[$1.1 + d]) } / total
            XCTAssertEqual(Double(actual[(token * qHeads + head) * dim + d]), expected, accuracy: 2e-6)
          }
        } }
      }
    } }
  }
}
