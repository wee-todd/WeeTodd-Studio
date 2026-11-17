import Foundation
import MLX
import XCTest
@testable import MOSSImageMLX

final class QuantizationTests: XCTestCase {
  func testNativeAffineConversionReloadsAndAdmitsExactLogicalShapes() throws {
    try Device.withDefaultDevice(.cpu) {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      let input = root.appendingPathComponent("input"), output = root.appendingPathComponent("output")
      try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: root) }
      let w = MLXArray((0..<256).map { sin(Float($0)) }).reshaped(2, 128).asType(.bfloat16)
      try MLX.save(arrays: ["layers.0.proj.q.weight": w], url: input.appendingPathComponent("weights.safetensors"))
      try Data("{}".utf8).write(to: input.appendingPathComponent("config.json"))
      let shapes = ["layers.0.proj.q.weight": [2, 128]]
      try MOSSQuantization.convert(input: input, output: output, bits: 8, component: "transformer", shapes: shapes)
      let index = try MOSSCheckpointIndex(directory: output)
      try MOSSImagePreflight.validate(index, shapes: shapes, selected: { _ in true })
      let source = MOSSWeightSource(index: index)
      let x = MLXArray((0..<128).map { cos(Float($0)) }).reshaped(1, 128).asType(.bfloat16)
      let actual = try source.linear(x, "layers.0.proj.q").asArray(Float.self)
      // Accumulate the original BF16 values in FP32: a BF16 output rounds
      // a large dot product too coarsely to assess affine quantization error.
      let expected = matmul(x.asType(.float32), w.asType(.float32).transposed()).asArray(Float.self)
      for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 0.06) }
      XCTAssertThrowsError(try MOSSImagePreflight.validate(index, shapes: ["layers.0.proj.q.weight": [3, 128]], selected: { _ in true }))
      XCTAssertThrowsError(try MOSSQuantization.convert(input: input, output: output, bits: 4, component: "transformer", shapes: shapes))
    }
  }
}
