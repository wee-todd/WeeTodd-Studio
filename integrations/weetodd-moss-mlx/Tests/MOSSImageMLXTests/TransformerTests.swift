import Foundation
import MLX
import XCTest
@testable import MOSSImageMLX

private final class MemoryWeights: MOSSWeightAccess {
  var arrays: [String: MLXArray]
  init(_ arrays: [String: MLXArray]) { self.arrays = arrays }
  func contains(_ name: String) -> Bool { arrays[name] != nil }
  func tensor(_ name: String) throws -> MLXArray {
    guard let value = arrays[name] else { throw MOSSError("Missing fixture \(name)") }; return value
  }
}

final class TransformerTests: XCTestCase {
  func testZeroGatedBlocksKeepLatentSignalAndReturnOnlyImageRows() throws {
    try Device.withDefaultDevice(.cpu) {
      let dimensions = MOSSDimensions(width: 8, feedForward: 8, heads: 1, depth: 2, textWidth: 8, latentChannels: 8, frequencyWidth: 4)
      var arrays = [String: MLXArray]()
      for (key, shape) in MOSSModelContract.transformerShapes(dimensions) {
        arrays[key] = key.contains("norm") ? .ones(shape) : .zeros(shape)
      }
      arrays["video_embedder.weight"] = MLXArray.eye(8)
      arrays["final_linear_video.weight"] = MLXArray.eye(8)
      let transformer = MOSSTransformer(weights: MemoryWeights(arrays), dimensions: dimensions)
      let context = MLXArray.zeros([1, 3, 8])
      let prepared = try transformer.prepare(context: context, height: 1, width: 1)
      let latent = MLXArray((1...8).map(Float.init)).reshaped(1, 8, 1, 1)
      let output = try transformer.predict(latents: latent, timestep: 1000, conditioning: prepared, skipLayers: [])
      XCTAssertEqual(output.shape, [1, 8, 1, 1])
      let expected: [Float] = [0.1980295, 0.396059, 0.59408855, 0.792118, 0.9901475, 1.1881771, 1.3862065, 1.584236]
      for (a, b) in zip(output.asArray(Float.self), expected) { XCTAssertEqual(a, b, accuracy: 2e-6) }
    }
  }
}
