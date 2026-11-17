import Foundation
import MLX
import XCTest
@testable import MOSSImageMLX

final class ReferenceTests: XCTestCase {
  func testSourcePreparationKeepsOrientationAndChecksIdentity() throws {
    try Device.withDefaultDevice(.cpu) {
      let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: directory) }
      let url = directory.appendingPathComponent("source.png")
      var pixels = [Float](repeating: -1, count: 16 * 32 * 3)
      for y in 0..<16 { for x in 0..<32 { pixels[(y * 32 + x) * 3 + (y < 8 ? 0 : 2)] = 1 } }
      try MOSSVAEOutput.writePNG(MLXArray(pixels).reshaped(1, 16, 32, 3), to: url)
      var reference = MOSSReferenceImage(path: url.path, sha256: try MOSSLayoutImage.sha256(url))
      let prepared = try reference.prepare(width: 64, height: 32)
      XCTAssertEqual(prepared.pixels.shape, [1, 32, 64, 3])
      let values = prepared.pixels.asArray(Float.self)
      XCTAssertGreaterThan(values[(4 * 64 + 32) * 3], 0.9)
      XCTAssertGreaterThan(values[(28 * 64 + 32) * 3 + 2], 0.9)
      XCTAssertEqual(prepared.sourceWidth, 32); XCTAssertEqual(prepared.sourceHeight, 16)
      reference.sha256 = String(repeating: "0", count: 64)
      XCTAssertThrowsError(try reference.prepare(width: 64, height: 32))
    }
  }
  func testReferenceAdmissionRejectsUnsupportedRecipesWithoutOverridingThem() throws {
    var request = MOSSImageRequest(transformerDirectory: "/models/sft", sharedDirectory: "/models/shared", prompt: "scene")
    request.sourceImage = .init(path: "/source.png", sha256: String(repeating: "a", count: 64))
    XCTAssertNoThrow(try request.effectivePrompt())
    request.configuration.quantizationBits = 4
    XCTAssertThrowsError(try request.effectivePrompt())
    request.configuration.quantizationBits = nil; request.modelVariant = "base"
    XCTAssertThrowsError(try request.effectivePrompt())
    request.modelVariant = "sft"; request.loras = [.init(path: "/adapter")]
    XCTAssertThrowsError(try request.effectivePrompt())
    request.loras[0].enabled = false
    XCTAssertNoThrow(try request.effectivePrompt())
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    object.removeValue(forKey: "sourceImage")
    XCTAssertNil(try JSONDecoder().decode(MOSSImageRequest.self, from: JSONSerialization.data(withJSONObject: object)).sourceImage)
  }
  func testEncoderSelectsGaussianMeanAndNormalizesThePackedSpace() throws {
    try Device.withDefaultDevice(.cpu) {
      let image = MLXArray.zeros([1, 16, 32, 3])
      let encoded = try MOSSSmallVAE(weights: EncoderFixture()).encode(image: image)
      XCTAssertEqual(encoded.shape, [1, 128, 1, 2])
      XCTAssertLessThan(abs(encoded - Float(0.8 / sqrt(0.4901))).max().item(Float.self), 1e-5)
    }
  }
  func testPackedSourceLatentsInvertDecoderNormalizationOnNonSquareGrid() {
    Device.withDefaultDevice(.cpu) {
      let values: [Float] = (0..<768).map { Float($0) / Float(200) }
      let means: [Float] = (0..<128).map { Float($0) / Float(100) }
      let source = MLXArray(values).reshaped(1, 4, 6, 32)
      let mean = MLXArray(means), variance = MLXArray.ones([128]) * Float(0.7)
      let packed = MOSSSmallVAE.pack(source, mean: mean, variance: variance)
      XCTAssertEqual(packed.shape, [1, 128, 2, 3])
      XCTAssertLessThan(abs(MOSSSmallVAE.unpack(packed, mean: mean, variance: variance) - source).max().item(Float.self), 1e-6)
    }
  }
  func testStridedBandsMatchDenseRightBottomPaddingAcrossEveryBoundary() {
    Device.withDefaultDevice(.cpu) {
      let values: [Float] = (0..<160).map { Float($0 % 13) / Float(13) }
      let weights: [Float] = (0..<54).map { Float($0 % 7) / Float(20) }
      let pixels = MLXArray(values).reshaped(1, 8, 10, 2)
      let weight = MLXArray(weights).reshaped(3, 3, 3, 2)
      let bias = MLXArray([Float(0.1), 0.2, 0.3])
      let expected = conv2d(padded(pixels, widths: [0, .init((0, 1)), .init((0, 1)), 0]), weight, stride: 2) + bias
      for inputRows in [1, 2, 3, 8] {
        for outputRows in [1, 2, 3, 4] {
          let value = MOSSImageBands(pixels, rowLimit: inputRows).downsampled(weight: weight, bias: bias, rowLimit: outputRows).materialized()
          XCTAssertEqual(value.shape, [1, 4, 5, 3])
          XCTAssertLessThan(abs(value - expected).max().item(Float.self), 1e-6)
        }
      }
    }
  }
  func testReferenceUsesActualStartSigmaAndNonzeroConfiguredSteps() throws {
    try Device.withDefaultDevice(.cpu) {
      let source = MLXArray([Float(2), 4]), noise = MLXArray([Float(6), 8])
      let result = try MOSSReferenceImage.initialize(source: source, noise: noise, sigma: 0.15)
      XCTAssertLessThan(abs(result - MLXArray([Float(2.6), 4.6])).max().item(Float.self), 1e-6)
      let sigmas = try MOSSReferenceImage.sigmas([1, 0.7, 0.2, 0], noiseLevel: 0.15)
      XCTAssertEqual(sigmas.count, 4); XCTAssertEqual(sigmas.first!, 0.15); XCTAssertEqual(sigmas.last!, 0)
      XCTAssertThrowsError(try MOSSReferenceImage.sigmas([1, 0.5, 0.5, 0], noiseLevel: 0.15))
      for invalid: Float in [0, 1, -.infinity, .nan] {
        XCTAssertThrowsError(try MOSSReferenceImage.initialize(source: source, noise: noise, sigma: invalid))
      }
      XCTAssertThrowsError(try MOSSReferenceImage.initialize(source: source, noise: MLXArray([Float(1)]), sigma: 0.15))
    }
  }
}

private final class EncoderFixture: MOSSWeightAccess {
  let shapes = MOSSModelContract.encoderShapes()
  func contains(_ name: String) -> Bool { shapes[name] != nil }
  func tensor(_ name: String) throws -> MLXArray {
    if name == "bn.running_mean" { return MLXArray.ones([128]) * Float(0.2) }
    if name == "bn.running_var" { return MLXArray.ones([128]) * Float(0.49) }
    guard let shape = shapes[name] else { throw MOSSError("Unknown encoder fixture tensor: \(name)") }
    if name == "quant_conv.weight" { return MLXArray.eye(64).reshaped(64, 64, 1, 1) }
    if name == "encoder.conv_out.bias" || (name.contains("norm") && name.hasSuffix("weight")) { return .ones(shape) }
    return .zeros(shape)
  }
}
