import Foundation
import ImageIO
import MLX
import XCTest
@testable import MOSSImageMLX

final class VAETests: XCTestCase {
  func testNonfiniteDecodedPixelsAreNotPublished() throws {
    try Device.withDefaultDevice(.cpu) {
      let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
      defer { try? FileManager.default.removeItem(at: file) }
      let image = MLXArray([Float.nan, 0, 0]).reshaped(1, 1, 1, 3)
      XCTAssertThrowsError(try MOSSVAEOutput.writePNG(image, to: file))
      XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
    }
  }
  func testRawDecoderInfinitiesAreRejectedBeforeClipping() throws {
    try Device.withDefaultDevice(.cpu) {
      for value in [Float.infinity, -Float.infinity, Float.nan] {
        XCTAssertThrowsError(try MOSSSmallVAE.finalize(MLXArray([value, 0, 0]).reshaped(1, 1, 1, 3)))
      }
      let finite = try MOSSSmallVAE.finalize(MLXArray([-2.0, 0, 2]).reshaped(1, 1, 1, 3))
      XCTAssertEqual(finite.asArray(Float.self), [-1, 0, 1])
    }
  }
  func testBandedConvolutionUsesRealNeighborRows() {
    Device.withDefaultDevice(.cpu) {
      let image = MOSSImageBands(MLXArray((1...9).map(Float.init)).reshaped(1, 3, 3, 1), rowLimit: 1)
      let result = image.convolved(weight: .ones([1, 3, 3, 1]), bias: .zeros([1]), rowLimit: 1).materialized()
      XCTAssertEqual(result.asArray(Float.self), [12, 21, 16, 27, 45, 33, 24, 39, 28])
    }
  }
  func testGroupNormUsesWholeImageStatisticsAcrossBands() {
    Device.withDefaultDevice(.cpu) {
      let pixels = MLXArray((0..<24).map(Float.init)).reshaped(1, 3, 2, 4)
      let bands = MOSSImageBands(pixels, rowLimit: 1).groupNormalized(groups: 2, weight: .ones([4]), bias: .zeros([4])).materialized()
      let whole = MOSSImageBands(pixels, rowLimit: 3).groupNormalized(groups: 2, weight: .ones([4]), bias: .zeros([4])).materialized()
      XCTAssertLessThan(abs(bands - whole).max().item(Float.self), 1e-6)
      let values = bands.asArray(Float.self)
      XCTAssertEqual(values[0], -1.532942, accuracy: 1e-6)
      XCTAssertEqual(values[1], -1.3869475, accuracy: 1e-6)
      XCTAssertEqual(values[22], 1.3869475, accuracy: 1e-6)
      XCTAssertEqual(values[23], 1.532942, accuracy: 1e-6)
    }
  }
  func testPackedBatchNormalizationPrecedesUnpatchifying() {
    Device.withDefaultDevice(.cpu) {
      let packed = MLXArray((0..<128).map(Float.init)).reshaped(1, 128, 1, 1)
      let decoded = MOSSSmallVAE.unpack(packed, mean: .zeros([128]), variance: .ones([128]))
      XCTAssertEqual(decoded.shape, [1, 2, 2, 32])
      XCTAssertEqual(decoded[0, 0, 1, 0].item(Float.self), 1.00005, accuracy: 1e-6)
      XCTAssertEqual(decoded[0, 1, 0, 0].item(Float.self), 2.0001, accuracy: 1e-6)
      XCTAssertEqual(decoded[0, 1, 1, 31].item(Float.self), 127.00635, accuracy: 0.00002)
    }
  }
  func testNativePNGHasExpectedDimensionsAndClampedRGB() throws {
    try Device.withDefaultDevice(.cpu) {
      let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: folder) }
      let file = folder.appendingPathComponent("image.png")
      let tensor = MLXArray([Float(-2), 0, 2, 1, -1, 0]).reshaped(1, 1, 2, 3)
      try MOSSVAEOutput.writePNG(tensor, to: file)
      let source = try XCTUnwrap(CGImageSourceCreateWithURL(file as CFURL, nil))
      let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any])
      XCTAssertEqual(properties[kCGImagePropertyPixelWidth as String] as? Int, 2)
      XCTAssertEqual(properties[kCGImagePropertyPixelHeight as String] as? Int, 1)
      XCTAssertEqual(MOSSVAEOutput.rgb8(tensor), [0, 128, 255, 255, 0, 128])
    }
  }
  func testStageGuardReleasesAfterFailureAndPreventsOverlap() throws {
    let lifetime = MOSSStageLifetime()
    XCTAssertThrowsError(try lifetime.withStage(.conditioning) {
      try lifetime.withStage(.decoding) { 0 }
    })
    XCTAssertNil(lifetime.active)
    XCTAssertThrowsError(try lifetime.withStage(.transformer) { throw CancellationError() })
    XCTAssertNil(lifetime.active)
    XCTAssertEqual(try lifetime.withStage(.decoding) { 42 }, 42)
  }
}
