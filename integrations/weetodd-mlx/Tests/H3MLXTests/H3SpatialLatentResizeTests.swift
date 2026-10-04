import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3SpatialLatentResizeTests: XCTestCase {
  func testNearestExactAndHalfPixelBilinearKeepChannelTimeAndPatchOrder() throws {
    try Device.withDefaultDevice(.cpu) {
      let source = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
      let target = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
      var values = [Float]()
      for time in 0..<source.videoLatentFrames {
        for value: Float in [0, 2, 4, 6] {
          for channel in 0..<24 { values.append(value + Float(time * 100 + channel * 10)) }
        }
      }
      let pixels = MLXArray(values, [1, source.videoLatentFrames, 2, 2, 24])
      let packed = pixels.reshaped([1, source.videoLatentFrames, 1, 2, 1, 2, 24])
        .transposed(0, 1, 2, 4, 6, 3, 5).reshaped([1, source.videoRows, 96])
      for (method, expected): (H3SpatialLatentResizeMethod, [Float]) in [
        (.nearestExact, [0,0,2,2,0,0,2,2,4,4,6,6,4,4,6,6]),
        (.bilinear, [0,0.5,1.5,2,1,1.5,2.5,3,3,3.5,4.5,5,4,4.5,5.5,6])
      ] {
        let result = try H3SpatialLatentResize.rows(packed, source: source, target: target, method: method)
        let unpacked = try H3LatentCodec.videoDecoderInput(rows: result,
          latentFrames: target.videoLatentFrames, latentHeight: 4, latentWidth: 4,
          mean: [Float](repeating: 0,count: 24), standardDeviation: [Float](repeating: 1,count: 24)).asArray(Float.self)
        for time in [0, source.videoLatentFrames - 1] {
          for pixel in 0..<16 {
            for channel in [0,23] {
              XCTAssertEqual(unpacked[(time * 16 + pixel) * 24 + channel],
                expected[pixel] + Float(time * 100 + channel * 10))
            }
          }
        }
      }
      let unchanged = try H3SpatialLatentResize.rows(packed, source: source, target: source, method: .lanczos3)
      XCTAssertEqual(unchanged.asArray(Float.self).map(\.bitPattern), packed.asArray(Float.self).map(\.bitPattern))
    }
  }
  func testCubicAndLanczosPreserveConstantLatentsAndNeverResizeAudioOrTime() throws {
    try Device.withDefaultDevice(.cpu) {
      let source = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
      let target = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
      let constant = MLXArray([Float](repeating: 2, count: source.videoRows * 96), [1, source.videoRows, 96])
      for method in [H3SpatialLatentResizeMethod.bicubic, .lanczos3] {
        let result = try H3SpatialLatentResize.rows(constant, source: source, target: target, method: method)
        XCTAssertEqual(result.shape, [1,target.videoRows,96])
        for value in result.asArray(Float.self) { XCTAssertEqual(value,2,accuracy: 2e-6) }
      }
      let wrongTime = try H3Geometry(width: 64,height: 64,durationSeconds: 5)
      XCTAssertThrowsError(try H3SpatialLatentResize.rows(constant, source: source,target:wrongTime,method:.bilinear))
    }
  }
}
