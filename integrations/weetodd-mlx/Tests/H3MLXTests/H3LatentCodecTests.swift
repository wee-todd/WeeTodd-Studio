import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3LatentCodecTests: XCTestCase {
  func testReferencePosteriorMeansPackWithVideoAndChannelMajorAudioOrder() throws {
    let pixels = (0..<4).flatMap { spatial in
      (0..<24).map { Float(spatial * 10 + $0) }
    }
    let video = MLXArray(pixels, [1, 1, 2, 2, 24])
    let rows = try H3LatentCodec.videoEncoderRows(latents: video,
      mean: [Float](repeating: 0, count: 24),
      standardDeviation: [Float](repeating: 1, count: 24))
    XCTAssertEqual(rows.shape, [1, 1, 96])
    XCTAssertEqual(rows[0, 0, 0].item(Float.self), 0)
    XCTAssertEqual(rows[0, 0, 1].item(Float.self), 10)
    XCTAssertEqual(rows[0, 0, 2].item(Float.self), 20)
    XCTAssertEqual(rows[0, 0, 3].item(Float.self), 30)
    XCTAssertEqual(rows[0, 0, 4].item(Float.self), 1)
    let audio = MLXArray([Float](repeating: 1, count: 32)
      + [Float](repeating: 2, count: 32)
      + [Float](repeating: 3, count: 32)
      + [Float](repeating: 4, count: 32), [2, 2, 32])
    let audioRows = try H3LatentCodec.audioEncoderRows(latents: audio,
      mean: [Float](repeating: 0, count: 32),
      standardDeviation: [Float](repeating: 1, count: 32))
    XCTAssertEqual(audioRows.shape, [1, 4, 32])
    for index in 0..<4 {
      XCTAssertEqual(audioRows[0, index, 0].item(Float.self), Float(index + 1))
    }
  }
  func testVideoRowsUnpatchifyAndDenormalizeIntoDecoderOrder() throws {
    let rows = MLXArray((0..<96).map(Float.init), [1, 1, 96])
    let mean = Array(repeating: Float(2), count: 24)
    let std = Array(repeating: Float(3), count: 24)
    let decoded = try H3LatentCodec.videoDecoderInput(rows: rows,
      latentFrames: 1, latentHeight: 2, latentWidth: 2,
      mean: mean, standardDeviation: std)
    XCTAssertEqual(decoded.shape, [1, 1, 2, 2, 24])
    let values = decoded.asArray(Float.self)
    for y in 0..<2 {
      for x in 0..<2 {
        for channel in 0..<24 {
          let expected = Float((channel * 4 + y * 2 + x) * 3 + 2)
          XCTAssertEqual(values[(y * 2 + x) * 24 + channel], expected)
        }
      }
    }
  }

  func testAudioRowsStayChannelMajorWhenDenormalized() throws {
    let rows = MLXArray((0..<128).map(Float.init), [1, 4, 32])
    let decoded = try H3LatentCodec.audioDecoderInput(rows: rows,
      latentFrames: 2, mean: Array(repeating: 1, count: 32),
      standardDeviation: Array(repeating: 2, count: 32))
    XCTAssertEqual(decoded.shape, [2, 2, 32])
    let values = decoded.asArray(Float.self)
    XCTAssertEqual(values[0], 1)
    XCTAssertEqual(values[32], 65)
    XCTAssertEqual(values[64], 129)
  }

  func testVideoPixelPresentationUsesImageNetChannelStatistics() throws {
    let normalized = MLXArray([Float(0), 0, 0], [1, 1, 1, 1, 3])
    let pixels = try H3LatentCodec.videoPixelsRGB8(normalized)
    XCTAssertEqual(pixels.shape, [1, 1, 1, 1, 3])
    XCTAssertEqual(pixels.asArray(UInt8.self), [124, 116, 104])
  }
}
