import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenImageProcessorTests: XCTestCase {
  func testReleasedStillNormalizationAndEntirePatchOrderOnCPU() throws {
    // The released H3 processor uses (RGB / 255 - 0.5) / 0.5.
    // Spatially distinct colors detect channel, temporal, patch and merge swaps.
    let width = 64
    var rgb = [UInt8](repeating: 0, count: width * width * 3)
    let colors: [[UInt8]] = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
      [0, 0, 0], [255, 255, 255], [64, 128, 192]]
    for y in 0..<width {
      for x in 0..<width {
        let color = colors[(x + 3 * y + x / 16 + y / 16) % colors.count]
        for channel in 0..<3 { rgb[(y * width + x) * 3 + channel] = color[channel] }
      }
    }
    let actual = try Device.withDefaultDevice(.cpu) {
      try H3QwenImageProcessor.packRGB8(image: Data(rgb), width: width, height: width)
    }
    XCTAssertEqual(actual.grid.temporal, 1)
    XCTAssertEqual(actual.grid.height, 4)
    XCTAssertEqual(actual.grid.width, 4)
    XCTAssertEqual(actual.pixels.shape, [16, 1536])
    let values = Device.withDefaultDevice(.cpu) { actual.pixels.asArray(Float.self) }
    // Invert the flattened processor contract independently for each element:
    // row = [blockY, blockX, mergeY, mergeX], column = [channel, time, y, x].
    for row in 0..<16 {
      let block = row / 4
      let merge = row % 4
      let originY = (block / 2 * 2 + merge / 2) * 16
      let originX = (block % 2 * 2 + merge % 2) * 16
      for column in 0..<1536 {
        let channel = column / 512
        let withinFrame = column % 256
        let y = originY + withinFrame / 16
        let x = originX + withinFrame % 16
        let source = rgb[(y * width + x) * 3 + channel]
        let expected = (Float(source) / 255 - 0.5) / 0.5
        XCTAssertEqual(values[row * 1536 + column], expected, accuracy: 0.0000001,
          "row \(row), column \(column), source (\(x),\(y),\(channel))")
      }
    }
  }

  func testStillPatchRowsMatchInstalledProcessorOracle() throws {
    let root = URL(fileURLWithPath: "/tmp/weetodd-h3-qwen-image-processor")
    guard FileManager.default.fileExists(atPath: root.appendingPathComponent("rgb.u8").path) else {
      throw XCTSkip("Qwen image processor oracle is not installed.")
    }
    let rgb = try Data(contentsOf: root.appendingPathComponent("rgb.u8"))
    let actual = try Device.withDefaultDevice(.cpu) {
      try H3QwenImageProcessor.packRGB8(image: rgb, width: 64, height: 64)
    }
    XCTAssertEqual(actual.grid.temporal, 1)
    XCTAssertEqual(actual.grid.height, 4)
    XCTAssertEqual(actual.grid.width, 4)
    let data = try Data(contentsOf: root.appendingPathComponent("pixels.f32"))
    let difference = Device.withDefaultDevice(.cpu) {
      let expected = data.withUnsafeBytes { MLXArray($0, [16, 1536], type: Float.self) }
      return max(abs(actual.pixels - expected)).item(Float.self)
    }
    XCTAssertLessThan(difference, 0.00001)
  }

  func testRejectsReferenceOutsideAdmittedPatchGeometry() {
    XCTAssertThrowsError(try H3QwenImageProcessor.packRGB8(
      image: Data(count: 16 * 16 * 3), width: 16, height: 16))
  }
}
