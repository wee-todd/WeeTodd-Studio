import Foundation
import XCTest
@testable import H3MLX

final class H3FL2VAMediaTests: XCTestCase {
  func testFirstAndLastCanvasKeepSourceTopAtTop() throws {
    // Independent 2×2 PNG fixture: red top row, blue bottom row.
    let png = Data(base64Encoded:
      "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII=")!
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".png")
    try png.write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    for (width, height, first) in [(32, 32, true), (32, 32, false),
      (1376, 768, true), (1376, 768, false)] {
      let loaded = try H3FL2VAMedia.load(path: file.path,
        width: width, height: height, first: first)
      let bytes = Array(loaded.image.rgb8)
      XCTAssertGreaterThan(Int(bytes[0]), Int(bytes[2]) + 100)
      let lower = ((height - 1) * width) * 3
      XCTAssertGreaterThan(Int(bytes[lower + 2]), Int(bytes[lower]) + 100)
    }
  }
}
