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
    for first in [true, false] {
      let loaded = try H3FL2VAMedia.load(path: file.path,
        width: 32, height: 32, first: first)
      let bytes = Array(loaded.image.rgb8)
      XCTAssertGreaterThan(Int(bytes[0]), Int(bytes[2]) + 100)
      let lower = (31 * 32 + 0) * 3
      XCTAssertGreaterThan(Int(bytes[lower + 2]), Int(bytes[lower]) + 100)
    }
  }
}
