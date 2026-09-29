import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import H3MLX

final class H3StillReferenceMediaTests: XCTestCase {
  func testReferenceThumbnailKeepsSourceTopAtTop() throws {
    // Independent 2×2 PNG fixture: red top row, blue bottom row.
    let png = Data(base64Encoded:
      "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII=")!
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".png")
    try png.write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    let loaded = try H3StillReferenceMedia.load(path: file.path)
    let bytes = Array(loaded.reference.rgb8)
    XCTAssertGreaterThan(Int(bytes[0]), Int(bytes[2]) + 100)
    let lower = (255 * 256) * 3
    XCTAssertGreaterThan(Int(bytes[lower + 2]), Int(bytes[lower]) + 100)
  }

  func testLargeSourceIsBoundedAndLetterboxedWithoutLosingIdentity() throws {
    let width = 1024
    let height = 512
    let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height,
      bitsPerComponent: 8, bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let image = try XCTUnwrap(context.makeImage())
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".png")
    defer { try? FileManager.default.removeItem(at: file) }
    let data = NSMutableData()
    let destination = try XCTUnwrap(CGImageDestinationCreateWithData(data,
      UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    try (data as Data).write(to: file)
    let loaded = try H3StillReferenceMedia.load(path: file.path)
    XCTAssertEqual(loaded.sourceWidth, width)
    XCTAssertEqual(loaded.sourceHeight, height)
    XCTAssertEqual(loaded.sourceSHA256.count, 64)
    XCTAssertEqual(loaded.reference.rgb8.count, 256 * 256 * 3)
    XCTAssertEqual(loaded.reference.width, 256)
    let top = Array(loaded.reference.rgb8.prefix(3))
    let center = Array(loaded.reference.rgb8[(128 * 256 + 128) * 3..<(128 * 256 + 128) * 3 + 3])
    XCTAssertLessThan(abs(Int(top[0]) - Int(top[1])), 3)
    XCTAssertGreaterThan(center[0], 220)
    XCTAssertLessThan(center[1], 64)
  }
}
