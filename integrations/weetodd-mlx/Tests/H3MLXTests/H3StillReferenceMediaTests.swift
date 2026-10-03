import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import H3MLX

final class H3StillReferenceMediaTests: XCTestCase {
  private func fixture(width: Int, height: Int, orientation: Int = 1) throws -> URL {
    // Independent source rows: white/green/blue/white corner fiducials on red.
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
      for x in 0..<width {
        var color: [UInt8] = [255, 0, 0]
        if x < width / 8 && y < height / 8 { color = [255, 255, 255] }
        if x >= width * 7 / 8 && y < height / 8 { color = [0, 255, 0] }
        if x < width / 8 && y >= height * 7 / 8 { color = [0, 0, 255] }
        if x >= width * 7 / 8 && y >= height * 7 / 8 { color = [255, 255, 255] }
        for c in 0..<3 { rgba[(y * width + x) * 4 + c] = color[c] }
      }
    }
    let provider = try XCTUnwrap(CGDataProvider(data: Data(rgba) as CFData))
    let image = try XCTUnwrap(CGImage(width: width, height: height,
      bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(
        rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + (orientation == 1 ? ".png" : ".tiff"))
    let encoded = NSMutableData()
    let destination = try XCTUnwrap(CGImageDestinationCreateWithData(encoded,
      (orientation == 1 ? UTType.png : UTType.tiff).identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image,
      [kCGImagePropertyOrientation: orientation] as CFDictionary)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    try (encoded as Data).write(to: file)
    return file
  }

  private func assertPixel(_ image: H3StillReference, x: Int, y: Int,
    expected: [UInt8], file: StaticString = #filePath, line: UInt = #line) {
    let index = (y * image.width + x) * 3
    let actual = Array(image.rgb8[index..<(index + 3)])
    for c in 0..<3 {
      XCTAssertLessThanOrEqual(abs(Int(actual[c]) - Int(expected[c])), 12,
        "pixel (\(x),\(y)) channel \(c)", file: file, line: line)
    }
  }

  func testPortraitFillsAspectCanvasAndKeepsCornerFiducials() throws {
    let file = try fixture(width: 448, height: 1344)
    defer { try? FileManager.default.removeItem(at: file) }
    let loaded = try H3StillReferenceMedia.load(path: file.path)
    let image = loaded.reference
    XCTAssertEqual(image.width, 128)
    XCTAssertEqual(image.height, 384)
    XCTAssertEqual(image.rgb8.count, 128 * 384 * 3)
    XCTAssertEqual(loaded.sourceWidth, 448)
    XCTAssertEqual(loaded.sourceHeight, 1344)
    XCTAssertEqual(loaded.sourceSHA256.count, 64)
    XCTAssertLessThanOrEqual(max(loaded.thumbnailWidth, loaded.thumbnailHeight), 384)
    assertPixel(image, x: 3, y: 3, expected: [255, 255, 255])
    assertPixel(image, x: 124, y: 3, expected: [0, 255, 0])
    assertPixel(image, x: 3, y: 380, expected: [0, 0, 255])
    assertPixel(image, x: 124, y: 380, expected: [255, 255, 255])
    assertPixel(image, x: 64, y: 192, expected: [255, 0, 0])
  }

  func testFourKSourceUsesBoundedThumbnailAndRetainsSourceCorners() throws {
    let file = try fixture(width: 3840, height: 2160)
    defer { try? FileManager.default.removeItem(at: file) }
    let loaded = try H3StillReferenceMedia.load(path: file.path)
    let canvas = try H3StillReferenceMedia.resolveCanvas(sourceWidth: 3840, sourceHeight: 2160)
    XCTAssertEqual(canvas.width, 288)
    XCTAssertEqual(canvas.height, 160)
    XCTAssertEqual(loaded.reference.width, canvas.width)
    XCTAssertEqual(loaded.reference.height, canvas.height)
    XCTAssertLessThanOrEqual(max(loaded.thumbnailWidth, loaded.thumbnailHeight),
      max(canvas.width, canvas.height))
    XCTAssertLessThanOrEqual(loaded.reference.rgb8.count, 65_536 * 3)
    XCTAssertLessThanOrEqual(max(canvas.width, canvas.height), 2048)
    assertPixel(loaded.reference, x: 3, y: 3, expected: [255, 255, 255])
    assertPixel(loaded.reference, x: canvas.width - 4, y: 3, expected: [0, 255, 0])
    assertPixel(loaded.reference, x: 3, y: canvas.height - 4, expected: [0, 0, 255])
  }

  func testExifRotationChangesCanvasAndMovesSourceCorners() throws {
    let file = try fixture(width: 448, height: 1344, orientation: 6)
    defer { try? FileManager.default.removeItem(at: file) }
    let loaded = try H3StillReferenceMedia.load(path: file.path)
    XCTAssertEqual(loaded.sourceOrientation, 6)
    XCTAssertEqual(loaded.reference.width, 384)
    XCTAssertEqual(loaded.reference.height, 128)
    assertPixel(loaded.reference, x: 3, y: 3, expected: [0, 0, 255])
    assertPixel(loaded.reference, x: 380, y: 3, expected: [255, 255, 255])
    assertPixel(loaded.reference, x: 380, y: 124, expected: [0, 255, 0])
  }

  func testGeometryKeepsSquareBudgetAndRejectsUpsamplingOrInvalidSources() throws {
    let square = try H3StillReferenceMedia.resolveCanvas(sourceWidth: 4096, sourceHeight: 4096)
    XCTAssertEqual(square.width, 256)
    XCTAssertEqual(square.height, 256)
    let small = try H3StillReferenceMedia.resolveCanvas(sourceWidth: 64, sourceHeight: 128)
    XCTAssertEqual(small.width, 64)
    XCTAssertEqual(small.height, 128)
    for orientation in 5...8 {
      let rotated = try H3StillReferenceMedia.resolveCanvas(sourceWidth: 448,
        sourceHeight: 1344, orientation: orientation)
      XCTAssertEqual(rotated.width, 384)
      XCTAssertEqual(rotated.height, 128)
    }
    for (width, height, orientation) in [(16, 64, 1), (32, 32, 1),
      (1000, 8000, 1), (4000, 4000, 9), (20_000, 20_000, 1)] {
      XCTAssertThrowsError(try H3StillReferenceMedia.resolveCanvas(sourceWidth: width,
        sourceHeight: height, orientation: orientation))
    }
  }

  func testNineStillAdmissionRetainsPreviousComputeBudgetBeforeWeights() throws {
    let image = H3StillReference(rgb8: Data(count: 256 * 256 * 3), width: 256, height: 256)
    try H3VideoReferencePreparation.validate(Array(repeating: .image(image), count: 9))
    let portrait = H3StillReference(rgb8: Data(count: 128 * 384 * 3), width: 128, height: 384)
    try H3VideoReferencePreparation.validate([.image(portrait)])
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate(
      Array(repeating: .image(image), count: 10)))
    let overBudget = H3StillReference(rgb8: Data(count: 256 * 800 * 3), width: 256, height: 800)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([.image(overBudget)]))
  }

  func testNineStillQwenBudgetIncludesPromptBeforeWeights() throws {
    var vocabulary = Dictionary(uniqueKeysWithValues: (33...126).map {
      (String(UnicodeScalar($0)!), Int32($0))
    })
    vocabulary["Ġ"] = 32
    let tokenizer = try H3QwenTokenizer(vocabulary: vocabulary, merges: [], regex: ".",
      specialIDs: ["<|vision_start|>": 151652, "<|vision_end|>": 151653,
        "<|image_pad|>": 151655])
    let grid = H3QwenRequest.Grid(temporal: 1, height: 16, width: 16)
    let refs = Array(repeating: H3QwenRequest.Reference.image(grid: grid), count: 9)
    let request = try H3QwenRequest.references(prompt: "x", references: refs,
      tokenizer: tokenizer)
    XCTAssertEqual(request.visualRanges.count, 9)
    XCTAssertEqual(request.tags.filter { $0 == 0 }.count, 9 * (64 + 2))
    XCTAssertLessThanOrEqual(request.tags.count, 1024)
    XCTAssertThrowsError(try H3QwenRequest.references(prompt: String(repeating: "x", count: 1000),
      references: refs, tokenizer: tokenizer))
  }
}
