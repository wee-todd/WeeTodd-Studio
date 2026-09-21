import CoreGraphics
import ImageIO
import XCTest
@testable import DrawThingsTransport

final class LocalVisionInputTests: XCTestCase {
  func testFourKOrientedReferenceUsesBoundedWholeImageGridWithoutMutatingOriginal() throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("local-vision-4k-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let imageURL = directory.appendingPathComponent("portrait-4k-oriented.jpg")
    try writeFourKFixture(to: imageURL, orientation: 6)
    let originalBytes = try Data(contentsOf: imageURL)
    let originalAttributes = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(
      try XCTUnwrap(CGImageSourceCreateWithURL(imageURL as CFURL, nil)), 0, nil) as? [CFString: Any])

    let input = LocalPromptImage(path: imageURL.path, label: "4K character reference")
    let prepared = try LocalVisionInput.prepare([input])
    let grid = try XCTUnwrap(prepared.grids.first)
    let request: [String: Any] = [
      "modelPath": "/missing/qwen_3.5_4b_i8x.ckpt", "systemPrompt": "Describe visible character details.",
      "prompt": "Return only supported observations.", "maxTokens": 256,
      "images": [["path": imageURL.path, "label": input.label]],
    ]
    let budget = try LocalTextGeneration.preflight(request)

    XCTAssertEqual(prepared.grids.count, 1)
    XCTAssertEqual(grid.t, 1)
    XCTAssertLessThanOrEqual(grid.h, 32, "The Qwen vision grid uses 16-pixel patches over a 512-pixel maximum edge.")
    XCTAssertLessThanOrEqual(grid.w, 32, "The Qwen vision grid uses 16-pixel patches over a 512-pixel maximum edge.")
    XCTAssertGreaterThan(grid.h, grid.w, "EXIF orientation 6 should make the prepared whole image portrait-shaped.")
    XCTAssertEqual(prepared.patches.shape[0], grid.t * grid.h * grid.w)
    XCTAssertEqual(budget["imagesUsed"] as? Int, 1)
    XCTAssertEqual(budget["imageTokens"] as? Int, grid.t * grid.h * grid.w / 4 - 1)
    XCTAssertLessThanOrEqual(budget["imageTokens"] as? Int ?? .max, 255)
    XCTAssertLessThan(budget["inputTokens"] as? Int ?? .max, budget["inputTokenLimit"] as? Int ?? 0)

    XCTAssertEqual(try Data(contentsOf: imageURL), originalBytes)
    let restoredSource = try XCTUnwrap(CGImageSourceCreateWithURL(imageURL as CFURL, nil))
    let restored = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(restoredSource, 0, nil) as? [CFString: Any])
    XCTAssertEqual(restored[kCGImagePropertyPixelWidth] as? Int, 4096)
    XCTAssertEqual(restored[kCGImagePropertyPixelHeight] as? Int, 2160)
    XCTAssertEqual(restored[kCGImagePropertyOrientation] as? Int, 6)
    XCTAssertEqual(restored[kCGImagePropertyPixelWidth] as? Int,
      originalAttributes[kCGImagePropertyPixelWidth] as? Int)
  }

  private func writeFourKFixture(to url: URL, orientation: Int) throws {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(data: nil, width: 4096, height: 2160, bitsPerComponent: 8,
            bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
      XCTFail("Could not allocate the synthetic 4K image."); return
    }
    context.setFillColor(CGColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 4096, height: 2160))
    context.setFillColor(CGColor(red: 0.15, green: 0.35, blue: 0.75, alpha: 1))
    context.fill(CGRect(x: 360, y: 240, width: 1150, height: 1680))
    context.setFillColor(CGColor(red: 0.80, green: 0.25, blue: 0.20, alpha: 1))
    context.fill(CGRect(x: 2520, y: 420, width: 1180, height: 1320))
    guard let image = context.makeImage(),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
      XCTFail("Could not encode the synthetic 4K image."); return
    }
    let properties: [CFString: Any] = [
      kCGImagePropertyOrientation: orientation,
      kCGImageDestinationLossyCompressionQuality: 0.8,
    ]
    CGImageDestinationAddImage(destination, image, properties as CFDictionary)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
  }
}
