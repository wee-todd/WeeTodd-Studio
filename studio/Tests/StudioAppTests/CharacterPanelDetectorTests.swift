import XCTest
import StudioCore
@testable import WeeToddStudio

final class CharacterPanelDetectorTests: XCTestCase {
  func testVisionBottomLeftRectangleMapsToTopLeftPixels() {
    let rect = CharacterPanelDetector.pixelRect(normalizedX: 0.20, normalizedY: 0.10,
      normalizedWidth: 0.30, normalizedHeight: 0.40, imageWidth: 1000, imageHeight: 500)
    XCTAssertEqual(rect, PanelPixelRect(x: 200, y: 250, width: 300, height: 200))
  }

  func testProxyRectScalesToOrientationNormalizedSourcePixels() {
    XCTAssertEqual(CharacterPanelDetector.sourceRect(
      PanelPixelRect(x: 800, y: 100, width: 400, height: 600),
      proxyWidth: 1600, proxyHeight: 907, sourceWidth: 1920, sourceHeight: 1088),
      PanelPixelRect(x: 960, y: 120, width: 480, height: 720))
  }

  func testFloat32VisionMaskUsesFormatAndRowStride() throws {
    var buffer: CVPixelBuffer?
    XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 2, 2,
      kCVPixelFormatType_OneComponent32Float, nil, &buffer), kCVReturnSuccess)
    let pixels = buffer!
    CVPixelBufferLockBaseAddress(pixels, [])
    let stride = CVPixelBufferGetBytesPerRow(pixels) / MemoryLayout<Float32>.size
    let values = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: Float32.self)
    values[0] = 0; values[1] = 0.5; values[stride] = 1; values[stride + 1] = 0.25
    CVPixelBufferUnlockBaseAddress(pixels, [])
    XCTAssertEqual(try VisionMaskPixels.bytes(from: pixels, expectedWidth: 2, expectedHeight: 2), [0, 128, 255, 64])
  }
}
