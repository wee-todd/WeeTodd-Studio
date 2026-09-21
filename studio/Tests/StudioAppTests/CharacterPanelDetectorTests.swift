import XCTest
import StudioCore
@testable import WeeToddStudio

final class CharacterPanelDetectorTests: XCTestCase {
  func testNarrowGapBetweenSeparateVisionInstancesRemainsAvailableForCropping() {
    let components = CharacterPanelDetector.mergedForegroundComponents([
      .init(x: 56, y: 30, width: 322, height: 840),
      .init(x: 425, y: 40, width: 189, height: 820),
      .init(x: 653, y: 30, width: 322, height: 840),
      .init(x: 978, y: 20, width: 622, height: 887),
      .init(x: 664, y: 50, width: 302, height: 810),
    ])
    let result = CharacterPanelLayout.detect(imageWidth: 1600, imageHeight: 907,
      foregroundComponents: components, edgeColumns: Array(repeating: 0, count: 1600), detectionRevision: 1)
    XCTAssertEqual(result.status, .detected)
    XCTAssertEqual(result.candidates.count, 4)
    XCTAssertEqual(result.candidates[2].sourcePixelRect.maxX, 976)
  }

  func testHumanDetectionsRecoverBodiesOmittedBySalientCloseUpMask() {
    let closeUp = PanelPixelRect(x: 1130, y: 20, width: 790, height: 1068)
    let bodies = [PanelPixelRect(x: 75, y: 60, width: 310, height: 980),
      PanelPixelRect(x: 510, y: 60, width: 190, height: 980),
      PanelPixelRect(x: 795, y: 60, width: 310, height: 980)]
    let components = CharacterPanelDetector.mergedForegroundComponents([closeUp] + bodies)
    let result = CharacterPanelLayout.detect(imageWidth: 1920, imageHeight: 1088,
      foregroundComponents: components, edgeColumns: Array(repeating: 0, count: 1920), detectionRevision: 1)
    XCTAssertEqual(result.status, .detected)
    XCTAssertEqual(result.candidates.count, 4)
    XCTAssertEqual(result.candidates.map(\.sourcePixelRect.maxX), [447, 747, 1117, 1920])
  }

  func testOverlappingHumanAndMaskBoundsFuseWithoutDuplicatingPanels() {
    let components = CharacterPanelDetector.mergedForegroundComponents([
      .init(x: 10, y: 10, width: 90, height: 180),
      .init(x: 20, y: 5, width: 75, height: 190),
      .init(x: 130, y: 10, width: 60, height: 180),
    ])
    XCTAssertEqual(components, [.init(x: 10, y: 5, width: 90, height: 190),
      .init(x: 130, y: 10, width: 60, height: 180)])
  }

  func testPixelGradientMeasuresEmptyGutterBetweenForegroundEdges() throws {
    let context = try XCTUnwrap(CGContext(data: nil, width: 100, height: 100, bitsPerComponent: 8,
      bytesPerRow: 400, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fill(CGRect(x: 10, y: 0, width: 20, height: 100))
    context.fill(CGRect(x: 60, y: 0, width: 20, height: 100))
    let edges = try CharacterPanelDetector.horizontalEdgeColumns(XCTUnwrap(context.makeImage()))
    XCTAssertGreaterThan(edges[10], 0.99)
    XCTAssertGreaterThan(edges[60], 0.99)
    XCTAssertEqual(edges[45], 0)
  }

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
