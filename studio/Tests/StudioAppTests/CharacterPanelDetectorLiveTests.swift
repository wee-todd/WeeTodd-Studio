import CoreGraphics
import ImageIO
import StudioCore
import XCTest
@testable import WeeToddStudio

/// Opt-in platform qualification. The source portrait remains outside the repository.
final class CharacterPanelDetectorLiveTests: XCTestCase {
  func testGeneratedSheetWithNarrowGutterKeepsSeparateVisionInstances() async throws {
    guard let sourcePath = ProcessInfo.processInfo.environment["WEETODD_NARROW_PANEL_FIXTURE"] else {
      throw XCTSkip("Set WEETODD_NARROW_PANEL_FIXTURE to the external narrow-gutter sheet.")
    }
    let result = try await CharacterPanelDetector().detect(source: URL(fileURLWithPath: sourcePath))
    print("Narrow-gutter generated sheet detection: \(result)")
    XCTAssertEqual(result.status, .detected, result.diagnostics.joined(separator: " "))
    XCTAssertEqual(result.candidates.count, 4)
    for (candidate, expected) in zip(result.candidates, [480, 745, 1177, 1920]) {
      XCTAssertLessThanOrEqual(abs(candidate.sourcePixelRect.maxX - expected), 45)
      XCTAssertEqual(candidate.sourcePixelRect.height, 1088)
    }
  }

  func testActualGeneratedSheetFindsFourContentBoundedPanels() async throws {
    guard let sourcePath = ProcessInfo.processInfo.environment["WEETODD_GENERATED_PANEL_FIXTURE"] else {
      throw XCTSkip("Set WEETODD_GENERATED_PANEL_FIXTURE to the external generated sheet.")
    }
    let result = try await CharacterPanelDetector().detect(source: URL(fileURLWithPath: sourcePath))
    print("Generated sheet panel detection: \(result)")
    XCTAssertEqual(result.status, .detected, result.diagnostics.joined(separator: " "))
    XCTAssertEqual(result.candidates.count, 4)
    let expectedBoundaries = [450, 764, 1128, 1920]
    for (candidate, expected) in zip(result.candidates, expectedBoundaries) {
      XCTAssertLessThanOrEqual(abs(candidate.sourcePixelRect.maxX - expected), 65)
      XCTAssertEqual(candidate.sourcePixelRect.height, 1088)
    }
  }

  func testActualVisionFindsUnequalSourceCoordinatePanels() async throws {
    guard let sourcePath = ProcessInfo.processInfo.environment["WEETODD_PANEL_VISION_FIXTURE"] else {
      throw XCTSkip("Set WEETODD_PANEL_VISION_FIXTURE to an external portrait image.")
    }
    let source = URL(fileURLWithPath: sourcePath)
    guard FileManager.default.fileExists(atPath: source.path) else {
      throw XCTSkip("The external Vision fixture does not exist at \(source.path).")
    }
    let output = FileManager.default.temporaryDirectory
      .appendingPathComponent("character-panel-vision-\(UUID().uuidString).png")
    defer { try? FileManager.default.removeItem(at: output) }

    let expected = [
      PanelPixelRect(x: 0, y: 0, width: 385, height: 1088),
      PanelPixelRect(x: 385, y: 0, width: 475, height: 1088),
      PanelPixelRect(x: 860, y: 0, width: 555, height: 1088),
      PanelPixelRect(x: 1415, y: 0, width: 505, height: 1088),
    ]
    try makeUnequalFourPanelFixture(source: source, output: output)
    let result = try await CharacterPanelDetector().detect(source: output)
    print("Actual Vision panel detection: status=\(result.status.rawValue) crops=\(result.candidates.map(\.sourcePixelRect)) diagnostics=\(result.diagnostics)")

    XCTAssertEqual(result.status, .detected, result.diagnostics.joined(separator: " "))
    XCTAssertEqual(result.candidates.count, 4, result.diagnostics.joined(separator: " "))
    XCTAssertEqual(result.candidates.map(\.role), [.front, .side, .back, .closeUp])
    for (actual, intended) in zip(result.candidates.map(\.sourcePixelRect), expected) {
      XCTAssertLessThanOrEqual(abs(actual.x - intended.x), 75, "actual crops: \(result.candidates.map(\.sourcePixelRect))")
      XCTAssertLessThanOrEqual(abs(actual.maxX - intended.maxX), 75, "actual crops: \(result.candidates.map(\.sourcePixelRect))")
      XCTAssertEqual(actual.y, 0)
      XCTAssertEqual(actual.height, 1088)
    }
    XCTAssertFalse(result.candidates.map(\.sourcePixelRect.width).allSatisfy { $0 == 480 },
      "Vision must return measured unequal crops, not fabricated image quarters.")
  }

  private func makeUnequalFourPanelFixture(source: URL, output: URL) throws {
    guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil),
          let portrait = CGImageSourceCreateImageAtIndex(imageSource, 0, nil),
          let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(data: nil, width: 1920, height: 1088, bitsPerComponent: 8,
            bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
      XCTFail("Could not construct the external live-test fixture.")
      return
    }
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 1920, height: 1088))
    let placements = [
      CGRect(x: 40, y: 110, width: 310, height: 860),
      CGRect(x: 420, y: 55, width: 400, height: 980),
      CGRect(x: 900, y: 85, width: 450, height: 920),
      CGRect(x: 1480, y: 165, width: 400, height: 760),
    ]
    for placement in placements { context.draw(portrait, in: placement) }
    guard let image = context.makeImage(),
          let destination = CGImageDestinationCreateWithURL(output as CFURL, "public.png" as CFString, 1, nil) else {
      XCTFail("Could not encode the external live-test fixture.")
      return
    }
    CGImageDestinationAddImage(destination, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
  }
}
