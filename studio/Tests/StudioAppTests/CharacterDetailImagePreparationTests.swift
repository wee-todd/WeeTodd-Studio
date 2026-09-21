import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import WeeToddStudio

final class CharacterDetailImagePreparationTests: XCTestCase {
  func testHeadContextIncludesScalpSidesAndUpperNeckWithinBounds() {
    let face = CGRect(x: 0.40, y: 0.35, width: 0.20, height: 0.25)
    let crop = CharacterDetailImagePreparation.headContextRect(
      normalizedVisionFace: face, imageWidth: 4000, imageHeight: 3000)
    XCTAssertEqual(crop, CGRect(x: 1080, y: 450, width: 1840, height: 2325))
    XCTAssertGreaterThan(crop.minX, 0)
    XCTAssertLessThanOrEqual(crop.maxX, 4000)
    XCTAssertLessThanOrEqual(crop.maxY, 3000)
  }

  func testEdgeFaceCropClampsToFourKSourceBounds() {
    let crop = CharacterDetailImagePreparation.headContextRect(
      normalizedVisionFace: CGRect(x: 0.01, y: 0.70, width: 0.16, height: 0.25),
      imageWidth: 2160, imageHeight: 3840)
    XCTAssertEqual(crop.minX, 0)
    XCTAssertEqual(crop.minY, 0)
    XCTAssertLessThanOrEqual(crop.maxX, 2160)
    XCTAssertLessThanOrEqual(crop.maxY, 3840)
  }

  func testMissingAndMultipleFacesReturnOverviewOnlyDiagnostics() async throws {
    let source = try makeJPEG(width: 64, height: 48)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let output = source.deletingLastPathComponent().appendingPathComponent("details")
    let missing = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [])
    XCTAssertNil(missing.detailImage)
    XCTAssertTrue(missing.diagnostics.contains { $0.contains("No face") })
    let multiple = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output,
      normalizedFaces: [CGRect(x: 0.1, y: 0.2, width: 0.2, height: 0.3),
                        CGRect(x: 0.6, y: 0.2, width: 0.2, height: 0.3)])
    XCTAssertNil(multiple.detailImage)
    XCTAssertTrue(multiple.diagnostics.contains { $0.contains("2 faces") })
  }

  func testSingleFaceProducesStableBoundedDetailAndManifest() async throws {
    let source = try makeJPEG(width: 4000, height: 3000, orientation: 6)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let output = source.deletingLastPathComponent().appendingPathComponent("details")
    let face = CGRect(x: 0.35, y: 0.36, width: 0.30, height: 0.28)
    let first = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face])
    let second = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face])
    let detail = try XCTUnwrap(first.detailImage)
    XCTAssertEqual(detail.label, "primary head and face detail")
    XCTAssertEqual(detail.path, second.detailImage?.path)
    XCTAssertEqual(detail.sha256, second.detailImage?.sha256)
    XCTAssertEqual(detail.sha256.count, 64)
    let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(
      try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: detail.path) as CFURL, nil)), 0, nil)
      as? [CFString: Any])
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
    XCTAssertTrue(FileManager.default.fileExists(atPath:
      URL(fileURLWithPath: detail.path).deletingLastPathComponent().appendingPathComponent("manifest.json").path))
  }

  func testFourKClothingDetailIsBoundedAndRequiresUniqueWearer() async throws {
    let source = try makeJPEG(width: 4000, height: 3000)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let output = source.deletingLastPathComponent().appendingPathComponent("details")
    let face = CGRect(x: 0.2, y: 0.65, width: 0.15, height: 0.2)
    let person = CGRect(x: 0.05, y: 0.05, width: 0.65, height: 0.9)
    let prepared = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face], normalizedPeople: [person])
    let clothing = try XCTUnwrap(prepared.clothingDetailImage)
    XCTAssertEqual(prepared.sourceDetailImages.count, 2)
    XCTAssertEqual(clothing.label, "primary torso and lap detail")
    let image = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: clothing.path) as CFURL, nil))
    let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any])
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
    let ambiguous = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face],
      normalizedPeople: [person, CGRect(x: 0.45, y: 0.05, width: 0.4, height: 0.7)])
    XCTAssertNil(ambiguous.clothingDetailImage)
    XCTAssertNotNil(ambiguous.detailImage)
    XCTAssertTrue(ambiguous.diagnostics.contains { $0.contains("ambiguous") })
  }

  func testClothingCropKeepsTorsoAndLapWithoutFaceOrOutsidePixels() {
    let crop = CharacterDetailImagePreparation.clothingContextRect(
      normalizedVisionFace: CGRect(x: 0.2, y: 0.65, width: 0.15, height: 0.2),
      normalizedPeople: [CGRect(x: 0.05, y: 0.05, width: 0.65, height: 0.9)],
      imageWidth: 4000, imageHeight: 3000)
    XCTAssertEqual(crop, CGRect(x: 200, y: 1050, width: 2600, height: 1800))
    XCTAssertNil(CharacterDetailImagePreparation.clothingContextRect(
      normalizedVisionFace: CGRect(x: 0.2, y: 0.65, width: 0.15, height: 0.2),
      normalizedPeople: [], imageWidth: 4000, imageHeight: 3000))
  }

  private func makeJPEG(width: Int, height: Int, orientation: Int = 1) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("source.jpg")
    let bytes = Data(repeating: 180, count: width * height * 3)
    let provider = try XCTUnwrap(CGDataProvider(data: bytes as CFData))
    let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8,
      bitsPerPixel: 24, bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
      url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, [
      kCGImageDestinationLossyCompressionQuality: 0.9,
      kCGImagePropertyOrientation: orientation,
    ] as CFDictionary)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    return url
  }
}
