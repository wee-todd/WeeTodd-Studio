import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import WeeToddStudio

final class CharacterDetailImagePreparationTests: XCTestCase {
  private enum SimulatedVisionError: Error { case segmentationFailed }

  func testConfiguredLiveAppleVisionSubjectIsolation() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let sourcePath = environment["WEETODD_CHARACTER_DETAIL_SOURCE"],
          let outputPath = environment["WEETODD_CHARACTER_DETAIL_OUTPUT"] else {
      throw XCTSkip("Set character detail source/output to run Apple Vision qualification.")
    }
    let result = try await CharacterDetailImagePreparation().prepare(
      source: URL(fileURLWithPath: sourcePath), outputDirectory: URL(fileURLWithPath: outputPath))
    let overview = try XCTUnwrap(result.subjectOverviewImage)
    XCTAssertTrue(FileManager.default.fileExists(atPath: overview.path))
    for image in [result.subjectOverviewImage, result.detailImage, result.clothingDetailImage].compactMap({ $0 }) {
      let source = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: image.path) as CFURL, nil))
      let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
      XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
      XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
    }
    print("CHARACTER_DETAIL_LIVE_RESULT \(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))")
  }
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
    XCTAssertNotNil(missing.subjectOverviewImage)
    XCTAssertTrue(missing.diagnostics.contains { $0.contains("No face") })
    let multiple = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output,
      normalizedFaces: [CGRect(x: 0.1, y: 0.2, width: 0.2, height: 0.3),
                        CGRect(x: 0.6, y: 0.2, width: 0.2, height: 0.3)])
    XCTAssertNil(multiple.detailImage)
    XCTAssertNotNil(multiple.subjectOverviewImage)
    XCTAssertTrue(multiple.diagnostics.contains { $0.contains("2 faces") })
  }

  func testPrimarySubjectMaskWhiteMattesOverviewAndDrivesSameSubjectDetails() async throws {
    let source = try makeJPEG(width: 100, height: 100)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let output = source.deletingLastPathComponent().appendingPathComponent("isolated")
    var mask = Array(repeating: UInt8(0), count: 10_000)
    for y in 10..<95 { for x in 20..<80 { mask[y * 100 + x] = 255 } }
    let result = try await CharacterDetailImagePreparation.prepare(source: source,
      outputDirectory: output,
      normalizedFaces: [CGRect(x: 0.40, y: 0.62, width: 0.20, height: 0.20)],
      normalizedPeople: [CGRect(x: 0.20, y: 0.05, width: 0.60, height: 0.90)],
      subjectMask: mask, subjectMaskWidth: 100, subjectMaskHeight: 100)
    let overview = try XCTUnwrap(result.subjectOverviewImage)
    XCTAssertEqual(overview.label, "isolated primary character overview")
    XCTAssertNotNil(result.detailImage)
    XCTAssertNotNil(result.clothingDetailImage)
    let imageSource = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: overview.path) as CFURL, nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(imageSource, 0, nil))
    XCTAssertLessThan(image.width, 100) // Subject crop removes unrelated horizontal background.
    let rgba = try rgbaBytes(image)
    XCTAssertGreaterThanOrEqual(rgba[0], 245)
    XCTAssertGreaterThanOrEqual(rgba[1], 245)
    XCTAssertGreaterThanOrEqual(rgba[2], 245)
    XCTAssertEqual(overview.sha256.count, 64)
  }

  func testMissingSubjectMaskUsesBoundedFallbackAndActionDiagnostic() async throws {
    let source = try makeJPEG(width: 4000, height: 3000)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let result = try await CharacterDetailImagePreparation.prepare(source: source,
      outputDirectory: source.deletingLastPathComponent().appendingPathComponent("fallback"),
      normalizedFaces: [CGRect(x: 0.4, y: 0.5, width: 0.2, height: 0.25)],
      normalizedPeople: [], subjectMask: nil, subjectMaskWidth: 0, subjectMaskHeight: 0)
    let overview = try XCTUnwrap(result.subjectOverviewImage)
    XCTAssertTrue(result.diagnostics.contains { $0.contains("could not isolate") })
    let imageSource = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: overview.path) as CFURL, nil))
    let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any])
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
  }

  func testVisionSegmentationFailureProducesBoundedOverviewDiagnostic() async throws {
    let source = try makeJPEG(width: 1200, height: 800)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let service = CharacterDetailImagePreparation { _ in
      throw SimulatedVisionError.segmentationFailed
    }
    let result = try await service.prepare(source: source,
      outputDirectory: source.deletingLastPathComponent().appendingPathComponent("vision-fallback"))
    let overview = try XCTUnwrap(result.subjectOverviewImage)
    XCTAssertEqual(overview.label, CharacterDetailImagePreparation.boundedOverviewLabel)
    XCTAssertTrue(result.diagnostics.contains { $0.contains("subject isolation failed") })
    XCTAssertTrue(result.diagnostics.contains { $0.contains("could not isolate") })
    let imageSource = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: overview.path) as CFURL, nil))
    let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any])
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
  }

  func testVisionCancellationRemainsFatalInsteadOfProducingFallback() async throws {
    let source = try makeJPEG(width: 64, height: 48)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let service = CharacterDetailImagePreparation { _ in throw CancellationError() }
    do {
      _ = try await service.prepare(source: source,
        outputDirectory: source.deletingLastPathComponent().appendingPathComponent("cancelled"))
      XCTFail("Cancellation must not be converted to a bounded fallback.")
    } catch is CancellationError {
      // Expected.
    }
  }

  func testSingleFaceProducesStableBoundedDetailAndManifest() async throws {
    let source = try makeJPEG(width: 4000, height: 3000, orientation: 6)
    defer { try? FileManager.default.removeItem(at: source.deletingLastPathComponent()) }
    let output = source.deletingLastPathComponent().appendingPathComponent("details")
    let face = CGRect(x: 0.35, y: 0.36, width: 0.30, height: 0.28)
    let mask = Array(repeating: UInt8(255), count: 1200 * 1600)
    let first = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face],
      subjectMask: mask, subjectMaskWidth: 1200, subjectMaskHeight: 1600)
    let second = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face],
      subjectMask: mask, subjectMaskWidth: 1200, subjectMaskHeight: 1600)
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
    let mask = Array(repeating: UInt8(255), count: 1600 * 1200)
    let prepared = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face], normalizedPeople: [person],
      subjectMask: mask, subjectMaskWidth: 1600, subjectMaskHeight: 1200)
    let clothing = try XCTUnwrap(prepared.clothingDetailImage)
    XCTAssertEqual(prepared.sourceDetailImages.count, 2)
    XCTAssertEqual(clothing.label, "primary torso and lap detail")
    let image = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: clothing.path) as CFURL, nil))
    let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any])
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelWidth] as? Int ?? 9999, 512)
    XCTAssertLessThanOrEqual(properties[kCGImagePropertyPixelHeight] as? Int ?? 9999, 512)
    let ambiguous = try await CharacterDetailImagePreparation.prepare(
      source: source, outputDirectory: output, normalizedFaces: [face],
      normalizedPeople: [person, CGRect(x: 0.45, y: 0.05, width: 0.4, height: 0.7)],
      subjectMask: mask, subjectMaskWidth: 1600, subjectMaskHeight: 1200)
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

  private func rgbaBytes(_ image: CGImage) throws -> [UInt8] {
    var bytes = Array(repeating: UInt8(0), count: image.width * image.height * 4)
    let context = try XCTUnwrap(CGContext(data: &bytes, width: image.width, height: image.height,
      bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return bytes
  }
}
