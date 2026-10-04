import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import XCTest
@testable import H3MLX

final class H3ReferencePreparationMediaTests: XCTestCase {
  func testExplicitStillAreaBudgetChangesRealPreparedCanvasWithoutChangingSource() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("source.png")
    let context = try XCTUnwrap(CGContext(data: nil, width: 640, height: 320,
      bitsPerComponent: 8, bytesPerRow: 640 * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 640, height: 320))
    let image = try XCTUnwrap(context.makeImage())
    let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL,
      "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
    let sourceBytes = try Data(contentsOf: file)
    let geometry = try H3Geometry(width: 384, height: 256, durationSeconds: 2.5)
    let legacy = try H3StillReferenceMedia.load(path: file.path)
    let explicit = try H3StillReferenceMedia.load(path: file.path,
      outputGeometry: geometry, pixelBudgetPercent: 100)
    XCTAssertEqual(legacy.reference.width, 320); XCTAssertEqual(legacy.reference.height, 160)
    XCTAssertEqual(explicit.reference.width, 448); XCTAssertEqual(explicit.reference.height, 224)
    XCTAssertEqual(explicit.reference.pixelBudgetPercent, 100)
    XCTAssertNotEqual(explicit.preparationPolicy, legacy.preparationPolicy)
    XCTAssertEqual(explicit.sourceSHA256, legacy.sourceSHA256)
    XCTAssertEqual(try Data(contentsOf: file), sourceBytes)
    XCTAssertNoThrow(try H3VideoReferencePreparation.validate([.image(explicit.reference)]))
    XCTAssertThrowsError(try H3StillReferenceMedia.load(path: file.path, pixelBudgetPercent: 100))
  }
  func testMoviePolicyPreservesAspectAndFullSourceWhileReducingOnlyPersistentFrames() throws {
    let ffmpeg = ProcessInfo.processInfo.environment["WEETODD_FFMPEG"] ?? "/opt/homebrew/bin/ffmpeg"
    guard FileManager.default.isExecutableFile(atPath: ffmpeg) else { throw XCTSkip("Native FFmpeg is unavailable.") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("source.mp4")
    let process = Process(); process.executableURL = URL(fileURLWithPath: ffmpeg)
    process.arguments = ["-v", "error", "-nostdin", "-f", "lavfi", "-i",
      "testsrc2=size=256x128:rate=24:duration=3", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", movie.path]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
    let geometry = try H3Geometry(width: 384, height: 256, durationSeconds: 2.5)
    let settings = try H3ReferencePreparationControls(videoSizePolicy: .matchOutput, temporalDensity: .half)
    let loaded = try H3VideoReferenceMedia.load(path: movie.path, ffmpeg: URL(fileURLWithPath: ffmpeg),
      outputGeometry: geometry, controls: settings)
    XCTAssertEqual(loaded.reference.width, 256); XCTAssertEqual(loaded.reference.height, 128)
    XCTAssertEqual(loaded.decodedFrames, 72); XCTAssertEqual(loaded.reference.frameCount, 56)
    XCTAssertEqual(loaded.reference.rgb8.count, 56 * 256 * 128 * 3)
    XCTAssertEqual(loaded.reference.persistentFrameCount, 22)
    XCTAssertEqual(loaded.reference.sourceLatentFrames, 17)
    XCTAssertEqual(loaded.reference.temporalDecision?.indices.first, 0)
    XCTAssertEqual(loaded.reference.temporalDecision?.indices.last, 55)
    XCTAssertNoThrow(try H3VideoReferencePreparation.validate([.video(loaded.reference)]))
    let legacy = try H3VideoReferenceMedia.load(path: movie.path, ffmpeg: URL(fileURLWithPath: ffmpeg))
    XCTAssertEqual(legacy.reference.width, 256); XCTAssertEqual(legacy.reference.height, 256)
    XCTAssertEqual(legacy.reference.persistentFrameCount, legacy.reference.frameCount)
    XCTAssertNil(legacy.reference.temporalDecision)
  }
}
