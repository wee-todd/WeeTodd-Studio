import AppKit
import CryptoKit
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import WeeToddStudio

private final class DigestReadCounter: @unchecked Sendable {
  private let lock = NSLock()
  private var count = 0
  func read(_ path: String) throws -> String {
    XCTAssertFalse(Thread.isMainThread, "File contents must be hashed off the UI thread")
    lock.lock(); count += 1; lock.unlock()
    return try AttachmentDigestStore.hashFile(path)
  }
  var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

final class MediaResourceTests: XCTestCase {
  private func folder() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }

  @MainActor func testDigestReadsAreSharedCachedAndInvalidatedByFileChanges() async throws {
    let file = try folder().appendingPathComponent("reference.bin")
    try Data("abc".utf8).write(to: file)
    let reads = DigestReadCounter()
    let cache = AttachmentDigestStore(reader: { try reads.read($0) })
    async let first: Void = cache.resolve([file.path])
    async let second: Void = cache.resolve([file.path])
    try await first; try await second
    XCTAssertTrue(cache.fingerprint(file.path).contains("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"))
    for _ in 0..<30 { _ = cache.fingerprint(file.path) }
    try await cache.resolve([file.path])
    XCTAssertEqual(reads.value, 1, "Display refreshes must reuse the file digest")
    let original = cache.fingerprint(file.path)
    try Data("abd".utf8).write(to: file, options: .atomic)
    try await cache.resolve([file.path])
    XCTAssertNotEqual(cache.fingerprint(file.path), original)
    XCTAssertEqual(reads.value, 2)
    try await cache.resolve([file.path], force: true)
    XCTAssertEqual(reads.value, 3, "Submission can demand a fresh content verification")
  }

  @MainActor func testUnreadableAttachmentCannotPassPreparation() async throws {
    let file = try folder().appendingPathComponent("missing.png")
    let cache = AttachmentDigestStore()
    do {
      try await cache.resolve([file.path])
      XCTFail("Missing media must fail before submission")
    } catch { XCTAssertTrue(error.localizedDescription.contains("missing.png")) }
    XCTAssertTrue(cache.fingerprint(file.path).contains("missing"))
  }

  @MainActor func testFingerprintLookupDoesNotSynchronouslyReadContents() throws {
    let file = try folder().appendingPathComponent("reference.bin")
    try Data("abc".utf8).write(to: file)
    let reads = DigestReadCounter()
    let cache = AttachmentDigestStore(reader: { try reads.read($0) })
    for _ in 0..<20 { _ = cache.fingerprint(file.path) }
    XCTAssertEqual(reads.value, 0)
  }

  @MainActor func testExplicitPreparationRetriesTransientReadFailure() async throws {
    let file = try folder().appendingPathComponent("reference.bin")
    try Data("abc".utf8).write(to: file)
    let attempts = TransientDigestReader()
    let cache = AttachmentDigestStore(reader: { try attempts.read($0) })
    do { try await cache.resolve([file.path]); XCTFail("First read should fail") } catch {}
    try await cache.resolve([file.path])
    XCTAssertTrue(cache.fingerprint(file.path).contains("ba7816bf"))
  }

  func testThumbnailBoundsDecodedPixelsAndReusesResult() async throws {
    let file = try folder().appendingPathComponent("large.png")
    let context = try XCTUnwrap(CGContext(data: nil, width: 1800, height: 900,
      bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    let source = try XCTUnwrap(context.makeImage())
    let writer = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(writer, source, nil)
    XCTAssertTrue(CGImageDestinationFinalize(writer))
    let cache = ImageThumbnailCache(byteLimit: 4 * 1024 * 1024)
    let firstValue = await cache.image(path: file.path, maximumPixelSize: 256)
    let first = try XCTUnwrap(firstValue)
    XCTAssertEqual(first.width, 256)
    XCTAssertEqual(first.height, 128)
    let second = await cache.image(path: file.path, maximumPixelSize: 256)
    XCTAssertTrue(first === second)
    let preview = await cache.image(path: file.path, maximumPixelSize: 512)
    XCTAssertEqual(preview?.width, 512)
  }

  func testThumbnailCacheBoundsResidentBytes() async throws {
    let file = try folder().appendingPathComponent("source.png")
    let context = try XCTUnwrap(CGContext(data: nil, width: 800, height: 800,
      bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    let writer = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(writer, try XCTUnwrap(context.makeImage()), nil)
    XCTAssertTrue(CGImageDestinationFinalize(writer))
    let cache = ImageThumbnailCache(byteLimit: 300_000)
    for size in [100, 150, 200, 256] { _ = await cache.image(path: file.path, maximumPixelSize: size) }
    let cost = await cache.residentBytes
    XCTAssertLessThanOrEqual(cost, 300_000)
  }
}

private final class TransientDigestReader: @unchecked Sendable {
  private let lock = NSLock()
  private var first = true
  func read(_ path: String) throws -> String {
    lock.lock(); let fail = first; first = false; lock.unlock()
    if fail { throw CocoaError(.fileReadUnknown) }
    return try AttachmentDigestStore.hashFile(path)
  }
}
