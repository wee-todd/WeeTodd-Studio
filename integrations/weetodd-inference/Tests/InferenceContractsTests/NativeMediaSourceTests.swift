import CryptoKit
import Foundation
import XCTest
@testable import InferenceContracts

final class NativeMediaSourceTests: XCTestCase {
  func testVerifiesFrozenRegularFileAndRejectsReplacement() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let movie = folder.appendingPathComponent("source.mov")
    let original = Data("original movie bytes".utf8)
    try original.write(to: movie)
    let digest = SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined()
    let source = try NativeMediaSource(path: movie.path, sha256: digest)
    try source.verify()
    try Data("different movie bytes".utf8).write(to: movie)
    XCTAssertThrowsError(try source.verify())
  }

  func testRejectsSymlinkAndInvalidDigestBeforeReading() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let movie = folder.appendingPathComponent("source.mov")
    let alias = folder.appendingPathComponent("alias.mov")
    let bytes = Data("movie".utf8)
    try bytes.write(to: movie)
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: movie)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    XCTAssertThrowsError(try NativeMediaSource(path: movie.path, sha256: "invalid"))
    XCTAssertThrowsError(try NativeMediaSource(path: alias.path, sha256: digest).verify())
  }
}
