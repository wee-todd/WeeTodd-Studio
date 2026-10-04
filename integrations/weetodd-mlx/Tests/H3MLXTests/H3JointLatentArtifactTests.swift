import CryptoKit
import Foundation
import XCTest
@testable import H3MLX

final class H3JointLatentArtifactTests: XCTestCase {
  private let identity = String(repeating: "a", count: 64)
  private func fixture() throws -> (URL, H3Geometry, H3JointLatentArtifact.Rows) {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let rows = H3JointLatentArtifact.Rows(video: (0..<(geometry.videoRows * 96)).map { Float($0) / 16 },
      audio: (0..<(geometry.audioRows * 32)).map { -Float($0) / 8 })
    return (directory, geometry, rows)
  }
  func testFullJointRowsRoundTripPreservesEveryChannelAndBitWithoutTailTruncation() throws {
    let (directory, geometry, rows) = try fixture(); defer { try? FileManager.default.removeItem(at: directory) }
    let hashes = try H3JointLatentArtifact.save(rows: rows, geometry: geometry,
      task: "ref2va", componentIdentity: identity, directory: directory)
    let manifest = directory.appendingPathComponent("manifest.json")
    let loaded = try H3JointLatentArtifact.load(manifestURL: manifest,
      expectedSHA256: hashes.manifestSHA256, expectedTask: "ref2va", expectedComponentIdentity: identity)
    XCTAssertEqual(loaded.0.generatedFrames, 73)
    XCTAssertEqual(loaded.1.video.map(\.bitPattern), rows.video.map(\.bitPattern))
    XCTAssertEqual(loaded.1.audio.map(\.bitPattern), rows.audio.map(\.bitPattern))
    XCTAssertThrowsError(try H3JointLatentArtifact.save(rows: rows, geometry: geometry,
      task: "ref2va", componentIdentity: identity, directory: directory))
    XCTAssertEqual(try Data(contentsOf: manifest).count > 0, true)
  }
  func testTaskIdentityPayloadMutationAndSymlinkAreRejectedWithoutWeights() throws {
    let (directory, geometry, rows) = try fixture(); defer { try? FileManager.default.removeItem(at: directory) }
    let hashes = try H3JointLatentArtifact.save(rows: rows, geometry: geometry,
      task: "t2va", componentIdentity: identity, directory: directory)
    let manifest = directory.appendingPathComponent("manifest.json")
    XCTAssertThrowsError(try H3JointLatentArtifact.load(manifestURL: manifest,
      expectedSHA256: hashes.manifestSHA256, expectedTask: "fl2va", expectedComponentIdentity: identity))
    XCTAssertThrowsError(try H3JointLatentArtifact.load(manifestURL: manifest,
      expectedSHA256: hashes.manifestSHA256, expectedTask: "t2va", expectedComponentIdentity: String(repeating: "b", count: 64)))
    let payload = directory.appendingPathComponent("joint-latents.f32")
    var changed = try Data(contentsOf: payload); changed[0] ^= 1; try changed.write(to: payload)
    XCTAssertThrowsError(try H3JointLatentArtifact.load(manifestURL: manifest,
      expectedSHA256: hashes.manifestSHA256, expectedTask: "t2va", expectedComponentIdentity: identity))
    let alias = directory.appendingPathComponent("alias.json")
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: manifest)
    XCTAssertThrowsError(try H3JointLatentArtifact.inspect(manifestURL: alias,
      expectedSHA256: hashes.manifestSHA256, expectedTask: "t2va", expectedComponentIdentity: identity))
  }
  func testContinuationTailAndNonfiniteSourcesCannotBePublishedAsCompleteJointLatents() throws {
    let (_, geometry, rows) = try fixture()
    XCTAssertThrowsError(try H3JointLatentArtifact.Rows(video: Array(rows.video.prefix(192)), audio: rows.audio).validate(geometry: geometry))
    var video = rows.video; video[0] = .infinity
    XCTAssertThrowsError(try H3JointLatentArtifact.Rows(video: video, audio: rows.audio).validate(geometry: geometry))
  }
}
