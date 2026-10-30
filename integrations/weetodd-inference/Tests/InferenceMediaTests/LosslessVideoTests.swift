import XCTest
import Foundation
import CryptoKit
@testable import InferenceMedia

final class LosslessVideoTests: XCTestCase {
  func testFFV1PreservesEveryRGBByteAndFractionalClock() throws {
    let binary = "/opt/homebrew/bin/ffmpeg"
    guard FileManager.default.isExecutableFile(atPath: binary) else { throw XCTSkip("Installed FFmpeg unavailable") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let video = root.appendingPathComponent("restored.mkv")
    let writer = try RawVideoWriter(ffmpeg: URL(fileURLWithPath: binary), output: video,
      width: 8, height: 4, frames: 7, fps: 30000.0/1001, encoding: .ffv1)
    var hash = SHA256()
    for frame in 0..<7 {
      let bytes = Data((0..<96).map { UInt8(($0*73 + frame*37) % 256) })
      hash.update(data: bytes); try writer.append(bytes, frame: frame)
    }
    try writer.finish()
    let sha = hash.finalize().map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(writer.rgbSHA256, sha)
    let proof = try LosslessVideoVerifier.verify(ffmpeg: URL(fileURLWithPath: binary), video: video,
      width: 8, height: 4, frames: 7, fps: 30000.0/1001, expectedRGBSHA256: sha)
    XCTAssertEqual(proof.frames, 7); XCTAssertEqual(proof.fpsNumerator, 30000); XCTAssertEqual(proof.fpsDenominator, 1001)
    XCTAssertThrowsError(try LosslessVideoVerifier.verify(ffmpeg: URL(fileURLWithPath: binary), video: video,
      width: 8, height: 4, frames: 6, fps: 30000.0/1001, expectedRGBSHA256: sha))
    XCTAssertThrowsError(try LosslessVideoVerifier.verify(ffmpeg: URL(fileURLWithPath: binary), video: video,
      width: 8, height: 4, frames: 7, fps: 24, expectedRGBSHA256: sha))
    XCTAssertThrowsError(try LosslessVideoVerifier.verify(ffmpeg: URL(fileURLWithPath: binary), video: video,
      width: 8, height: 4, frames: 7, fps: 30000.0/1001, expectedRGBSHA256: String(repeating: "0", count: 64)))
  }
}
