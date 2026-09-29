import Foundation
import XCTest
@testable import InferenceMedia

final class MediaOutputTests: XCTestCase {
  func testStereoWAVRetainsNativeH3AndLTXSampleRates() throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-wav-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    for rate in [32_000, 48_000] {
      let url = directory.appendingPathComponent("\(rate).wav")
      try MediaOutput.writeWAV(samples: [0.25, 0.5, -0.25, -0.5],
        sampleRate: rate, channels: 2, to: url)
      let bytes = try Data(contentsOf: url)
      XCTAssertEqual(bytes.count, 74)
      XCTAssertEqual(Array(bytes[24..<28]), withUnsafeBytes(of: UInt32(rate).littleEndian,
        Array.init))
      XCTAssertEqual(Array(bytes[28..<32]), withUnsafeBytes(of: UInt32(rate * 8).littleEndian,
        Array.init))
    }
  }
}
