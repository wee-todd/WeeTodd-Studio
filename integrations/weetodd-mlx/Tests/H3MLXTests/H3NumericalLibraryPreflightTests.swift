import CryptoKit
import Foundation
import MLX
import XCTest
@testable import H3MLX

/// Explicit library binding and a small seeded numerical oracle before weighted qualification.
final class H3NumericalLibraryPreflightTests: XCTestCase {
  func testExplicitLibraryAndReleasedSeededOracle() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_NUMERICAL_PREFLIGHT_TEST"] == "1" else {
      throw XCTSkip("Explicit numerical/library qualification was not requested.")
    }
    let location = try XCTUnwrap(environment["WEETODD_H3_TEST_METALLIB"])
    let expectedSHA = try XCTUnwrap(environment["WEETODD_H3_TEST_METALLIB_SHA256"])
    let library = URL(fileURLWithPath: location)
    let bytes = try Data(contentsOf: library)
    let actualSHA = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(actualSHA, expectedSHA)
    guard actualSHA == expectedSHA else { return }
    GPU.metallib = library
    XCTAssertEqual(GPU.metallib?.standardizedFileURL, library.standardizedFileURL)
    let rows = try H3Noise.makeWithCondition(seed: 1234, conditionRows: 7,
      videoLatentFrames: 7, latentHeight: 2, latentWidth: 2, audioLatentFrames: 10)
    let expected: [(MLXArray, [Float])] = [
      (rows.condition, [0.39139548, 0.6809802, -2.8445895, -0.39998114]),
      (rows.video, [0.18150127, -0.40931788, 1.1606829, 0.06083998]),
      (rows.audio, [-0.54947805, 1.1330395, -0.1210604, -0.46126363])]
    for (array, oracle) in expected {
      let values = Array(array.reshaped([-1]).asArray(Float.self).prefix(4))
      for index in oracle.indices { XCTAssertEqual(values[index], oracle[index], accuracy: 0.000001) }
      XCTAssertTrue(array.asArray(Float.self).allSatisfy(\.isFinite))
    }
  }
}
