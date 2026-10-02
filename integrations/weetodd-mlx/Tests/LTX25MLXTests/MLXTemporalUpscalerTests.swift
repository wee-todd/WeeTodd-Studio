import XCTest
import Foundation
import MLX
import TensorIO
@testable import LTX25MLX

final class MLXTemporalUpscalerTests: XCTestCase {
  func testTemporalShuffleDropsTheCausalLeadingFrame() throws {
    let packed = MLXArray((0..<8).map(Float.init), [2, 1, 1, 4])
    let shuffled = MLXTemporalUpscaler.temporalShuffle(packed)
    XCTAssertEqual(shuffled.shape, [4, 1, 1, 2])
    XCTAssertEqual(shuffled.asArray(Float.self), [0, 2, 1, 3, 4, 6, 5, 7])
    XCTAssertEqual(shuffled[1..<shuffled.shape[0]].asArray(Float.self), [1, 3, 4, 6, 5, 7])
  }

  func testTemporalAdmissionRejectsExcessiveVolumeBeforeWeightRead() throws {
    XCTAssertEqual(try MLXTemporalUpscaler.admit(shape: [7, 4, 4, 128],
      maximumActivationBytes: 2 * 1024 * 1024 * 1024), [13, 4, 4, 128])
    XCTAssertThrowsError(try MLXTemporalUpscaler.admit(shape: [7, 4, 4, 128],
      maximumActivationBytes: 1))
    XCTAssertThrowsError(try MLXTemporalUpscaler.admit(shape: [Int.max, 4, 4, 128],
      maximumActivationBytes: Int.max))
  }

  func testInstalledTemporalNetworkMatchesIndependentOracle() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_TEMPORAL_UPSCALER"],
      let stats = env["WEETODD_VIDEO_VAE"],
      let oracle = env["WEETODD_TEMPORAL_UPSCALER_ORACLE"] else {
      throw XCTSkip("Installed temporal upscaler parity is opt-in.")
    }
    let fixture = try SafeTensorFile(url: URL(fileURLWithPath: oracle))
    let input = try MLXWeight.read(fixture, "input")
    let expected = try fixture.readFloat32(named: "output")
    let model = try MLXTemporalUpscaler(checkpoint: URL(fileURLWithPath: checkpoint),
      statisticsCheckpoint: URL(fileURLWithPath: stats))
    XCTAssertThrowsError(try model.upscale(input, maximumActivationBytes: 1))
    let oldCache = Memory.cacheLimit
    XCTAssertThrowsError(try model.upscale(input, progress: { _ in throw CancellationError() }))
    XCTAssertEqual(model.residentWeightBytes, 0)
    XCTAssertEqual(Memory.cacheLimit, oldCache)
    let actual = try model.upscale(input).asArray(Float.self)
    let maxError = zip(actual, expected).map { abs($0 - $1) }.max()!
    let squared = zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
    let norm = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
    print("MLX_TEMPORAL_UPSCALER maxabs=\(maxError) rel=\(sqrt(squared / norm))")
    XCTAssertEqual(actual.count, expected.count)
    XCTAssertLessThan(maxError, 0.002)
    XCTAssertLessThan(sqrt(squared / norm), 0.0001)
  }
}
