import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3NoiseTests: XCTestCase {
  func testFL2VAConditionDrawPrecedesTargetVideoAndAudio() throws {
    let rows = try H3Noise.makeWithCondition(seed: 1234,
      conditionRows: 7, videoLatentFrames: 7,
      latentHeight: 2, latentWidth: 2, audioLatentFrames: 10)
    XCTAssertEqual(rows.condition.shape, [1, 7, 96])
    XCTAssertEqual(rows.video.shape, [1, 7, 96])
    XCTAssertEqual(rows.audio.shape, [1, 20, 32])
    // Independent Python MLX oracle: draw condition, then channel-major video,
    // then stereo audio from seed 1234, with H3's patchification/row order.
    let expected: [(MLXArray, [Float])] = [
      (rows.condition, [0.39139548, 0.6809802, -2.8445895, -0.39998114]),
      (rows.video, [0.18150127, -0.40931788, 1.1606829, 0.06083998]),
      (rows.audio, [-0.54947805, 1.1330395, -0.1210604, -0.46126363])
    ]
    for (actual, oracle) in expected {
      let prefix = Array(actual.reshaped([-1]).asArray(Float.self).prefix(4))
      for index in 0..<4 {
        XCTAssertEqual(prefix[index], oracle[index], accuracy: 0.000001)
      }
    }
  }

  func testSeededAVNoiseMatchesReleasedRowOrdering() throws {
    let fixture = URL(fileURLWithPath: "/tmp/weetodd-h3-noise")
    guard FileManager.default.fileExists(atPath:
      fixture.appendingPathComponent("video.f32").path) else {
      throw XCTSkip("H3 seeded noise reference fixture is unavailable.")
    }
    let result = try H3Noise.make(seed: 1234,
      videoLatentFrames: 7, latentHeight: 2, latentWidth: 2,
      audioLatentFrames: 10)
    XCTAssertEqual(result.video.shape, [1, 7, 96])
    XCTAssertEqual(result.audio.shape, [1, 20, 32])
    let video = try Data(contentsOf: fixture.appendingPathComponent("video.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 7, 96], type: Float.self) }
    let audio = try Data(contentsOf: fixture.appendingPathComponent("audio.f32"))
      .withUnsafeBytes { MLXArray($0, [1, 20, 32], type: Float.self) }
    // Swift and Python MLX differ by one float32 rounding step on this seed.
    XCTAssertLessThan(max(abs(result.video - video)).item(Float.self), 1e-7)
    XCTAssertLessThan(max(abs(result.audio - audio)).item(Float.self), 1e-7)
  }
}
