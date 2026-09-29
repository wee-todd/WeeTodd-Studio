import Foundation
import XCTest
@testable import H3MLX

final class H3AudioVAELayoutTests: XCTestCase {
  func testInstalledCheckpointAdmitsStereoDecoderAndStatistics() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_VAE"] else {
      throw XCTSkip("Set installed H3 audio VAE checkpoint path.")
    }
    let layout = try H3AudioVAELayout(url: URL(fileURLWithPath: checkpoint))
    XCTAssertEqual(layout.sampleRate, 32_000)
    XCTAssertEqual(layout.samplesPerLatent, 800)
    XCTAssertEqual(layout.latentsMean.count, 32)
    XCTAssertEqual(layout.latentsStandardDeviation.count, 32)
    XCTAssertEqual(layout.upsampleRates, [5, 5, 2, 2, 2, 2, 2])
  }
}
