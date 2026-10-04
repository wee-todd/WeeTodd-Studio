import XCTest
@testable import LTX25MLX

final class MLXMovieSourceAudioContractTests: XCTestCase {
  func testOriginalRateAndSamplesRetainedWhileMonoDuplicatesAndConditioningIsSeparate() throws {
    let p = try MLXMovieSourceAudioContract(videoFrames: 98, fps: 24,
      sampleRate: 44100, samples: 180075, channels: 1)
    XCTAssertEqual(p.publicationSampleRate, 44100); XCTAssertEqual(p.publicationSamples, 180075)
    XCTAssertTrue(p.duplicateMono); XCTAssertEqual(p.publicationChannels, 2)
    XCTAssertEqual(p.conditioningSampleRate, 16000); XCTAssertTrue(p.sourceSupplied)
    let a = try p.publicationSampleBounds(startFrame: 0, endFrame: 49, fps: 24)
    let b = try p.publicationSampleBounds(startFrame: 49, endFrame: 98, fps: 24)
    XCTAssertEqual(a, 0..<90038); XCTAssertEqual(a.upperBound, b.lowerBound)
    XCTAssertEqual(b.upperBound, 180075)
    XCTAssertThrowsError(try p.publicationSampleBounds(startFrame: 0, endFrame: 49, fps: 25))
    XCTAssertThrowsError(try p.publicationSampleBounds(startFrame: 0, endFrame: 99, fps: 24))
  }
  func testNoSourceSynthesizesExactSilenceButSuppliedBadDurationIsNeverRepaired() throws {
    let p = try MLXMovieSourceAudioContract(videoFrames: 98, fps: 24)
    XCTAssertFalse(p.sourceSupplied); XCTAssertEqual(p.publicationSampleRate, 48000)
    XCTAssertEqual(p.publicationSamples, 196000)
    XCTAssertThrowsError(try MLXMovieSourceAudioContract(videoFrames: 98, fps: 24,
      sampleRate: 44100, samples: 44100, channels: 2))
    XCTAssertThrowsError(try MLXMovieSourceAudioContract(videoFrames: 98, fps: 24, sampleRate: 44100))
    XCTAssertThrowsError(try MLXMovieSourceAudioContract(videoFrames: 98, fps: 24,
      sampleRate: 44100, samples: 180075, channels: 6))
  }
}
