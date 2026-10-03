import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3ReferenceNoiseTests: XCTestCase {
  func testVisualAugmentationConsumesFirstSeededDrawBeforeTargetAV() throws {
    try Device.withDefaultDevice(.cpu) {
      let clean = (0..<(7 * 96)).map { Float($0 % 17) - 8 }
      let sound = (0..<(4 * 32)).map { Float($0 % 11) / 8 }
      let rows = try H3Noise.makeReference(seed: 1234,
        conditionVideo: clean, conditionAudio: sound,
        videoLatentFrames: 7, latentHeight: 2, latentWidth: 2,
        audioLatentFrames: 10)
      XCTAssertEqual(rows.video.shape, [1, 14, 96])
      XCTAssertEqual(rows.audio.shape, [1, 24, 32])
      let video = rows.video.asArray(Float.self)
      let audio = rows.audio.asArray(Float.self)
      // Independent owned Python MLX oracle from the existing FL2VA noise
      // fixture: a packed [1,7,96] draw, channel-major video, stereo audio.
      // This is WeeTodd's migrated seed policy, not PyTorch RNG parity.
      let condition: [Float] = [0.39139548, 0.6809802, -2.8445895, -0.39998114]
      let targetVideo: [Float] = [0.18150127, -0.40931788, 1.1606829, 0.06083998]
      let targetAudio: [Float] = [-0.54947805, 1.1330395, -0.1210604, -0.46126363]
      let strength = Float(0.999)
      for index in 0..<4 {
        XCTAssertEqual(video[index], strength * clean[index] +
          (1 - strength) * condition[index], accuracy: 0.000001)
        XCTAssertEqual(video[clean.count + index], targetVideo[index], accuracy: 0.000001)
        XCTAssertEqual(audio[sound.count + index], targetAudio[index], accuracy: 0.000001)
      }
      XCTAssertEqual(Array(audio.prefix(sound.count)), sound)
      XCTAssertNotEqual(Array(video.prefix(clean.count)), clean)
      XCTAssertTrue(video.allSatisfy(\.isFinite))
      XCTAssertTrue(audio.allSatisfy(\.isFinite))
    }
  }

  func testAudioOnlyReferenceDoesNotAdvanceTargetSeedStream() throws {
    try Device.withDefaultDevice(.cpu) {
      let sound = [Float](repeating: 0.125, count: 4 * 32)
      let reference = try H3Noise.makeReference(seed: 1234,
        conditionVideo: [], conditionAudio: sound,
        videoLatentFrames: 7, latentHeight: 2, latentWidth: 2,
        audioLatentFrames: 10)
      let target = try H3Noise.make(seed: 1234, videoLatentFrames: 7,
        latentHeight: 2, latentWidth: 2, audioLatentFrames: 10)
      XCTAssertEqual(reference.video.asArray(Float.self), target.video.asArray(Float.self))
      XCTAssertEqual(Array(reference.audio.asArray(Float.self).dropFirst(sound.count)),
        target.audio.asArray(Float.self))
      XCTAssertEqual(Array(reference.audio.asArray(Float.self).prefix(sound.count)), sound)
    }
  }

  func testEntireSeededPackAgainstIndependentChannelIndexOracle() throws {
    try Device.withDefaultDevice(.cpu) {
      // The owned Python sampler draws a two-dimensional condition pack;
      // construct its target patch/stereo ordering using scalar indices.
      MLXRandom.seed(1234)
      let condition = MLXRandom.normal([7, 96]).asArray(Float.self)
      let rawVideo = MLXRandom.normal([1, 24, 7, 2, 2]).asArray(Float.self)
      let rawAudio = MLXRandom.normal([2, 32, 10]).asArray(Float.self)
      let clean = (0..<(7 * 96)).map { Float($0 % 17) - 8 }
      let strength = Float(0.999)
      var expectedVideo = zip(clean, condition).map { strength * $0 + (1 - strength) * $1 }
      for frame in 0..<7 {
        for channel in 0..<24 {
          for subpixel in 0..<4 {
            expectedVideo.append(rawVideo[(channel * 7 + frame) * 4 + subpixel])
          }
        }
      }
      var expectedAudio: [Float] = []
      for stereo in 0..<2 {
        for frame in 0..<10 {
          for channel in 0..<32 {
            expectedAudio.append(rawAudio[(stereo * 32 + channel) * 10 + frame])
          }
        }
      }
      let result = try H3Noise.makeReference(seed: 1234,
        conditionVideo: clean, conditionAudio: [], videoLatentFrames: 7,
        latentHeight: 2, latentWidth: 2, audioLatentFrames: 10)
      let video = result.video.asArray(Float.self)
      let audio = result.audio.asArray(Float.self)
      XCTAssertEqual(video.count, expectedVideo.count)
      XCTAssertEqual(audio.count, expectedAudio.count)
      XCTAssertLessThanOrEqual(zip(video, expectedVideo).map { abs($0 - $1) }.max()!, 0.000001)
      XCTAssertEqual(audio, expectedAudio)
    }
  }

  func testVisualAndAudioConditionTimestepsAgreeWithAugmentation() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1], references: [
      .image(latentHeight: 2, latentWidth: 2),
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 2,
        audioLatents: 2, sourceLatentFrames: 2, targetFrame: 0),
      .audio(latents: 2),
    ])
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let plan = try H3ReferenceRowSchedule(layout: layout, video: video, audio: audio)
    for step in video.timesteps.indices {
      for row in layout.conditionVideoIndices {
        XCTAssertEqual(plan.table[Int(plan.indicesByStep[step][row])],
          max(video.timesteps[step], Float(0.999)))
      }
      for row in layout.conditionAudioIndices {
        XCTAssertEqual(plan.table[Int(plan.indicesByStep[step][row])], 1)
      }
    }
  }

  func testMultiReferenceNoiseUsesRefBudgetRatherThanFL2VATwoAnchorBudget() throws {
    try Device.withDefaultDevice(.cpu) {
      let clean = [Float](repeating: 0.25, count: 4097 * 96)
      let rows = try H3Noise.makeReference(seed: 42,
        conditionVideo: clean, conditionAudio: [], videoLatentFrames: 7,
        latentHeight: 2, latentWidth: 2, audioLatentFrames: 10)
      XCTAssertEqual(rows.video.shape, [1, 4097 + 7, 96])
      XCTAssertTrue(rows.video.asArray(Float.self).prefix(8).contains { $0 != 0.25 })
      XCTAssertThrowsError(try H3Noise.makeReference(seed: 42,
        conditionVideo: [1], conditionAudio: [], videoLatentFrames: 7,
        latentHeight: 2, latentWidth: 2, audioLatentFrames: 10))
      XCTAssertThrowsError(try H3Noise.makeReference(seed: 42,
        conditionVideo: [], conditionAudio: [.nan] + [Float](repeating: 0, count: 31),
        videoLatentFrames: 7, latentHeight: 2, latentWidth: 2, audioLatentFrames: 10))
    }
  }
}
