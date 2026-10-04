import Foundation
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3ReferenceControlsTests: XCTestCase {
  func testStrictScalarAdmissionAndGeneratedFrameBounds() throws {
    XCTAssertNil(try H3ReferenceNoiseControls.parse([:]))
    XCTAssertEqual(try H3ReferenceNoiseControls.parse(["audio_condition_strength": 0.5]), try .init(visual: 0.999, audio: 0.5))
    for value in ([true, NSNull(), "0.5", -0.1, 1.1] as [Any]) {
      XCTAssertThrowsError(try H3ReferenceNoiseControls.parse(["visual_condition_strength": value]))
    }
    XCTAssertEqual(try H3ReferencePlacement.frame("last", frames: 73), 72)
    XCTAssertEqual(try H3ReferencePlacement.frame(25, frames: 73), 25)
    XCTAssertThrowsError(try H3ReferenceNoiseControls(visual: .nan))
  }
  func testReferenceStrengthNoiseIsExactAndAudioKeyCannotShiftTargetStream() throws {
    try Device.withDefaultDevice(.cpu) {
      let cleanVideo = [Float](repeating: 0.375, count: 96)
      let cleanAudio = [Float](repeating: -0.25, count: 64)
      for visual in ([0, 0.5, 0.999, 1] as [Float]) {
        let controls = try H3ReferenceNoiseControls(visual: visual, audio: 0.5)
        let actual = try H3Noise.makeReference(seed: 42, conditionVideo: cleanVideo,
          conditionAudio: cleanAudio, videoLatentFrames: 7, latentHeight: 2, latentWidth: 2,
          audioLatentFrames: 17, referenceNoise: controls)
        // Independent owned sequential draw contract, not a zero-input shortcut.
        MLXRandom.seed(42)
        let c = MLXRandom.normal([1, 1, 96]).asType(.float32)
        let v = MLXRandom.normal([1, 24, 7, 2, 2]).asType(.float32)
          .transposed(0, 2, 1, 3, 4).reshaped([1, 7, 96])
        let a = MLXRandom.normal([2, 32, 17]).asType(.float32).transposed(0, 2, 1).reshaped([1, 34, 32])
        let an = MLXRandom.normal([1, 2, 32], key: MLXRandom.key(43)).asType(.float32)
        let expectedV = concatenated([visual * MLXArray(cleanVideo, [1, 1, 96]) + (Float(1)-visual) * c, v], axis: 1)
        let expectedA = concatenated([Float(0.5) * MLXArray(cleanAudio, [1, 2, 32]) + Float(0.5) * an, a], axis: 1)
        eval(actual.video, actual.audio, expectedV, expectedA)
        XCTAssertEqual(actual.video.asArray(Float.self).map(\.bitPattern), expectedV.asArray(Float.self).map(\.bitPattern))
        XCTAssertEqual(actual.audio.asArray(Float.self).map(\.bitPattern), expectedA.asArray(Float.self).map(\.bitPattern))
      }
    }
  }
  func testStrengthClocksUseMaximumOfConditionAndTargetClock() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1], references: [
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 2, audioLatents: 2, sourceLatentFrames: 2, targetFrame: 20)])
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let rows = try H3ReferenceRowSchedule(layout: layout, video: video, audio: audio,
      visualConditionStrength: 0.5, audioConditionStrength: 0.25)
    for step in video.timesteps.indices {
      for index in layout.conditionVideoIndices {
        XCTAssertEqual(rows.table[Int(rows.indicesByStep[step][index])], max(video.timesteps[step], 0.5))
      }
      for index in layout.conditionAudioIndices {
        XCTAssertEqual(rows.table[Int(rows.indicesByStep[step][index])], max(audio.timesteps[step], 0.25))
      }
    }
  }

}
