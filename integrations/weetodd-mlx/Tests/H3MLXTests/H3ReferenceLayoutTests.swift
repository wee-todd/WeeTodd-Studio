import MLX
import XCTest
@testable import H3MLX

final class H3ReferenceLayoutTests: XCTestCase {
  func testStillOnlyReferenceScheduleDoesNotRequireAudioConditionTimestep() throws {
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: [1, 0, 1], references: [.image(latentHeight: 4, latentWidth: 4)])
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let plan = try H3ReferenceRowSchedule(layout: layout, video: video, audio: audio)
    XCTAssertEqual(layout.conditionAudioIndices.count, 0)
    XCTAssertEqual(plan.indicesByStep.count, video.timesteps.count)
    XCTAssertTrue(plan.indicesByStep.allSatisfy({ $0.count == layout.tags.count }))
  }

  func testMixedReferencesRetainPackedOrderAndSeparateLatentOrder() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1, 0, 1], references: [
      .image(latentHeight: 2, latentWidth: 4),
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 4,
        audioLatents: 3, sourceLatentFrames: 2),
      .audio(latents: 2),
    ])
    XCTAssertEqual(layout.tags.count, 307)
    XCTAssertEqual(Array(layout.tags[3..<5]), [0, 0])
    XCTAssertEqual(Array(layout.tags[5..<11]), [2, 2, 2, 2, 2, 2])
    XCTAssertEqual(Array(layout.tags[11..<15]), [0, 0, 0, 0])
    XCTAssertEqual(Array(layout.tags[15..<19]), [2, 2, 2, 2])
    XCTAssertEqual(layout.conditionVideoIndices, [3, 4, 11, 12, 13, 14])
    XCTAssertEqual(layout.conditionAudioIndices, [5, 6, 7, 8, 9, 10, 15, 16, 17, 18])
    XCTAssertEqual(layout.videoIndices.prefix(6), [3, 4, 11, 12, 13, 14])
    XCTAssertEqual(layout.audioIndices.prefix(10), [5, 6, 7, 8, 9, 10, 15, 16, 17, 18])
    XCTAssertEqual(layout.targetAudioIndices.first, 19)
    XCTAssertEqual(layout.targetVideoIndices.first, 263)
    XCTAssertEqual(layout.positions[3].x, 3, accuracy: 0.001)
    XCTAssertEqual(layout.positions[5].x, 4, accuracy: 0.001)
    XCTAssertEqual(layout.positions[11].x, 4, accuracy: 0.001)
    XCTAssertEqual(layout.positions[13].x, 5.666_667, accuracy: 0.001)
    XCTAssertEqual(layout.positions[15].x, 12.333_333, accuracy: 0.001)
    XCTAssertEqual(layout.positions[19].x, 14.333_333, accuracy: 0.001)
  }

  func testTimedGuideUsesTargetClockWithoutExtendingReferencePrefix() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [0, 1], references: [
      .image(latentHeight: 2, latentWidth: 4),
      .image(latentHeight: 2, latentWidth: 4, targetFrame: 12),
    ])
    XCTAssertEqual(layout.positions[2].x, 2, accuracy: 0.001)
    XCTAssertEqual(layout.positions[4].x, 23, accuracy: 0.001)
    XCTAssertEqual(layout.positions[6].x, 3, accuracy: 0.001)
  }

  func testRejectsUnsupportedCountsAudioOnlyAndOverBudgetRows() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    XCTAssertThrowsError(try H3ReferenceLayout(geometry: geometry,
      textTags: [0], references: [.audio(latents: 2)]))
    XCTAssertThrowsError(try H3ReferenceLayout(geometry: geometry,
      textTags: [0], references: Array(repeating: .image(latentHeight: 2,
        latentWidth: 4), count: 10)))
    XCTAssertThrowsError(try H3ReferenceLayout(geometry: geometry,
      textTags: [0], references: [.image(latentHeight: 2, latentWidth: 4,
        targetFrame: 1_000)]))
    XCTAssertThrowsError(try H3ReferenceLayout(geometry: geometry,
      textTags: [0], references: [.image(latentHeight: 512, latentWidth: 512)]))
    XCTAssertThrowsError(try H3ReferenceLayout(geometry: geometry,
      textTags: [0], references: [.video(latentFrames: 1, latentHeight: 2,
        latentWidth: 4, audioLatents: 0, sourceLatentFrames: 4_097)]))
  }

  func testReferenceAudioAndVideoStayPinnedOnSeparateSamplingClocks() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1, 0, 1], references: [
      .image(latentHeight: 2, latentWidth: 4),
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 4,
        audioLatents: 3, sourceLatentFrames: 2),
      .audio(latents: 2),
    ])
    let video = try H3Schedule(requestedSteps: 5, shift: 12)
    let audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let plan = try H3ReferenceRowSchedule(layout: layout, video: video,
      audio: audio, visualConditionStrength: 0.999, audioConditionStrength: 0.8)
    func timestep(_ step: Int, _ row: Int) -> Float {
      plan.table[Int(plan.indicesByStep[step][row])]
    }
    XCTAssertEqual(timestep(0, 0), 0, accuracy: 0.0001)
    XCTAssertEqual(timestep(0, 3), 0.999, accuracy: 0.0001)
    XCTAssertEqual(timestep(0, 5), 0.8, accuracy: 0.0001)
    XCTAssertEqual(timestep(0, 19), 0, accuracy: 0.0001)
    XCTAssertEqual(timestep(3, 3), 0.999, accuracy: 0.0001)
    XCTAssertEqual(timestep(3, 15), max(audio.timesteps[3], 0.8), accuracy: 0.0001)
    XCTAssertEqual(timestep(3, 19), audio.timesteps[3], accuracy: 0.0001)
    XCTAssertEqual(timestep(3, 263), video.timesteps[3], accuracy: 0.0001)
  }

  func testPackedEmbeddingsInterleaveReferencesWithoutChangingLatentTensorOrder() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1, 0, 1], references: [
      .image(latentHeight: 2, latentWidth: 4),
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 4,
        audioLatents: 3, sourceLatentFrames: 2),
      .audio(latents: 2),
    ])
    let width = 5376
    let text = MLXArray([Float](repeating: 1, count: 3 * width), [1, 3, width]).asType(.bfloat16)
    let video = MLXArray([Float](repeating: 2, count: 6 * width)
      + [Float](repeating: 5, count: 44 * width), [1, 50, width]).asType(.bfloat16)
    let audio = MLXArray([Float](repeating: 3, count: 10 * width)
      + [Float](repeating: 4, count: 244 * width), [1, 254, width]).asType(.bfloat16)
    let packed = try H3PackedSequence(layout: layout, text: text,
      video: video, audio: audio, timestepIndices: [Int32](repeating: 2, count: 307))
    for (row, expected): (Int, Float) in [(0, 1), (3, 2), (5, 3),
      (11, 2), (15, 3), (19, 4), (263, 5)] {
      XCTAssertEqual(packed.embeddings[0, row, 0].item(Float.self), expected)
    }
    XCTAssertEqual(packed.videoIndices.asArray(Int32.self).prefix(6),
      [3, 4, 11, 12, 13, 14])
    XCTAssertEqual(packed.audioIndices.asArray(Int32.self).prefix(10),
      [5, 6, 7, 8, 9, 10, 15, 16, 17, 18])
    XCTAssertEqual(packed.modulationIndices.asArray(Int32.self)[5], 8)
  }
}
