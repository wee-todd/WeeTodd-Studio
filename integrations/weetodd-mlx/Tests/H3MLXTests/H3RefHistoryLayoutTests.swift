import Foundation
import XCTest
@testable import H3MLX

final class H3RefHistoryLayoutTests: XCTestCase {
  func testPhysicalReferenceOrderAndModalityContextPrefixAreIndependent() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let original = try H3ReferenceLayout(geometry: geometry, textTags: [1, 1], references: [
      .image(latentHeight: 2, latentWidth: 2),
      .video(latentFrames: 2, latentHeight: 2, latentWidth: 2, audioLatents: 2, sourceLatentFrames: 2),
      .audio(latents: 3, targetFrame: 4)])
    let history = try H3ReferenceLayout(referenceLayout: original, geometry: geometry, contextFrames: 5)
    // Independent shape witness: reference physical rows2...14;16context
    // audio rows15...30;2context video rows31...32;244target audio33...276;
    //22target video277...298. Prefixes are not physical row spans.
    XCTAssertEqual(history.conditionVideoIndices, [31, 32, 2, 7, 8])
    XCTAssertEqual(history.conditionAudioIndices, Array(15...30) + Array(3...6) + Array(9...14))
    XCTAssertEqual(history.targetAudioIndices, Array(33...276))
    XCTAssertEqual(history.targetVideoIndices, Array(277...298))
    XCTAssertEqual(Array(history.positions.prefix(15)), Array(original.positions.prefix(15)))
    for index in 0..<8 {
      XCTAssertEqual(history.positions[15 + index], original.positions[15 + index])
      XCTAssertEqual(history.positions[23 + index], original.positions[15 + 122 + index])
    }
    XCTAssertEqual(history.positions[31], history.positions[277])
    XCTAssertEqual(history.positions[32], history.positions[278])
    XCTAssertEqual(Set(history.videoIndices + history.audioIndices + [0, 1]).count, history.tags.count)
  }
  func testCleanAudioAndVideoAreIndependentOfLowerReferenceAugmentation() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let original = try H3ReferenceLayout(geometry: geometry, textTags: [1], references: [.audio(latents: 2, targetFrame: 3)])
    let layout = try H3ReferenceLayout(referenceLayout: original, geometry: geometry, contextFrames: 5)
    let video = try H3Schedule(requestedSteps: 5, shift: 12), audio = try H3Schedule(requestedSteps: 5, shift: 3)
    let rows = try H3ReferenceRowSchedule(layout: layout, video: video, audio: audio, visualConditionStrength: 0.25,
      audioConditionStrength: 0.5, cleanVideoPrefixRows: 2, cleanAudioPrefixRows: 16)
    for step in video.timesteps.indices {
      for index in layout.conditionVideoIndices { XCTAssertEqual(rows.table[Int(rows.indicesByStep[step][index])], 1) }
      for index in layout.conditionAudioIndices.prefix(16) { XCTAssertEqual(rows.table[Int(rows.indicesByStep[step][index])], 1) }
      for index in layout.conditionAudioIndices.dropFirst(16) { XCTAssertEqual(rows.table[Int(rows.indicesByStep[step][index])], max(audio.timesteps[step], 0.5)) }
    }
  }
  func testOversizedContextAndMismatchedGeometryRejectBeforeWeights() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let original = try H3ReferenceLayout(geometry: geometry, textTags: [1], references: [.image(latentHeight: 2, latentWidth: 2)])
    XCTAssertThrowsError(try H3ReferenceLayout(referenceLayout: original, geometry: geometry, contextFrames: 6))
    let other = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    XCTAssertThrowsError(try H3ReferenceLayout(referenceLayout: original, geometry: other, contextFrames: 5))
  }
}
