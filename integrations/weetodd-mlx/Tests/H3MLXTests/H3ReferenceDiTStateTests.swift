import MLX
import XCTest
@testable import H3MLX

final class H3ReferenceDiTStateTests: XCTestCase {
  func testReferenceStateRejectsWrongTextRowsBeforeOpeningWeights() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: [1, 1, 1, 1], references: [.image(latentHeight: 2, latentWidth: 2)])
    XCTAssertThrowsError(try H3ReferenceDiTState(
      checkpointURL: URL(fileURLWithPath: "/missing"), layout: layout,
      textEmbeddings: MLXArray.zeros([1, 3, 5120]),
      timestepTable: [0.1, 0.5, 1]))
  }

  func testInstalledOneBlockReferenceStateUsesTheSharedTransformer() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_PATCH_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint and input projection oracle paths.")
    }
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: [1, 1, 1, 1], references: [.image(latentHeight: 2, latentWidth: 2)])
    let data = try Data(contentsOf: URL(fileURLWithPath: fixture)
      .appendingPathComponent("condition_proj-input.u16"))
    let input = data.withUnsafeBytes {
      MLXArray($0, [1, 4, 5120], type: UInt16.self).view(dtype: .bfloat16)
    }
    let state = try H3ReferenceDiTState(
      checkpointURL: URL(fileURLWithPath: checkpoint), layout: layout,
      textEmbeddings: input, timestepTable: [0.1, 0.5, 1], blockCount: 1)
    XCTAssertTrue(state.isResident)
    let video = MLXArray([Float](repeating: 0.01,
      count: (geometry.videoRows + 1) * 96), [1, geometry.videoRows + 1, 96])
    let audio = MLXArray([Float](repeating: 0.01,
      count: geometry.audioRows * 32), [1, geometry.audioRows, 32])
    let output = try state.predict(videoLatents: video, audioLatents: audio,
      timestepIndices: [Int32](repeating: 1, count: layout.tags.count)) { _, _ in }
    XCTAssertEqual(output.video.shape, video.shape)
    XCTAssertEqual(output.audio.shape, audio.shape)
    state.unload()
    XCTAssertFalse(state.isResident)
  }
}
