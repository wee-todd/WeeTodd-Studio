import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3DiTStateTests: XCTestCase {
  func testInstalledAllFiftyBlocksRunOneJointAVPrediction() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_FULL_TEST"] == "1",
      let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_PATCH_ORACLE"] else {
      throw XCTSkip("Enable the installed full H3 transformer qualification explicitly.")
    }
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry,
      textTags: [1, 1, 1, 1], anchors: [])
    let inputData = try Data(contentsOf: URL(fileURLWithPath: fixture)
      .appendingPathComponent("condition_proj-input.u16"))
    let input = inputData.withUnsafeBytes {
      MLXArray($0, [1, 4, 5120], type: UInt16.self).view(dtype: .bfloat16)
    }
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let preparationStarted = Date()
    let state = try H3DiTState(checkpointURL: URL(fileURLWithPath: checkpoint),
      layout: layout, textEmbeddings: input,
      timestepTable: [0.1, 0.5, 1]) { completed, total in
      if completed.isMultiple(of: 10) || completed == total {
        print("H3 AdaLN prepared \(completed)/\(total)")
      }
    }
    let preparationSeconds = Date().timeIntervalSince(preparationStarted)
    defer { state.unload() }
    let video = MLXArray([Float](repeating: 0.01,
      count: geometry.videoRows * 96), [1, geometry.videoRows, 96])
    let audio = MLXArray([Float](repeating: 0.01,
      count: geometry.audioRows * 32), [1, geometry.audioRows, 32])
    var completed = 0
    let predictionStarted = Date()
    let result = try state.predict(videoLatents: video, audioLatents: audio,
      timestepIndices: [Int32](repeating: 1, count: layout.tags.count)) {
      completed = $0
      XCTAssertEqual($1, 50)
      if completed.isMultiple(of: 10) || completed == 50 {
        print("H3 transformer evaluated \(completed)/50")
      }
    }
    print("H3 fifty-block prep=\(preparationSeconds) "
      + "prediction=\(Date().timeIntervalSince(predictionStarted)) "
      + "peak_mlx=\(Memory.peakMemory)")
    XCTAssertEqual(completed, 50)
    XCTAssertEqual(result.video.shape, [1, geometry.videoRows, 96])
    XCTAssertEqual(result.audio.shape, [1, geometry.audioRows, 32])
    XCTAssertTrue(result.video.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    XCTAssertTrue(result.audio.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testInvalidPreparedInputsFailBeforeOpeningCheckpoint() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry,
      textTags: [1, 1, 1, 1], anchors: [])
    XCTAssertThrowsError(try H3DiTState(checkpointURL: URL(fileURLWithPath: "/missing"),
      layout: layout, textEmbeddings: MLXArray.zeros([1, 3, 5120]),
      timestepTable: [0, 0.5, 1]))
    XCTAssertThrowsError(try H3DiTState(checkpointURL: URL(fileURLWithPath: "/missing"),
      layout: layout, textEmbeddings: MLXArray.zeros([1, 4, 5120]),
      timestepTable: [0.5, 0.5, 1]))
  }

  func testInstalledOneBlockStateProducesSynchronizedVelocitiesAndUnloads() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_PATCH_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint and input projection oracle paths.")
    }
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry,
      textTags: [1, 1, 1, 1], anchors: [])
    let inputData = try Data(contentsOf: URL(fileURLWithPath: fixture)
      .appendingPathComponent("condition_proj-input.u16"))
    let input = inputData.withUnsafeBytes {
      MLXArray($0, [1, 4, 5120], type: UInt16.self).view(dtype: .bfloat16)
    }
    let state = try H3DiTState(checkpointURL: URL(fileURLWithPath: checkpoint),
      layout: layout, textEmbeddings: input,
      timestepTable: [0.1, 0.5, 1], blockCount: 1)
    XCTAssertTrue(state.isResident)
    XCTAssertGreaterThan(state.residentActivationBytes, 0)
    let video = MLXArray([Float](repeating: 0.01,
      count: geometry.videoRows * 96), [1, geometry.videoRows, 96])
    let audio = MLXArray([Float](repeating: 0.01,
      count: geometry.audioRows * 32), [1, geometry.audioRows, 32])
    let result = try state.predict(videoLatents: video, audioLatents: audio,
      timestepIndices: [Int32](repeating: 1, count: layout.tags.count))
    XCTAssertEqual(result.video.shape, [1, geometry.videoRows, 96])
    XCTAssertEqual(result.audio.shape, [1, geometry.audioRows, 32])
    state.unload()
    XCTAssertFalse(state.isResident)
    XCTAssertEqual(state.residentActivationBytes, 0)
    XCTAssertThrowsError(try state.predict(videoLatents: video, audioLatents: audio,
      timestepIndices: [Int32](repeating: 1, count: layout.tags.count)))
  }
}
