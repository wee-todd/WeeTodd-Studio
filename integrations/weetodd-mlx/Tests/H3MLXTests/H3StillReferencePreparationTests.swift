import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3StillReferencePreparationTests: XCTestCase {
  func testInstalledStillReferencesReachJointTransformerWithoutDroppingRows() throws {
    guard let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let qwenPages = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let qwenVision = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_COMPACT"],
      let videoVAE = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"],
      let transformer = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set installed H3 Qwen, video VAE and transformer paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
    let refs = [UInt8(16), UInt8(208)].map { value in
      H3StillReference(rgb8: Data(repeating: value, count: 64 * 64 * 3),
        width: 64, height: 64)
    }
    let prepared = try H3StillReferencePreparation.prepare(
      prompt: "Two people meet.", geometry: geometry,
      references: refs, tokenizerURL: URL(fileURLWithPath: tokenizer))
    let qwen = try H3QwenTextEncoder.encodeReferences(
      prompt: "Two people meet.", pixels: prepared.qwenPixels,
      references: prepared.qwenGrids.map { .image(grid: $0) },
      checkpointRoot: URL(fileURLWithPath: qwenPages),
      visionCheckpointURL: URL(fileURLWithPath: qwenVision),
      tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(qwen.tags, Array(prepared.layout.tags.prefix(qwen.tags.count)))
    let referenceRows = try prepared.encodeVideoRows(
      videoVAEURL: URL(fileURLWithPath: videoVAE))
    let noise = try H3Noise.make(seed: 12,
      videoLatentFrames: geometry.videoLatentFrames,
      latentHeight: geometry.height / 16, latentWidth: geometry.width / 16,
      audioLatentFrames: geometry.audioLatentFrames)
    let video = concatenated([referenceRows, noise.video], axis: 1)
    let schedule = try H3ReferenceRowSchedule(layout: prepared.layout,
      video: H3Schedule(requestedSteps: 5, shift: 12),
      audio: H3Schedule(requestedSteps: 5, shift: 3))
    let state = try H3ReferenceDiTState(
      checkpointURL: URL(fileURLWithPath: transformer), layout: prepared.layout,
      textEmbeddings: qwen.hidden.reshaped([1, qwen.tags.count, 5120]),
      timestepTable: schedule.table, blockCount: 1)
    defer { state.unload() }
    let output = try state.predict(videoLatents: video,
      audioLatents: noise.audio, timestepIndices: schedule.indicesByStep[0])
    XCTAssertEqual(output.video.shape, video.shape)
    XCTAssertEqual(output.audio.shape, noise.audio.shape)
    XCTAssertTrue(output.video.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    XCTAssertTrue(output.audio.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testInstalledTwoStillEncodesRetainAdmittedOrderAndRows() throws {
    guard let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let videoVAE = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set installed Qwen tokenizer and H3 video VAE paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
    let refs = [UInt8(16), UInt8(208)].map { value in
      H3StillReference(rgb8: Data(repeating: value, count: 64 * 64 * 3),
        width: 64, height: 64)
    }
    let prepared = try H3StillReferencePreparation.prepare(
      prompt: "Two people meet.", geometry: geometry,
      references: refs, tokenizerURL: URL(fileURLWithPath: tokenizer))
    let rows = try prepared.encodeVideoRows(videoVAEURL: URL(fileURLWithPath: videoVAE))
    XCTAssertEqual(rows.shape, [1, 8, 96])
    let first = rows[0, 0..<4, 0..<96]
    let second = rows[0, 4..<8, 0..<96]
    XCTAssertGreaterThan(max(abs(first - second)).item(Float.self), 0.01)
    XCTAssertTrue(rows.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testTwoPreparedStillsKeepOrderAcrossQwenAndReferenceRows() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set installed Qwen tokenizer path.")
    }
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
    let first = H3StillReference(rgb8: Data(repeating: 16, count: 64 * 64 * 3),
      width: 64, height: 64)
    let second = H3StillReference(rgb8: Data(repeating: 208, count: 64 * 64 * 3),
      width: 64, height: 64)
    let prepared = try H3StillReferencePreparation.prepare(
      prompt: "Two people meet.", geometry: geometry,
      references: [first, second], tokenizerURL: URL(fileURLWithPath: path))
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 8)
    XCTAssertEqual(prepared.qwenPixels.shape, [32, 1536])
    XCTAssertEqual(prepared.qwenRequest.visualRanges.count, 2)
    XCTAssertEqual(prepared.qwenGrids.count, 2)
    let firstValue = prepared.qwenPixels[0, 0].item(Float.self)
    let secondValue = prepared.qwenPixels[16, 0].item(Float.self)
    XCTAssertLessThan(firstValue, secondValue)
  }

  func testRejectsMalformedOrExcessiveStillsBeforeWeights() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set installed Qwen tokenizer path.")
    }
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 2.5)
    let valid = H3StillReference(rgb8: Data(repeating: 0, count: 64 * 64 * 3),
      width: 64, height: 64)
    let bad = H3StillReference(rgb8: Data(count: 5), width: 64, height: 64)
    let tokenizer = URL(fileURLWithPath: path)
    XCTAssertThrowsError(try H3StillReferencePreparation.prepare(
      prompt: "One person.", geometry: geometry,
      references: [bad], tokenizerURL: tokenizer))
    XCTAssertThrowsError(try H3StillReferencePreparation.prepare(
      prompt: "Ten people.", geometry: geometry,
      references: Array(repeating: valid, count: 10), tokenizerURL: tokenizer))
  }
}
