import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VideoReferencePreparationTests: XCTestCase {
  func testRejectsMalformedVideoBeforeWeights() throws {
    let still = H3StillReference(rgb8: Data(repeating: 16,
      count: 64 * 64 * 3), width: 64, height: 64)
    let malformed = H3VideoReference(rgb8: Data(repeating: 128,
      count: 6 * 64 * 64 * 3), frameCount: 6, width: 64, height: 64)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([
      .image(still), .video(malformed)]))
    let valid = H3VideoReference(rgb8: Data(repeating: 128,
      count: 5 * 64 * 64 * 3), frameCount: 5, width: 64, height: 64)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate(
      Array(repeating: .video(valid), count: 4)))
    let sound = H3AudioReference(samples: [Float](repeating: 0, count: 3_200),
      frames: 1_600)
    XCTAssertThrowsError(try H3VideoReferencePreparation.validate([.audio(sound)]),
      "Untimed sound needs a visual reference")
  }

  func testTimedAudioDriverCanBeTheOnlyReference() throws {
    let sound = H3AudioReference(samples: [Float](repeating: 0,
      count: 2 * 80_000), frames: 80_000)
    let references: [H3Ref2VAReference] = [.timedAudio(sound, frame: 0)]
    try H3VideoReferencePreparation.validate(references)
    let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"]
    guard let tokenizer else { throw XCTSkip("Set an installed H3 tokenizer.") }
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 2.5)
    let prepared = try H3VideoReferencePreparation.prepare(prompt: "A person speaks.",
      geometry: geometry, references: references,
      tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertTrue(prepared.qwenRequest.visualRanges.isEmpty)
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 0)
    XCTAssertEqual(prepared.layout.conditionAudioIndices.count, 200)
    let rows = try H3VideoReferencePreparation.encodeVideoRows(
      references: references, layout: prepared.layout,
      videoVAEURL: URL(fileURLWithPath: "/not-loaded-for-audio-only"))
    XCTAssertEqual(rows.shape, [1, 0, 96])
  }

  func testInstalledImageAndAudioReferencePreservesOrderedRows() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let tokenizer = environment["WEETODD_H3_QWEN_TOKENIZER"],
      let audioVAE = environment["WEETODD_H3_AUDIO_VAE"] else {
      throw XCTSkip("Set installed tokenizer and audio VAE paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 2.5)
    let still = H3StillReference(rgb8: Data(repeating: 16,
      count: 64 * 64 * 3), width: 64, height: 64)
    let sound = H3AudioReference(samples: [Float](repeating: 0,
      count: 2 * 1_600), frames: 1_600)
    let references: [H3Ref2VAReference] = [.image(still), .audio(sound)]
    let prepared = try H3VideoReferencePreparation.prepare(
      prompt: "A character speaks.", geometry: geometry,
      references: references, tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(prepared.qwenRequest.visualRanges.count, 1)
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 4)
    XCTAssertEqual(prepared.layout.conditionAudioIndices.count, 4)
    let rows = try H3VideoReferencePreparation.encodeAudioRows(
      references: references, layout: prepared.layout,
      audioVAEURL: URL(fileURLWithPath: audioVAE))
    XCTAssertEqual(rows.shape, [1, 4, 32])
    XCTAssertTrue(rows.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testInstalledSoundtrackVideoUsesOneQwenAudioLabelAndPairedLatents() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let tokenizer = environment["WEETODD_H3_QWEN_TOKENIZER"],
      let audioVAE = environment["WEETODD_H3_AUDIO_VAE"] else {
      throw XCTSkip("Set installed tokenizer and audio VAE paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 2.5)
    let sound = H3AudioReference(samples: [Float](repeating: 0,
      count: 2 * 1_600), frames: 1_600)
    let video = H3VideoReference(rgb8: Data(repeating: 128,
      count: 5 * 64 * 64 * 3), frameCount: 5, width: 64,
      height: 64, audio: sound)
    let references: [H3Ref2VAReference] = [.video(video)]
    let prepared = try H3VideoReferencePreparation.prepare(
      prompt: "A character speaks.", geometry: geometry,
      references: references, tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 8)
    XCTAssertEqual(prepared.layout.conditionAudioIndices.count, 4)
    XCTAssertEqual(prepared.qwenRequest.visualRanges.count, 1)
    let rows = try H3VideoReferencePreparation.encodeAudioRows(
      references: references, layout: prepared.layout,
      audioVAEURL: URL(fileURLWithPath: audioVAE))
    XCTAssertEqual(rows.shape, [1, 4, 32])
  }

  func testInstalledMixedVideoReferencePreservesQwenAndLatentOrder() throws {
    guard let tokenizer = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"],
      let videoVAE = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set installed H3 tokenizer and video VAE paths.")
    }
    let geometry = try H3Geometry(width: 64, height: 64,
      durationSeconds: 2.5)
    let still = H3StillReference(rgb8: Data(repeating: 16,
      count: 64 * 64 * 3), width: 64, height: 64)
    let video = H3VideoReference(rgb8: Data(repeating: 208,
      count: 22 * 64 * 64 * 3), frameCount: 22, width: 64, height: 64)
    let references: [H3Ref2VAReference] = [.image(still), .video(video)]
    let prepared = try H3VideoReferencePreparation.prepare(
      prompt: "A person walks.", geometry: geometry,
      references: references, tokenizerURL: URL(fileURLWithPath: tokenizer))
    XCTAssertEqual(prepared.layout.conditionVideoIndices.count, 32)
    XCTAssertEqual(prepared.qwenRequest.visualRanges.count, 2)
    XCTAssertEqual(prepared.qwenGrids.count, 2)
    XCTAssertEqual(prepared.qwenPixels.shape, [32, 1536])
    XCTAssertLessThan(prepared.qwenPixels[0, 0].item(Float.self),
      prepared.qwenPixels[16, 0].item(Float.self))
    let rows = try H3VideoReferencePreparation.encodeVideoRows(
      references: references, layout: prepared.layout,
      videoVAEURL: URL(fileURLWithPath: videoVAE))
    XCTAssertEqual(rows.shape, [1, 32, 96])
    XCTAssertTrue(rows.asArray(Float.self).allSatisfy(\.isFinite))
  }
}
