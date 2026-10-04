import CryptoKit
import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FL2VAContinuationTests: XCTestCase {
  private func recipe(source: Bool = true) -> [String: Any] {
    var continuation: [String: Any] = ["version": 3, "context_frames": 22, "save_context": true]
    if source {
      continuation["source_context"] = "/tmp/context/manifest.json"
      continuation["source_manifest_sha256"] = String(repeating: "a", count: 64)
    }
    return ["components": ["task": "fl2va"], "config": ["duration_seconds": 119.0 / 24],
      "continuation": continuation,
      "conditioning": ["task": "fflf", "inputs": [
        ["id": "front", "role": "first", "frame_index": 0],
        ["id": "end", "role": "last", "frame_index": "last"]]]]
  }
  private func prepare(_ recipe: [String: Any]) throws -> H3FL2VAContinuationRecipe.Prepared {
    try H3FL2VAContinuationRecipe.prepare(data: JSONSerialization.data(withJSONObject: recipe))
  }

  func testVisibleAnchorsOffsetOnceAndSaveOnlyKeepsModelEnd() throws {
    let loaded = try prepare(recipe())
    XCTAssertEqual(loaded.plan.generatedFrames, 141)
    XCTAssertEqual(loaded.plan.publishedFrames, 119)
    XCTAssertEqual(loaded.plan.overlapFrames, 22)
    XCTAssertEqual(loaded.visibleAnchors, [0, 118])
    XCTAssertEqual(loaded.sampledAnchors, [22, 140])
    let sampled = try XCTUnwrap(JSONSerialization.jsonObject(with: loaded.sampledRecipe) as? [String: Any])
    XCTAssertNil(sampled["continuation"])
    let inputs = try XCTUnwrap((sampled["conditioning"] as? [String: Any])?["inputs"] as? [[String: Any]])
    XCTAssertEqual(inputs[0]["role"] as? String, "keyframe")
    XCTAssertEqual(inputs.map { $0["frame_index"] as? Int }, [22, 140])
    XCTAssertThrowsError(try H3FL2VAContinuationRecipe.prepare(data: loaded.sampledRecipe))
    let saved = try prepare(recipe(source: false))
    XCTAssertEqual(saved.plan.generatedFrames, 124)
    XCTAssertEqual(saved.plan.publishedFrames, 124)
    XCTAssertEqual(saved.sampledAnchors, [0, 123])
  }

  func testMalformedControlsAndVisibleBoundsFailBeforeMedia() throws {
    for (key, value) in [("version", true as Any), ("version", 2),
      ("context_frames", true), ("context_frames", 21), ("save_context", 1),
      ("source_context", "relative/manifest.json"), ("source_context", "/tmp/bad\0file")] {
      var root = recipe()
      var fields = root["continuation"] as! [String: Any]
      fields[key] = value; root["continuation"] = fields
      XCTAssertThrowsError(try prepare(root), key)
    }
    for anchors in [[0, 119], [22, 0], [0, 0]] {
      var root = recipe()
      root["conditioning"] = ["task": "fflf", "inputs": anchors.map {
        ["role": "keyframe", "frame_index": $0]
      }]
      XCTAssertThrowsError(try prepare(root))
    }
    var trimmed = recipe()
    trimmed["config"] = ["duration_seconds": 118.0 / 24]
    XCTAssertThrowsError(try prepare(trimmed))
  }

  func testPackedOrderPositionsAndModulationMatchIndependentWitness() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 141.0 / 24)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1, 0, 1],
      anchors: [.frame(22), .frame(140)], contextFrames: 22)
    XCTAssertEqual(layout.tags.count, 649)
    XCTAssertEqual(layout.conditionVideoIndices, Array(3..<21))
    XCTAssertEqual(layout.conditionAudioIndices, Array(21..<95))
    XCTAssertEqual(layout.targetAudioIndices, Array(95..<565))
    XCTAssertEqual(layout.targetVideoIndices, Array(565..<649))
    // Independently derive the complete primary packing formula, including
    // separate channel-major condition and target audio clocks.
    let root = sqrt(8.0), height = Float((1 - 2 / root) * 16)
    let widths: [Float] = [Float((1 - 4 / root) * 16), 16]
    var expected = (0..<3).map { SIMD3<Float>(Float($0), 0, 0) }
    func video(_ times: [Double]) {
      for time in times { for width in widths { expected.append(SIMD3(Float(time), height, width)) } }
    }
    func times(_ count: Int) -> [Double] {
      var current = 3.0, result: [Double] = []
      for i in 0..<count { result.append(current); current += (i % 5 == 0 ? 1.0 : 4.0) * 5 / 3 }
      return result
    }
    video(times(7)); video([3 + 22.0 * 5 / 3, 3 + 140.0 * 5 / 3])
    for count in [37, 235] { for width in widths { for t in 0..<count {
      expected.append(SIMD3(Float(3 + t), 0, width))
    } } }
    video(times(42))
    // Float conversion only after the Float64 position accumulation.
    for (actual, oracle) in zip(layout.positions, expected) {
      XCTAssertEqual(actual.x.bitPattern, oracle.x.bitPattern)
      XCTAssertEqual(actual.y.bitPattern, oracle.y.bitPattern)
      XCTAssertEqual(actual.z.bitPattern, oracle.z.bitPattern)
    }
    let videoSchedule = try H3Schedule(requestedSteps: 5, shift: 12)
    let audioSchedule = try H3Schedule(requestedSteps: 5, shift: 3)
    let schedule = try H3ReferenceRowSchedule(layout: layout,
      video: videoSchedule, audio: audioSchedule, cleanVideoPrefixRows: 14)
    for step in 0..<4 {
      let rows = schedule.indicesByStep[step]
      for index in 3..<17 { XCTAssertEqual(schedule.table[Int(rows[index])], 1) }
      for index in 17..<21 { XCTAssertEqual(schedule.table[Int(rows[index])], 0.999) }
      for index in 21..<95 { XCTAssertEqual(schedule.table[Int(rows[index])], 1) }
      XCTAssertEqual(schedule.table[Int(rows[95])], audioSchedule.timesteps[step])
      XCTAssertEqual(schedule.table[Int(rows[565])], videoSchedule.timesteps[step])
    }
    XCTAssertThrowsError(try H3ReferenceRowSchedule(layout: layout,
      video: videoSchedule, audio: audioSchedule, cleanVideoPrefixRows: 19))
  }

  func testArtifactTaskSeparationRepeatedLoadAndLegacyManifestBytes() throws {
    let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 124.0 / 24)
    let rows = try H3Continuation.tail(video: (0..<(geometry.videoRows * 96)).map(Float.init),
      audio: (0..<(geometry.audioRows * 32)).map(Float.init), geometry: geometry, contextFrames: 22)
    let plan = try H3Continuation.Plan(contextFrames: 22, requestedDuration: 5,
      sourceManifest: nil, sourceSHA256: nil, saveContext: true)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let identity = String(repeating: "a", count: 64)
    let saved = try H3Continuation.save(rows, plan: plan, width: 32, height: 32,
      identity: identity, directory: root.appendingPathComponent("fl"), task: "fl2va")
    let before = try Data(contentsOf: saved.manifest)
    for _ in 0..<2 {
      let loaded = try XCTUnwrap(H3Continuation.load(manifestURL: saved.manifest,
        expectedSHA256: saved.sha256, contextFrames: 22, width: 32, height: 32,
        identity: identity, task: "fl2va"))
      XCTAssertEqual(loaded.video, rows.video); XCTAssertEqual(loaded.audio, rows.audio)
    }
    XCTAssertEqual(try Data(contentsOf: saved.manifest), before)
    XCTAssertThrowsError(try H3Continuation.load(manifestURL: saved.manifest,
      expectedSHA256: saved.sha256, contextFrames: 22, width: 32, height: 32, identity: identity))
    let legacy = try H3Continuation.save(rows, plan: plan, width: 32, height: 32,
      identity: identity, directory: root.appendingPathComponent("t2v"))
    let old = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: legacy.manifest)) as? [String: Any])
    XCTAssertNil(old["task"])
    let expected: [String: Any] = ["format": "weetodd-h3-swift-continuation-v2", "contextFrames": 22,
      "width": 32, "height": 32, "generatedFrames": 124, "publishedFrames": 124, "overlapFrames": 0,
      "identity": identity, "payloadBytes": (rows.video.count + rows.audio.count) * 4,
      "payloadSHA256": legacy.payloadSHA256]
    XCTAssertEqual(try Data(contentsOf: legacy.manifest),
      try JSONSerialization.data(withJSONObject: expected, options: [.sortedKeys]))
    XCTAssertThrowsError(try H3Continuation.load(manifestURL: legacy.manifest,
      expectedSHA256: legacy.sha256, contextFrames: 22, width: 32, height: 32,
      identity: identity, task: "fl2va"))
  }

  func testFLFingerprintBindsVisionAndSeparatesTaskWithoutChangingT2V() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let component = root.appendingPathComponent("base"), vision = root.appendingPathComponent("vision")
    try Data([1]).write(to: component); try Data([2]).write(to: vision)
    let base = try H3T2VARequest(prompt: "Walk", width: 32, height: 32, durationSeconds: 2.5,
      seed: 42, requestedSteps: 5, transformer: component, qwenPages: component,
      tokenizer: component, videoVAE: component, audioVAE: component)
    let request = try H3FL2VARequest(base: base, vision: vision,
      images: [H3StillReference(rgb8: Data(repeating: 0, count: 32 * 32 * 3), width: 32, height: 32)],
      anchors: [.first])
    let t2v = try H3Continuation.fingerprint(base), fl = try H3Continuation.fingerprint(request)
    XCTAssertNotEqual(t2v, fl)
    try Data([2, 3]).write(to: vision)
    XCTAssertNotEqual(try H3Continuation.fingerprint(request), fl)
    XCTAssertEqual(try H3Continuation.fingerprint(base), t2v)
  }
  private final class ConstantVelocity: H3ReferenceVelocityPredictor {
    let layout: H3ReferenceLayout
    init(_ layout: H3ReferenceLayout) { self.layout = layout }
    func predict(videoLatents: MLXArray, audioLatents: MLXArray,
      timestepIndices: [Int32], progress: (Int, Int) -> Void) throws -> H3FinalLayer.Output {
      H3FinalLayer.Output(video: MLXArray.ones(videoLatents.shape), audio: MLXArray.ones(audioLatents.shape))
    }
  }
  func testSharedSamplerKeepsBothContextAndTimedImagePrefixImmutableAcrossRepeatedUses() throws {
    try Device.withDefaultDevice(.cpu) {
      let geometry = try H3Geometry(width: 32, height: 32, durationSeconds: 141.0 / 24)
      let layout = try H3ReferenceLayout(geometry: geometry, textTags: [1],
        anchors: [.frame(22)], contextFrames: 22)
      let vs = try H3Schedule(requestedSteps: 5, shift: 12)
      let aus = try H3Schedule(requestedSteps: 5, shift: 3)
      let rs = try H3ReferenceRowSchedule(layout: layout, video: vs, audio: aus, cleanVideoPrefixRows: 7)
      let cv = (0..<(8 * 96)).map { Float($0) / 8 }
      let ca = (0..<(74 * 32)).map { -Float($0) / 4 }
      for method in [H3SamplingMethod.euler, .resMultistep] {
        for _ in 0..<2 {
          let video = concatenated([MLXArray(cv, [1, 8, 96]), MLXArray.zeros([1, geometry.videoRows, 96])], axis: 1)
          let audio = concatenated([MLXArray(ca, [1, 74, 32]), MLXArray.zeros([1, geometry.audioRows, 32])], axis: 1)
          let output = try H3ReferenceSampler.run(predictor: ConstantVelocity(layout),
            videoSchedule: vs, audioSchedule: aus, rowSchedule: rs,
            videoLatents: video, audioLatents: audio, samplingMethod: method)
          XCTAssertEqual(output.video[0..<1, 0..<8, 0..<96].asArray(Float.self).map(\.bitPattern), cv.map(\.bitPattern))
          XCTAssertEqual(output.audio[0..<1, 0..<74, 0..<32].asArray(Float.self).map(\.bitPattern), ca.map(\.bitPattern))
          XCTAssertEqual(output.video[0, 8, 0].item(Float.self), 1, accuracy: 0.000001)
          XCTAssertEqual(output.audio[0, 74, 0].item(Float.self), 1, accuracy: 0.000001)
        }
      }
    }
  }

}
