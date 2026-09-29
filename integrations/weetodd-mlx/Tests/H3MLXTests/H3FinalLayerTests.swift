import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FinalLayerTests: XCTestCase {
  func testInstalledOutputHeadsMatchReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let refiner = ProcessInfo.processInfo.environment["WEETODD_H3_REFINER_ORACLE"],
      let time = ProcessInfo.processInfo.environment["WEETODD_H3_TIME_ORACLE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_FINAL_ORACLE"] else {
      throw XCTSkip("Set installed H3 checkpoint, refiner, time and final-layer oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    func array<T: HasDType>(_ url: URL, shape: [Int], type: T.Type) throws -> MLXArray {
      let data = try Data(contentsOf: url)
      return data.withUnsafeBytes { MLXArray($0, shape, type: type) }
    }
    let input = try array(URL(fileURLWithPath: refiner)
      .appendingPathComponent("final.u16"), shape: [1, 4, 5376],
      type: UInt16.self).view(dtype: .bfloat16)
    let timeEmbedding = try array(URL(fileURLWithPath: time)
      .appendingPathComponent("output.f32"), shape: [3, 2688], type: Float.self)
    let timestepIndices = try array(root.appendingPathComponent("timestep-indices.i32"),
      shape: [4], type: Int32.self)
    let videoIndices = try array(root.appendingPathComponent("video-indices.i32"),
      shape: [2], type: Int32.self)
    let audioIndices = try array(root.appendingPathComponent("audio-indices.i32"),
      shape: [2], type: Int32.self)
    let output = try H3FinalLayer.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      input: input, timeEmbeddings: timeEmbedding,
      timestepIndices: timestepIndices, videoIndices: videoIndices,
      audioIndices: audioIndices) { name, value in
      let shape = name == "modulation" ? [3, 10752] : [1, 4, 5376]
      let expected = try array(root.appendingPathComponent("\(name).u16"),
        shape: shape, type: UInt16.self).view(dtype: .bfloat16)
      XCTAssertEqual(max(abs(value.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, name)
    }
    let video = try array(root.appendingPathComponent("video.f32"),
      shape: [1, 2, 96], type: Float.self)
    let audio = try array(root.appendingPathComponent("audio.f32"),
      shape: [1, 2, 32], type: Float.self)
    XCTAssertEqual(max(abs(output.video - video)).item(Float.self), 0)
    XCTAssertEqual(max(abs(output.audio - audio)).item(Float.self), 0)
  }
}
