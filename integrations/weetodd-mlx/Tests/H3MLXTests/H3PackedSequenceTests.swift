import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3PackedSequenceTests: XCTestCase {
  func testReferenceAudioAndTargetRowsRetainReleasedOrder() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry,
      textTags: [1, 0, 1], anchors: [.first])
    let text = MLXArray([Float](repeating: 1, count: 3 * 5376),
      [1, 3, 5376]).asType(.bfloat16)
    let videoCount = layout.conditionVideoRows + geometry.videoRows
    let video = MLXArray([Float](repeating: 2, count: videoCount * 5376),
      [1, videoCount, 5376]).asType(.bfloat16)
    let audio = MLXArray([Float](repeating: 3, count: geometry.audioRows * 5376),
      [1, geometry.audioRows, 5376]).asType(.bfloat16)
    let timesteps = [Int32](repeating: 2, count: layout.tags.count)
    let packed = try H3PackedSequence(layout: layout, text: text,
      video: video, audio: audio, timestepIndices: timesteps)
    XCTAssertEqual(packed.embeddings.shape, [1, layout.tags.count, 5376])
    XCTAssertEqual(packed.positions.shape, [layout.tags.count, 3])
    XCTAssertEqual(packed.modulationIndices.shape, [layout.tags.count])
    XCTAssertEqual(packed.videoIndices.shape, [videoCount])
    XCTAssertEqual(packed.audioIndices.shape, [geometry.audioRows])
    XCTAssertEqual(packed.embeddings[0, 0, 0].item(Float.self), 1)
    XCTAssertEqual(packed.embeddings[0, layout.audioStart - 1, 0].item(Float.self), 2)
    XCTAssertEqual(packed.embeddings[0, layout.audioStart, 0].item(Float.self), 3)
    XCTAssertEqual(packed.embeddings[0, layout.videoStart, 0].item(Float.self), 2)
    XCTAssertEqual(packed.modulationIndices.asArray(Int32.self)[0], 7)
    XCTAssertEqual(packed.modulationIndices.asArray(Int32.self)[1], 6)
    XCTAssertEqual(packed.modulationIndices.asArray(Int32.self)[layout.audioStart], 8)
    XCTAssertEqual(packed.videoIndices.asArray(Int32.self).first, 3)
    XCTAssertEqual(packed.videoIndices.asArray(Int32.self).last, Int32(layout.tags.count - 1))
  }
}
