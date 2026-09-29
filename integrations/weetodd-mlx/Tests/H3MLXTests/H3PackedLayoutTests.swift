import XCTest
@testable import H3MLX

final class H3PackedLayoutTests: XCTestCase {
  func testFirstLastAndTimedRowsMatchH3Oracle() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry, textTags: [1, 1],
      anchors: [.first, .last, .frame(12)])
    XCTAssertEqual(layout.positions.count, 296)
    XCTAssertEqual(layout.tags.count, 296)
    XCTAssertEqual(layout.conditionVideoRows, 6)
    XCTAssertEqual(layout.audioStart, 8)
    XCTAssertEqual(layout.videoStart, 252)
    XCTAssertEqual(layout.tags[0], 1)
    XCTAssertEqual(layout.tags[2], 0)
    XCTAssertEqual(layout.tags[8], 2)
    XCTAssertEqual(layout.tags[252], 0)
    XCTAssertEqual(layout.positions[2].x, 2)
    XCTAssertEqual(layout.positions[4].x, 122)
    XCTAssertEqual(layout.positions[6].x, 22)
    XCTAssertEqual(layout.positions[2].y, 4.6862917, accuracy: 0.000001)
    XCTAssertEqual(layout.positions[2].z, -6.627417, accuracy: 0.000001)
    XCTAssertEqual(layout.positions[3].z, 16)
    XCTAssertEqual(layout.positions[8].x, 2)
    XCTAssertEqual(layout.positions[130].z, 16)
    XCTAssertEqual(layout.positions[252].x, 2)
    XCTAssertEqual(layout.positions[254].x, 3.6666667, accuracy: 0.000001)
  }

  func testInvalidTagsAndTimedAnchorsFailBeforePacking() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    XCTAssertThrowsError(try H3PackedLayout(geometry: geometry, textTags: [], anchors: []))
    XCTAssertThrowsError(try H3PackedLayout(geometry: geometry, textTags: [2], anchors: []))
    XCTAssertThrowsError(try H3PackedLayout(geometry: geometry, textTags: [1], anchors: [.frame(-1)]))
    XCTAssertThrowsError(try H3PackedLayout(geometry: geometry, textTags: [1], anchors: [.frame(73)]))
  }
}
