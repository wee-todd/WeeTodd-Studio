import XCTest
@testable import ImageLayout

final class EditingTests: XCTestCase {
  func testUndoRegionRetainsLaterSceneAndDimensionChanges() throws {
    let region = ImageLayoutRegion(type: .object, caption: "Bottle", bounds: .init(x: 0.2, y: 0.2, width: 0.4, height: 0.4))
    let before = ImageLayoutDraft(width: 1024, height: 1024, caption: "A", regions: [region])
    var moved = before; moved.regions[0].bounds.x = 0.3
    var history = ImageLayoutHistory(); history.record(before: before, after: moved)
    var external = moved; external.summary.caption = "B"; external.width = 896; external.height = 1152
    let undone = try XCTUnwrap(history.undo(current: external))
    XCTAssertEqual(undone.summary.caption, "B"); XCTAssertEqual(undone.width, 896); XCTAssertEqual(undone.height, 1152)
    XCTAssertEqual(undone.regions, before.regions)
    let redone = try XCTUnwrap(history.redo(current: undone))
    XCTAssertEqual(redone.summary.caption, "B"); XCTAssertEqual(redone.regions, moved.regions)
  }
  func testMoveClampsWithoutChangingRegionSize() {
    let rect = NormalizedLayoutRect(x: 0.2, y: 0.3, width: 0.4, height: 0.5)
    XCTAssertEqual(ImageLayoutEditing.move(rect, dx: 2, dy: -2), .init(x: 0.6, y: 0, width: 0.4, height: 0.5))
  }
  func testResizeKeepsOppositeCornerAndEnforcesMinimum() {
    let rect = NormalizedLayoutRect(x: 0.2, y: 0.3, width: 0.4, height: 0.5)
    let resized = ImageLayoutEditing.resize(rect, handle: .topLeft, dx: 0.8, dy: -1, minimumWidth: 0.01, minimumHeight: 0.01)
    XCTAssertEqual(resized.x, 0.59, accuracy: 1e-12)
    XCTAssertEqual(resized.width, 0.01, accuracy: 1e-12)
    XCTAssertEqual(resized.y, 0)
    XCTAssertEqual(resized.height, 0.8, accuracy: 1e-12)
  }
  func testOneDragIsOneUndoAndNewEditDropsRedo() {
    let original = ImageLayoutDraft(width: 1024, height: 1024, caption: "Original")
    var edited = original; edited.summary.caption = "Changed"
    var history = ImageLayoutHistory()
    history.record(before: original, after: edited)
    XCTAssertEqual(history.undo(current: edited), original)
    XCTAssertEqual(history.redo(current: original), edited)
    _ = history.undo(current: edited)
    var another = original; another.width = 896
    history.record(before: original, after: another)
    XCTAssertNil(history.redo(current: another))
  }
}
