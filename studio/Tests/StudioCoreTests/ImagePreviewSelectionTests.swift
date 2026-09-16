import XCTest
@testable import StudioCore

final class ImagePreviewSelectionTests: XCTestCase {
  func testSameFilenameAndSubjectStillHaveDistinctIdentitiesAndLabels() {
    let paths = ["/images/take-a/frames/00000000.png", "/images/take-b/frames/00000000.png"]
    let items = ImagePreviewSelection.references(paths: paths, subject: "cat woman")
    XCTAssertEqual(items.map(\.title), ["cat woman · Reference 1", "cat woman · Reference 2"])
    XCTAssertNotEqual(items[0].id, items[1].id)
    XCTAssertEqual(items.map(\.path), paths)
  }

  func testRemovingFirstReferenceKeepsSecondIdentityAndCapturedSelection() {
    var paths = ["/images/a.png", "/images/b.png"]
    let selected = ImagePreviewSelection.references(paths: paths, subject: "Cat")[1]
    paths.removeFirst()
    let remaining = ImagePreviewSelection.references(paths: paths, subject: "Cat")[0]
    XCTAssertEqual(remaining.id, selected.id)
    XCTAssertEqual(selected.title, "Cat · Reference 2")
    XCTAssertEqual(selected.path, "/images/b.png")
    XCTAssertEqual(remaining.title, "Cat · Reference 1")
  }

  func testCanonicalPathIdentityDoesNotUseMutableTitle() {
    let first = ImagePreviewSelection(path: "/images/take/../b.png", title: "Before")
    let renamed = ImagePreviewSelection(path: "/images/b.png", title: "After")
    XCTAssertEqual(first.id, renamed.id)
    XCTAssertEqual(ImagePreviewSelection.references(paths: ["/images/b.png", "/images/b.png"], subject: "Cat").count, 1)
  }
}
