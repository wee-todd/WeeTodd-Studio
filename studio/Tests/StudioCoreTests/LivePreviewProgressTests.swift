import XCTest
@testable import StudioCore

final class LivePreviewProgressTests: XCTestCase {
  func testPreviewRevisionSurvivesSplitProgressRecords() throws {
    var stream = BridgeProgressStream()
    XCTAssertTrue(stream.append(Data("{\"event\":\"progress\",\"message\":\"Sampling 2/8\",\"preview".utf8)).isEmpty)
    let events = stream.append(Data("Path\":\"/tmp/job/live-preview.png\",\"previewRevision\":2}\n".utf8))
    XCTAssertEqual(events.first?.previewPath, "/tmp/job/live-preview.png")
    XCTAssertEqual(events.first?.previewRevision, 2)
  }
}
