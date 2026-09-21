import XCTest
@testable import StudioCore

final class CharacterPanelLayoutTests: XCTestCase {
  func testUntrustedCropCoordinatesCannotOverflowOrEscapeImage() {
    XCTAssertFalse(PanelPixelRect(x: Int.max, y: 0, width: 2, height: 1).isValid(inWidth: 1920, height: 1088))
    XCTAssertFalse(PanelPixelRect(x: 2, y: 0, width: Int.max, height: 1).isValid(inWidth: 1920, height: 1088))
    XCTAssertEqual(PanelPixelRect(x: Int.max, y: 0, width: 2, height: 1).maxX, Int.max)
  }

  func testUnequalSubjectsProduceFourGutterBoundedPanels() {
    let components = [
      PanelPixelRect(x: 35, y: 40, width: 330, height: 1000),
      PanelPixelRect(x: 430, y: 50, width: 455, height: 990),
      PanelPixelRect(x: 955, y: 35, width: 485, height: 1010),
      PanelPixelRect(x: 1530, y: 100, width: 350, height: 850),
    ]
    var edges = Array(repeating: 0.8, count: 1920)
    for range in [366..<425, 886..<950, 1441..<1525] { for x in range { edges[x] = 0.01 } }

    let result = CharacterPanelLayout.detect(imageWidth: 1920, imageHeight: 1088,
      foregroundComponents: components, edgeColumns: edges, detectionRevision: 7)

    XCTAssertEqual(result.status, .detected)
    XCTAssertEqual(result.candidates.map(\.role), [.front, .side, .back, .closeUp])
    XCTAssertEqual(result.candidates.map { $0.sourcePixelRect.x + $0.sourcePixelRect.width }, [395, 917, 1482, 1920])
    XCTAssertEqual(result.candidates.map(\.sourcePixelRect.x), [0, 395, 917, 1482])
    XCTAssertTrue(zip(result.candidates, components).allSatisfy { $0.sourcePixelRect.contains($1) })
    XCTAssertFalse(result.candidates.map(\.sourcePixelRect.width).allSatisfy { $0 == 480 })
  }

  func testMergedSubjectsRequireReviewWithoutFabricatedQuarters() {
    let result = CharacterPanelLayout.detect(imageWidth: 1920, imageHeight: 1088,
      foregroundComponents: [
        PanelPixelRect(x: 20, y: 20, width: 350, height: 1040),
        PanelPixelRect(x: 410, y: 20, width: 1020, height: 1040),
        PanelPixelRect(x: 1510, y: 80, width: 370, height: 900),
      ], edgeColumns: Array(repeating: 0.1, count: 1920), detectionRevision: 1)
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertTrue(result.candidates.isEmpty)
    XCTAssertTrue(result.diagnostics.contains { $0.contains("exactly four") })
  }

  func testFourSubjectsWithoutMeasuredGuttersNeedReview() {
    let result = CharacterPanelLayout.detect(imageWidth: 400, imageHeight: 200,
      foregroundComponents: [
        .init(x: 10, y: 10, width: 70, height: 180), .init(x: 110, y: 10, width: 70, height: 180),
        .init(x: 210, y: 10, width: 70, height: 180), .init(x: 310, y: 10, width: 70, height: 180),
      ], edgeColumns: Array(repeating: 1, count: 400), detectionRevision: 1)
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertEqual(result.candidates.count, 4)
    XCTAssertTrue(result.diagnostics.contains { $0.contains("gutter") })
  }

  func testPixelRectRejectsOutOfBoundsAndOverlap() {
    XCTAssertFalse(PanelPixelRect(x: -1, y: 0, width: 4, height: 4).isValid(inWidth: 10, height: 10))
    XCTAssertTrue(PanelPixelRect(x: 1, y: 2, width: 3, height: 4).isValid(inWidth: 10, height: 10))
    XCTAssertTrue(PanelPixelRect(x: 1, y: 1, width: 4, height: 4).intersects(PanelPixelRect(x: 4, y: 2, width: 3, height: 3)))
  }
}
