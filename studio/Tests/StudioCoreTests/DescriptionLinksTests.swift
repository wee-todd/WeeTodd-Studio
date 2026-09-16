import XCTest
@testable import StudioCore

final class DescriptionLinksTests: XCTestCase {
  func testMatchesLinkedNamesAliasesAndIDsWithUnicodeRanges() {
    let text = "🌙 Mara wears Copper Jacket; jacket_1 hangs beside a copper coat."
    let target = DescriptionLinkTarget(id: "jacket_1", name: "Copper Jacket", aliases: ["copper coat"], description: "High collar.")
    let links = DescriptionLinks.ranges(in: text, targets: [target])
    XCTAssertEqual(links.map { (text as NSString).substring(with: $0.range) }, ["Copper Jacket", "jacket_1", "copper coat"])
    XCTAssertTrue(links.allSatisfy { $0.targetID == "jacket_1" && $0.tooltip == "Copper Jacket\n\nHigh collar." })
  }
  func testLongestMatchBoundariesAndAmbiguity() {
    let targets = [DescriptionLinkTarget(id: "a", name: "Pool", description: "Water"),
                   DescriptionLinkTarget(id: "b", name: "Pool chair", description: "Wicker")]
    XCTAssertEqual(DescriptionLinks.ranges(in: "Pool chair near Pool, not Poolside", targets: targets).map(\.targetID), ["b", "a"])
    let duplicate = DescriptionLinkTarget(id: "c", name: "Pool", description: "Other pool")
    XCTAssertTrue(DescriptionLinks.ranges(in: "Pool", targets: targets + [duplicate]).isEmpty)
    XCTAssertTrue(DescriptionLinks.ranges(in: "Unlinked chair", targets: []).isEmpty)
  }
}
