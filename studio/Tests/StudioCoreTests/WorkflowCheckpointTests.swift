import Foundation
import XCTest
@testable import StudioCore

final class WorkflowCheckpointTests: XCTestCase {
  private func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }
  private struct Evidence: Decodable { let revision: String; let evidence: String }
  func testLargeCheckpointRetainsReviewEvidence() throws {
    let file = try directory().appendingPathComponent("run.json")
    let evidence = String(repeating: "Original lyric and timing evidence. ", count: 70_000)
    try JSONSerialization.data(withJSONObject: ["revision": "reviewed", "evidence": evidence]).write(to: file)
    let value = try WorkflowCheckpoint.read(Evidence.self, at: file)
    XCTAssertEqual(value.revision, "reviewed")
    XCTAssertEqual(value.evidence, evidence)
  }
  func testOversizeAndSymbolicLinksAreRejected() throws {
    let dir = try directory(), file = dir.appendingPathComponent("run.json")
    try Data(repeating: 32, count: 16 * 1024 * 1024 + 1).write(to: file)
    XCTAssertThrowsError(try WorkflowCheckpoint.read(Evidence.self, at: file))
    try Data(#"{"revision":"r1","evidence":"retained"}"#.utf8).write(to: file)
    let link = dir.appendingPathComponent("linked.json")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
    XCTAssertThrowsError(try WorkflowCheckpoint.read(Evidence.self, at: link))
  }
}
