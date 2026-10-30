import Foundation
import XCTest
@testable import StudioCore

final class NativeRestorationCLITests: XCTestCase {
  func testProcessingRequestRejectsUnknownAndUnusedRuntimeAndKeepsTypedSequence() throws {
    var h3 = H3RegenerateSettings(); h3.targetWidth = 128; h3.targetHeight = 64
    let recipe = VideoProcessingRecipe(h3Regenerate: h3)
    var request = NativeVideoProcessingRequest(sourcePath: "/movies/source.mov", recipe: recipe, h3Runtime: .init())
    let bytes = try JSONEncoder().encode(request)
    let decoded = try JSONDecoder().decode(NativeVideoProcessingRequest.self, from: bytes)
    XCTAssertEqual(decoded.recipe, recipe)
    var object = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
    object["pythonPath"] = "/unwanted/python"
    XCTAssertThrowsError(try JSONDecoder().decode(NativeVideoProcessingRequest.self, from: JSONSerialization.data(withJSONObject: object)))
    request.runtime = .init(); XCTAssertThrowsError(try request.validate())
    request.runtime = nil; request.sourceSHA256 = "not-a-digest"; XCTAssertThrowsError(try request.validate())
  }
  func testCLIRequestReaderIsBoundedBeforeDecoding() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("oversized.json")
    try Data(count: 1024 * 1024 + 1).write(to: url)
    XCTAssertThrowsError(try NativeRestorationCLI.data(url.path))
  }
}
