import XCTest
@testable import StudioCore

final class AssistantModelSetupTests: XCTestCase {
  func testDiscoveryIncludesConfiguredExternalModelWithoutDuplicate() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let standard = root.appendingPathComponent("standard"), external = root.appendingPathComponent("external")
    try FileManager.default.createDirectory(at: standard, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
    let model = external.appendingPathComponent(LocalPromptModels.filenames[0])
    try Data("fixture".utf8).write(to: model)
    XCTAssertEqual(LocalPromptModels.discover(in: standard, including: model.path), [model])
    XCTAssertEqual(LocalPromptModels.discover(in: external, including: model.path), [model])
    XCTAssertTrue(LocalPromptModels.discover(in: standard, including: root.appendingPathComponent("missing.ckpt").path).isEmpty)
  }
  func testFormatCheckAloneCannotSelectAnUnverifiedModel() throws {
    func inspection(_ status: String, _ inference: Bool, _ required: Bool) throws -> AssistantModelInspection {
      let data = try JSONSerialization.data(withJSONObject: ["path": "/model", "status": status,
        "inferenceChecked": inference, "requiresHealthCheck": required, "vision": true, "message": "test"])
      return try JSONDecoder().decode(AssistantModelInspection.self, from: data)
    }
    XCTAssertFalse(try inspection("format_checked", false, true).selectable)
    XCTAssertFalse(try inspection("inference_checked", false, false).selectable)
    XCTAssertTrue(try inspection("checksum_verified", false, false).selectable)
    XCTAssertTrue(try inspection("inference_checked", true, false).selectable)
  }
  func testUnknownFilenameIsNotMislabeledAsSupportedNineBillion() {
    XCTAssertEqual(LocalPromptModels.label(for: "/some/encoder.ckpt"), "Unsupported checkpoint")
    XCTAssertEqual(LocalPromptModels.studioDirectory.lastPathComponent, "AssistantModels")
  }
}
