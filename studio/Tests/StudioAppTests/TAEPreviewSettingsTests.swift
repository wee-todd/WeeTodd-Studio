import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TAEPreviewSettingsTests: XCTestCase {
  func testLegacyRuntimeKeepsPreviewOffWithoutAddingSavedKeys() throws {
    let runtime = try settings([:])
    let object = try encoded(runtime)
    XCTAssertNil(object["ltx25PreviewDecoder"])
    XCTAssertNil(object["ltx25PreviewCheckpoint"])
    XCTAssertNil(object["ltx25PreviewWideCheckpoint"])
  }

  func testBothWeightLinksSurviveSettingsRestorationButDoNotChangeRenderIdentity() throws {
    for mode in ["taeltx2_3", "taeltx2_3_wide"] {
      let runtime = try settings(["ltx25PreviewDecoder": mode,
        "ltx25PreviewCheckpoint": "/models/standard.safetensors",
        "ltx25PreviewWideCheckpoint": "/models/wide.safetensors"])
      let restored = RuntimeSettings.restoring(try JSONEncoder().encode(runtime), defaults: try settings([:]))
      let object = try encoded(restored)
      XCTAssertEqual(object["ltx25PreviewDecoder"] as? String, mode)
      XCTAssertEqual(object["ltx25PreviewCheckpoint"] as? String, "/models/standard.safetensors")
      XCTAssertEqual(object["ltx25PreviewWideCheckpoint"] as? String, "/models/wide.safetensors")
      let renderIdentity = try encoded(restored.generationSettings)
      XCTAssertNil(renderIdentity["ltx25PreviewDecoder"])
      XCTAssertNil(renderIdentity["ltx25PreviewCheckpoint"])
      XCTAssertNil(renderIdentity["ltx25PreviewWideCheckpoint"])
      XCTAssertTrue(NSDictionary(dictionary: renderIdentity).isEqual(to: try encoded(settings([:]).generationSettings)),
        "Changing the live display decoder must preserve prepared render identity")
    }
  }

  @MainActor func testChangingPreviewModeKeepsPreparedRequestKey() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.runtime = try settings([:])
    store.addClip(.ltx25)
    let clip = try XCTUnwrap(store.selectedClip)
    let key = store.generationRequestKey(for: clip)
    for mode in ["taeltx2_3", "taeltx2_3_wide", "off"] {
      store.runtime.ltx25PreviewDecoder = mode
      store.runtime.ltx25PreviewCheckpoint = "/models/standard.safetensors"
      store.runtime.ltx25PreviewWideCheckpoint = "/models/wide.safetensors"
      XCTAssertEqual(store.generationRequestKey(for: clip), key)
    }
  }

  func testEnabledPreviewAddsOnlyEnvelopePreferenceAndPreservesRecipeBytesAndHash() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe = root.appendingPathComponent("recipe.json")
    let original = Data("abc".utf8)
    try original.write(to: recipe)
    let standard = root.appendingPathComponent("standard.safetensors")
    let wide = root.appendingPathComponent("wide.safetensors")
    try Data("standard".utf8).write(to: standard)
    try Data("wide".utf8).write(to: wide)
    let payload: [String: Any] = ["recipePath": recipe.path]
    let output = root.appendingPathComponent("output")
    let baseline = try NativeVideoJobRequest.ltx(payload: payload, runtime: settings([:]), output: output)
    for (mode, selected) in [("taeltx2_3", standard), ("taeltx2_3_wide", wide)] {
      let runtime = try settings(["ltx25PreviewDecoder": mode,
        "ltx25PreviewCheckpoint": mode == "taeltx2_3" ? standard.path : "/missing/unselected-standard",
        "ltx25PreviewWideCheckpoint": mode == "taeltx2_3_wide" ? wide.path : "/missing/unselected-wide"])
      var envelope = try NativeVideoJobRequest.ltx(payload: payload, runtime: runtime, output: output)
      let preview = try XCTUnwrap(envelope.removeValue(forKey: "preview") as? [String: String])
      XCTAssertEqual(preview, ["decoder": mode, "checkpointPath": selected.path])
      envelope["jobID"] = baseline["jobID"]
      XCTAssertTrue(NSDictionary(dictionary: envelope).isEqual(to: baseline))
      XCTAssertEqual(envelope["recipeSHA256"] as? String,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
      XCTAssertEqual(try Data(contentsOf: recipe), original)
    }
  }

  func testExplicitPreviewRejectsUnknownModeAndUnavailableSelectedWeight() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe = root.appendingPathComponent("recipe.json")
    try Data("abc".utf8).write(to: recipe)
    let invalidSettings: [[String: Any]] = [
      ["ltx25PreviewDecoder": "future-decoder"],
      ["ltx25PreviewDecoder": "taeltx2_3"],
      ["ltx25PreviewDecoder": "taeltx2_3", "ltx25PreviewCheckpoint": "  "],
      ["ltx25PreviewDecoder": "taeltx2_3", "ltx25PreviewCheckpoint": "relative.safetensors"],
      ["ltx25PreviewDecoder": "taeltx2_3", "ltx25PreviewCheckpoint": root.path],
      ["ltx25PreviewDecoder": "taeltx2_3_wide", "ltx25PreviewCheckpoint": recipe.path],
      ["ltx25PreviewDecoder": "taeltx2_3_wide", "ltx25PreviewWideCheckpoint": root.appendingPathComponent("missing").path],
    ]
    for fields in invalidSettings {
      XCTAssertThrowsError(try NativeVideoJobRequest.ltx(payload: ["recipePath": recipe.path],
        runtime: settings(fields), output: root.appendingPathComponent("output"))) { error in
        XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("preview"), error.localizedDescription)
      }
    }
  }

  func testOffAndH3IgnoreUnselectedPreviewLinks() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let recipe = root.appendingPathComponent("recipe.json")
    try Data("abc".utf8).write(to: recipe)
    let payload: [String: Any] = ["recipePath": recipe.path]
    let output = root.appendingPathComponent("output")
    for fields: [String: Any] in [[:], ["ltx25PreviewDecoder": "off",
      "ltx25PreviewCheckpoint": "/missing/standard", "ltx25PreviewWideCheckpoint": "/missing/wide"]] {
      XCTAssertNil(try NativeVideoJobRequest.ltx(payload: payload, runtime: settings(fields), output: output)["preview"])
    }
    let active = try settings(["ltx25PreviewDecoder": "taeltx2_3", "ltx25PreviewCheckpoint": "/missing/standard"])
    let h3 = try NativeVideoJobRequest.h3(payload: payload, runtime: active, output: output)
    XCTAssertNil(h3["preview"])
    XCTAssertEqual(h3["engine"] as? String, "h3")
  }

  private func settings(_ additions: [String: Any]) throws -> RuntimeSettings {
    let runtime = RuntimeSettings(root: "/source", pythonPath: "/python", profilesDirectory: "/profiles")
    var object = try encoded(runtime)
    object.merge(additions) { _, value in value }
    return try JSONDecoder().decode(RuntimeSettings.self, from: JSONSerialization.data(withJSONObject: object))
  }

  private func encoded(_ value: RuntimeSettings) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
  }

  private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("tae-preview-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }
}
