import Foundation
import ModelZoo
import XCTest
@testable import DrawThingsTransport

final class ConfigurationTests: XCTestCase {
  var request: [String: Any] {
    ["modelID": "flux_2_klein_4b_q8p.ckpt", "operation": "image",
     "configuration": ["width": 512, "height": 512, "steps": 4, "seed": 42],
     "inputs": [], "loras": []]
  }

  func testOfflineEstimateUsesCanonicalPixelDimensions() throws {
    let result = try ComputeEstimate.evaluate(request)
    XCTAssertEqual(result["cu"] as? Int, 208)
    let effective = try XCTUnwrap(result["configuration"] as? [String: Any])
    XCTAssertEqual(effective["width"] as? Int, 512)
    XCTAssertEqual(effective["steps"] as? Int, 4)
    XCTAssertEqual(effective["seed"] as? Int, 42)
  }

  func testKrea2CanvasStrengthAndReferenceCapabilities() throws {
    let model = "krea_2_turbo_q8p.ckpt"
    let old = ModelZoo.overrideMapping[model]
    defer { ModelZoo.overrideMapping[model] = old }
    ModelZoo.overrideMapping[model] = ModelZoo.Specification(name: "Krea 2 test", file: model, prefix: "", version: .krea2)
    var value = request
    value["modelID"] = "krea_2_turbo_q8p.ckpt"
    value["configuration"] = ["width": 512, "height": 512, "steps": 8, "seed": 42, "strength": 0.35]
    let result = try ComputeEstimate.evaluate(value)
    XCTAssertEqual((result["configuration"] as? [String: Any])?["strength"] as? Double ?? 0, 0.35, accuracy: 0.001)
    let rules = try XCTUnwrap(Capabilities.rules(for: "krea_2_turbo_q8p.ckpt"))
    let image = ((rules["operations"] as? [String: Any])?["image"] as? [String: Any])
    XCTAssertEqual(image?["inputRoleCombinations"] as? [[String]], [[], ["canvas"]])
  }

  func testCustomSDXLAndQwenEditReferencesUseDeclaredCapabilities() throws {
    let custom = "user-custom-sdxl.ckpt", edit = "user-qwen-edit.ckpt"
    let previousCustom = ModelZoo.overrideMapping[custom], previousEdit = ModelZoo.overrideMapping[edit]
    defer { ModelZoo.overrideMapping[custom] = previousCustom; ModelZoo.overrideMapping[edit] = previousEdit }
    ModelZoo.overrideMapping[custom] = ModelZoo.Specification(name: "Custom SDXL", file: custom, prefix: "", version: .sdxlBase)
    var value = request; value["modelID"] = custom
    XCTAssertEqual(try Configuration.resolve(value).model, custom)
    XCTAssertNotNil(Capabilities.rules(for: custom))
    XCTAssertFalse(Capabilities.supportsMoodboard(custom))
    var specification = ModelZoo.Specification(name: "Custom Qwen", file: edit, prefix: "", version: .qwenImage)
    specification.modifier = .qwenimageEdit2511; ModelZoo.overrideMapping[edit] = specification
    XCTAssertTrue(Capabilities.supportsMoodboard(edit))
    value["modelID"] = edit
    value["inputs"] = [["role": "moodboard", "path": "/reference.png", "sha256": String(repeating: "a", count: 64), "strength": 0.6, "fit": "fit"]]
    XCTAssertEqual(try Configuration.resolve(value).model, edit)
    value["modelID"] = custom
    XCTAssertThrowsError(try Configuration.resolve(value))
    specification.modifier = .none; ModelZoo.overrideMapping[edit] = specification
    XCTAssertFalse(Capabilities.supportsMoodboard(edit))
  }

  func testRejectsUnrepresentableDimensionsAndBooleans() {
    for width: Any in [513, 0, -64, true, 512.5, 4194304] {
      var value = request
      value["configuration"] = ["width": width, "height": 512, "steps": 4, "seed": 42]
      XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
    }
  }

  func testUnknownModelDoesNotBecomeDefaultModel() {
    var value = request
    value["modelID"] = "unrecognized-model"
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testUnknownConfigurationCannotBeSilentlyIgnored() {
    var value = request
    value["configuration"] = ["width": 512, "height": 512, "mysteryOption": 1]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testConditioningNeedsAnImplementedMapping() {
    var value = request
    value["inputs"] = [["role": "reference", "path": "image.png"]]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
    value = request
    value["loras"] = [["modelID": "style", "strength": 0.5]]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testVideoRequiresExplicitTiming() {
    var value = request
    value["operation"] = "video"
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }
}
