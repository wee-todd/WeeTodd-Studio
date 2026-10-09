import Foundation
import XCTest
@testable import StudioCore

extension NativeLTXPreparationTests {
  func dfrFixture(temporal: Bool = true) throws -> (URL, StudioProject, [String: Any]) {
    let (root, original, runtime) = try fixture()
    let file = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
    var components = recipe["components"] as! [String: Any]
    components["loras"] = []
    recipe["components"] = components
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("ordinary.json"))
    var config = recipe["config"] as! [String: Any]
    config["dfr_enabled"] = true
    config["dfr_detailing_lora_path"] = "/models/detail.safetensors"
    config["dfr_detailing_lora_strength"] = 0.5
    config["dfr_temporal_rounds"] = temporal ? 1 : 0
    config["dfr_temporal_upsampler_path"] = temporal ? "/models/temporal.safetensors" : ""
    recipe["config"] = config
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    var project = original
    project.clips[0].profileID = file.path
    return (root, project, runtime)
  }

  func selectDFR(_ project: StudioProject, enabled: Bool = true, rounds: Int = 2,
    strength: Double = 0.7, optIn: Bool = true) throws -> StudioProject {
    var value = project
    var selection = try JSONSerialization.jsonObject(with: JSONEncoder().encode(
      value.clips[0].generationSelection!)) as! [String: Any]
    selection["ltx25DFR"] = ["enabled": enabled, "temporalRounds": rounds,
      "detailingStrength": strength, "experimentalEnabled": optIn]
    value.clips[0].generationSelection = try JSONDecoder().decode(GenerationSelection.self,
      from: JSONSerialization.data(withJSONObject: selection))
    return value
  }

  func testLTXRejectsH3PrecisionAndWeightCacheBeforePreparation() throws {
    let (_, original, runtime) = try fixture()
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request: request(original, runtime)))
    for precision in [true, false] {
      var project = original
      if precision { project.clips[0].generationSelection?.h3VideoDecodePrecision = .float16 }
      else { project.clips[0].generationSelection?.h3TransformerWeightCacheGB = 8 }
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)),
        "LTX must reject H3-only settings rather than silently dropping them")
    }
  }

  func testDFRSelectionPersistsOverridesAndLeavesLegacyRecipeUnchanged() throws {
    let (_, original, runtime) = try dfrFixture()
    let legacy = try NativeLTXPreparation.compose(request: request(original, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((legacy["config"] as! [String: Any])["dfr_temporal_rounds"] as? Int, 1)
    let project = try selectDFR(original)
    let reopened = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(project))
    let result = try NativeLTXPreparation.compose(request: request(reopened, runtime))
    let config = (result["recipe"] as! [String: Any])["config"] as! [String: Any]
    XCTAssertEqual(config["dfr_temporal_rounds"] as? Int, 2)
    XCTAssertEqual(config["dfr_detailing_lora_strength"] as? Double, 0.7)
    XCTAssertTrue(reopened.clips[0].generationSelection!.isModified)
    var reset = reopened
    reset.clips[0].generationSelection?.resetOverrides()
    let restored = try NativeLTXPreparation.compose(request: request(reset, runtime))["recipe"] as! [String: Any]
    XCTAssertEqual((restored["config"] as! [String: Any])["dfr_temporal_rounds"] as? Int, 1)
  }

  func testDFRAutomaticSelectionRequiresMatchingTemporalComponents() throws {
    let (_, original, runtime) = try dfrFixture(temporal: false)
    var project = try selectDFR(original)
    project.clips[0].profileID = "auto"
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project = try selectDFR(original, rounds: 0)
    project.clips[0].profileID = "auto"
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    XCTAssertEqual(((result["recipe"] as! [String: Any])["config"] as! [String: Any])["dfr_enabled"] as? Bool, true)
  }

  func testDFRInvalidSettingsRejectBeforeMediaResolution() throws {
    let (_, original, runtime) = try dfrFixture()
    for project in [try selectDFR(original, optIn: false), try selectDFR(original, rounds: -1),
      try selectDFR(original, rounds: 3), try selectDFR(original, strength: 0),
      try selectDFR(original, strength: 3.1)] {
      XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    }
  }

  func testDFROffDoesNotSilentlyRunAnExplicitDFRProfile() throws {
    let (_, original, runtime) = try dfrFixture()
    var project = try selectDFR(original, enabled: false)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    project.clips[0].profileID = "auto"
    let result = try NativeLTXPreparation.compose(request: request(project, runtime))
    XCTAssertNotEqual(((result["recipe"] as! [String: Any])["config"] as! [String: Any])["dfr_enabled"] as? Bool, true)
  }

  func testDFRDependenciesAndEffectiveOutputFrameRate() throws {
    let (root, original, runtime) = try dfrFixture()
    let project = try selectDFR(original)
    let description = try NativeLTXPreparation.describe(request: request(project, runtime))
    let paths = description["sourcePaths"] as! [String]
    XCTAssertTrue(paths.contains("/models/detail.safetensors"))
    XCTAssertTrue(paths.contains("/models/temporal.safetensors"))
    let file = root.appendingPathComponent("model.json")
    var recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
    var config = recipe["config"] as! [String: Any]
    config["frame_rate"] = 60.0; recipe["config"] = config
    try JSONSerialization.data(withJSONObject: recipe).write(to: file)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request: request(project, runtime)))
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request: request(
      try selectDFR(original, rounds: 1), runtime)))
  }
}
