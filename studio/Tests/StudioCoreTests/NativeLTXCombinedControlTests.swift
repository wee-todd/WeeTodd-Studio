import Foundation
import CryptoKit
import XCTest
@testable import StudioCore

extension NativeLTXPreparationTests {
  func combinedControlFixture(_ family: String) throws -> (URL, StudioProject, [String: Any]) {
    let (root, original, runtime) = try fixture()
    try specializedControlProfile(root, family: family)
    var project = original
    project.clips[0].generationSelection = GenerationSelection(task: "control")
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method: .cfgpp,
      negativeSchedule: .speed, experimentalEnabled: true)
    project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount: 2, experimentalEnabled: true)
    project.clips[0].generationWidth = 576; project.clips[0].generationHeight = 320
    project.clips[0].negativePrompt = "blur"
    var attachments: [Attachment] = []
    for role in family == "crossview_warp" ? ["warp", "source"] : ["control"] {
      let url = root.appendingPathComponent(role + ".mp4"); try Data([1, 2]).write(to: url)
      let asset = MediaAsset(name: role, kind: .video, path: url.path); project.assets.append(asset)
      var attachment = Attachment(assetID: asset.id, role: .control)
      attachment.controlType = family == "union" ? "pose_skeleton" : family
      if family == "crossview_warp" { attachment.referenceRole = role }
      attachments.append(attachment)
    }
    for role: MediaRole in [.first, .keyframe, .last] {
      let url = root.appendingPathComponent(role.rawValue + ".png"); try Data([1]).write(to: url)
      let asset = MediaAsset(name: role.rawValue, kind: .image, path: url.path); project.assets.append(asset)
      var attachment = Attachment(assetID: asset.id, role: role)
      if role == .keyframe { attachment.time = 1 }
      attachments.append(attachment)
    }
    project.clips[0].attachments = attachments
    return (root, project, runtime)
  }

  func testCombinedControlsPreserveOrderedGuidesAndTimedImages() throws {
    for family in ["union", "motion_track", "crossview_warp"] {
      let (_, project, runtime) = try combinedControlFixture(family)
      let frozen = project
      let result = try NativeLTXPreparation.compose(request: request(project, runtime))
      let recipe = result["recipe"] as! [String: Any], config = recipe["config"] as! [String: Any]
      let condition = recipe["conditioning"] as! [String: Any], inputs = condition["inputs"] as! [[String: Any]]
      XCTAssertEqual(config["stage2_steps"] as? Int, 0)
      XCTAssertEqual(config["generated_keyframes"] as? Int, 2)
      XCTAssertEqual(inputs.filter { $0["role"] as? String == "keyframe" }.count, 3)
      XCTAssertEqual(inputs.filter { $0["role"] as? String == "control" }.count, family == "crossview_warp" ? 2 : 1)
      XCTAssertEqual(project, frozen)
      let description = try NativeLTXPreparation.describe(request: request(project, runtime))
      let generation = description["generation"] as! [String: Any]
      XCTAssertEqual(generation["singleStageEnabled"] as? Bool, true)
      XCTAssertEqual(generation["ordinaryKeyframesAvailable"] as? Bool, true)
      XCTAssertEqual((generation["controls"] as! [String: Any])["evaluations"] as? Int, 10)
      XCTAssertEqual((result["report"] as! [String: Any])["productionQualified"] as? Bool, false)
    }
  }

  func testResolvedRendererSummaryFollowsSingleStageScheduleAndInheritedProfile() throws {
    let (root, original, runtime) = try combinedControlFixture("union")
    var project = original
    for (schedule, evaluations) in [(LTX25NegativeSchedule.full, 15), (.balanced, 12), (.speed, 10)] {
      project.clips[0].generationSelection?.ltx25SingleStage?.negativeSchedule = schedule
      let result = try NativeLTXPreparation.describe(request: request(project, runtime))
      let descriptor = try JSONDecoder().decode(GenerationDescriptor.self,
        from: JSONSerialization.data(withJSONObject: result["generation"]!))
      XCTAssertEqual(descriptor.ltx25ExecutionSummary, "single-stage · \(evaluations) evaluations")
    }
    let recipe = try NativeLTXPreparation.compose(request: request(project, runtime))["recipe"]!
    try JSONSerialization.data(withJSONObject: recipe).write(to: root.appendingPathComponent("model.json"))
    project.clips[0].generationSelection?.ltx25SingleStage = nil
    let inherited = try NativeLTXPreparation.describe(request: request(project, runtime))
    let descriptor = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: inherited["generation"]!))
    XCTAssertEqual(descriptor.ltx25ExecutionSummary, "single-stage · 10 evaluations")
  }

  func testCombinedControlPreparationFreezesActiveGuideGridWithoutConvertingAnchors() async throws {
    for family in ["union", "motion_track"] {
      let (root, original, runtime) = try combinedControlFixture(family)
      let movie = try await continuityMovie(root)
      var project = original
      let guideID = project.clips[0].attachments[0].assetID
      project.assets[project.assets.firstIndex(where: { $0.id == guideID })!].path = movie.path
      project.clips[0].duration = 4.0 / 3
      let destination = root.appendingPathComponent("prepared")
      let result = try await NativeLTXPreparation.prepareWithMedia(request: request(project, runtime), destination: destination)
      let recipe = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: result["recipePath"] as! String))) as! [String: Any]
      let inputs = (recipe["conditioning"] as! [String: Any])["inputs"] as! [[String: Any]]
      let guide = inputs.first(where: { $0["role"] as? String == "control" })!
      let bytes = try Data(contentsOf: URL(fileURLWithPath: guide["path"] as! String))
      XCTAssertEqual(bytes.count, 33 * 288 * 160 * 3)
      XCTAssertEqual(guide["sha256"] as? String, SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
      let anchors = inputs.filter { $0["role"] as? String == "keyframe" }
      XCTAssertEqual(anchors.count, 3)
      XCTAssertTrue(anchors.allSatisfy { $0["kind"] as? String == "image" && $0["format"] == nil })
      XCTAssertEqual(anchors.map { $0["path"] as! String }, original.clips[0].attachments.dropFirst().map { item in
        original.assets.first(where: { $0.id == item.assetID })!.path
      })
      let reports = (result["report"] as! [String: Any])["controlGuides"] as! [[String: Any]]
      XCTAssertEqual(reports.count, 1); XCTAssertEqual(reports[0]["width"] as? Int, 288)
      XCTAssertEqual(reports[0]["height"] as? Int, 160)
    }
  }

  func testImportedSingleStageControlProfilePreservesItsEffectiveSampler() throws {
    for family in ["union", "motion_track", "crossview_warp"] {
      let (root, original, runtime) = try combinedControlFixture(family)
      let composed = try NativeLTXPreparation.compose(request: request(original, runtime))["recipe"] as! [String: Any]
      try JSONSerialization.data(withJSONObject: composed).write(to: root.appendingPathComponent("model.json"))
      var project = original
      project.clips[0].generationSelection?.ltx25SingleStage = nil
      let result = try NativeLTXPreparation.compose(request: request(project, runtime))
      let config = (result["recipe"] as! [String: Any])["config"] as! [String: Any]
      XCTAssertEqual(config["stage1_sampler"] as? String, "euler_ancestral_cfg_pp")
      XCTAssertEqual(config["cfg_pp_schedule"] as? String, "speed")
      XCTAssertEqual(config["negative_prompt"] as? String, "blur")
      XCTAssertEqual(config["stage2_steps"] as? Int, 0)
      project.clips[0].negativePrompt = ""
      let inherited = try NativeLTXPreparation.compose(request: request(project, runtime))["recipe"] as! [String: Any]
      XCTAssertEqual((inherited["config"] as! [String: Any])["negative_prompt"] as? String, "blur")
    }
  }
}
