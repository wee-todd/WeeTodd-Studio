import Foundation
import XCTest
@testable import StudioCore

final class NativeLTXSingleStagePreparationTests:XCTestCase {
  private func fixture() throws -> (URL,StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":"old",
      "components":["transformer_path":"/models/transformer"],
      "config":["pipeline_mode":"distilled","stage1_steps":8,"stage2_steps":3,"frame_rate":24,
        "width":768,"height":448,"seed":1,"duration_seconds":5],
      "conditioning":["version":1,"task":"t2v","inputs":[]]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("model.json"))
    var project=StudioProject(),clip=Clip(engine:.ltx25)
    clip.prompt="A brass cup on a stone ledge.";clip.duration=2
    clip.generationWidth=768;clip.generationHeight=448;clip.generationSelection=GenerationSelection(task:"t2v")
    project.clips=[clip]
    return(root,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  private func compose(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    try NativeLTXPreparation.compose(request:["project":JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "clipID":project.clips[0].id.uuidString,"runtime":runtime])
  }
  func testOldSelectionBytesAndRecipeRemainStableWhileNewControlsFreezeActualSchedule() throws {
    let(_,original,runtime)=try fixture();var project=original
    let old=try compose(project,runtime)["recipe"] as! [String:Any]
    XCTAssertNil(original.clips[0].generationSelection?.ltx25SingleStage)
    let data=try JSONEncoder().encode(original.clips[0].generationSelection!)
    XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("ltx25SingleStage"))
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.cfgpp,negativeSchedule:.speed,experimentalEnabled:true)
    project.clips[0].negativePrompt="plastic skin";project.clips[0].generationWidth=480;project.clips[0].generationHeight=288
    let before=project,output=try compose(project,runtime),content=output["recipe"] as! [String:Any],config=content["config"] as! [String:Any]
    XCTAssertEqual(config["stage1_sampler"] as? String,"euler_ancestral_cfg_pp")
    XCTAssertEqual(config["stage2_steps"] as? Int,0);XCTAssertEqual(config["stage1_steps"] as? Int,8)
    XCTAssertEqual(config["cfg_pp_schedule"] as? String,"speed");XCTAssertEqual(config["cfg_pp_batched"] as? Bool,false)
    XCTAssertEqual(config["negative_prompt"] as? String,"plastic skin")
    XCTAssertEqual(project,before)
    XCTAssertEqual((output["report"] as! [String:Any])["nativePreparation"] as? String,"swift")
    let describe=try NativeLTXPreparation.describe(request:["project":JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "clipID":project.clips[0].id.uuidString,"runtime":runtime])
    let generation=describe["generation"] as! [String:Any]
    XCTAssertEqual(generation["singleStageEnabled"] as? Bool,true)
    XCTAssertEqual((generation["controls"] as! [String:Any])["evaluations"] as? Int,10)
    XCTAssertEqual((try compose(original,runtime)["recipe"] as! [String:Any])["config"] as? NSDictionary,old["config"] as? NSDictionary)
    var selection=project.clips[0].generationSelection!
    XCTAssertEqual(try JSONDecoder().decode(GenerationSelection.self,from:JSONEncoder().encode(selection)),selection)
    selection.resetOverrides();XCTAssertNil(selection.ltx25SingleStage)
  }
  func testUnapprovedAndUnsupportedSettingsRejectBeforePreparation() throws {
    let(_,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.ltx25SingleStage = .init()
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25SingleStage?.experimentalEnabled=true
    project.clips[0].negativePrompt="ignored text"
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].negativePrompt="";project.clips[0].generationSelection?.ltx25SingleStage?.negativeSchedule = .speed
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25SingleStage?.negativeSchedule = .full
    project.clips[0].generationSelection?.ltx25AutomaticDuration = .init(experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25AutomaticDuration=nil
    project.clips[0].generationSelection?.ltx25Guidance = .init(mode:.guidedHQ,experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime))
  }
}
