import Foundation
import XCTest
@testable import StudioCore

final class NativeLTXIngredientsSamplingTests:XCTestCase {
  private func fixture(sampler:String?=nil) throws -> (URL,StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    var config:[String:Any]=["pipeline_mode":"distilled","stage1_steps":8,"stage2_steps":3,
      "frame_rate":24,"width":768,"height":448,"seed":1,"duration_seconds":5,"ic_lora_single_stage":true]
    if let sampler { config["single_stage_sampler"]=sampler }
    if sampler == "euler_ancestral" {
      config["stage2_steps"]=0;config["stage1_sampler"]="euler_ancestral";config["stage2_sampler"]="euler"
      config["stage1_eta"]=1;config["stage1_s_noise"]=1;config["ancestral_seed_offset"]=10000
      config["cfg_pp_schedule"]="full";config["cfg_pp_batched"]=false;config["negative_prompt"]=""
    }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":"profile prompt",
      "components":["transformer_path":"/models/transformer","loras":[],"ic_loras":[["/models/ingredients.safetensors",1.2]]],
      "config":config,"conditioning":["version":1,"task":"control","inputs":[]]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("ingredients.json"))
    let file=root.appendingPathComponent("sheet.png");try Data([1]).write(to:file)
    let asset=MediaAsset(name:"Two character sheet",kind:.image,path:file.path)
    var attachment=Attachment(assetID:asset.id,role:.control)
    attachment.controlType="ingredients_reference_sheet";attachment.description="A warrior on the left and a crowned king on the right."
    var project=StudioProject(),clip=Clip(engine:.ltx25)
    clip.prompt="The warrior speaks to the king in the hall.";clip.duration=5
    clip.generationWidth=768;clip.generationHeight=448;clip.generationSelection=GenerationSelection(task:"control")
    clip.attachments=[attachment];project.clips=[clip];project.assets=[asset]
    return(root,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  private func request(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),"clipID":project.clips[0].id.uuidString,"runtime":runtime]
  }
  private func compose(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    try NativeLTXPreparation.compose(request:request(project,runtime))
  }
  func testDefaultIsStableAndIngredientsAdvertisesOnlySupportedSamplerChoices() throws {
    let(_,project,runtime)=try fixture()
    let result=try compose(project,runtime),recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    let config=try XCTUnwrap(recipe["config"] as? [String:Any])
    XCTAssertNil(config["single_stage_sampler"]);XCTAssertEqual(config["stage2_steps"] as? Int,3)
    XCTAssertFalse(String(decoding:try JSONEncoder().encode(project.clips[0].generationSelection!),as:UTF8.self).contains("ltx25SingleStage"))
    let generation=try XCTUnwrap((result["report"] as? [String:Any])?["generation"] as? [String:Any])
    XCTAssertEqual(generation["referenceFamily"] as? String,"ingredients")
    XCTAssertEqual(generation["singleStageAvailable"] as? Bool,true)
    XCTAssertEqual(generation["singleStageMethods"] as? [String],["euler_ancestral","euler_ancestral_cfg_pp"])
    XCTAssertEqual(generation["ordinaryKeyframesAvailable"] as? Bool,false)
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,8)
  }
  func testExplicitAncestralUsesEightNoiseUpdatesAndResetRestoresInheritedRecipe() throws {
    let(_,original,runtime)=try fixture();var project=original
    let inherited=try compose(original,runtime)["recipe"] as! [String:Any]
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.ancestral,experimentalEnabled:true)
    let before=project,result=try compose(project,runtime),recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    let config=try XCTUnwrap(recipe["config"] as? [String:Any])
    XCTAssertEqual(config["single_stage_sampler"] as? String,"euler_ancestral")
    XCTAssertEqual(config["stage1_sampler"] as? String,"euler_ancestral")
    XCTAssertEqual(config["stage1_steps"] as? Int,8);XCTAssertEqual(config["stage2_steps"] as? Int,0)
    XCTAssertEqual(config["stage1_eta"] as? Int,1);XCTAssertEqual(config["stage1_s_noise"] as? Int,1)
    XCTAssertEqual(config["ancestral_seed_offset"] as? Int,10000)
    XCTAssertEqual(config["cfg_pp_schedule"] as? String,"full");XCTAssertEqual(config["negative_prompt"] as? String,"")
    let conditioning=try XCTUnwrap(recipe["conditioning"] as? [String:Any]),inputs=try XCTUnwrap(conditioning["inputs"] as? [[String:Any]])
    XCTAssertEqual(inputs.count,1);XCTAssertEqual(inputs[0]["control_type"] as? String,"ingredients_reference_sheet")
    XCTAssertEqual((inputs[0]["sha256"] as? String)?.count,64)
    XCTAssertEqual(inputs[0]["description"] as? String,"A warrior on the left and a crowned king on the right.")
    let generation=try XCTUnwrap((result["report"] as? [String:Any])?["generation"] as? [String:Any])
    XCTAssertEqual(generation["referenceFamily"] as? String,"ingredients")
    XCTAssertEqual(generation["singleStageEnabled"] as? Bool,true)
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,8)
    XCTAssertEqual((generation["controls"] as? [String:Any])?["refinementSteps"] as? Int,0)
    XCTAssertEqual(project,before)
    XCTAssertEqual(try JSONDecoder().decode(StudioProject.self,from:JSONEncoder().encode(project)),project)
    project.clips[0].generationSelection?.resetOverrides()
    XCTAssertEqual((try compose(project,runtime)["recipe"] as! [String:Any])["config"] as? NSDictionary,inherited["config"] as? NSDictionary)
  }
  func testCFGPPOverrideRetainsSixteenEvaluationsAndExistingImportedRecipe() throws {
    let(_,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.cfgpp,experimentalEnabled:true)
    let result=try compose(project,runtime),recipe=try XCTUnwrap(result["recipe"] as? [String:Any]),config=try XCTUnwrap(recipe["config"] as? [String:Any])
    XCTAssertEqual(config["single_stage_sampler"] as? String,"euler_ancestral_cfg_pp")
    XCTAssertEqual(config["stage2_steps"] as? Int,3)
    XCTAssertEqual(config["stage1_sampler"] as? String,"euler_ancestral")
    XCTAssertEqual(config["negative_prompt"] as? String,"")
    let generation=try XCTUnwrap((result["report"] as? [String:Any])?["generation"] as? [String:Any])
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,16)
    let(_,imported,importedRuntime)=try fixture(sampler:"euler_ancestral_cfg_pp")
    let importedRecipe=try compose(imported,importedRuntime)["recipe"] as! [String:Any]
    XCTAssertEqual((importedRecipe["config"] as? [String:Any])?["stage2_steps"] as? Int,3)
    XCTAssertEqual((importedRecipe["config"] as? [String:Any])?["single_stage_sampler"] as? String,"euler_ancestral_cfg_pp")
    XCTAssertNil(imported.clips[0].generationSelection?.ltx25SingleStage)
  }
  func testImportedAncestralRecipeStaysIngredientsWithoutInventingAnEditorOverride() throws {
    let(_,project,runtime)=try fixture(sampler:"euler_ancestral")
    let before=project,result=try compose(project,runtime),recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    let config=try XCTUnwrap(recipe["config"] as? [String:Any])
    XCTAssertEqual(config["single_stage_sampler"] as? String,"euler_ancestral")
    XCTAssertEqual(config["stage2_steps"] as? Int,0);XCTAssertEqual(config["stage1_eta"] as? Int,1)
    let generation=try XCTUnwrap((result["report"] as? [String:Any])?["generation"] as? [String:Any])
    XCTAssertEqual(generation["referenceFamily"] as? String,"ingredients")
    XCTAssertEqual(generation["singleStageEnabled"] as? Bool,true)
    XCTAssertEqual(generation["supportedTasks"] as? [String],["control"])
    XCTAssertEqual(generation["ordinaryKeyframesAvailable"] as? Bool,false)
    XCTAssertEqual((generation["controls"] as? [String:Any])?["evaluations"] as? Int,8)
    XCTAssertNil(project.clips[0].generationSelection?.ltx25SingleStage)
    XCTAssertEqual(project,before)
  }
  func testIngredientsRejectsUnsupportedOverridesBeforeMediaWork() throws {
    let(_,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.ltx25SingleStage = .init()
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.euler,experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime))
    for schedule in [LTX25NegativeSchedule.balanced,.speed] {
      project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.cfgpp,negativeSchedule:schedule,experimentalEnabled:true)
      XCTAssertThrowsError(try compose(project,runtime))
    }
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.ancestral,experimentalEnabled:true)
    project.clips[0].negativePrompt="unused negative"
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25SingleStage = .init(method:.cfgpp,experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].negativePrompt="";project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:1,experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25Keyframes=nil
    var anchor=project.clips[0].attachments[0];anchor.role = .first
    project.clips[0].attachments.append(anchor)
    XCTAssertThrowsError(try compose(project,runtime))
  }
}
