import XCTest
import LTX25Engine
@testable import LTX25MLX

final class MLXSingleStageSamplingTests:XCTestCase {
  func testSchedulesCountActualNonterminalNegativePasses() throws {
    for (schedule,indices,count) in [(MLXSingleStageSampling.NegativeSchedule.full,Set(0...6),15),(.balanced,Set([0,2,4,6]),12),(.speed,Set([0,4]),10)] {
      let policy=try MLXSingleStageSampling(method:.cfgpp,negativeSchedule:schedule,negativePrompt:"blur")
      XCTAssertEqual(policy.negativeSchedule.indices,indices)
      XCTAssertEqual(policy.transformerEvaluations,count)
      XCTAssertEqual(try policy.schedule().eta,1)
      XCTAssertEqual(try JSONDecoder().decode(MLXSingleStageSampling.self,from:JSONEncoder().encode(policy)),policy)
    }
    XCTAssertEqual(try MLXSingleStageSampling(method:.euler).schedule().eta,0)
    XCTAssertEqual(try MLXSingleStageSampling(method:.ancestral).transformerEvaluations,8)
    XCTAssertThrowsError(try MLXSingleStageSampling(method:.euler,negativeSchedule:.speed))
    XCTAssertThrowsError(try MLXSingleStageSampling(method:.ancestral,negativePrompt:"silently ignored"))
    XCTAssertThrowsError(try JSONDecoder().decode(MLXSingleStageSampling.self,from:Data(#"{"method":"euler","negative_schedule":"full","negative_prompt":"","ignored":true}"#.utf8)))
  }
  func testCompilerAdmitsThirtyTwoGridSingleStageWithoutAnUpscalerAndKeepsOldContract() throws {
    var recipe=StudioRecipeTests().fixture(),config=recipe["config"] as! [String:Any],components=recipe["components"] as! [String:Any]
    let old=try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:recipe),outputDirectory:"/out")
    XCTAssertNil(old.singleStageSampling)
    config["ic_lora_single_stage"]=true;config["stage2_steps"]=0
    config["stage1_sampler"]="euler_ancestral_cfg_pp";config["stage1_eta"]=1
    config["cfg_pp_schedule"]="speed";config["negative_prompt"]="plastic skin"
    config["width"]=480;config["height"]=288;config["generated_keyframes"]=2
    components["spatial_upscaler_path"]="";components["loras"]=[["/models/style.safetensors",0.8]]
    recipe["config"]=config;recipe["components"]=components
    let data=try JSONSerialization.data(withJSONObject:recipe)
    let request=try MLXStudioRecipe.compile(data:data,outputDirectory:"/out")
    XCTAssertEqual(request.version,15);XCTAssertEqual(request.singleStageSampling?.transformerEvaluations,10)
    XCTAssertEqual(request.stageOneLoras.count,1);XCTAssertTrue(request.stageTwoLoras.isEmpty)
    XCTAssertEqual(try request.recipe().low.width,480);XCTAssertEqual(try request.recipe().high.width,480)
    XCTAssertEqual(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request)).singleStageSampling,request.singleStageSampling)
    config["cfg_pp_batched"]=true;recipe["config"]=config
    XCTAssertThrowsError(try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:recipe),outputDirectory:"/out"))
    config["cfg_pp_batched"]=false;config["dfr_enabled"]=true;recipe["config"]=config
    XCTAssertThrowsError(try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:recipe),outputDirectory:"/out"))
  }
}
