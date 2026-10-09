import XCTest
import Foundation
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXIngredientsAncestralTests:XCTestCase {
  private func recipe() -> [String:Any] {
    var value=StudioRecipeTests().fixture(),config=value["config"] as! [String:Any]
    config["duration_seconds"]=5.0;config["ic_lora_single_stage"]=true
    config["stage1_sampler"]="euler_ancestral";config["stage2_steps"]=0
    config["stage1_eta"]=1;config["stage1_s_noise"]=1;config["ancestral_seed_offset"]=10000
    config["cfg_pp_schedule"]="full";config["cfg_pp_batched"]=false;config["negative_prompt"]=""
    config["single_stage_sampler"]="euler_ancestral";value["config"]=config
    var components=value["components"] as! [String:Any]
    components["ic_loras"]=[["/models/ingredients.safetensors",1.2]]
    components["spatial_upscaler_path"]="";value["components"]=components
    value["conditioning"]=["version":1,"task":"control","audio_policy":"generated","inputs":[
      ["id":"sheet","kind":"image","role":"control","control_type":"ingredients_reference_sheet",
       "path":"/sheet.png","sha256":String(repeating:"a",count:64),"strength":1.0,"description":"Two distinct aliens"]]]
    return value
  }
  private func compile(_ value:[String:Any]) throws -> MLXDistilledRequest {
    try StudioRecipeTests().compile(value)
  }
  func testExplicitPlainAncestralCompilesEightPositivePredictionsAndRoundTripsV18() throws {
    let request=try compile(recipe())
    XCTAssertEqual(request.version,18)
    XCTAssertEqual(request.ingredientsSampling.rawValue,"euler_ancestral_bf16_v1")
    XCTAssertEqual(request.ingredientsSampling.transformerEvaluations,8)
    XCTAssertNil(request.guidedSampling);XCTAssertNil(request.singleStageSampling)
    XCTAssertTrue(request.stageOneLoras.isEmpty);XCTAssertTrue(request.stageTwoLoras.isEmpty)
    let replay=try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONEncoder().encode(request))
    XCTAssertEqual(replay.version,18);XCTAssertEqual(replay.ingredientsSampling,request.ingredientsSampling)
    XCTAssertEqual(replay.ingredientsSheet?.adapterStrength,1.2)
  }
  func testPlainAncestralCannotReinterpretLegacyOrCFGPPRequests() throws {
    let request=try compile(recipe())
    var object=try JSONSerialization.jsonObject(with:JSONEncoder().encode(request)) as! [String:Any]
    for version in [6,11,17] {
      var legacy=object;legacy["version"]=version
      if version == 11 {
        for key in ["guided_sampling","automatic_duration","generated_keyframes","single_stage_sampling"] { legacy.removeValue(forKey:key) }
      }
      XCTAssertThrowsError(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:legacy)))
    }
    object["version"]=18
    for policy in ["deterministic_bf16_v1","euler_ancestral_cfg_pp_float32_v1"] {
      object["ingredients_sampling"]=policy
      XCTAssertThrowsError(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:object)))
    }
    object["ingredients_sampling"]="euler_ancestral_bf16_v1";object["task"]="t2v"
    XCTAssertThrowsError(try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:object)))
  }
  func testPlainAncestralRejectsRefinementGuidanceNoiseOverridesAndReducedSheets() throws {
    for (key,value):(String,Any) in [("stage2_steps",3),("stage2_steps",false),("stage1_steps",7),
      ("stage1_sampler","euler"),("stage1_eta",0),("stage1_s_noise",0.5),
      ("ancestral_seed_offset",9999),("video_cfg_scale",4),("negative_prompt","blur"),
      ("cfg_pp_schedule","speed"),("single_stage_sampler",true)] {
      var input=recipe(),config=input["config"] as! [String:Any]
      config[key]=value;input["config"]=config
      XCTAssertThrowsError(try compile(input),key)
    }
    var input=recipe(),conditioning=input["conditioning"] as! [String:Any]
    var sheet=(conditioning["inputs"] as! [[String:Any]])[0]
    sheet["reference_size_policy"]="speed";conditioning["inputs"]=[sheet];input["conditioning"]=conditioning
    XCTAssertThrowsError(try compile(input))
  }
  func testPlainAncestralRejectsAutomaticDurationAndGeneratedSlotsInsteadOfDroppingIntent() throws {
    for (key,value):(String,Any) in [("duration_mode","automatic"),("duration_mode",true),
      ("generated_keyframes",1),("generated_keyframes",true),("generated_keyframes",false),
      ("generated_keyframes",0.5),("generated_keyframes",NSNull())] {
      var input=recipe(),config=input["config"] as! [String:Any]
      config[key]=value;input["config"]=config
      if key == "duration_mode",value as? String == "automatic" {
        var components=input["components"] as! [String:Any]
        components["duration_head_path"]="/models/duration-head.safetensors"
        components["duration_head_header_sha256"]=String(repeating:"b",count:64)
        input["components"]=components
      }
      XCTAssertThrowsError(try compile(input),"v18 must reject rather than erase \(key)=\(value)")
    }
    for zero:Any in [0,0.0] {
      var input=recipe(),config=input["config"] as! [String:Any]
      config["duration_mode"]="manual";config["generated_keyframes"]=zero;input["config"]=config
      XCTAssertEqual(try compile(input).version,18)
    }
  }
  func testNoOverridePreservesTheExistingDeterministicIngredientsRequest() throws {
    var input=recipe(),config=input["config"] as! [String:Any]
    config.removeValue(forKey:"single_stage_sampler");config["stage2_steps"]=3;input["config"]=config
    let legacy=try compile(input)
    XCTAssertEqual(legacy.version,6);XCTAssertEqual(legacy.ingredientsSampling.rawValue,"deterministic_bf16_v1")
    config["single_stage_sampler"]="euler_ancestral_cfg_pp";input["config"]=config
    let cfgpp=try compile(input)
    XCTAssertEqual(cfgpp.version,11);XCTAssertEqual(cfgpp.ingredientsSampling.transformerEvaluations,16)
  }
  func testPlainAncestralUsesPythonSplitNoiseBF16StateAndOnePredictionPerUpdate() throws {
    let url=try XCTUnwrap(Bundle.module.url(forResource:"ingredients-plain-ancestral",
      withExtension:"json",subdirectory:"Fixtures"))
    let fixture=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
    let initial=try XCTUnwrap(fixture["initial"] as? [String:[NSNumber]])
    let expected=try XCTUnwrap(fixture["expectedStates"] as? [[String:[[NSNumber]]]])
    let draws=try XCTUnwrap(fixture["noise"] as? [[String:Any]])
    let policy=try XCTUnwrap(MLXIngredientsSampling(rawValue:"euler_ancestral_bf16_v1"))
    try Device.withDefaultDevice(.gpu) {
      let source=MLXSamplingTests(),f=try source.fixture()
      var inputs=f.inputs.mapValues { MLXArray($0) }
      inputs["video_latent"]=MLXArray(initial["video"]!.map(\.floatValue),[5,128])
      inputs["audio_latent"]=MLXArray(initial["audio"]!.map(\.floatValue),[3,128])
      let condition=try MLXVideoDenoiseCondition(clean:inputs["video_latent"]!,mask:[1,0,1,0,1])
      let schedule=try MLXSingleStageRipple.schedule(sampling:policy)
      XCTAssertEqual(schedule.sigmas,fixture["sigmas"] as? [Double])
      XCTAssertEqual(schedule.eta,1);XCTAssertEqual(schedule.noiseStrength,1)
      let generator=MLXSingleStageRipple.ancestralNoise(seed:43)
      let runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
      var drawIndex=0,evaluations:[Int]=[],updates=0
      func zero(_ name:String,_ shape:[Int]) throws -> MLXWeight {
        try MLXWeight(dense:MLXArray.zeros(shape))
      }
      let result=try runner.evaluate(inputs,schedule:schedule,videoConditioning:condition,
        bfloat16State:["video","audio"],fixedWeights:zero,
        blockWeights:{ _,_,shape in try zero("",shape) },noise:{ index,name,shape in
          XCTAssertEqual(index,drawIndex/2);XCTAssertEqual(name,draws[drawIndex]["modality"] as? String)
          XCTAssertEqual([1]+shape,draws[drawIndex]["shape"] as? [Int])
          let value=try generator(index,name,shape)
          let rows=try XCTUnwrap(draws[drawIndex]["values"] as? [[NSNumber]])
          let wanted=rows.flatMap { $0.map(\.floatValue) }
          let actual=value[0..<shape[0],0..<8].asArray(Float.self)
          XCTAssertEqual(actual.count,wanted.count)
          for (a,b) in zip(actual,wanted) { XCTAssertEqual(a,b,accuracy:0.000002) }
          drawIndex+=1;return value
        },stageProgress:{ evaluation,event in
          if event.stage == "transformer" { evaluations.append(evaluation) }
        },preview:{ output,event in
          updates+=1
          for name in ["video","audio"] {
            let value=output[name]!
            XCTAssertEqual(value.asArray(Float.self),value.asType(.bfloat16).asType(.float32).asArray(Float.self))
            let wanted=expected[event.completedSteps-1][name]!.flatMap { $0.map(\.floatValue) }
            let actual=value[0..<value.shape[0],0..<8].asArray(Float.self)
            for (a,b) in zip(actual,wanted) { XCTAssertEqual(a,b,accuracy:0.000001,"update \(event.completedSteps) \(name)") }
          }
          for row in [1,3] { XCTAssertEqual(output["video"]![row].asArray(Float.self),inputs["video_latent"]![row].asArray(Float.self)) }
          XCTAssertEqual(runner.residentWeightBytes,0)
        })
      XCTAssertEqual(drawIndex,14);XCTAssertEqual(updates,8)
      XCTAssertEqual(evaluations,[1,2,3,4,5,6,7,8]);XCTAssertEqual(runner.residentWeightBytes,0)
      for row in [1,3] { XCTAssertEqual(result["video"]![row].asArray(Float.self),inputs["video_latent"]![row].asArray(Float.self)) }
      let geometry=try AVGeometry(width:768,height:448,frames:121,fps:24)
      XCTAssertEqual(MLXSingleStageRipple.leadingMarkerRows(geometry:geometry,task:.ingredients,sampling:policy),0)
    }
  }

}
