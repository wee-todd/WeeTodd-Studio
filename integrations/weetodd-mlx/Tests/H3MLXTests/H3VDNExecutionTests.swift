import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VDNExecutionTests:XCTestCase {
  func testVDNRequestRequiresItsEulerScheduleAndRejectsOtherAdapters() throws {
    let selection=try H3VDNSelection(stage:URL(fileURLWithPath:"/model/stage"),variant:.eightStep)
    func request(_ steps:Int,_ method:H3SamplingMethod = .euler,_ adapters:[H3LoRAAdapter] = []) throws -> H3T2VARequest {
      try H3T2VARequest(prompt:"A man nods.",width:672,height:384,durationSeconds:5,seed:42,
        requestedSteps:steps,transformer:URL(fileURLWithPath:"/model/base.safetensors"),
        qwenPages:URL(fileURLWithPath:"/model/qwen"),tokenizer:URL(fileURLWithPath:"/model/tokenizer.json"),
        videoVAE:URL(fileURLWithPath:"/model/video.safetensors"),audioVAE:URL(fileURLWithPath:"/model/audio.safetensors"),
        loRAAdapters:adapters,vdn:selection,samplingMethod:method)
    }
    XCTAssertEqual(try request(9).vdn?.variant,.eightStep)
    XCTAssertThrowsError(try request(5))
    XCTAssertThrowsError(try request(9,.resMultistep))
    XCTAssertThrowsError(try request(9,.euler,[H3LoRAAdapter(url:URL(fileURLWithPath:"/model/other.safetensors"),strength:1)]))
  }

  func testVDNLayoutUsesOnlyTargetGridAndKeepsPromptAndAudioGlobal() throws {
    let geometry=try H3Geometry(width:64,height:64,durationSeconds:2.5)
    let packed=try H3PackedLayout(geometry:geometry,textTags:[1,1,1],anchors:[])
    let layout=try H3VDNLayout(packed:packed)
    XCTAssertEqual(layout.videoStart,packed.videoStart)
    XCTAssertEqual(layout.frames,geometry.videoLatentFrames)
    XCTAssertEqual(layout.tokensPerFrame,4)
    XCTAssertEqual(layout.textLength,3)
    let anchored=try H3PackedLayout(geometry:geometry,textTags:[1,1,1],anchors:[.first])
    XCTAssertThrowsError(try H3VDNLayout(packed:anchored))
  }
  func testSharedBlockConsumesProjectedHybridAttentionExactlyOnce() throws {
    let input=MLXArray((1...16).map(Float.init),[1,2,8]).asType(.bfloat16)
    var slots=[Float](repeating:0,count:18*8)
    for i in 16..<24 { slots[i]=0.5 }
    let modulation=MLXArray(slots,[1,144]).asType(.bfloat16)
    let angles=H3RotaryAngles(rows:2,cosine:MLXArray.ones([1,1,2,4],dtype:.bfloat16),
      sine:MLXArray.zeros([1,1,2,4],dtype:.bfloat16))
    let result=try H3TransformerBlock.evaluateKernel(input:input,modulation:modulation,
      modulationIndices:MLXArray.zeros([2],dtype:.int32),angles:angles,
      hiddenWidth:8,heads:2,headWidth:4,feedWidth:4,rotaryWidth:4,
      read:{ _,shape in MLXArray.ones(shape,dtype:.bfloat16) },
      project:{ activation,_,rows,_,_ in MLXArray.zeros([1,activation.shape[1],rows],dtype:.bfloat16) },
      hybridAttention:{ first,_,_,_,_ in first*2 })
    let expected=input + MLXFast.rmsNorm(input,weight:MLXArray.ones([8],dtype:.bfloat16),eps:1e-5)
    XCTAssertEqual(result.asArray(Float.self),expected.asArray(Float.self))
    XCTAssertGreaterThan(max(abs(result-input)).item(Float.self),1)
  }

  private func recipe(variant:String="8_step") throws -> [String:Any] {
    let stage=variant == "8_step" ? "stage-dmd-step-250" : "stage-b-step-2000"
    let root="/models/vdn",base=root+"/"+stage
    var adapters:[[String:Any]]=[["path":base+"/adapters/default/adapter_model.safetensors","strength":1.0,
      "profile":"standard","qkv_layout":"contiguous_qkv","start_after_evaluations":0]]
    if variant == "8_step" { adapters.append(["path":base+"/adapters/turbo/adapter_model.safetensors","strength":1.0,
      "profile":"turbo","qkv_layout":"contiguous_qkv","start_after_evaluations":0]) }
    return ["format":"weetodd-headless-v2","engine":"h3","prompt":"A man nods.",
      "components":["task":"t2va","transformer":"/models/base.safetensors","text_encoder":"/models/qwen",
        "tokenizer":"/models/tokenizer.json","video_vae":"/models/video.safetensors","audio_vae":"/models/audio.safetensors"],
      "config":["width":672,"height":384,"duration_seconds":5.0,"seed":42,"steps":variant == "8_step" ? 9 : 51],
      "conditioning":["version":1,"task":"t2v","inputs":[],"audio_policy":"generated"],
      "loras":["version":1,"adapters":adapters],
      "vdn":["repository":root,"checkpoint":base,"model_spec":base+"/model_spec.json",
        "linear_branch":base+"/linear_branch/model.safetensors","default_adapter":adapters[0]["path"]!,
        "turbo_adapter":variant == "8_step" ? adapters[1]["path"]! : NSNull(),
        "schedule_points":variant == "8_step" ? 9 : 51,"stage":stage,"inference_backend":"verified"]]
  }

  func testVDNRecipePreservesReleasedStagesAndRejectsChangedStackAndControls() throws {
    for variant in ["8_step","50_step"] {
      let root=try recipe(variant:variant)
      let request=try H3VDNStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:root))
      XCTAssertEqual(request.vdn?.variant.rawValue,variant)
      XCTAssertEqual(request.requestedSteps,variant == "8_step" ? 9 : 51)
      XCTAssertTrue(request.loRAAdapters.isEmpty)
    }
    var root=try recipe()
    func compile() throws -> H3T2VARequest { try H3VDNStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:root)) }
    root["continuation"]=["version":2,"context_frames":22]
    XCTAssertThrowsError(try compile());root.removeValue(forKey:"continuation")
    var vdn=root["vdn"] as! [String:Any];vdn["linear_branch"]="/other/model.safetensors";root["vdn"]=vdn
    XCTAssertThrowsError(try compile());root=try recipe()
    var stack=root["loras"] as! [String:Any];var adapters=stack["adapters"] as! [[String:Any]]
    adapters[1]["strength"]=0.5;stack["adapters"]=adapters;root["loras"]=stack
    XCTAssertThrowsError(try compile());root=try recipe()
    var config=root["config"] as! [String:Any];config["steps"]=5;root["config"]=config
    XCTAssertThrowsError(try compile())
    XCTAssertThrowsError(try H3StudioRecipe.compile(data:JSONSerialization.data(withJSONObject:try recipe())))
  }

  func testVDNRejectsShadowedPrimaryLoRAAndDownstreamContextIdentity() throws {
    let selection=try H3VDNSelection(stage:URL(fileURLWithPath:"/models/stage"),variant:.eightStep)
    func request(primary:URL?=nil) throws -> H3T2VARequest {
      try H3T2VARequest(prompt:"A man nods.",width:64,height:64,durationSeconds:2.5,seed:42,
        requestedSteps:9,transformer:URL(fileURLWithPath:"/models/base.safetensors"),
        qwenPages:URL(fileURLWithPath:"/models/qwen"),tokenizer:URL(fileURLWithPath:"/models/tokenizer.json"),
        videoVAE:URL(fileURLWithPath:"/models/video.safetensors"),audioVAE:URL(fileURLWithPath:"/models/audio.safetensors"),
        turboLoRA:primary,loRAAdapters:[],vdn:selection)
    }
    XCTAssertThrowsError(try request(primary:URL(fileURLWithPath:"/models/ordinary.safetensors")))
    let valid=try request()
    XCTAssertThrowsError(try H3ContinuationRunner.preflight(valid,contextFrames:22))
    XCTAssertThrowsError(try H3JointLatentArtifact.componentIdentity(base:valid,task:"t2va")) {
      XCTAssertEqual($0 as? H3CheckpointError,.invalid("H3 full latent component binding requires the exact task and vision components."))
    }
  }

}
