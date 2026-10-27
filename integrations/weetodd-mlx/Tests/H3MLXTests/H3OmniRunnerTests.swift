import CryptoKit
import InferenceContracts
import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3OmniRunnerTests: XCTestCase {
  func testEightRealStrategyUpdatesPreserveConditionsAndUseModalityTimes() throws {
    let plan = try H3PDDPlan(contract: H3OmniContract(data: omniContractData()))
    let geometry = try H3Geometry(width: 64, height: 64, durationSeconds: 5)
    let layout = try H3ReferenceLayout(geometry: geometry, textTags: [0], references: [.image(latentHeight: 4, latentWidth: 4), .audio(latents: 1)])
    let rows = try H3ReferenceRowSchedule(layout: layout, omni: plan)
    var calls = 0
    let output = try H3OmniRunner.sample(layout: layout, plan: plan, rowSchedule: rows,
      videoNoise: MLXArray.ones([1, layout.videoIndices.count, 96]),
      audioNoise: MLXArray.ones([1, layout.audioIndices.count, 32]) * 2) { step, video, audio, indices in
        XCTAssertEqual(step, calls)
        XCTAssertEqual(try plan.block(step: step, modality: .video).headRange, (calls * 4)..<(calls * 4 + 4))
        XCTAssertEqual(rows.table[Int(indices[layout.targetVideoIndices[0]])], plan.timesteps(for: .video)[step])
        XCTAssertEqual(rows.table[Int(indices[layout.targetAudioIndices[0]])], plan.timesteps(for: .audio)[step])
        XCTAssertEqual(rows.table[Int(indices[layout.conditionVideoIndices[0]])], 0.999)
        XCTAssertEqual(rows.table[Int(indices[layout.conditionAudioIndices[0]])], 1)
        XCTAssertEqual(video[0,0,0].item(Float.self), 1)
        XCTAssertEqual(audio[0,0,0].item(Float.self), 2)
        func sigma(_ shift:Float) -> Float { let u=0.999*Double(32-step*4)/32;return Float(Double(shift)*u/(1+(Double(shift)-1)*u/0.999)) }
        XCTAssertEqual(video[0,4,0].item(Float.self),0.999 + (0.999-sigma(12)),accuracy:0.00001)
        XCTAssertEqual(audio[0,2,0].item(Float.self),1.998 + 3*(0.999-sigma(3)),accuracy:0.00001)
        if step == 0 {
          XCTAssertEqual(video[0,4,0].item(Float.self), 0.999)
          XCTAssertEqual(audio[0,2,0].item(Float.self), 1.998)
        }
        calls += 1
        return H3FinalLayer.Output(video: MLXArray.ones(video.shape), audio: MLXArray.ones(audio.shape) * 3)
      }
    XCTAssertEqual(calls, 8)
    XCTAssertEqual(output.video[0,4,0].item(Float.self), 1.998, accuracy: 0.00001)
    XCTAssertEqual(output.audio[0,2,0].item(Float.self), 4.995, accuracy: 0.00001)
  }

  func testUntouchedWrapperRejectsUnsupportedCombinationsBeforeSourceRead() throws {
    var root: [String: Any] = ["fasth3_omni": ["version": 1, "contract_path": "/missing", "contract_sha256": String(repeating: "a", count: 64)],
      "components": ["task": "ref2va"], "config": ["width":64, "height":64, "duration_seconds":5, "steps":9, "seed":0],
      "conditioning": ["version":1, "task":"ref2va", "inputs": [["kind":"image", "role":"reference", "path":"/image", "sha256":String(repeating:"b",count:64)]]]]
    for key in ["still_image", "continuation", "joint_refinement", "two_stage", "fasth3", "vdn", "motion_fidelity", "loras", "control"] {
      var invalid = root; invalid[key] = [:]
      XCTAssertThrowsError(try H3OmniRecipe.prepare(data: JSONSerialization.data(withJSONObject: invalid))) { error in
        XCTAssertFalse(String(describing:error).contains("No such file"))
      }
    }
    root["fasth3_omni"] = ["version":true, "contract_path":"/missing", "contract_sha256":String(repeating:"a",count:64)]
    XCTAssertThrowsError(try H3OmniRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
  }
}

private let officialOmniContract = Data(("""
{
  "attention_backend": "VIDEO_SPARSE_ATTN_H3",
  "audio_scheduler_shift": 3.0,
  "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
  "conditioning": "fixed_ordered_references_target_only_flow",
  "grid_max_t": 0.999,
  "guidance_scale": 1.0,
  "model_type": "ref2va",
  "num_inference_steps": 8,
  "pdd_step_indices": [
    0,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32
  ],
  "pdd_steps": 32,
  "schema": "fasth3-inference-contract-v1",
  "schema_version": "fasth3-inference-contract-v1",
  "transformer_component": "transformer_ref",
  "transformer_forwards": 8,
  "video_scheduler_shift": 12.0,
  "vsa_ref_keep_rate": 0.1,
  "vsa_ref_policy": "p2_multi_region",
  "vsa_sparsity": 0.9,
  "vsa_tile_size": 128
}
""" + "\n").utf8)

extension H3OmniRunnerTests {
  private func fixture(_ operation: ([String:Any],URL) throws -> Void) throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:folder) }
    let contract = folder.appendingPathComponent("contract.json")
    try officialOmniContract.write(to:contract)
    let root: [String:Any] = ["format":"weetodd-headless-v2","engine":"h3","prompt":"A person walks.",
      "fasth3_omni":["version":1,"contract_path":contract.path,"contract_sha256":"68577f23f0c866e329aa8a578b2455e4ca9a26a3ca1d15ddd6d0e99091007e18"],
      "components":["task":"ref2va","transformer":"/missing","text_encoder":"/missing","vision_encoder":"/missing","tokenizer":"/missing","video_vae":"/missing","audio_vae":"/missing"],
      "config":["width":64,"height":64,"duration_seconds":5,"steps":9,"seed":7,"sampling_method":"euler"],
      "conditioning":["version":1,"task":"ref2va","audio_policy":"generated","inputs":[["id":"image-1","kind":"image","role":"reference","path":"/image","sha256":String(repeating:"b",count:64)]]]]
    try operation(root,contract)
  }
  private func prepare(_ root:[String:Any]) throws -> H3OmniRecipe.Prepared {
    guard let result = try H3OmniRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)) else { throw H3CheckpointError.invalid("Missing Omni selection.") }; return result
  }
  private func compile(_ root:[String:Any], calls:inout Int) throws -> H3Ref2VAStillRequest {
    let prepared = try prepare(root)
    return try H3StudioRecipe.compileMediaReferences(data:prepared.ordinaryRecipe,omni:prepared.selection) { _,kind,_,_,controls in
      calls += 1
      if kind == "image" { return .image(H3StillReference(rgb8:Data(count:64*64*3),width:64,height:64,pixelBudgetPercent:controls?.imagePixelBudgetPercent)) }
      if kind == "video" {
        let pixels=Data(count:5*64*64*3)
        let decision=try controls.map { try H3ReferenceTemporalPolicy.resolve(rgb8:pixels,frames:5,width:64,height:64,policy:$0.temporalDensity ?? .full) }
        return .video(H3VideoReference(rgb8:pixels,frameCount:5,width:64,height:64,controls:controls,temporalDecision:decision))
      }
      return .audio(H3AudioReference(samples:[Float](repeating:0,count:1600),frames:800))
    }
  }
  func testPinnedBytesAndStrictSchemaRejectEquivalentRewrittenContract() throws {
    try fixture { root,file in
      let selection = try prepare(root).selection
      XCTAssertEqual(selection.metadata["evaluations"] as? Int,8)
      var altered = root
      let bytes = try omniContractData()
      try bytes.write(to:file)
      var wrapper = root["fasth3_omni"] as! [String:Any]
      wrapper["contract_sha256"] = SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
      altered["fasth3_omni"] = wrapper
      XCTAssertThrowsError(try prepare(altered))
      XCTAssertThrowsError(try selection.checkUnchanged())
    }
  }
  func testEveryUnsupportedSettingRejectsBeforeMediaResolution() throws {
    try fixture { root,_ in
      var calls = 0
      XCTAssertNotNil(try compile(root,calls:&calls).omni)
      XCTAssertEqual(calls,1)
      let badConfig:[String:[Any]] = ["width":[true,63,0],"height":[false,65],"duration_seconds":[true,4.9,15.1],"steps":[true,8,32],"seed":[true,-1],"sampling_method":["res_multistep"],"drop_adaln":[1,false],"attention_policy":["dense","sol_experimental"],"transformer_backend":["nnc_experimental"],"inference_optimization":["tea_cache"],"transformer_weight_cache_gb":[4,true],"paging_cache_gb":[1,true],"prefix_cache":[true],"visual_condition_strength":[0.999],"audio_condition_strength":[1],"unknown":[0]]
      for (key,values) in badConfig {
        for value in values {
          var altered=root, config=root["config"] as! [String:Any]
          config[key]=value;altered["config"]=config;calls=0
          XCTAssertThrowsError(try compile(altered,calls:&calls),"\(key)=\(value)")
          XCTAssertEqual(calls,0,key)
        }
      }
      for key in ["still_image","fasth3","vdn","continuation","refinement","joint_latents","joint_refinement","two_stage","motion_fidelity","loras","control"] {
        var altered=root;altered[key]=[:];calls=0
        XCTAssertThrowsError(try compile(altered,calls:&calls),key);XCTAssertEqual(calls,0,key)
      }
      for key in ["frame_index","unknown","temporal_density","image_pixel_budget_percent"] {
        var altered=root, conditioning=root["conditioning"] as! [String:Any]
        var inputs=conditioning["inputs"] as! [[String:Any]];inputs[0][key]=0;conditioning["inputs"]=inputs;altered["conditioning"]=conditioning;calls=0
        XCTAssertThrowsError(try compile(altered,calls:&calls),key);XCTAssertEqual(calls,0,key)
      }
      for path in ["/fake-turbo"] {
        var altered=root,components=root["components"] as! [String:Any];components["loras"]=[[path,1]];altered["components"]=components;calls=0
        XCTAssertThrowsError(try compile(altered,calls:&calls));XCTAssertEqual(calls,0)
      }
    }
  }
  func testReferenceCountsAndActualRoleOrder() throws {
    try fixture { root,_ in
      let input = (root["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
      for kinds in [[],["audio"],Array(repeating:"image",count:10),Array(repeating:"video",count:4),["image"]+Array(repeating:"audio",count:4),Array(repeating:"image",count:9)+Array(repeating:"video",count:3)+["audio"]] {
        var altered=root, conditioning=root["conditioning"] as! [String:Any]
        conditioning["inputs"]=kinds.enumerated().map { index,kind in var value=input[0];value["kind"]=kind;value["id"]="ref-\(index)";value["path"]="/media-\(index)";return value }
        altered["conditioning"]=conditioning;var calls=0
        XCTAssertThrowsError(try compile(altered,calls:&calls));XCTAssertEqual(calls,0)
      }
    }
    let geometry = try H3Geometry(width:64,height:64,durationSeconds:5)
    let audio=H3AudioReference(samples:[Float](repeating:0,count:1600),frames:800)
    let refs:[H3Ref2VAReference] = [.image(H3StillReference(rgb8:Data(count:64*64*3),width:64,height:64)),.video(H3VideoReference(rgb8:Data(count:5*64*64*3),frameCount:5,width:64,height:64,audio:audio)),.audio(audio)]
    let layout = try H3ReferenceLayout(geometry:geometry,textTags:[0,1],references:[.image(latentHeight:4,latentWidth:4),.video(latentFrames:2,latentHeight:4,latentWidth:4,audioLatents:1,sourceLatentFrames:2),.audio(latents:1)])
    let plan = try H3OmniRunner.attention(references:refs,geometry:geometry,textRows:2,layout:layout)
    XCTAssertEqual(plan.prefixSegments,[.dense(role:.text,rows:2),.dense(role:.image,rows:4),.dense(role:.referenceAudio,rows:2),.referenceVideo(grid:[2,2,2]),.dense(role:.referenceAudio,rows:2),.dense(role:.targetAudio,rows:414)])
    XCTAssertEqual(plan.rows,layout.tags.count)
  }
}

extension H3OmniRunnerTests {
  func testActualTaskCancellationStopsTheRealPDDLoop() async throws {
    let task=Task { () throws -> Int in
      let plan=try H3PDDPlan(contract:H3OmniContract(data:omniContractData()))
      let geometry=try H3Geometry(width:64,height:64,durationSeconds:5)
      let layout=try H3ReferenceLayout(geometry:geometry,textTags:[0],references:[.image(latentHeight:2,latentWidth:2)])
      let rows=try H3ReferenceRowSchedule(layout:layout,omni:plan)
      var calls=0
      do {
        _ = try H3OmniRunner.sample(layout:layout,plan:plan,rowSchedule:rows,
          videoNoise:MLXArray.ones([1,layout.videoIndices.count,96]),audioNoise:MLXArray.ones([1,layout.audioIndices.count,32])) { _,video,audio,_ in
          calls += 1
          if calls == 3 { withUnsafeCurrentTask { $0?.cancel() } }
          return H3FinalLayer.Output(video:MLXArray.zeros(video.shape),audio:MLXArray.zeros(audio.shape))
        }
        return -1
      } catch is CancellationError { return calls }
    }
    let count=try await task.value
    XCTAssertEqual(count,3)
  }
  func testWrapperShapeTypesAndHarmlessDecoderControls() throws {
    try fixture { root,_ in
      for (key,values) in ["version":[true,"1",2],"contract_path":["relative",17],"contract_sha256":["BAD",true],"unknown":[1]] as [String:[Any]] {
        for value in values {
          var bad=root, fields=root["fasth3_omni"] as! [String:Any];fields[key]=value;bad["fasth3_omni"]=fields
          XCTAssertThrowsError(try prepare(bad),key)
        }
      }
      for key in ["version","contract_path","contract_sha256"] {
        var bad=root,fields=root["fasth3_omni"] as! [String:Any];fields.removeValue(forKey:key);bad["fasth3_omni"]=fields
        XCTAssertThrowsError(try prepare(bad),key)
      }
      var accepted=root,config=root["config"] as! [String:Any]
      config["memory_mode"]="low_memory_bf16";config["video_decode_precision"]="float16";config["drop_adaln"]=true;accepted["config"]=config
      var calls=0
      XCTAssertNotNil(try compile(accepted,calls:&calls).omni);XCTAssertEqual(calls,1)
    }
  }
  func testLegacyWrapperEntryPointsRejectOmniBeforeTheyRewriteSettings() throws {
    func reject(_ root:[String:Any], _ compiler:(Data)throws->Void) throws {
      try compiler(JSONSerialization.data(withJSONObject:root))
      var mixed=root;mixed["fasth3_omni"]=["version":1]
      XCTAssertThrowsError(try compiler(JSONSerialization.data(withJSONObject:mixed)))
    }
    try reject(["components":["task":"t2va"],"config":["width":1536,"height":864,"duration_seconds":5,"steps":21,"seed":42],"conditioning":["task":"t2v"],"two_stage":["version":1,"draft_scale":2.0,"strength":0.35,"evaluations":3,"resize_method":"bilinear","refinement_adapter_policy":"standard"]]) { _ = try H3TwoStageRecipe.prepare(data:$0) }
    try reject(["config":["width":64,"height":64,"duration_seconds":5,"steps":20],"joint_latents":["version":1,"save_full":true]]) { _ = try H3JointRefinementRecipe.prepare(data:$0) }
    try reject(["engine":"h3","components":["task":"t2va"],"config":["steps":20],"conditioning":["task":"t2v","inputs":[]],"motion_fidelity":["version":1,"source_video":"/source.mp4","source_sha256":String(repeating:"a",count:64),"source_in":0.125,"duration_seconds":2.5,"ffprobe":"/ffprobe","mode":"adaptive","strength":0.5,"max_hold":2,"sensitivity":0.5,"seed":42,"max_frames":345]]) { _ = try H3MotionFidelityRecipe.prepare(data:$0) }
    try reject(["components":["task":"ref2va"],"config":["duration_seconds":68.0/24],"conditioning":["task":"ref2va","inputs":[["kind":"image","role":"reference"]]],"continuation":["version":4,"context_frames":22,"save_context":true]]) { _ = try H3Ref2VAContinuationRecipe.prepare(data:$0) }
    try reject(["components":["task":"fl2va"],"config":["duration_seconds":119.0/24],"conditioning":["task":"fflf","inputs":[["id":"first","role":"first","frame_index":0]]],"continuation":["version":3,"context_frames":22,"save_context":true]]) { _ = try H3FL2VAContinuationRecipe.prepare(data:$0) }
  }
}

extension H3OmniRunnerTests {
  private var componentAdmissionError: H3CheckpointError {
    .invalid("FastH3 Omni components require native Ref2VA paths and no compatibility controls.")
  }

  func testOmniRejectsCompatibilityFieldBeforeCommonCompilerCanRemoveIt() throws {
    try fixture { root,_ in
      // This otherwise valid pinned fixture must admit before adding the field.
      XCTAssertEqual(try prepare(root).selection.contractSHA256,H3OmniRecipe.contractSHA256)
      for value:Any in [0,1,false,true,"false",NSNull(),[],[:]] {
        var altered=root,components=root["components"] as! [String:Any]
        components["allow_fl2va_weights_for_ref2va"]=value;altered["components"]=components
        XCTAssertThrowsError(try prepare(altered),String(describing:value)) { error in
          XCTAssertEqual(error as? H3CheckpointError,componentAdmissionError)
        }
      }
    }
  }

  func testOmniComponentSchemaRejectsMistypedPathsAndRemovedVisionSelection() throws {
    try fixture { root,_ in
      let required=["transformer","text_encoder","tokenizer","video_vae","audio_vae"]
      let optional=["vision_encoder","checkpoint","processor"]
      for key in required + optional {
        for value:Any in [0,false,NSNull(),"relative","/nul\0path"] {
          var altered=root,components=root["components"] as! [String:Any]
          components[key]=value;altered["components"]=components
          XCTAssertThrowsError(try prepare(altered),"\(key)=\(value)") { error in
            XCTAssertEqual(error as? H3CheckpointError,componentAdmissionError)
          }
        }
      }
      for key in required {
        var altered=root,components=root["components"] as! [String:Any]
        components.removeValue(forKey:key);altered["components"]=components
        XCTAssertThrowsError(try prepare(altered),key) { error in
          XCTAssertEqual(error as? H3CheckpointError,componentAdmissionError)
        }
      }
      for (key,value) in [("task",true as Any),("task","fl2va"),("unknown",0),("loras",false),("loras",[["/adapter",1]])] {
        var altered=root,components=root["components"] as! [String:Any]
        components[key]=value;altered["components"]=components
        XCTAssertThrowsError(try prepare(altered),key) { error in
          XCTAssertEqual(error as? H3CheckpointError,componentAdmissionError)
        }
      }
    }
  }

  func testOmniOptionalPathsAndOrdinaryCompatibilityRemainSeparate() throws {
    try fixture { root,_ in
      var omni=root,components=root["components"] as! [String:Any]
      components.removeValue(forKey:"vision_encoder")
      components["text_encoder"]="/shared-text-vision"
      components["checkpoint"]="/checkpoint";components["processor"]="/processor";components["loras"]=[]
      omni["components"]=components
      var calls=0
      let inherited=try compile(omni,calls:&calls)
      XCTAssertEqual(calls,1);XCTAssertEqual(inherited.qwenVision.path,"/shared-text-vision")
      components["vision_encoder"]="/dedicated-vision";omni["components"]=components
      let explicit=try compile(omni,calls:&calls)
      XCTAssertEqual(explicit.qwenVision.path,"/dedicated-vision")
      var ordinary=omni;ordinary.removeValue(forKey:"fasth3_omni")
      components["allow_fl2va_weights_for_ref2va"]=false;ordinary["components"]=components
      let request=try H3StudioRecipe.compileMediaReferences(data:JSONSerialization.data(withJSONObject:ordinary)) { _,_,_,_,_ in
        .image(H3StillReference(rgb8:Data(count:64*64*3),width:64,height:64))
      }
      XCTAssertNil(request.omni);XCTAssertEqual(request.qwenVision.path,"/dedicated-vision")
    }
  }
}


extension H3OmniRunnerTests {
  func testOmniReferenceSizingAdmitsOwnedPreparationAndRejectsUnsupportedControls() throws {
    try fixture { root,_ in
      func withInput(_ kind:String,_ extra:[String:Any]) -> [String:Any] {
        var changed=root,c=root["conditioning"] as! [String:Any]
        var input=(c["inputs"] as! [[String:Any]])[0];input["kind"]=kind
        input.merge(extra) { _,new in new };c["inputs"]=[input];changed["conditioning"]=c;return changed
      }
      for (kind,extra) in [("image",["image_pixel_budget_percent":400] as [String:Any]),
        ("image",["image_pixel_budget_percent":50]),
        ("video",["size_policy":"match_output","temporal_density":"full"]),
        ("video",["size_policy":"native_h3","temporal_density":"full"])] {
        let prepared=try prepare(withInput(kind,extra))
        let saved=try JSONSerialization.jsonObject(with:prepared.ordinaryRecipe) as! [String:Any]
        let input=((saved["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[0]
        XCTAssertEqual(NSDictionary(dictionary:input).object(forKey:extra.keys.first!) as? NSObject,extra.values.first as? NSObject)
        var calls=0;XCTAssertNoThrow(try compile(withInput(kind,extra),calls:&calls));XCTAssertEqual(calls,1)
      }
      for (kind,extra) in [("image",["image_pixel_budget_percent":true] as [String:Any]),
        ("image",["image_pixel_budget_percent":401]),("image",["image_pixel_budget_percent":"400"]),
        ("image",["size_policy":"native_h3"]),("audio",["image_pixel_budget_percent":100]),
        ("video",["image_pixel_budget_percent":400]),("video",["size_policy":true]),
        ("video",["temporal_density":"half"]),("video",["temporal_density":"quarter"]),
        ("video",["temporal_density":"automatic"]),("video",["temporal_density":true]),
        ("image",["frame_index":0]),("image",["noise":0.1])] {
        var calls=0;XCTAssertThrowsError(try compile(withInput(kind,extra),calls:&calls),"\(kind) \(extra)");XCTAssertEqual(calls,0)
      }
    }
  }
}
