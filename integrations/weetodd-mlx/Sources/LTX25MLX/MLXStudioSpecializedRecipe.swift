import Foundation
import CoreFoundation
import LTX25Engine

/// Converts the existing Studio conditioning contract to the already-shared
/// single-stage engines. Every task control is validated before media loading.
enum MLXStudioSpecializedRecipe {
  static func compile(data:Data,outputDirectory:String) throws -> MLXDistilledRequest {
    if let root=try JSONSerialization.jsonObject(with:data) as? [String:Any],
      let condition=root["conditioning"] as? [String:Any],
      let inputs=condition["inputs"] as? [[String:Any]],
      inputs.contains(where:{ ["motion_track","crossview_warp"].contains($0["control_type"] as? String ?? "") }) {
      return try compileIC(root:root,outputDirectory:outputDirectory)
    }
    func invalid(_ text:String) -> LTXError { .invalid("Swift LTX reference recipe: "+text) }
    guard data.count<=1024*1024,
      var root=try JSONSerialization.jsonObject(with:data) as? [String:Any],
      var components=root["components"] as? [String:Any],
      var config=root["config"] as? [String:Any],
      let conditioning=root["conditioning"] as? [String:Any],
      Set(conditioning.keys).isSubset(of:["version","task","audio_policy","inputs"]),
      conditioning["version"] as? Int == 1,
      (conditioning["audio_policy"] as? String ?? "generated") == "generated",
      let task=conditioning["task"] as? String,["ref2va","control"].contains(task),
      let inputs=conditioning["inputs"] as? [[String:Any]],
      (config["dfr_enabled"] as? Bool ?? false) == false,
      (components["loras"] as? [Any] ?? []).isEmpty,
      let adapters=components["ic_loras"] as? [[Any]],adapters.count == 1,
      adapters[0].count == 2,let adapter=adapters[0][0] as? String,
      let strength=adapters[0][1] as? NSNumber,
      CFGetTypeID(strength) != CFBooleanGetTypeID(),
      strength.doubleValue.isFinite,(0...3).contains(strength.doubleValue),strength.doubleValue>0,
      let prompt=root["prompt"] as? String else {
      throw invalid("requires one dedicated adapter, generated audio, and distilled single-stage sampling")
    }
    if let flag=config["ic_lora_single_stage"] as? NSNumber,CFGetTypeID(flag) != CFBooleanGetTypeID() {
      throw invalid("ic_lora_single_stage must be a boolean")
    }
    let single=config["ic_lora_single_stage"] as? Bool ?? false
    let ingredientsCFGPP:Bool
    if let sampler=config["single_stage_sampler"] {
      guard task == "control",single,sampler as? String == "euler_ancestral_cfg_pp" else {
        throw invalid("single_stage_sampler requires Ingredients and euler_ancestral_cfg_pp")
      }
      ingredientsCFGPP=true
      config.removeValue(forKey:"single_stage_sampler")
    } else { ingredientsCFGPP=false }
    if task == "control",!single {
      guard inputs.count == 1,(components["msr_lora_path"] as? String ?? "").isEmpty,
        strength.doubleValue <= 2 else { throw invalid("Union Control needs one guide and its stage-one adapter") }
      let input=inputs[0]
      guard Set(input.keys).isSubset(of:["id","kind","role","path","sha256","strength","control_type","format"]),
        let id=input["id"] as? String,!id.isEmpty,
        input["kind"] as? String == "video",input["role"] as? String == "control",
        input["format"] as? String == "rgb24",
        ["canny_edges","depth_map","pose_skeleton"].contains(input["control_type"] as? String ?? ""),
        let path=input["path"] as? String,let digest=input["sha256"] as? String else {
        throw invalid("Union Control needs a frozen RGB24 Canny, depth or pose guide prepared by Studio")
      }
      components["ic_loras"]=[];root["components"]=components
      root["conditioning"]=["version":1,"task":"t2v","audio_policy":"generated","inputs":[]]
      var request=try MLXStudioRecipe.compileFields(data:JSONSerialization.data(withJSONObject:root),outputDirectory:outputDirectory)
      guard (request["width"] as! Int)%128 == 0,(request["height"] as! Int)%128 == 0 else {
        throw invalid("Union's quarter-canvas guide requires final dimensions divisible by 128")
      }
      request["version"]=5;request["task"]="union_control"
      request["audio_reference"]=NSNull()
      request["union_control_guide"]=["path":path,"source_sha256":digest,"adapter_path":adapter,
        "adapter_strength":strength,"reference_strength":input["strength"] ?? 1]
      return try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:request))
    }
    guard single else { throw invalid("MSR and Ingredients require single-stage sampling") }
    var ids=Set<String>()
    for input in inputs {
      guard let id=input["id"] as? String,!id.isEmpty,ids.insert(id).inserted,
        input["kind"] as? String == "image" else { throw invalid("requires unique still-image references") }
    }
    if task == "ref2va" {
      let msrStrength=components["msr_lora_strength"] ?? 1
      guard (1...5).contains(inputs.count),components["msr_lora_path"] as? String == adapter,
        let number=msrStrength as? NSNumber,CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue == strength.doubleValue else {
        throw invalid("MSR adapter path and strength must match the single IC adapter")
      }
    } else {
      guard inputs.count == 1,(components["msr_lora_path"] as? String ?? "").isEmpty else {
        throw invalid("Ingredients needs exactly one sheet and no MSR adapter")
      }
    }
    config["ic_lora_single_stage"]=false
    components["ic_loras"]=[];components["msr_lora_path"]="";components["msr_lora_strength"]=1
    if components["spatial_upscaler_path"] == nil { components["spatial_upscaler_path"]="" }
    root["components"]=components;root["config"]=config
    root["conditioning"]=["version":1,"task":"t2v","audio_policy":"generated","inputs":[]]
    var request=try MLXStudioRecipe.compileFields(data:JSONSerialization.data(withJSONObject:root),
      outputDirectory:outputDirectory,requiresSpatialUpscaler:false)
    request["version"]=task == "ref2va" ? 7 : ingredientsCFGPP ? 11 : 6
    request["task"]=task == "ref2va" ? "msr" : "ingredients"
    request["audio_reference"]=NSNull();request["union_control_guide"]=NSNull()
    request["ingredients_sheet"]=NSNull()
    if ingredientsCFGPP {
      request["msr"]=NSNull();request["dfr"]=NSNull();request["ic_control"]=NSNull()
      request["ingredients_sampling"]=MLXIngredientsSampling.ancestralCFGPP.rawValue
    }
    if task == "ref2va" {
      let ordered=inputs.filter { $0["reference_role"] as? String != "background" }
        + inputs.filter { $0["reference_role"] as? String == "background" }
      let references=try ordered.map { input -> [String:Any] in
        guard Set(input.keys).isSubset(of:["id","kind","role","path","sha256","strength",
          "description","reference_role","reference_priority","reference_frames","reference_size_policy","attention_strength"]),
          input["role"] as? String == "reference",
          let path=input["path"] as? String,let digest=input["sha256"] as? String,
          let description=input["description"] as? String,!description.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
          let role=input["reference_role"] as? String else { throw invalid("MSR input fields or description are incomplete") }
        return ["path":path,"source_sha256":digest,"role":role,
          "priority":input["reference_priority"] ?? "auto","reference_frames":input["reference_frames"] ?? "auto",
          "size_policy":input["reference_size_policy"] ?? "sol_auto","strength":input["strength"] ?? 1,
          "attention_strength":input["attention_strength"] ?? 1]
      }
      request["msr"]=["adapter_path":adapter,"adapter_strength":strength,"references":references]
      let guide=ordered.enumerated().map { index,input in
        "Image \(index+1) provides the \(input["reference_role"] as! String): \((input["description"] as! String).trimmingCharacters(in:.whitespacesAndNewlines))"
      }.joined(separator:"\n")
      request["prompt"]=prompt.hasPrefix(guide) ? prompt : guide+"\n"+prompt
    } else {
      let input=inputs[0]
      guard Set(input.keys).isSubset(of:["id","kind","role","path","sha256","strength","description","control_type"]),
        input["role"] as? String == "control",input["control_type"] as? String == "ingredients_reference_sheet",
        let path=input["path"] as? String,let digest=input["sha256"] as? String,
        let description=input["description"] as? String,!description.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
        throw invalid("Ingredients requires a frozen and described reference sheet")
      }
      request["ingredients_sheet"]=["path":path,"source_sha256":digest,"adapter_path":adapter,
        "adapter_strength":strength,"reference_strength":input["strength"] ?? 1]
      if !prompt.hasPrefix("Reference sheet:") && !prompt.hasPrefix("### Reference Sheet Description") {
        request["prompt"]="Reference sheet: "+description+"\n\nGenerated video: "+prompt
      }
    }
    return try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:request))
  }
  private static func compileIC(root original:[String:Any],outputDirectory:String) throws -> MLXDistilledRequest {
    func invalid(_ message:String)->LTXError { .invalid("Swift LTX IC control: "+message) }
    guard var components=original["components"] as? [String:Any],
      let config=original["config"] as? [String:Any],
      let condition=original["conditioning"] as? [String:Any],
      Set(condition.keys).isSubset(of:["version","task","audio_policy","inputs","control_family","publication_audio"]),
      let version=condition["version"] as? NSNumber,
      CFGetTypeID(version) != CFBooleanGetTypeID(),version.doubleValue == 1,
      condition["task"] as? String == "control",
      let inputs=condition["inputs"] as? [[String:Any]],
      let pairs=components["ic_loras"] as? [[Any]],
      config["ic_lora_single_stage"] as? Bool != true,
      (config["dfr_enabled"] as? Bool ?? false) == false,
      (components["msr_lora_path"] as? String ?? "").isEmpty else {
      throw invalid("requires a two-stage control recipe with explicit ordered guides")
    }
    let family:String,roles:[String],families:[String]
    if inputs.count == 1,inputs[0]["control_type"] as? String == "motion_track" {
      family="motion_track";roles=["control"];families=["motion_track"]
    } else if inputs.count == 2 || inputs.count == 3 {
      family=inputs.count == 2 ? "crossview_warp" : "crossview_ingredients"
      roles=inputs.count == 2 ? ["warp","source"] : ["warp","source","ingredients"]
      families=inputs.count == 2 ? ["crossview_warp"] : ["crossview_warp","ingredients_reference_sheet"]
    } else { throw invalid("MotionTrack needs one guide; CrossView needs ordered warp/source guides and an optional Ingredients guide") }
    if let declared=condition["control_family"] {
      guard let value=declared as? String,value == family else {
        throw invalid("declared control family must match the attached guides")
      }
    }
    guard pairs.count == families.count,
      condition["audio_policy"] as? String == (family == "motion_track" ? "generated" : "source") else {
      throw invalid("adapter count and audio policy must match the selected control family")
    }
    var ids=Set<String>()
    let guides=try inputs.enumerated().map { index,input->[String:Any] in
      guard Set(input.keys).isSubset(of:["id","kind","role","path","sha256","strength","control_type","format","reference_role"]),
        let id=input["id"] as? String,!id.isEmpty,ids.insert(id).inserted,
        input["kind"] as? String == "video",input["role"] as? String == "control",
        input["format"] as? String == "rgb24",
        input["control_type"] as? String == (roles[index] == "ingredients" ? "ingredients_reference_sheet" : family == "motion_track" ? "motion_track" : "crossview_warp"),
        (input["reference_role"] as? String ?? "control") == roles[index],
        let path=input["path"] as? String,let digest=input["sha256"] as? String else {
        throw invalid("guide roles, format or ordering differ from the trained layout")
      }
      return ["path":path,"source_sha256":digest,"role":roles[index],"strength":input["strength"] ?? 1]
    }
    let adapters=try pairs.enumerated().map { index,pair->[String:Any] in
      guard pair.count == 2,let path=pair[0] as? String,
        let value=pair[1] as? NSNumber,CFGetTypeID(value) != CFBooleanGetTypeID() else {
        throw invalid("IC adapters require path/strength pairs")
      }
      return ["path":path,"family":families[index],"strength":value]
    }
    var root=original
    components["ic_loras"]=[];root["components"]=components
    root["conditioning"]=["version":1,"task":"t2v","audio_policy":"generated","inputs":[]]
    var request=try MLXStudioRecipe.compileFields(data:JSONSerialization.data(withJSONObject:root),outputDirectory:outputDirectory)
    request["version"]=10;request["task"]="ic_control"
    for key in ["audio_reference","union_control_guide","ingredients_sheet","msr","dfr"] { request[key]=NSNull() }
    var audio:Any=NSNull()
    if var source=condition["publication_audio"] as? [String:Any] {
      guard Set(source.keys) == ["path","sha256","source_start_seconds","source_duration_seconds"] else {
        throw invalid("source audio requires a frozen path, checksum and explicit interval")
      }
      source["source_sha256"]=source.removeValue(forKey:"sha256");audio=source
    }
    request["ic_control"]=["family":family,"adapters":adapters,"guides":guides,"publication_audio":audio]
    return try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:request))
  }

}
