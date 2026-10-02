import Foundation
import CoreFoundation
import LTX25Engine
import InferenceContracts

/// Strict adapter for Studio's resolved recipe. This owns no inference or prompt rewriting.
public enum MLXStudioRecipe {
  public static func compile(data:Data,outputDirectory:String) throws -> MLXDistilledRequest {
    guard data.count <= 1024*1024,
      let root=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {
      throw LTXError.invalid("Studio recipe must be a JSON object of at most 1 MiB.")
    }
    func reject(_ field:String) -> LTXError { .invalid("Native Swift LTX does not support this recipe setting: \(field). Select the Python renderer explicitly for advanced workflows.") }
    func keys(_ object:[String:Any],_ allowed:Set<String>,_ label:String) throws {
      if let extra=Set(object.keys).subtracting(allowed).sorted().first { throw reject(label+"."+extra) }
    }
    func number(_ object:[String:Any],_ key:String) throws -> Double {
      guard let value=object[key] as? NSNumber,CFGetTypeID(value) != CFBooleanGetTypeID(),value.doubleValue.isFinite else { throw reject(key) }
      return value.doubleValue
    }
    func integer(_ object:[String:Any],_ key:String,_ range:ClosedRange<Int>) throws -> Int {
      let value=try number(object,key)
      guard value.rounded() == value,value >= Double(range.lowerBound),value <= Double(range.upperBound) else { throw reject(key) }
      return Int(value)
    }
    try keys(root,["format","engine","components","config","prompt","candidate","conditioning","ffmpeg","ffprobe"],"recipe")
    guard root["engine"] as? String == "ltx25",root["format"] as? String == "weetodd-headless-v2",
      let config=root["config"] as? [String:Any],let components=root["components"] as? [String:Any],
      let prompt=root["prompt"] as? String else { throw reject("format/engine/config/components/prompt") }
    // Inactive controls are accepted only at their neutral values. Distilled
    // CFG=1 cannot evaluate a negative prompt, so reject it before weights load.
    let dfrEnabled:Bool
    if let value=config["dfr_enabled"] {
      guard let flag=value as? NSNumber,CFGetTypeID(flag) == CFBooleanGetTypeID() else { throw reject("config.dfr_enabled") }
      dfrEnabled=flag.boolValue
    } else { dfrEnabled=false }
    var fixed:[String:Any] = ["pipeline_mode":"distilled","duration_mode":"manual","stage1_steps":8,"stage2_steps":3,
      "stage1_sampler":"euler_ancestral","stage2_sampler":"euler","stage1_eta":1,"stage1_s_noise":1,"ancestral_seed_offset":10000,
      "video_cfg_scale":1,"audio_cfg_scale":1,"stg_scale":0,"video_rescale_scale":0,"audio_rescale_scale":0,"modality_scale":1,
      "stg_blocks":[],"low_memory":true,"low_ram_streaming":false,"prompt_context":"official_1024","feed_forward_backend":"reference_fp32",
      "generated_keyframes":0,"dfr_enabled":false,"dfr_detailing_lora_path":"","dfr_detailing_lora_strength":1,
      "dfr_prebaked_transformer_path":"","dfr_temporal_upsampler_path":"","dfr_temporal_rounds":0,
      "ic_lora_single_stage":false,"cfg_pp_batched":false,"cfg_pp_schedule":"full","sol_attention_profile":"disabled",
      "diffvae_optimization":"combined","diffvae_query_chunk_size":512,"diffvae_context_width_chunks":4,"diffvae_stage4_tile_width":0,
      "auto_duration_min_seconds":1,"auto_duration_max_seconds":20]
    if dfrEnabled {
      for key in ["dfr_enabled","dfr_detailing_lora_path","dfr_detailing_lora_strength",
        "dfr_temporal_upsampler_path","dfr_temporal_rounds"] { fixed.removeValue(forKey:key) }
    }
    try keys(config,Set(fixed.keys).union(["width","height","duration_seconds","frame_rate","seed","negative_prompt",
      "dfr_enabled","dfr_detailing_lora_path","dfr_detailing_lora_strength","dfr_temporal_upsampler_path","dfr_temporal_rounds"]),"config")
    for (key,expected) in fixed where config[key] != nil {
      let actual=config[key]!
      // JSON equality must distinguish booleans from 0/1.
      let a=try JSONSerialization.data(withJSONObject:[actual],options:.sortedKeys)
      let b=try JSONSerialization.data(withJSONObject:[expected],options:.sortedKeys)
      guard a == b else { throw reject("config."+key) }
    }
    guard config["pipeline_mode"] as? String == "distilled",
      try integer(config,"stage1_steps",8...8) == 8,try integer(config,"stage2_steps",3...3) == 3 else { throw reject("distilled 8+3 schedule") }
    if let negative=config["negative_prompt"] {
      guard let value=negative as? String,value.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
        throw reject("negative_prompt is not evaluated by distilled sampling")
      }
    }
    let width=try integer(config,"width",64...4096),height=try integer(config,"height",64...4096)
    let fps=try number(config,"frame_rate"),duration=try number(config,"duration_seconds")
    guard (1...120).contains(fps),(0.01...20).contains(duration) else { throw reject("duration/frame rate") }
    let frames=max(1,Int((duration*fps/8).rounded(.toNearestOrEven)))*8+1
    // The legacy recipe contract uses a 32-bit seed; rejecting wider numbers avoids JSON Double loss.
    let seed=try integer(config,"seed",0...Int(UInt32.max))
    let paths=["transformer_path","text_encoder_path","video_vae_path","audio_vae_path","spatial_upscaler_path"]
    try keys(components,Set(paths).union(["duration_head_path","distilled_lora_path","loras","ic_loras","msr_lora_path","msr_lora_strength"]),"components")
    for name in ["duration_head_path","distilled_lora_path","msr_lora_path"] {
      if let value=components[name],value as? String != "" { throw reject(name) }
    }
    if let value=components["ic_loras"],(value as? [Any])?.isEmpty != true { throw reject("ic_loras") }
    if components["msr_lora_strength"] != nil,try number(components,"msr_lora_strength") != 1 { throw reject("msr_lora_strength") }
    var resolved:[String:String]=[:]
    for key in paths {
      guard let value=components[key] as? String,!value.isEmpty else { throw reject(key) }
      resolved[key]=value
    }
    var adapters:[[String:Any]]=[]
    if let value=components["loras"] {
      guard let pairs=value as? [[Any]] else { throw reject("loras") }
      for pair in pairs {
        guard pair.count == 2,let path=pair[0] as? String else { throw reject("LoRA path/strength pair") }
        let strength=try number(["strength":pair[1]],"strength")
        guard strength > 0 else { throw reject("LoRA strength") }
        adapters.append(["path":path,"strength":strength,"enabled":true])
      }
    }
    var dfr:[String:Any]?
    if dfrEnabled {
      guard adapters.isEmpty else { throw reject("DFR cannot combine with ordinary LoRAs") }
      if config["generated_keyframes"] != nil {
        guard try integer(config,"generated_keyframes",0...0) == 0 else { throw reject("generated_keyframes") }
      }
      guard let adapter=config["dfr_detailing_lora_path"] as? String,
        adapter.hasPrefix("/"),!adapter.utf8.contains(0) else { throw reject("dfr_detailing_lora_path") }
      let strength=try number(config,"dfr_detailing_lora_strength")
      guard strength > 0,strength <= 3 else { throw reject("dfr_detailing_lora_strength") }
      let rounds=config["dfr_temporal_rounds"] == nil ? 0 : try integer(config,"dfr_temporal_rounds",0...2)
      let temporal=config["dfr_temporal_upsampler_path"] as? String ?? ""
      guard (rounds == 0 && temporal.isEmpty) ||
        (rounds > 0 && temporal.hasPrefix("/") && !temporal.utf8.contains(0) && fps*Double(1 << rounds) <= 120) else {
        throw reject("dfr_temporal_upsampler_path/dfr_temporal_rounds")
      }
      dfr=["adapter_path":adapter,"adapter_strength":strength]
      if rounds > 0 { dfr!["temporal_upscaler_path"]=temporal; dfr!["temporal_rounds"]=rounds }
    }
    var references:[[String:Any]]=[],task="t2v",audioReference:[String:Any]?
    if let value=root["conditioning"] {
      guard let c=value as? [String:Any] else { throw reject("conditioning") }
      guard let name=c["task"] as? String,["t2v","fflf","a2v"].contains(name) else {
        throw reject("conditioning task")
      }
      let inputs=try ConditioningV1.inputs(c,task:name,
        audioPolicy:name == "a2v" ? "source" : "generated",
        count:name == "t2v" ? 0...0 : 1...2)
      if name == "a2v" {
        guard inputs.filter({ $0["role"] as? String == "audio_driver" }).count == 1,
          inputs.count <= 2,
          let input=inputs.first(where: { $0["role"] as? String == "audio_driver" }) else {
          throw reject("A2V requires one audio driver and optional first image")
        }
        try keys(input,["id","kind","role","path","strength","source_start_seconds","source_duration_seconds"],"conditioning input")
        guard input["kind"] as? String == "audio",input["role"] as? String == "audio_driver",
          let path=input["path"] as? String else { throw reject("A2V audio_driver") }
        let strength=input["strength"] == nil ? 1 : try number(input,"strength")
        guard strength == 1 else { throw reject("A2V audio strength") }
        let start=try number(input,"source_start_seconds")
        let sourceDuration=try number(input,"source_duration_seconds")
        audioReference=["path":path,"source_start_seconds":start,"source_duration_seconds":sourceDuration]
        if let first=inputs.first(where: { $0["role"] as? String != "audio_driver" }) {
          try keys(first,["id","kind","role","path","strength","frame_index"],"conditioning input")
          guard first["kind"] as? String == "image",first["role"] as? String == "keyframe",
            let imagePath=first["path"] as? String,
            try integer(first,"frame_index",0...0) == 0 else {
            throw reject("A2V optional image must be the first frame")
          }
          let imageStrength=first["strength"] == nil ? 1 : try number(first,"strength")
          references=[["role":"first","path":imagePath,"strength":imageStrength,"crf":33]]
        }
        task="a2v"
      } else {
      task=inputs.isEmpty ? "t2v" : inputs.count == 1 ? "i2v":"fflf"
      for input in inputs {
        try keys(input,["id","kind","role","path","strength","frame_index"],"conditioning input")
        guard input["kind"] as? String == "image",input["role"] as? String == "keyframe",
          let path=input["path"] as? String else { throw reject("reference kind/role/path") }
        let frame=input["frame_index"] as? String == "last" ? frames-1 : try integer(input,"frame_index",0...frames-1)
        guard frame == 0 || frame == frames-1 else { throw reject("only first and last endpoints") }
        let strength=input["strength"] == nil ? 1 : try number(input,"strength")
        references.append(["role":frame == 0 ? "first":"last","path":path,"strength":strength,"crf":33])
      }
      references.sort { ($0["role"] as! String) < ($1["role"] as! String) }
      guard references.isEmpty || (references.first?["role"] as? String == "first" &&
        (references.count != 2 || references.last?["role"] as? String == "last")) else {
        throw reject("first anchor and optional last anchor")
      }
      }
    }
    if dfrEnabled && task == "a2v" { throw reject("DFR does not support A2V") }
    var request:[String:Any] = ["version":dfrEnabled ? ((dfr!["temporal_rounds"] as? Int ?? 0) > 0 ? 9 : 8) : (task == "a2v" ? 4 : 3),
      "engine":"ltx25","task":dfrEnabled ? "dfr" : task,"prompt":prompt,"width":width,"height":height,"frames":frames,
      "fps":fps,"seed":seed,"output_directory":outputDirectory,"gemma_root":resolved["text_encoder_path"]!,
      "transformer_root":resolved["transformer_path"]!,"connector_checkpoint":resolved["transformer_path"]!+"/pages/fixed.safetensors",
      "video_checkpoint":resolved["video_vae_path"]!,"audio_checkpoint":resolved["audio_vae_path"]!,
      "spatial_upscaler_checkpoint":resolved["spatial_upscaler_path"]!,"stage_one_loras":adapters,"stage_two_loras":adapters,
      "reference_images":references,"noise_policy":"mlx_threefry_bf16_v1"]
    if let audioReference { request["audio_reference"]=audioReference }
    if let dfr {
      request["audio_reference"]=NSNull()
      request["union_control_guide"]=NSNull()
      request["ingredients_sheet"]=NSNull()
      request["msr"]=NSNull()
      request["dfr"]=dfr
    }
    return try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:request))
  }
}
