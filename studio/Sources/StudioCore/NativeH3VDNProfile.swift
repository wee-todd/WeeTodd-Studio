import Foundation

/// Weight-free editor contract for the released eight/fifty-evaluation routes.
/// Complete payload/task admission remains authoritative in the shared worker.
enum NativeH3VDNProfile {
  struct Packet { let ordinary:[String:Any];let fields:[String:Any];let stack:[String:Any]
    var evaluations:Int { fields["stage"] as? String == "stage-dmd-step-250" ? 8:50 }
  }
  private static func integer(_ value:Any?,_ expected:Int) -> Bool {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
    return n.doubleValue == Double(expected)
  }
  static func packet(_ recipe:[String:Any]) throws -> Packet {
    guard let fields=recipe["vdn"] as? [String:Any],
      Set(fields.keys)==["repository","checkpoint","model_spec","linear_branch","default_adapter","turbo_adapter","schedule_points","stage","inference_backend"],
      let repository=fields["repository"] as? String,repository.hasPrefix("/"),!repository.utf8.contains(0),
      let stageName=fields["stage"] as? String,["stage-dmd-step-250","stage-b-step-2000"].contains(stageName),fields["inference_backend"] as? String == "verified",
      integer(fields["schedule_points"],stageName=="stage-dmd-step-250" ? 9:51),
      let components=recipe["components"] as? [String:Any],components["task"] as? String == "t2va",
      components["fun_controlnet"]==nil,
      components["loras"]==nil || (components["loras"] as? [Any])?.isEmpty==true,
      let config=recipe["config"] as? [String:Any],integer(config["steps"],stageName=="stage-dmd-step-250" ? 9:51),
      config["sampling_method"] as? String == "euler",
      let stack=recipe["loras"] as? [String:Any],Set(stack.keys).isSubset(of:["version","adapters"]),
      stack["version"]==nil || integer(stack["version"],1),
      let adapters=stack["adapters"] as? [[String:Any]],adapters.count==(stageName=="stage-dmd-step-250" ? 2:1) else {
      throw StudioError.invalid("Select a released eight- or fifty-step VDN T2VA profile with its full-strength adapter stack and Euler schedule.")
    }
    let stage=URL(fileURLWithPath:repository).appendingPathComponent(stageName)
    let paths=[stage.appendingPathComponent("adapters/default/adapter_model.safetensors").path,
      stage.appendingPathComponent("adapters/turbo/adapter_model.safetensors").path]
    for (key,path) in ["checkpoint":stage.path,"model_spec":stage.appendingPathComponent("model_spec.json").path,
      "linear_branch":stage.appendingPathComponent("linear_branch/model.safetensors").path,
      "default_adapter":paths[0],"turbo_adapter":paths[1]] {
      if key=="turbo_adapter",stageName=="stage-b-step-2000" {
        guard fields[key] is NSNull else { throw StudioError.invalid("Fifty-step VDN must not use a Turbo adapter.") };continue
      }
      guard fields[key] as? String == path else { throw StudioError.invalid("VDN component paths must belong to the selected released stage.") }
    }
    for (index,a) in adapters.enumerated() {
      guard Set(a.keys).isSubset(of:["path","strength","profile","qkv_layout","adaln_input_grid","start_after_evaluations"]),
        a["path"] as? String == paths[index],integer(a["strength"],1),
        a["profile"] as? String == (index==0 ? "standard":"turbo"),a["qkv_layout"] as? String == "contiguous_qkv",
        a["start_after_evaluations"]==nil || integer(a["start_after_evaluations"],0),
        a["adaln_input_grid"]==nil || a["adaln_input_grid"] is NSNull ||
          (index==1 && (a["adaln_input_grid"] as? String).map { $0.hasPrefix("/") && !$0.utf8.contains(0) } == true) else {
        throw StudioError.invalid("VDN requires its released adapters at strength 1, immediate activation and correct timestep input coordinates.")
      }
    }
    var ordinary=recipe;ordinary.removeValue(forKey:"vdn");ordinary.removeValue(forKey:"loras")
    return Packet(ordinary:ordinary,fields:fields,stack:stack)
  }
  static func validate(clip:Clip,project:StudioProject,motion:NativeH3MotionPlan?,recipe:[String:Any]) throws {
    let packet=try packet(recipe)
    guard ["t2v","t2va"].contains(clip.inferredTask),clip.continuityMode=="independent",motion==nil,
      !project.shouldSaveContinuityContext(for:clip),clip.attachments.isEmpty,
      clip.generationSelection?.steps==nil || clip.generationSelection?.steps==packet.evaluations,
      clip.generationSelection?.h3SamplingMethod==nil || clip.generationSelection?.h3SamplingMethod == .euler,
      clip.generationSelection?.h3Joint==nil,clip.generationSelection?.h3MotionFidelity==nil else {
      throw StudioError.invalid("VDN uses its released Euler evaluation count and fixed adapter stack. References, extra LoRAs, continuity, latent refinement and motion repair are not supported by this Studio route.")
    }
  }
  static func sources(_ recipe:[String:Any]) -> [String] {
    guard let packet=try? packet(recipe) else { return [] }
    let adapters=packet.stack["adapters"] as! [[String:Any]]
    return ["model_spec","linear_branch","default_adapter","turbo_adapter"].compactMap { packet.fields[$0] as? String }
      + adapters.compactMap { ($0["path"] as? String).map { URL(fileURLWithPath:$0).deletingLastPathComponent().appendingPathComponent("adapter_config.json").path } }
      + adapters.compactMap { $0["adaln_input_grid"] as? String }
  }
}
