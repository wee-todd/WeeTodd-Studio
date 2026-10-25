import CoreFoundation
import Foundation

/// Explicit automatic generation wrapper; ordinary/saved-source recipes are unchanged.
public enum H3TwoStageRecipe {
  public struct Prepared {
    public let ordinaryRecipe: Data
    public let plan: H3TwoStagePlan
    public let resizeMethod: H3SpatialLatentResizeMethod?
    public let learnedUpscaler: URL?
    public let learnedUpscalerHeaderSHA256: String?
    public let saveFullLatents: Bool
  }
  private static func number(_ value:Any?) throws -> Double {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite else {
      throw H3CheckpointError.invalid("H3 two-stage needs finite numeric settings.")
    };return n.doubleValue
  }
  private static func integer(_ value:Any?) throws -> Int {
    let n=try number(value)
    guard n.rounded()==n,(0...4096).contains(n) else {throw H3CheckpointError.invalid("Invalid bounded H3 two-stage integer.")}
    return Int(n)
  }
  public static func prepare(data:Data) throws -> Prepared? {
    guard data.count<=1_048_576,var root=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {
      throw H3CheckpointError.invalid("Invalid bounded H3 two-stage recipe.")
    }
    guard let raw=root.removeValue(forKey:"two_stage") else {return nil}
    guard let fields=raw as? [String:Any],
      Set(fields.keys).isSubset(of:["version","draft_scale","strength","evaluations","resize_method","learned_upscaler_path","learned_upscaler_header_sha256","refinement_adapter_policy"]),
      try integer(fields["version"])==1,fields["refinement_adapter_policy"] as? String=="standard",
      ["refinement","joint_refinement","continuation","motion_fidelity","fasth3","vdn"].allSatisfy({root[$0]==nil}),
      let config=root["config"] as? [String:Any],let components=root["components"] as? [String:Any],
      components["fun_controlnet"]==nil,
      ["t2va","fl2va","ref2va"].contains(components["task"] as? String ?? ""),
      ["t2v","fflf","ref2va","a2v"].contains((root["conditioning"] as? [String:Any])?["task"] as? String ?? ""),
      (config["projection_backend"] as? String ?? "mlx")=="mlx",
      (config["transformer_backend"] as? String ?? "mlx")=="mlx",
      (config["attention_policy"] as? String ?? "dense")=="dense",
      try config["transformer_weight_cache_gb"].map({try number($0)}) ?? 0 == 0 else {
      throw H3CheckpointError.invalid("Automatic H3 two-stage requires ordinary dense MLX generation, standard refinement adapters and no continuation, control, FastH3, VDN, motion repair or weight cache.")
    }
    let plan=try H3TwoStagePlan(width:integer(config["width"]),height:integer(config["height"]),
      durationSeconds:number(config["duration_seconds"]),draftScale:number(fields["draft_scale"]),
      strength:number(fields["strength"]),evaluations:integer(fields["evaluations"]))
    var method:H3SpatialLatentResizeMethod?,url:URL?,hash:String?
    if let value=fields["resize_method"] {
      guard let text=value as? String,let selected=H3SpatialLatentResizeMethod(rawValue:text),
        fields["learned_upscaler_path"]==nil,fields["learned_upscaler_header_sha256"]==nil else {
        throw H3CheckpointError.invalid("Choose an exclusive H3 latent resize method or learned model.")
      };method=selected
    } else {
      guard let path=fields["learned_upscaler_path"] as? String,path.hasPrefix("/"),path.utf8.count<=4096,
        !path.utf8.contains(0),!path.contains("://"),
        let digest=fields["learned_upscaler_header_sha256"] as? String,digest.utf8.count==64,
        digest.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}) else {
        throw H3CheckpointError.invalid("A learned H3 upscaler needs its saved absolute path and header SHA-256.")
      };url=URL(fileURLWithPath:path);hash=digest
    }
    var save=false
    if let raw=root.removeValue(forKey:"joint_latents") {
      guard let fields=raw as? [String:Any],Set(fields.keys)==["version","save_full"],try integer(fields["version"])==1,
        let flag=fields["save_full"] as? NSNumber,CFGetTypeID(flag)==CFBooleanGetTypeID(),flag.boolValue else {
        throw H3CheckpointError.invalid("H3 full-latent publication requires explicit version 1 save_full true.")
      };save=true
    }
    return Prepared(ordinaryRecipe:try JSONSerialization.data(withJSONObject:root,options:[.sortedKeys]),plan:plan,
      resizeMethod:method,learnedUpscaler:url,learnedUpscalerHeaderSHA256:hash,saveFullLatents:save)
  }
}
