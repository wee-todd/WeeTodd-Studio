import Foundation

public enum NativeH3TwoStagePreparation {
  public static func apply(_ settings:H3TwoStageSettings,to ordinary:[String:Any]) throws -> [String:Any] {
    guard ordinary["fasth3"]==nil,ordinary["vdn"]==nil,ordinary["continuation"]==nil,
      ordinary["motion_fidelity"]==nil,ordinary["refinement"]==nil,ordinary["two_stage"]==nil,
      let config=ordinary["config"] as? [String:Any],let components=ordinary["components"] as? [String:Any],
      components["fun_controlnet"]==nil,
      ["t2va","fl2va","ref2va"].contains(components["task"] as? String ?? ""),
      (config["attention_policy"] as? String ?? "dense")=="dense",
      (config["projection_backend"] as? String ?? "mlx")=="mlx",
      (config["transformer_backend"] as? String ?? "mlx")=="mlx",
      (config["transformer_weight_cache_gb"] as? Int ?? 0)==0,
      let width=config["width"] as? Int,let height=config["height"] as? Int,
      let duration=config["duration_seconds"] as? Double else {
      throw StudioError.invalid("Automatic H3 upscale requires independent standard MLX T2V, I2V or Ref2VA with dense attention and no continuation, motion repair, Fun control, FastH3, VDN or transformer weight cache.")
    }
    if settings.useTurboDraft {
      guard config["steps"] as? Int == 5,(config["sampling_method"] as? String ?? "euler")=="euler" else {
        throw StudioError.invalid("The default H3 two-stage draft requires four Euler evaluations and its linked Turbo adapter.")
      }
    }
    _ = try settings.draftCanvas(width:width,height:height,durationSeconds:duration)
    let learned=settings.learnedUpscalerPath != nil
    guard (settings.resizeMethod != nil) != learned,
      learned || settings.learnedUpscalerHeaderSHA256==nil else {
      throw StudioError.invalid("Link the learned H3 latent upscaler, or explicitly choose interpolation for a custom recipe.")
    }
    var fields:[String:Any]=["version":1,"draft_scale":settings.draftScale,"strength":settings.strength,
      "evaluations":settings.evaluations,"refinement_adapter_policy":"standard"]
    if let method=settings.resizeMethod {fields["resize_method"]=method.rawValue}
    if let path=settings.learnedUpscalerPath {
      let model=try NativeH3LearnedUpscalerMetadata.inspect(path:path,expectedHeaderSHA256:settings.learnedUpscalerHeaderSHA256)
      fields["learned_upscaler_path"]=model.path;fields["learned_upscaler_header_sha256"]=model.headerSHA256
    }
    var recipe=ordinary;recipe["two_stage"]=fields;return recipe
  }
}
