import Foundation
import CoreFoundation

/// Applies only explicit full-latent intent after ordinary native composition.
public enum NativeH3JointPreparation {
  public static func apply(_ settings:H3JointSettings?,to ordinary:[String:Any],continuityMode:String) throws -> [String:Any] {
    guard let settings else { return ordinary }
    guard settings.saveFullLatents || settings.refinement != nil else { throw StudioError.invalid("Remove inactive H3 full-latent controls or enable Save full latents.") }
    guard continuityMode=="independent",ordinary["continuation"]==nil,ordinary["motion_fidelity"]==nil,
      ordinary["joint_latents"]==nil,ordinary["refinement"]==nil,
      let config=ordinary["config"] as? [String:Any],let components=ordinary["components"] as? [String:Any],
      components["fun_controlnet"]==nil,let task=components["task"] as? String,
      ["t2va","fl2va","ref2va"].contains(task) else { throw StudioError.invalid("H3 full-latent refinement cannot combine continuation, motion fidelity or Fun control.") }
    var recipe=ordinary
    if settings.saveFullLatents { recipe["joint_latents"]=["version":1,"save_full":true] }
    if let selected=settings.refinement {
      guard selected.strength.isFinite,selected.strength>0,selected.strength<=1,
        selected.startVideoSigma==nil || (selected.startVideoSigma!.isFinite && selected.startVideoSigma!>0 && selected.startVideoSigma!<=1),
        selected.evaluations==nil || (selected.startVideoSigma != nil && (1...64).contains(selected.evaluations!)) else { throw StudioError.invalid("H3 full-latent refinement requires valid strength and explicit noise/evaluation controls.") }
      func integer(_ key:String)->Int? {
        guard let n=config[key] as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
          n.doubleValue.rounded()==n.doubleValue,(32...4096).contains(n.doubleValue) else {return nil};return n.intValue
      }
      guard let width=integer("width"),let height=integer("height"),
        let duration=config["duration_seconds"] as? Double,duration.isFinite,(2.5...15).contains(duration) else { throw StudioError.invalid("H3 refinement requires a valid target canvas and duration.") }
      var frames=Int((duration*24).rounded(.toNearestOrEven));while frames%17 != 5 { frames+=1 }
      guard selected.source.task==task,selected.source.generatedFrames==frames else { throw StudioError.invalid("H3 full-latent source task or complete duration differs from the target.") }
      let expanded=selected.expandedSpatialTarget == true
      let learned=selected.learnedUpscalerPath != nil
      var learnedInspection:NativeH3LearnedUpscalerMetadata.Inspection?
      if selected.mode == .spatial {
        guard width>selected.source.width,height>selected.source.height,width<=2*selected.source.width,height<=2*selected.source.height,
          (selected.resizeMethod != nil) != learned,selected.startVideoSigma==nil,selected.evaluations==nil,
          learned || selected.learnedUpscalerHeaderSHA256==nil else { throw StudioError.invalid("H3 spatial refinement requires both axes enlarged by at most 2×, a resize method and the owned schedule suffix.") }
      } else {
        guard width*height<=1376*768 else {throw StudioError.invalid("Initialized same-canvas refinement remains within the ordinary 1 MP budget. Expanded full-latent artifacts can be stored, but this mode cannot reuse a larger canvas.")}
        guard width==selected.source.width,height==selected.source.height,selected.resizeMethod==nil,
          selected.expandedSpatialTarget==nil,selected.learnedUpscalerPath==nil,selected.learnedUpscalerHeaderSHA256==nil else { throw StudioError.invalid("H3 initialized refinement requires the exact source canvas and no resize method.") }
      }
      if selected.mode == .spatial {
        guard !learned || expanded else {throw StudioError.invalid("The learned H3 upscaler requires explicit spatial v2 admission.")}
        let pixels=width*height
        if expanded {
          let targetRows=((frames-5)/17*5+2)*(width/32)*(height/32)+2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))
          guard width%32==0,height%32==0,max(width,height)<=1920,min(width,height)<=1088,
            pixels<=1920*1088,targetRows<=64_000 else {
            throw StudioError.invalid("Spatial v2 requires a 32-pixel grid, longest edge at most 1920, shortest edge at most 1088, area at most 1920×1088 and complete packed rows at most 64000. Text and conditions must also fit the worker budget.")
          }
        } else {
          guard pixels<=1376*768 else {throw StudioError.invalid("Enable spatial v2 explicitly for a target above the historical 1 MP budget.")}
        }
        if let path=selected.learnedUpscalerPath {
          learnedInspection=try NativeH3LearnedUpscalerMetadata.inspect(path:path,expectedHeaderSHA256:selected.learnedUpscalerHeaderSHA256)
        }
      }
      try selected.source.verify()
      var fields:[String:Any]=["version":expanded ? 2 : 1,"mode":selected.mode.rawValue,"source_manifest":selected.source.manifest,
        "source_manifest_sha256":selected.source.manifestSHA256,"strength":selected.strength,"preserve_audio":selected.preserveAudio]
      if let sigma=selected.startVideoSigma { fields["start_video_sigma"]=sigma }
      if let evaluations=selected.evaluations { fields["evaluations"]=evaluations }
      if let method=selected.resizeMethod { fields["resize_method"]=method.rawValue }
      if let model=learnedInspection {fields["learned_upscaler_path"]=model.path;fields["learned_upscaler_header_sha256"]=model.headerSHA256}
      recipe["refinement"]=fields
    }
    return recipe
  }
  public static func sourcePaths(_ settings:H3JointSettings?) -> [String] {
    guard let source=settings?.refinement?.source else { return [] };return [source.manifest,source.payloadPath]
  }
}
