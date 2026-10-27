import CoreFoundation
import Foundation

/// Editor admission for the experimental fixed Omni profile. Complete tensor,
/// contract-byte and packed media admission remain authoritative in the worker.
public enum NativeH3OmniProfile {
  public static let presetID = "swift-h3-omni"
  public static let contractSHA256 = "68577f23f0c866e329aa8a578b2455e4ca9a26a3ca1d15ddd6d0e99091007e18"
  public static let sourceSHA256 = "44fff6e29adde8ec485b1dddad02d9a134d75c52df2218e19bbf760199080ba6"
  public static let sourceBytes: Int64 = 37_304_406_432
  public static let requirements = "Experimental FastH3 Omni · 8 actual Euler evaluations · CFG 1. Ordered appearance references require an image or movie. Maximum 12 total: 9 images, 3 videos and 3 audio sources including movie soundtracks. Canvas: 32-pixel axes up to 2048; duration: 5–15 seconds. The worker admits all text, target and media within 40,000 packed rows."
  public static let advancedRequirements = "Fixed model requirements: PDD 32; boundaries 0,4,8,12,16,20,24,28,32; grid maximum 0.999; video/audio shifts 12/3; trained VSA tile 128, sparsity 0.9; p2_multi_region, reference keep rate 0.1. CPU reference preparation supports image pixel budgets 50–400%, movie match_output/native_h3 sizing and full temporal density. No timed placement, anchors, continuation, refinement, extra LoRAs, Sol or approximation caches."

  private static func number(_ value:Any?, integer:Bool = false) -> Double? {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
      !integer || !["f","d"].contains(String(cString:n.objCType)) else { return nil }
    return n.doubleValue
  }
  private static func path(_ value:Any?) -> Bool {
    guard let s=value as? String else {return false};return s.hasPrefix("/") && !s.utf8.contains(0)
  }
  static func ordinary(_ recipe:[String:Any]) throws -> [String:Any] {
    let required:Set<String>=["transformer","text_encoder","tokenizer","video_vae","audio_vae"]
    let optional:Set<String>=["vision_encoder","checkpoint","processor"]
    guard let wrapper=recipe["fasth3_omni"] as? [String:Any],Set(wrapper.keys)==["version","contract_path","contract_sha256"],
      number(wrapper["version"],integer:true)==1,path(wrapper["contract_path"]),wrapper["contract_sha256"] as? String==contractSHA256,
      Set(recipe.keys).isSubset(of:["format","engine","candidate","components","config","prompt","conditioning","ffmpeg","block_residency","negative_prompt","fasth3_omni"]),
      let components=recipe["components"] as? [String:Any],components["task"] as? String=="ref2va",
      Set(components.keys).isSubset(of:required.union(optional).union(["task","loras"])),
      required.allSatisfy({path(components[$0])}),optional.allSatisfy({components[$0]==nil || path(components[$0])}),
      components["loras"]==nil || (components["loras"] as? [Any])?.isEmpty==true,
      let config=recipe["config"] as? [String:Any],number(config["steps"],integer:true)==9,
      (config["drop_adaln"]==nil || (config["drop_adaln"] as? NSNumber).map { CFGetTypeID($0)==CFBooleanGetTypeID() && $0.boolValue } == true),
      let seed=number(config["seed"],integer:true),(0...Double(UInt32.max)).contains(seed),
      config["attention_policy"]==nil,config["visual_condition_strength"]==nil,config["audio_condition_strength"]==nil,
      config["sampling_method"]==nil || config["sampling_method"] as? String=="euler",
      config["transformer_backend"]==nil || config["transformer_backend"] as? String=="mlx",
      config["inference_optimization"]==nil || config["inference_optimization"] as? String=="off",
      config["paging_cache_gb"]==nil || number(config["paging_cache_gb"])==0,
      config["transformer_weight_cache_gb"]==nil || number(config["transformer_weight_cache_gb"])==0,
      let width=number(config["width"],integer:true),let height=number(config["height"],integer:true),
      width>0,height>0,width<=2048,height<=2048,Int(width)%32==0,Int(height)%32==0,
      let duration=number(config["duration_seconds"]),(5...15).contains(duration),
      let conditioning=recipe["conditioning"] as? [String:Any],number(conditioning["version"],integer:true)==1,conditioning["task"] as? String=="ref2va",
      let inputs=conditioning["inputs"] as? [[String:Any]],inputs.count<=12,
      inputs.allSatisfy({ Set($0.keys).isSubset(of:["id","kind","role","path","strength","sha256","soundtrack_path","soundtrack_sha256","image_pixel_budget_percent","size_policy","temporal_density"]) && $0["role"] as? String=="reference" }) else {
      throw StudioError.invalid("FastH3 Omni requires its pinned contract, native Ref2VA components and fixed 8-evaluation profile. Clear incompatible profile settings; the worker validates source bytes before inference.")
    }
    for input in inputs {
      let kind=input["kind"] as? String
      if let budget=input["image_pixel_budget_percent"] {
        guard kind=="image",let value=number(budget),value.rounded(.towardZero)==value,(50...400).contains(value) else {
          throw StudioError.invalid("Omni image pixel budget requires an image and an integer 50–400 percent.")
        }
      }
      if let size=input["size_policy"] {
        guard kind=="video",let value=size as? String,["match_output","native_h3"].contains(value) else {
          throw StudioError.invalid("Omni movie size policy requires match_output or native_h3 on a video.")
        }
      }
      if let density=input["temporal_density"] {
        guard kind=="video",density as? String=="full" else {
          throw StudioError.invalid("Omni supports full reference movie temporal density only.")
        }
      }
    }
    try canvas(width:Int(width),height:Int(height),duration:duration)
    var ordinary=recipe;ordinary.removeValue(forKey:"fasth3_omni");return ordinary
  }
  static func canvas(width:Int,height:Int,duration:Double) throws {
    guard duration.isFinite,(5...15).contains(duration),(32...2048).contains(width),(32...2048).contains(height),
      width%32==0,height%32==0 else { throw StudioError.invalid("Omni needs 5–15 seconds and 32-pixel axes up to 2048. Change the canvas explicitly.") }
    var frames=Int((duration*24).rounded(.toNearestOrEven));while frames%17 != 5 {frames+=1}
    let rows=((frames-5)/17*5+2)*(width/32)*(height/32)+2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))
    guard rows<40_000 else { throw StudioError.invalid("Omni's target alone exceeds its 40,000 packed-row budget. Reduce the canvas or duration; text and references also need room.") }
  }
  /// Pure editor checks: no file reads or model work on the presentation thread.
  public static func conflicts(clip:Clip) -> [String] {
    var issues:[String]=[];let s=clip.generationSelection
    if clip.inferredTask != "ref2va" {issues.append("Use Appearance reference inputs (Ref2VA).")}
    if s?.steps != nil && s?.steps != 8 || s?.h3SamplingMethod != nil && s?.h3SamplingMethod != .euler || s?.cfg != nil || s?.shift != nil {issues.append("Apply Omni's required 8 Euler evaluations and CFG 1.")}
    if s?.h3Joint != nil || s?.h3MotionFidelity != nil || s?.h3Reference != nil || s?.refinementSteps != nil {issues.append("Clear Turbo/two-stage, refinement, motion and reference-noise overrides.")}
    if s?.h3AttentionPolicy != nil || s?.h3TransformerWeightCacheGB != nil && s?.h3TransformerWeightCacheGB != 0 || clip.h3PagingCacheGB != nil && clip.h3PagingCacheGB != 0 || s?.transformerBackend != nil && s?.transformerBackend != "mlx" {issues.append("Clear attention, alternate backend and cache overrides; Omni owns its trained VSA.")}
    if clip.continuityMode != "independent" || !clip.extensionDirection.isEmpty {issues.append("Choose independent generation without continuation.")}
    if clip.attachments.contains(where: {$0.isEnabled && $0.role == .lora}) {issues.append("Disable or remove additional LoRAs in the LoRA controls.")}
    if clip.attachments.contains(where: {$0.role != .reference && $0.role != .lora}) {issues.append("Remove anchors, timed keyframes, controls and audio drivers; use ordered appearance references.")}
    if clip.attachments.contains(where: {$0.h3ReferencePlacement?.frame != nil || $0.h3ReferencePlacement?.videoTemporalDensity != nil && $0.h3ReferencePlacement?.videoTemporalDensity != .full}) {issues.append("Clear timed reference placement and non-full movie temporal density.")}
    return issues
  }
  /// Explicit clearing preserves supported CPU preparation and soundtrack selection.
  public static func clearUnsupportedReferenceSettings(_ placement:inout H3ReferencePlacement) {
    placement.frame=nil
    if placement.videoTemporalDensity != .full { placement.videoTemporalDensity=nil }
  }
  /// Explicitly invoked by the user. Authored inputs, LoRAs, prompt, canvas and
  /// duration survive; unsupported input purposes must be edited separately.
  public static func applyRequiredSettings(to clip:inout Clip) {
    var s=clip.generationSelection ?? GenerationSelection(task:clip.inferredTask)
    s.steps=8;s.h3SamplingMethod = .euler;s.cfg=nil;s.shift=nil;s.refinementSteps=nil
    s.h3Joint=nil;s.h3MotionFidelity=nil;s.h3Reference=nil;s.h3AttentionPolicy=nil
    s.h3TransformerWeightCacheGB=nil;s.transformerBackend=nil;s.projectionBackend=nil;s.memoryPolicy=nil
    clip.generationSelection=s;clip.h3PagingCacheGB=nil
  }
  static func validate(clip:Clip,project:StudioProject,motion:NativeH3MotionPlan?,recipe:[String:Any]) throws {
    _=try ordinary(recipe)
    let issues=conflicts(clip:clip)
    guard issues.isEmpty,motion==nil,!project.shouldSaveContinuityContext(for:clip) else {
      throw StudioError.invalid((["FastH3 Omni cannot execute retained incompatible settings."]+issues).joined(separator:" "))
    }
    try canvas(width:clip.generationWidth,height:clip.generationHeight,duration:clip.duration)
  }
}
