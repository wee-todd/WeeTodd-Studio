import CoreFoundation
import Foundation
import LTX25Engine
import InferenceContracts

/// Strict adapter for Studio's existing source-video extension contract.
/// The normalized distilled request owns the generated audiovisual window;
/// source identity and publication range remain explicit here.
public struct MLXStudioExtensionRecipe {
  public let request:MLXDistilledRequest
  public let window:LTX25ExtensionWindow
  public let source:URL
  public let sourceSHA256:String

  public static func compile(data:Data,outputDirectory:String) throws -> Self {
    guard data.count<=1024*1024,
      var root=try JSONSerialization.jsonObject(with:data) as? [String:Any],
      let condition=root["conditioning"] as? [String:Any],
      Set(condition.keys).isSubset(of:["version","task","inputs","audio_policy","extension"]),
      condition["version"] as? Int == 1,condition["task"] as? String == "extension",
      (condition["audio_policy"] as? String ?? "source_reencoded_and_generated_extension") ==
        "source_reencoded_and_generated_extension",
      let inputs=condition["inputs"] as? [[String:Any]],inputs.count == 1,
      let input=inputs.first,
      Set(input.keys).isSubset(of:["id","kind","role","path","strength","sha256"]),
      let id=input["id"] as? String,!id.isEmpty,
      input["kind"] as? String == "video",input["role"] as? String == "reference",
      let path=input["path"] as? String,
      let digest=input["sha256"] as? String,
      input["strength"] == nil || (input["strength"] as? NSNumber).map({
        CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue == 1
      }) == true,
      let extensionValue=condition["extension"] as? [String:Any],
      Set(extensionValue.keys) == ["direction","context_frames","additional_frames"],
      extensionValue["direction"] as? String == "after",
      let context=extensionValue["context_frames"] as? Int,
      let additional=extensionValue["additional_frames"] as? Int,
      let config=root["config"] as? [String:Any],
      let width=config["width"] as? Int,let height=config["height"] as? Int,
      let fpsNumber=config["frame_rate"] as? NSNumber,
      CFGetTypeID(fpsNumber) != CFBooleanGetTypeID() else {
      throw LTXError.invalid("Swift LTX extension needs one frozen source movie, after direction, and explicit aligned timing.")
    }
    let frozen=try NativeMediaSource(path:path,sha256:digest)
    let window=try LTX25ExtensionWindow(contextFrames:context,additionalFrames:additional,
      width:width,height:height,fps:fpsNumber.doubleValue)
    guard window.totalFrames<=1501,Double(window.totalFrames-1)/window.geometry.fps<=20 else {
      throw LTXError.invalid("Swift two-stage LTX extension currently admits at most 20 seconds including source context.")
    }
    var normalized=config
    normalized["duration_seconds"]=Double(window.totalFrames-1)/window.geometry.fps
    root["config"]=normalized
    root["conditioning"]=["version":1,"task":"t2v","inputs":[]]
    let request=try MLXStudioRecipe.compile(data:JSONSerialization.data(withJSONObject:root),
      outputDirectory:outputDirectory)
    guard request.frames == window.totalFrames,request.noisePolicy == .releasedMLX else {
      throw LTXError.invalid("LTX extension's normalized recipe differs from its causal source window.")
    }
    return Self(request:request,window:window,source:URL(fileURLWithPath:frozen.path),
      sourceSHA256:frozen.sha256)
  }
}
