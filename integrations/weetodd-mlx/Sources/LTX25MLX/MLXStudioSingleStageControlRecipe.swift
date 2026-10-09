import Foundation
import CoreFoundation
import LTX25Engine

/// The ordinary compiler owns sampling and image anchors. This bridge validates
/// the dedicated Union guide, then attaches it to that same full-resolution job.
enum MLXStudioSingleStageControlRecipe {
  static func compileUnion(root original: [String: Any], outputDirectory: String) throws -> MLXDistilledRequest {
    func invalid(_ text: String) -> LTXError { .invalid("Swift LTX single-stage Union: " + text) }
    guard var components=original["components"] as? [String:Any],
      let condition=original["conditioning"] as? [String:Any],
      Set(condition.keys).isSubset(of:["version","task","audio_policy","inputs","control_family"]),
      let version=condition["version"] as? NSNumber,
      CFGetTypeID(version) != CFBooleanGetTypeID(),version.doubleValue == 1,
      condition["task"] as? String == "control",
      (condition["audio_policy"] as? String ?? "generated") == "generated",
      condition["control_family"] == nil || condition["control_family"] as? String == "union",
      let inputs=condition["inputs"] as? [[String:Any]],
      let pairs=components["ic_loras"] as? [[Any]],pairs.count == 1,pairs[0].count == 2,
      let adapter=pairs[0][0] as? String,let strength=pairs[0][1] as? NSNumber,
      CFGetTypeID(strength) != CFBooleanGetTypeID(),
      strength.doubleValue.isFinite,strength.doubleValue>0,strength.doubleValue<=2,
      (components["msr_lora_path"] as? String ?? "").isEmpty else {
      throw invalid("requires one dedicated adapter and generated audio")
    }
    let guides=inputs.filter { $0["role"] as? String == "control" }
    let images=inputs.filter { $0["role"] as? String == "keyframe" }
    guard guides.count == 1,guides.count+images.count == inputs.count,
      inputs.compactMap({ $0["id"] as? String }).count == inputs.count,
      Set(inputs.compactMap({ $0["id"] as? String })).count == inputs.count else {
      throw invalid("requires one guide and ordinary anchors with unique IDs")
    }
    let guide=guides[0]
    guard Set(guide.keys).isSubset(of:["id","kind","role","path","sha256","strength","control_type","format"]),
      let id=guide["id"] as? String,!id.isEmpty,
      guide["kind"] as? String == "video",guide["format"] as? String == "rgb24",
      ["canny_edges","depth_map","pose_skeleton"].contains(guide["control_type"] as? String ?? ""),
      let path=guide["path"] as? String,let digest=guide["sha256"] as? String else {
      throw invalid("requires a frozen RGB24 Canny, depth or pose guide")
    }
    var root=original
    components["ic_loras"]=[];root["components"]=components
    root["conditioning"]=["version":1,"task":images.isEmpty ? "t2v":"fflf",
      "audio_policy":"generated","inputs":images]
    var request=try MLXStudioRecipe.compileFields(data:JSONSerialization.data(withJSONObject:root),outputDirectory:outputDirectory)
    guard request["version"] as? Int == 15 else { throw invalid("requires the full-resolution single-stage sampler") }
    request["task"]="union_control"
    request["union_control_guide"]=["path":path,"source_sha256":digest,"adapter_path":adapter,
      "adapter_strength":strength,"reference_strength":guide["strength"] ?? 1]
    return try JSONDecoder().decode(MLXDistilledRequest.self,from:JSONSerialization.data(withJSONObject:request))
  }
}
