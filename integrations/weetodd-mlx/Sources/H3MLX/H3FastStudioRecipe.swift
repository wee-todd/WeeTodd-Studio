import Foundation

/// Explicit preview-v1 admission. Generic H3 recipes cannot accidentally select
/// a distilled or trained sparse checkpoint through a filename alone.
public enum H3FastStudioRecipe {
  public static func compile(data:Data) throws -> H3T2VARequest {
    guard data.count <= 1_048_576,
      var root = try JSONSerialization.jsonObject(with:data) as? [String:Any],
      let fast = root.removeValue(forKey:"fasth3") as? [String:Any],
      Set(fast.keys) == ["variant"],let name = fast["variant"] as? String,
      let variant = H3FastVariant(rawValue:name),
      root["continuation"] == nil,root["motion_fidelity"] == nil,root["joint_refinement"] == nil,
      root["vdn"] == nil,root["loras"] == nil else {
      throw H3CheckpointError.invalid("FastH3 needs an explicit dense-v1 or vsa-v1 T2VA recipe without combined controls.")
    }
    let base = try H3StudioRecipe.compile(data:JSONSerialization.data(withJSONObject:root))
    return try H3T2VARequest(prompt:base.prompt,width:base.geometry.width,height:base.geometry.height,
      durationSeconds:base.durationSeconds,seed:base.seed,requestedSteps:base.requestedSteps,
      transformer:base.transformer,qwenPages:base.qwenPages,tokenizer:base.tokenizer,
      videoVAE:base.videoVAE,audioVAE:base.audioVAE,turboLoRA:base.turboLoRA,
      additionalLoRAs:base.additionalLoRAs,loRAAdapters:base.loRAAdapters,funControl:base.funControl,
      fastVariant:variant,videoDecodeMemoryMode:base.videoDecodeMemoryMode,samplingMethod:base.samplingMethod)
  }
}
