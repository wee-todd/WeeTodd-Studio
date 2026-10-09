import Foundation

/// Explicit experimental T2VA admission. Ordinary recipes retain their strict
/// compiler; raw VDN PEFT descriptors never enter the ordinary LoRA loader.
public enum H3VDNStudioRecipe {
  public static func compile(data: Data) throws -> H3T2VARequest {
    func integer(_ value: Any?, _ expected: Int) -> Bool {
      guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
      return n.doubleValue == Double(expected)
    }
    guard data.count <= 1024 * 1024,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let fields = root.removeValue(forKey: "vdn") as? [String: Any],
      Set(fields.keys) == ["repository", "checkpoint", "model_spec", "linear_branch",
        "default_adapter", "turbo_adapter", "schedule_points", "stage", "inference_backend"],
      fields["inference_backend"] as? String == "verified",
      let repository = fields["repository"] as? String,
      repository.hasPrefix("/"), !repository.utf8.contains(0),
      let stageName = fields["stage"] as? String,
      ["stage-dmd-step-250", "stage-b-step-2000"].contains(stageName),
      root["continuation"] == nil, root["motion_fidelity"] == nil,
      root["joint_refinement"] == nil,
      let components = root["components"] as? [String: Any],
      components["loras"] == nil || (components["loras"] as? [Any])?.isEmpty == true,
      let stack = root.removeValue(forKey: "loras") as? [String: Any],
      Set(stack.keys).isSubset(of: ["version", "adapters"]),
      stack["version"] == nil || integer(stack["version"], 1),
      let adapters = stack["adapters"] as? [[String: Any]] else {
      throw H3CheckpointError.invalid("Swift VDN requires a released verified T2VA stage and its unmodified named adapter descriptors; combined controls are not qualified.")
    }
    let stage = URL(fileURLWithPath: repository).appendingPathComponent(stageName)
    let turbo = stageName == "stage-dmd-step-250"
    let expectedPaths = [stage.appendingPathComponent("adapters/default/adapter_model.safetensors").path]
      + (turbo ? [stage.appendingPathComponent("adapters/turbo/adapter_model.safetensors").path] : [])
    guard fields["checkpoint"] as? String == stage.path,
      fields["model_spec"] as? String == stage.appendingPathComponent("model_spec.json").path,
      fields["linear_branch"] as? String == stage.appendingPathComponent("linear_branch/model.safetensors").path,
      fields["default_adapter"] as? String == expectedPaths[0],
      turbo ? fields["turbo_adapter"] as? String == expectedPaths[1] : fields["turbo_adapter"] is NSNull,
      integer(fields["schedule_points"], turbo ? 9 : 51),
      adapters.count == expectedPaths.count else {
      throw H3CheckpointError.invalid("VDN stage paths, adapter stack or schedule differ from the released contract.")
    }
    var inputGrid: URL?
    for (index, adapter) in adapters.enumerated() {
      guard Set(adapter.keys).isSubset(of: ["path", "strength", "profile", "adaln_input_grid", "qkv_layout", "start_after_evaluations"]),
        adapter["path"] as? String == expectedPaths[index],
        integer(adapter["strength"], 1),
        adapter["profile"] as? String == (index == 0 ? "standard" : "turbo"),
        adapter["qkv_layout"] as? String == "contiguous_qkv",
        adapter["adaln_input_grid"] == nil || adapter["adaln_input_grid"] is NSNull ||
          (index == 1 && (adapter["adaln_input_grid"] as? String).map({ $0.hasPrefix("/") && !$0.utf8.contains(0) }) == true),
        adapter["start_after_evaluations"] == nil || integer(adapter["start_after_evaluations"], 0) else {
        throw H3CheckpointError.invalid("Swift VDN needs immediate full-strength released adapters and original-width timestep coordinates; only the Turbo modulation adapter may declare an input grid.")
      }
      if let path = adapter["adaln_input_grid"] as? String { inputGrid = URL(fileURLWithPath:path) }
    }
    guard try H3TransformerCachePlan.budget(config:root["config"] as? [String:Any] ?? [:]) == 0 else {
      throw H3CheckpointError.invalid("Transformer weight cache is not admitted for FastH3 or VDN.")
    }
    let base = try H3StudioRecipe.compile(data: JSONSerialization.data(withJSONObject: root))
    return try H3T2VARequest(prompt: base.prompt, width: base.geometry.width,
      height: base.geometry.height, durationSeconds: base.durationSeconds,
      seed: base.seed, requestedSteps: base.requestedSteps,
      transformer: base.transformer, qwenPages: base.qwenPages, tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      vdn: H3VDNSelection(stage: stage, variant: turbo ? .eightStep : .fiftyStep, adalnInputGrid:inputGrid),
      videoDecodeMemoryMode: base.videoDecodeMemoryMode, videoDecodePrecision: base.videoDecodePrecision, samplingMethod: base.samplingMethod)
  }
}
