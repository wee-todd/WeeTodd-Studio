/// Maps actual H3 transformer work into the worker's 10–83% sampling span.
/// A requested step count contains one more sigma point than evaluations.
public enum H3WorkerProgress {
  public static func fraction(stage: String, completed: Int, total: Int,
    evaluations: Int) -> Double? {
    guard evaluations > 0, total > 0,
      (0...total).contains(completed) else { return nil }
    if stage == "transformer_prepare" {
      guard total == 50 else { return nil }
      return 0.08 + 0.02 * Double(completed) / Double(total)
    }
    if stage == "sampling" {
      guard total == evaluations else { return nil }
      return 0.1 + 0.73 * Double(completed) / Double(evaluations)
    }
    let prefix = "sampling_block_"
    guard stage.hasPrefix(prefix), total == 50,
      let step = Int(stage.dropFirst(prefix.count)),
      (1...evaluations).contains(step) else { return nil }
    return 0.1 + 0.73 * (Double(step - 1)
      + Double(completed) / Double(total)) / Double(evaluations)
  }
}

/// Boundaries used by the native worker to report disjoint reference-task stage times.
public enum H3WorkerStageBoundary {
  public static func tracks(task: String) -> Bool {
    task == "ref2va" || task == "fl2va"
  }

  public static func name(stage: String, completed: Int,
    total: Int) -> String? {
    guard total > 0, completed == total else { return nil }
    switch stage {
    case "text_weights_released": return "qwen"
    case "reference_video_weights_released": return "referenceVideoEncode"
    case "keyframe_video_weights_released": return "keyframeVideoEncode"
    case "transformer_prepare" where total == 50:
      return "transformerPreparation"
    case "transformer_weights_released": return "sampling"
    case "video_weights_released": return "videoDecode"
    case "audio_weights_released": return "audioDecode"
    default: return nil
    }
  }
}
