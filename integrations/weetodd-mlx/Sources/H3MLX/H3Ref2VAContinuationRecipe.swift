import CoreFoundation
import Foundation

/// Explicit v4 publication/loading of a reference-conditioned AV history.
/// Inputs retain their visible target coordinates until this single bridge
/// places timed media after the saved overlap. Untimed identity stays untimed.
public enum H3Ref2VAContinuationRecipe {
  public struct Prepared {
    public let plan: H3Continuation.Plan
    public let sampledRecipe: Data
  }
  private static func integer(_ value: Any?) throws -> Int {
    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
      n.doubleValue.isFinite, n.doubleValue.rounded() == n.doubleValue,
      n.doubleValue >= 0, n.doubleValue < Double(Int.max) else {
      throw H3CheckpointError.invalid("Invalid Ref2VA continuation integer.")
    }
    return n.intValue
  }
  public static func prepare(data: Data) throws -> Prepared {
    try Task.checkCancellation()
    guard data.count <= 1_048_576,
      var recipe = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      (recipe["components"] as? [String: Any])?["task"] as? String == "ref2va",
      var conditioning = recipe["conditioning"] as? [String: Any],
      ["ref2va", "a2v"].contains(conditioning["task"] as? String ?? ""),
      var config = recipe["config"] as? [String: Any],
      let duration = config["duration_seconds"] as? NSNumber,
      CFGetTypeID(duration) != CFBooleanGetTypeID(),
      let fields = recipe.removeValue(forKey: "continuation") as? [String: Any],
      Set(fields.keys).isSubset(of: ["version", "context_frames", "source_context", "source_manifest_sha256", "save_context"]),
      try integer(fields["version"]) == 4,
      let inputs = conditioning["inputs"] as? [[String: Any]], !inputs.isEmpty,
      (fields["source_context"] == nil || fields["source_context"] is String),
      (fields["source_context"] as? String).map({ $0.hasPrefix("/") && !$0.utf8.contains(0) && !$0.contains("://") }) ?? true,
      (fields["source_manifest_sha256"] == nil || fields["source_manifest_sha256"] is String),
      (fields["save_context"] == nil || (fields["save_context"] as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() }) == true) else {
      throw H3CheckpointError.invalid("Ref2VA continuation v4 needs a compatible reference recipe and explicit synchronized context.")
    }
    let plan = try H3Continuation.Plan(contextFrames: integer(fields["context_frames"]), requestedDuration: duration.doubleValue,
      sourceManifest: (fields["source_context"] as? String).map { URL(fileURLWithPath: $0) },
      sourceSHA256: fields["source_manifest_sha256"] as? String, saveContext: fields["save_context"] as? Bool ?? false)
    let shifted = try inputs.map { source -> [String: Any] in
      var input = source
      // The A2V compiler places its implicit driver origin from this plan;
      // only authored explicit media coordinates are rewritten here.
      let position = input["frame_index"]
      if let frame = try H3ReferencePlacement.frame(position, frames: plan.publishedFrames) {
        input["frame_index"] = frame + plan.overlapFrames
      }
      return input
    }
    conditioning["inputs"] = shifted; recipe["conditioning"] = conditioning
    if plan.sourceManifest != nil { config["duration_seconds"] = min(Double(plan.generatedFrames) / 24, 15); recipe["config"] = config }
    return Prepared(plan: plan, sampledRecipe: try JSONSerialization.data(withJSONObject: recipe, options: [.sortedKeys]))
  }
}
