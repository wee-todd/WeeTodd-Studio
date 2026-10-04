import CoreFoundation
import Foundation

/// FL2VA v3 keeps image indices in the new visible interval. Only this bridge
/// offsets them into the overlap plus sampled window; T2VA v2 is unchanged.
public enum H3FL2VAContinuationRecipe {
  public struct Prepared {
    public let plan: H3Continuation.Plan
    public let sampledRecipe: Data
    public let visibleAnchors: [Int]
    public let sampledAnchors: [Int]
  }
  private static func invalid() -> H3CheckpointError {
    .invalid("H3 FL2VA continuation v3 needs a compatible FL2VA recipe, legal synchronized context and visible keyframe indices.")
  }
  private static func integer(_ value: Any?) throws -> Int {
    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
      n.doubleValue.isFinite, n.doubleValue.rounded() == n.doubleValue,
      n.doubleValue >= 0, n.doubleValue < Double(Int.max) else { throw invalid() }
    return n.intValue
  }
  public static func prepare(data: Data) throws -> Prepared {
    try Task.checkCancellation()
    guard data.count <= 1_048_576,
      var recipe = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      (recipe["components"] as? [String: Any])?["task"] as? String == "fl2va",
      var conditioning = recipe["conditioning"] as? [String: Any],
      conditioning["task"] as? String == "fflf",
      var config = recipe["config"] as? [String: Any],
      let duration = config["duration_seconds"] as? NSNumber,
      CFGetTypeID(duration) != CFBooleanGetTypeID(),
      let fields = recipe.removeValue(forKey: "continuation") as? [String: Any],
      Set(fields.keys).isSubset(of: ["version", "context_frames", "source_context",
        "source_manifest_sha256", "save_context"]),
      try integer(fields["version"]) == 3,
      let inputs = conditioning["inputs"] as? [[String: Any]], (1...8).contains(inputs.count),
      (fields["source_context"] == nil || fields["source_context"] is String),
      (fields["source_context"] as? String).map({
        $0.hasPrefix("/") && !$0.utf8.contains(0) && !$0.contains("://")
      }) ?? true,
      (fields["source_manifest_sha256"] == nil || fields["source_manifest_sha256"] is String),
      (fields["save_context"] == nil || (fields["save_context"] as? NSNumber).map({
        CFGetTypeID($0) == CFBooleanGetTypeID()
      }) == true) else { throw invalid() }
    let plan = try H3Continuation.Plan(contextFrames: integer(fields["context_frames"]),
      requestedDuration: duration.doubleValue,
      sourceManifest: (fields["source_context"] as? String).map { URL(fileURLWithPath: $0) },
      sourceSHA256: fields["source_manifest_sha256"] as? String,
      saveContext: fields["save_context"] as? Bool ?? false)
    var visible: [Int] = [], sampled: [Int] = [], shifted: [[String: Any]] = []
    for var input in inputs {
      let anchor: Int
      if input["frame_index"] as? String == "last" { anchor = plan.publishedFrames - 1 }
      else { anchor = try integer(input["frame_index"]) }
      guard (0..<plan.publishedFrames).contains(anchor), visible.last.map({ $0 < anchor }) ?? true else {
        throw invalid()
      }
      let role = input["role"] as? String
      guard (role == "first" && anchor == 0) || role == "last" || role == "keyframe" else {
        throw invalid()
      }
      let position = anchor + plan.overlapFrames
      visible.append(anchor); sampled.append(position)
      input["frame_index"] = position
      // A visible first frame becomes an interior keyframe after the overlap.
      if role == "first" && position != 0 { input["role"] = "keyframe" }
      shifted.append(input)
    }
    conditioning["inputs"] = shifted; recipe["conditioning"] = conditioning
    if plan.sourceManifest != nil {
      config["duration_seconds"] = min(Double(plan.generatedFrames) / 24, 15)
      recipe["config"] = config
    }
    return Prepared(plan: plan,
      sampledRecipe: try JSONSerialization.data(withJSONObject: recipe, options: [.sortedKeys]),
      visibleAnchors: visible, sampledAnchors: sampled)
  }
}
