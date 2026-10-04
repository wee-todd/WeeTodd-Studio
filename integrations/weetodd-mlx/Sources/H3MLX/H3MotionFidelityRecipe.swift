import CoreFoundation
import Foundation

/// Versioned opt-in movie workflow. Old recipes have no additional media reads.
public enum H3MotionFidelityRecipe {
  public struct Prepared: Sendable {
    public let ordinaryRecipe: Data
    public let sourcePath: String
    public let sourceSHA256: String
    public let sourceIn: Double
    public let durationSeconds: Double
    public let ffprobe: URL
    public let settings: H3MotionFidelitySettings
  }
  public static func prepare(data: Data) throws -> Prepared? {
    guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3CheckpointError.invalid("H3 motion recipe must be a JSON object.")
    }
    guard let fields = root.removeValue(forKey: "motion_fidelity") else { return nil }
    guard root["engine"] as? String == "h3",
      root["joint_latents"] == nil, root["refinement"] == nil, root["continuation"] == nil,
      let components = root["components"] as? [String: Any],
      (components["task"] as? String ?? "t2va") == "t2va", components["fun_controlnet"] == nil,
      let conditioning = root["conditioning"] as? [String: Any],
      (conditioning["task"] as? String ?? "t2v") == "t2v",
      (conditioning["inputs"] as? [Any] ?? []).isEmpty,
      let config = root["config"] as? [String: Any], let steps = integer(config["steps"]), steps >= 16,
      let object = fields as? [String: Any],
      Set(object.keys).isSubset(of: ["version", "source_video", "source_sha256", "source_in", "duration_seconds",
        "ffprobe", "mode", "strength", "max_hold", "sensitivity", "seed", "max_frames", "evaluations"]),
      integer(object["version"]) == 1,
      let path = object["source_video"] as? String, validPath(path),
      let hash = object["source_sha256"] as? String, hash.utf8.count == 64,
      hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      let probe = object["ffprobe"] as? String, validPath(probe),
      let duration = number(object["duration_seconds"]),
      let modeString = object["mode"] as? String,
      let mode = H3MotionFidelitySettings.Mode(rawValue: modeString),
      let strength = number(object["strength"]),
      let hold = integer(object["max_hold"]),
      let sensitivity = number(object["sensitivity"]),
      let seed = integer(object["seed"]), seed >= 0,
      let maximum = integer(object["max_frames"]),
      object["source_in"] == nil || number(object["source_in"]) != nil,
      object["evaluations"] == nil || integer(object["evaluations"]) != nil else {
      throw H3CheckpointError.invalid("Invalid H3 Motion Fidelity v1 recipe, source or settings.")
    }
    let start = number(object["source_in"]) ?? 0
    guard start.isFinite, start >= 0, duration.isFinite, (2.5...Double(345) / 24).contains(duration),
      abs(start * 24 - (start * 24).rounded(.toNearestOrEven)) <= 0.001,
      abs(duration * 24 - (duration * 24).rounded(.toNearestOrEven)) <= 0.001 else {
      throw H3CheckpointError.invalid("Motion Fidelity requires a bounded native frame-aligned source trim.")
    }
    let settings = try H3MotionFidelitySettings(mode: mode, strength: strength, maxHold: hold,
      sensitivity: sensitivity, seed: UInt64(seed), maxFrames: maximum, evaluations: integer(object["evaluations"]))
    return Prepared(ordinaryRecipe: try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]),
      sourcePath: path, sourceSHA256: hash, sourceIn: start, durationSeconds: duration,
      ffprobe: URL(fileURLWithPath: probe), settings: settings)
  }
  private static func validPath(_ value: String) -> Bool {
    value.hasPrefix("/") && !value.utf8.contains(0) && !value.contains("://") && value.utf8.count <= 4096
  }
  private static func number(_ value: Any?) -> Double? {
    guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite else { return nil }
    return value.doubleValue
  }
  private static func integer(_ value: Any?) -> Int? {
    guard let value = number(value), value.rounded(.towardZero) == value,
      value >= Double(Int.min), value < Double(Int.max) else { return nil }
    return Int(value)
  }
}
