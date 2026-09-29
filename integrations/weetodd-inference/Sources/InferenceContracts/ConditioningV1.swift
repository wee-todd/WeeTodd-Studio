import CoreFoundation
import Foundation

/// Weight-free admission for the fields common to native H3 and LTX media recipes.
/// Engine-specific roles, timing, and image/audio decoding remain with each adapter.
public enum ConditioningV1 {
  public static func inputs(_ value: [String: Any], task: String,
    audioPolicy: String, count: ClosedRange<Int>) throws -> [[String: Any]] {
    guard Set(value.keys).isSubset(of: ["version", "task", "inputs", "audio_policy"]),
      let version = value["version"] as? NSNumber,
      CFGetTypeID(version) != CFBooleanGetTypeID(), version.intValue == 1,
      version.doubleValue == 1,
      !["f", "d"].contains(String(cString: version.objCType)),
      value["task"] as? String == task,
      (value["audio_policy"] == nil || value["audio_policy"] as? String == audioPolicy),
      let media = value["inputs"] as? [[String: Any]], count.contains(media.count) else {
      throw ContractError.invalid("Unsupported version-one conditioning task, audio policy, or input count.")
    }
    var identities = Set<String>()
    for item in media {
      guard let id = item["id"] as? String,
        !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        identities.insert(id).inserted else {
        throw ContractError.invalid("Conditioning inputs need unique nonempty IDs.")
      }
    }
    return media
  }
}
