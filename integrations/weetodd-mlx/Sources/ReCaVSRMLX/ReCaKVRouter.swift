import Foundation

public enum ReCaKVRouter {
  static let actions = ["none", "none", "recent_1", "none", "recent_2", "none", "recent_2",
    "recent_4", "none", "recent_2_anchor", "recent_4", "recent_4_anchor", "recent_2",
    "recent_4_anchor", "anchor", "recent_4_anchor", "recent_2", "recent_4_anchor",
    "recent_2_anchor", "recent_4", "recent_2", "none", "recent_2_anchor", "recent_2",
    "none", "recent_2", "none", "recent_1", "none", "none"]
  static func policy(layer: Int) -> (recent: Int, anchors: Bool) {
    let action = actions[layer]
    let recent = action.contains("recent_4") ? 4 : action.contains("recent_2") ? 2 : action.contains("recent_1") ? 1 : 0
    return (recent, action.contains("anchor"))
  }
  public static func capacity(layer: Int) -> Int {
    let p = policy(layer: layer); return p.recent + (p.anchors ? 2 : 0)
  }
  public static func retainedIDs(candidates: [Int], layer: Int) throws -> [Int] {
    guard (0..<30).contains(layer), candidates == Array(Set(candidates)).sorted(),
      candidates.allSatisfy({ $0 >= 0 }) else { throw ReCaError.invalid("Invalid ReCaVSR KV plane IDs.") }
    let p = policy(layer: layer)
    var selected = Set(candidates.suffix(p.recent))
    if p.anchors { selected.formUnion(candidates.filter { $0 >= 5 && ($0 - 5) % 6 == 0 }.suffix(2)) }
    return candidates.filter(selected.contains)
  }
}
