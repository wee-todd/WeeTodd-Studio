import Foundation

public struct GenerationPresetChange: Equatable, Identifiable {
  public var path, label, saved, current: String
  public var id: String { path }
}
public enum GenerationPresetDiff {
  public static func changes(from saved: GenerationPresetSnapshot, to current: GenerationPresetSnapshot) -> [GenerationPresetChange] {
    var changes: [GenerationPresetChange] = []
    func add(_ path: String, _ label: String, _ before: String?, _ after: String?) {
      if before != after { changes.append(.init(path: path, label: label, saved: before ?? "Not specified", current: after ?? "Not specified")) }
    }
    func model(_ value: GenerationModelRef?) -> String? {
      value.map { [$0.backend, $0.family, $0.variant, $0.precision, $0.id, $0.sourceIdentity].compactMap { $0 }.joined(separator: " · ") }
    }
    add("model", "Model / precision", model(saved.model), model(current.model))
    if saved.context != current.context { add("context", "Workspace / task", saved.context.workspace.rawValue + "/" + saved.context.task, current.context.workspace.rawValue + "/" + current.context.task) }
    let before = saved.adapters ?? [], after = current.adapters ?? []
    if saved.adapters == nil || current.adapters == nil {
      add("adapters", "LoRA stack", saved.adapters.map { $0.isEmpty ? "None" : $0.map { $0.reference.id }.joined(separator: ", ") }, current.adapters.map { $0.isEmpty ? "None" : $0.map { $0.reference.id }.joined(separator: ", ") })
    } else {
      add("adapters.order", "LoRA order", before.map { $0.reference.id }.joined(separator: ", "), after.map { $0.reference.id }.joined(separator: ", "))
    }
    for id in Set(before.map { $0.reference.id } + after.map { $0.reference.id }).sorted() {
      let a = before.first { $0.reference.id == id }, b = after.first { $0.reference.id == id }
      let prefix = "adapters." + id
      if a == nil || b == nil {
        func summary(_ value: GenerationPresetAdapter) -> String { value.reference.family + " · " + String(value.strength) + " · " + (value.enabled ? "On" : "Off") }
        add(prefix, "LoRA " + id, a.map(summary), b.map(summary))
        for key in Set(Array((a?.fields ?? [:]).keys) + Array((b?.fields ?? [:]).keys)).sorted() { walk(prefix + "." + key, a?.fields[key], b?.fields[key]) }
        continue
      }
      add(prefix + ".enabled", "LoRA enabled · " + id, a!.enabled ? "On" : "Off", b!.enabled ? "On" : "Off")
      add(prefix + ".strength", "LoRA strength · " + id, String(a!.strength), String(b!.strength))
      if a!.reference != b!.reference {
        func identity(_ ref: GenerationLoRARef) -> String { [ref.backend, ref.family, ref.id, ref.sourceIdentity].compactMap { $0 }.joined(separator: " · ") }
        add(prefix + ".identity", "LoRA source · " + id, identity(a!.reference), identity(b!.reference))
      }
      for key in Set(Array(a!.fields.keys) + Array(b!.fields.keys)).sorted() {
        walk(prefix + "." + key, a!.fields[key], b!.fields[key])
      }
    }
    func walk(_ path: String, _ a: GenerationPresetValue?, _ b: GenerationPresetValue?) {
      if a == b { return }
      if path.hasSuffix(".model") {
        let before = try? a?.decoded(GenerationModelRef.self), after = try? b?.decoded(GenerationModelRef.self)
        if before != nil || after != nil { add(path, "Model / precision", model(before), model(after)); return }
      }
      let aObject: [String: GenerationPresetValue]? = { if case .object(let v) = a { return v }; return nil }()
      let bObject: [String: GenerationPresetValue]? = { if case .object(let v) = b { return v }; return nil }()
      if (aObject != nil && (bObject != nil || b == nil)) || (bObject != nil && a == nil) {
        for key in Set(Array((aObject ?? [:]).keys) + Array((bObject ?? [:]).keys)).sorted() { walk(path + "." + key, aObject?[key], bObject?[key]) }
      } else { add(path, path.replacingOccurrences(of: "fields.", with: "").replacingOccurrences(of: ".", with: " › "), a?.display, b?.display) }
    }
    for key in Set(Array(saved.fields.keys) + Array(current.fields.keys)).sorted() { walk("fields." + key, saved.fields[key], current.fields[key]) }
    return changes
  }
}
