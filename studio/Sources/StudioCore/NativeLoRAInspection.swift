import CoreFoundation
import Darwin
import Foundation

/// Payload-free library inspection. Training provenance is a filter, not a promise
/// that a selected checkpoint/task supports every projection or adapter profile.
public enum NativeLoRAInspection {
  private struct Tensor { let shape: [UInt64]; let dtype: String }
  private struct Pair { let schema: String; var a: Tensor?; var b: Tensor? }
  private static let suffixes: [(String, Bool, String)] = [
    (".lora_A.weight", true, "ab"), (".lora_B.weight", false, "ab"),
    (".lora_A.default.weight", true, "default"), (".lora_B.default.weight", false, "default"),
    (".lora_A.turbo.weight", true, "turbo"), (".lora_B.turbo.weight", false, "turbo"),
    (".lora_down.weight", true, "down_up"), (".lora_up.weight", false, "down_up"),
    (".lora_a.weight", true, "lowercase_ab"), (".lora_b.weight", false, "lowercase_ab")]
  private static func invalid(_ message: String) -> StudioError { .invalid(message) }
  private static func integer(_ value: Any) throws -> UInt64 {
    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
      n.doubleValue.isFinite, n.doubleValue >= 0, n.doubleValue < Double(Int64.max),
      n.doubleValue == Double(n.int64Value) else { throw invalid("Invalid SafeTensors integer.") }
    return UInt64(n.int64Value)
  }
  private static func same(_ a: stat, _ b: stat) -> Bool {
    a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size
      && a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec
      && a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
  }
  private static func header(_ url: URL) throws -> ([String: String], [String: Tensor], stat) {
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw invalid("Cannot open the LoRA file.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
    var before = stat()
    guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG, before.st_size >= 10,
      let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
      throw invalid("Choose a regular SafeTensors LoRA file.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
    guard length >= 2, length <= 4 * 1024 * 1024, UInt64(before.st_size) >= length + 8,
      let bytes = try file.read(upToCount: Int(length)), bytes.count == Int(length),
      var document = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw invalid("Invalid SafeTensors header or header exceeds 4 MiB.")
    }
    var metadata: [String: String] = [:]
    if let raw = document.removeValue(forKey: "__metadata__"), !(raw is NSNull) {
      guard let strings = raw as? [String: String] else { throw invalid("SafeTensors metadata must contain strings.") }
      metadata = strings
    }
    let sizes: [String: UInt64] = ["BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
      "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
      "F8_E4M3": 1, "F8_E5M2": 1]
    let payload = UInt64(before.st_size) - 8 - length
    var tensors: [String: Tensor] = [:], ranges: [(UInt64, UInt64)] = []
    for (name, raw) in document {
      guard let entry = raw as? [String: Any], let dimensions = entry["shape"] as? [Any],
        let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
        let dtype = entry["dtype"] as? String, let size = sizes[dtype] else {
        throw invalid("Invalid SafeTensors descriptor: \(name).")
      }
      let shape = try dimensions.map(integer), start = try integer(offsets[0]), end = try integer(offsets[1])
      guard start <= end, end <= payload else { throw invalid("Invalid tensor payload offsets: \(name).") }
      var count: UInt64 = 1
      for axis in shape {
        let product = count.multipliedReportingOverflow(by: axis)
        guard !product.overflow else { throw invalid("Tensor shape overflows: \(name).") }; count = product.partialValue
      }
      let bytes = count.multipliedReportingOverflow(by: size)
      guard !bytes.overflow, bytes.partialValue == end - start else { throw invalid("Tensor shape/byte size mismatch: \(name).") }
      tensors[name] = Tensor(shape: shape, dtype: dtype); ranges.append((start, end))
    }
    var end: UInt64 = 0
    for range in ranges.sorted(by: { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }) {
      guard range.0 == end else { throw invalid("Tensor payload contains overlaps or gaps.") }; end = range.1
    }
    guard end == payload else { throw invalid("SafeTensors payload has unaccounted bytes.") }
    var after = stat(), linked = stat()
    guard fstat(fd, &after) == 0, Darwin.lstat(url.path, &linked) == 0, same(before, after), same(before, linked) else {
      throw invalid("File changed during inspection. Refresh to retry.")
    }
    return (metadata, tensors, before)
  }
  private static func model(_ metadata: [String: String], hint: LoRAModel?) throws -> LoRAModel? {
    var models = Set<LoRAModel>(), families = Set<String>()
    let pattern = try NSRegularExpression(pattern: "(?:^|[^0-9])2[._-]([0-9]+)(?:[^0-9]|$)")
    for key in ["model_version", "base_model", "base_model_name_or_path", "ss_base_model_version", "modelspec.architecture", "converted_layout"] {
      let value = (metadata[key] ?? "").lowercased()
      if value.contains("minimax") && value.contains("h3") { models.insert(.h3); families.insert("h3") }
      if value.contains("ltx") { families.insert("ltx") }
      if ["flux", "wan", "hunyuan", "stable-diffusion", "stable_diffusion", "sdxl"].contains(where: {
        value.range(of: "(?:^|[^a-z0-9])" + $0 + "(?:[^a-z0-9]|$)", options: .regularExpression) != nil
      }) { throw invalid("LoRA training metadata declares a different model family.") }
      if key == "model_version" || value.contains("ltx"),
        let match = pattern.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)),
        let range = Range(match.range(at: 1), in: value) {
        let minor = String(value[range])
        guard ["3", "5"].contains(minor) else { throw invalid("Unsupported LoRA training version: \(value).") }
        models.insert(minor == "3" ? .ltx23 : .ltx25)
      }
    }
    guard models.count <= 1, families.count <= 1 else { throw invalid("LoRA has conflicting training-model metadata.") }
    if models.isEmpty, let family = families.first, let hint,
      family != (hint == .h3 ? "h3" : "ltx") {
      throw invalid("The selected training model conflicts with the declared model family.")
    }
    return models.first ?? hint
  }
  private static func scaling(_ metadata: [String: String]) throws {
    for (names, minimum) in [(["lora_rank", "ss_network_dim", "network_dim"], 1.0),
      (["lora_alpha", "ss_network_alpha", "network_alpha"], 0.0)] {
      var values = Set<Double>()
      for key in names {
        guard let raw = metadata[key] else { continue }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["", "none", "null", "dynamic", "baked", "baked_scale"].contains(value) { continue }
        guard let number = Double(value), number.isFinite, number >= minimum else { throw invalid("Invalid adapter scaling metadata: \(key).") }
        values.insert(number)
      }
      guard values.count <= 1 else { throw invalid("Conflicting adapter scaling metadata.") }
    }
  }
  private static func h3(_ metadata: [String: String], targets: [String], selectedProfile: String?) throws -> [String: Any] {
    func trim(_ key: String) -> String { (metadata[key] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
    var profiles = Set<String>(), counts = Set<Int>(), layouts = Set<String>()
    for key in ["adapter_profile", "profile", "distillation_profile"] {
      let value = trim(key); if value.isEmpty { continue }
      if ["standard", "base", "quality"].contains(value) { profiles.insert("standard") }
      else if value == "turbo" { profiles.insert("turbo") }
      else { throw invalid("This specialized LoRA profile belongs in a model/task recipe.") }
    }
    if trim("adapter_role") == "turbo" { profiles.insert("turbo") }
    for key in ["inference_steps", "num_inference_steps", "steps", "transformer_evaluations", "schedule_points"] {
      guard let raw = metadata[key] else { continue }
      let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard value.range(of: "^[1-9][0-9]*$", options: .regularExpression) != nil,
        let count = Int(value) else { throw invalid("Invalid H3 LoRA sampling metadata: \(key).") }
      counts.insert(count - (key == "schedule_points" ? 1 : 0))
    }
    guard counts.count <= 1 else { throw invalid("LoRA has conflicting sampling-count metadata.") }
    if let count = counts.first, (1...8).contains(count) { profiles.insert("turbo") }
    guard profiles.count <= 1 else { throw invalid("LoRA has conflicting profile metadata.") }
    if (profiles.first ?? selectedProfile) == "turbo", let count = counts.first, count != 4 {
      throw invalid("This H3 Turbo adapter needs a dedicated model/task recipe; the library supports 4 evaluations (5 schedule points).")
    }
    for key in ["qkv_layout", "qkv_fusion", "conversion"] {
      let value = trim(key); if value.isEmpty || (key == "conversion" && !value.contains("qkv")) { continue }
      var recognized = false
      if ["contiguous", "concat", "block diagonal", "block-diag"].contains(where: value.contains) {
        layouts.insert("contiguous_qkv"); recognized = true
      }
      if ["interleaved", "per-head"].contains(where: value.contains) { layouts.insert("native_interleaved"); recognized = true }
      guard key != "qkv_layout" || recognized || value == "auto" else { throw invalid("Unsupported H3 LoRA QKV layout metadata.") }
    }
    guard layouts.count <= 1 else { throw invalid("LoRA has conflicting QKV layout metadata.") }
    var result: [String: Any] = ["loraRequiresAdalnGrid": targets.contains(where: { $0.contains("adaln_proj") })]
    if let count = counts.first { result["h3DeclaredEvaluations"] = count }
    if let profile = profiles.first { result["loraProfile"] = profile }
    if let layout = layouts.first { result["loraLayout"] = layout }
    return result
  }
  /// Shared admission for an explicitly selected or header-declared Turbo adapter.
  /// Unknown generic adapters retain the caller's ordinary schedule unchanged.
  public static func validateH3Sampling(path: String, selectedProfile: String?, schedulePoints: Int) throws -> [String: Any] {
    let info = try inspect(URL(fileURLWithPath: path), modelHint: .h3, selectedH3Profile: selectedProfile)
    guard info["loraModel"] as? String == "h3" else { throw invalid("The adapter declares a different training model.") }
    if let selectedProfile, let declared = info["loraProfile"] as? String, declared != selectedProfile {
      throw invalid("The selected H3 adapter profile conflicts with its declared sampling profile.")
    }
    if selectedProfile == "turbo" || info["loraProfile"] as? String == "turbo" {
      guard schedulePoints == 5 else { throw invalid("H3 Turbo requires 4 evaluations (5 schedule points); select Steps 4.") }
      if let declared = info["h3DeclaredEvaluations"] as? Int, declared != 4 {
        throw invalid("H3 Turbo sampling metadata must declare 4 evaluations (5 schedule points).")
      }
    }
    return info
  }
  public static func inspect(_ url: URL, modelHint: LoRAModel? = nil, selectedH3Profile: String? = nil) throws -> [String: Any] {
    let source = url.standardizedFileURL.resolvingSymlinksInPath()
    guard source.pathExtension.lowercased() == "safetensors" else { throw invalid("Choose a SafeTensors LoRA file.") }
    let (metadata, tensors, identity) = try header(source)
    var pairs: [String: Pair] = [:], alphas = Set<String>()
    for (name, tensor) in tensors {
      if let suffix = suffixes.first(where: { name.hasSuffix($0.0) }) {
        let target = String(name.dropLast(suffix.0.count)); var pair = pairs[target] ?? Pair(schema: suffix.2)
        guard !target.isEmpty, pair.schema == suffix.2, (suffix.1 ? pair.a : pair.b) == nil,
          ["F16", "BF16", "F32", "F64"].contains(tensor.dtype) else { throw invalid("Ambiguous pair or unsupported adapter dtype: \(name).") }
        if suffix.1 { pair.a = tensor } else { pair.b = tensor }; pairs[target] = pair
      } else if let suffix = [".alpha", ".lora_alpha", ".alpha.weight"].first(where: name.hasSuffix) {
        let target = String(name.dropLast(suffix.count))
        guard tensor.shape.allSatisfy({ $0 == 1 }), alphas.insert(target).inserted else { throw invalid("Adapter alpha must be one unique scalar: \(name).") }
      } else { throw invalid("Unsupported adapter tensor; refusing partial application: \(name).") }
    }
    guard !pairs.isEmpty else { throw invalid("No supported LoRA A/B pairs.") }
    guard alphas.isSubset(of: Set(pairs.keys)) else { throw invalid("Alpha without adapter pair.") }
    for (target, pair) in pairs {
      guard let a = pair.a, let b = pair.b, a.shape.count == 2, b.shape.count == 2,
        (a.shape + b.shape).allSatisfy({ $0 > 0 }), a.shape[0] == b.shape[1] else { throw invalid("Incomplete pair or invalid adapter rank/shape: \(target).") }
    }
    try scaling(metadata)
    let model = try model(metadata, hint: modelHint)
    let trim: (String) -> String = { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
    let role = trim(metadata["adapter_role"] ?? "standard")
    guard ["standard", "transformer_lora", "style", "character"].contains(role) || (model == .h3 && role == "turbo"),
      !metadata.keys.contains(where: { $0.hasPrefix("reference_") }) else {
      throw invalid("This specialized adapter belongs in a model/task recipe, not a style LoRA group.")
    }
    var result: [String: Any] = ["kind": "lora", "path": source.path]
    if let model {
      result["loraModel"] = model.rawValue
      if model == .h3 {
        guard selectedH3Profile == nil || ["standard", "turbo"].contains(selectedH3Profile!) else {
          throw invalid("Select a supported H3 LoRA profile.")
        }
        result.merge(try h3(metadata, targets: Array(pairs.keys), selectedProfile: selectedH3Profile)) { _, new in new }
      }
      else { try NativeLTXLoRAMetadata.validate(path: source.path, model: model.rawValue) }
    } else {
      guard ["adapter_profile", "profile", "distillation_profile"].allSatisfy({
        ["", "standard", "base", "quality"].contains(trim(metadata[$0] ?? ""))
      }) else { throw invalid("This specialized adapter requires an explicit model/task recipe.") }
    }
    var after = stat()
    guard Darwin.lstat(source.path, &after) == 0, same(identity, after) else {
      throw invalid("File changed during inspection. Refresh to retry.")
    }
    return result
  }
}
