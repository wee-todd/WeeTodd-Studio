import Darwin
import Foundation

/// Reads only SafeTensors metadata. Projection/shape validation remains in the worker.
enum NativeLTXLoRAMetadata {
  static func validate(path: String, model: String) throws {
    func invalid(_ text: String) -> StudioError { .invalid("LoRA \(URL(fileURLWithPath: path).lastPathComponent): \(text)") }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw invalid("cannot open adapter.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      let lengthBytes = try file.read(upToCount: 8), lengthBytes.count == 8 else {
      throw invalid("expected a regular SafeTensors file.")
    }
    let length = lengthBytes.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
    guard length > 1, length <= 4 * 1024 * 1024, UInt64(status.st_size) >= length + 8,
      let bytes = try file.read(upToCount: Int(length)), bytes.count == Int(length),
      let header = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw invalid("invalid header or metadata exceeds 4 MiB.")
    }
    let metadata: [String: String]
    if let raw = header["__metadata__"] {
      guard let strings = raw as? [String: String] else { throw invalid("metadata must contain strings.") }
      metadata = strings
    } else { metadata = [:] }
    var declared = Set<String>()
    for key in ["model_version", "base_model", "base_model_name_or_path", "ss_base_model_version", "modelspec.architecture", "converted_layout"] {
      let value = (metadata[key] ?? "").lowercased()
      if value.contains("minimax"), value.contains("h3") { declared.insert("h3") }
      if key == "model_version" || value.contains("ltx") {
        let expression = try NSRegularExpression(pattern: "(?:^|[^0-9])2[._-]([0-9]+)(?:[^0-9]|$)")
        if let match = expression.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)),
          let range = Range(match.range(at: 1), in: value) {
          let minor = String(value[range])
          guard ["3", "5"].contains(minor) else { throw invalid("unsupported training version \(value).") }
          declared.insert("ltx2" + minor)
        }
      }
    }
    guard declared.isEmpty || declared == [model] else { throw invalid("training metadata conflicts with the library model; reimport the adapter.") }
    let trim: (String) -> String = { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
    let role = trim(metadata["adapter_role"] ?? "standard")
    guard ["standard", "transformer_lora", "style", "character"].contains(role),
      !metadata.keys.contains(where: { $0.hasPrefix("reference_") }),
      ["adapter_profile", "profile", "distillation_profile"].allSatisfy({
        ["", "standard", "base", "quality"].contains(trim(metadata[$0] ?? ""))
      }) else { throw invalid("specialized adapters require their own model/task recipe.") }
  }
}
