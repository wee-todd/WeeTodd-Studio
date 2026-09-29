import Darwin
import Foundation
import LTX25Engine
import TensorIO

/// Developer qualification commands. No generation capability is advertised until
/// an entire weighted pipeline has passed numerical and lifecycle qualification.
@main
struct InferenceMain {
  static func main() {
    do {
      let arguments = Array(CommandLine.arguments.dropFirst())
      guard arguments.count == 2, ["inspect-checkpoint", "inspect-ltx-standard-lora"].contains(arguments[0]) else {
        throw CheckpointError.invalid("Usage: WeeToddInference inspect-checkpoint|inspect-ltx-standard-lora CHECKPOINT")
      }
      let file = try SafeTensorFile(url: URL(fileURLWithPath: arguments[1]))
      var result: [String: Any] = ["format": "weetodd-swift-checkpoint-inspection-v1",
        "scope": "header-validation", "file_bytes": file.fileByteCount,
        "tensor_count": file.tensors.count,
        "dtype_counts": Dictionary(grouping: file.tensors.values, by: \.dtype).mapValues(\.count)]
      if arguments[0] == "inspect-ltx-standard-lora" {
        let plan = try LTXAdapterCompatibility.standardPlan(file: file, strength: 1)
        result["scope"] = "standard-ltx-transformer-adapter-structure-only"
        result["source_model_version"] = file.metadata["model_version"] ?? "unspecified"
        result["adapter_pairs"] = plan.pairs.count
        result["adapter_ranks"] = Set(plan.pairs.map(\.rank)).sorted()
      }
      let json = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
      try FileHandle.standardOutput.write(contentsOf: json + Data([10]))
    } catch {
      try? FileHandle.standardError.write(contentsOf: Data("\(error)\n".utf8))
      exit(2)
    }
  }
}
