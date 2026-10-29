import Drama3Contracts
import Foundation
import MLX
import MLXNN

/// Retains MLX arrays in the released sharded layout; expert tensors remain quantized.
final class Drama3Weights {
  var arrays: [String: MLXArray] = [:]
  let config: [String: Any]
  let quant: [String: Any]
  init(at root: URL, files: [String]) throws {
    config =
      try JSONSerialization.jsonObject(
        with: Data(contentsOf: root.appendingPathComponent("config.json"))) as! [String: Any]
    quant = config["quantization"] as? [String: Any] ?? [:]
    for file in files {
      for (name, value) in try loadArrays(url: root.appendingPathComponent(file)) {
        guard arrays[name] == nil else {
          throw Drama3Error.invalid("Duplicate native tensor: \(name)")
        }
        arrays[name] = value
      }
    }
    let extra = root.appendingPathComponent("control_head.safetensors")
    if FileManager.default.fileExists(atPath: extra.path) {
      for (k, v) in try loadArrays(url: extra) { arrays[k] = v }
    }
  }
  func get(_ name: String) throws -> MLXArray {
    guard let a = arrays[name] else { throw Drama3Error.invalid("Checkpoint is missing \(name).") }
    return a
  }
  func linear(_ name: String, _ x: MLXArray, indices: MLXArray? = nil) throws -> MLXArray {
    let w = try get(name + ".weight")
    let out: MLXArray
    if let scales = arrays[name + ".scales"] {
      let q = quant[name] as? [String: Any] ?? quant
      let bits = q["bits"] as? Int ?? 8
      let group = q["group_size"] as? Int ?? 64
      if let indices {
        out = gatherQuantizedMM(
          x, w, scales: scales, biases: arrays[name + ".biases"], rhsIndices: indices,
          transpose: true, groupSize: group, bits: bits)
      } else {
        out = quantizedMM(
          x, w, scales: scales, biases: arrays[name + ".biases"], transpose: true, groupSize: group,
          bits: bits)
      }
    } else if let indices {
      out = gatherMM(x, w.transposed(0, 2, 1), rhsIndices: indices)
    } else {
      out = matmul(x, w.transposed(-1, -2))
    }
    return arrays[name + ".bias"].map { out + $0 } ?? out
  }
  func norm(_ name: String, _ x: MLXArray) throws -> MLXArray {
    Drama3Math.norm(x, weight: try get(name + ".weight"))
  }
  func mlp(_ name: String, _ x: MLXArray) throws -> MLXArray {
    try linear(
      name + ".down_proj", silu(try linear(name + ".gate_proj", x)) * linear(name + ".up_proj", x))
  }
}
