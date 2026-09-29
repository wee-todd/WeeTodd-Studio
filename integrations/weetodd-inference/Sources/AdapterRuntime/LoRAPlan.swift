import Foundation
import TensorIO

/// A complete, validated dense adapter mapping. This inspects headers and tiny alpha
/// scalars only; matrices retain their source names and are loaded by the active stage.
/// Structural compatibility does not qualify an adapter's task or sampling schedule.
public struct LoRAPlan: Sendable {
  public struct Pair: Sendable {
    public let target: String
    public let downTensor: String
    public let upTensor: String
    public let rank: UInt64
    public let shape: [UInt64]
    public let scale: Float
  }
  public let pairs: [Pair]
  public let strength: Float

  public init(file: SafeTensorFile, strength: Float, targetShapes: [String: [UInt64]],
    normalize: (String) -> String? = { $0 }) throws {
    guard strength.isFinite else { throw AdapterError.invalid("LoRA strength must be finite.") }
    let schemas = [
      (".lora_A.weight", ".lora_B.weight"),
      (".lora_A.default.weight", ".lora_B.default.weight"),
      (".lora_A.turbo.weight", ".lora_B.turbo.weight"),
      (".lora_down.weight", ".lora_up.weight"),
      (".lora_a.weight", ".lora_b.weight"),
    ]
    let alphaSuffixes = [".alpha", ".lora_alpha", ".alpha.weight"]
    let floatTypes: Set<String> = ["F16", "BF16", "F32", "F64"]
    struct Group { var schema: Int; var down: String?; var up: String? }
    var groups: [String: Group] = [:]
    var alphas: [String: String] = [:]
    for name in file.tensors.keys.sorted() {
      var matched = false
      for (index, suffixes) in schemas.enumerated() {
        let isDown = name.hasSuffix(suffixes.0)
        let isUp = name.hasSuffix(suffixes.1)
        guard isDown || isUp else { continue }
        let target = String(name.dropLast((isDown ? suffixes.0 : suffixes.1).count))
        var group = groups[target] ?? Group(schema: index)
        guard !target.isEmpty, group.schema == index,
              (isDown ? group.down : group.up) == nil,
              floatTypes.contains(file.tensors[name]!.dtype) else {
          throw AdapterError.invalid("Ambiguous LoRA pair or unsupported dtype: \(name)")
        }
        if isDown { group.down = name } else { group.up = name }
        groups[target] = group; matched = true; break
      }
      if matched { continue }
      guard let suffix = alphaSuffixes.first(where: name.hasSuffix) else {
        throw AdapterError.invalid("Unsupported LoRA tensor; refusing partial application: \(name)")
      }
      let target = String(name.dropLast(suffix.count))
      guard alphas[target] == nil else { throw AdapterError.invalid("Multiple LoRA alpha tensors: \(target)") }
      alphas[target] = name
    }
    guard !groups.isEmpty, Set(alphas.keys).isSubset(of: Set(groups.keys)) else {
      throw AdapterError.invalid("LoRA has no supported pairs or contains orphan alpha tensors.")
    }
    let globalRank = try Self.scaling(file.metadata,
      aliases: ["lora_rank", "ss_network_dim", "network_dim"], minimum: 1)
    let globalAlpha = try Self.scaling(file.metadata,
      aliases: ["lora_alpha", "ss_network_alpha", "network_alpha"], minimum: 0)
    var targets: Set<String> = []
    var result: [Pair] = []
    for source in groups.keys.sorted() {
      let group = groups[source]!
      guard let down = group.down, let up = group.up else {
        throw AdapterError.invalid("Incomplete LoRA pair: \(source)")
      }
      let a = file.tensors[down]!.shape, b = file.tensors[up]!.shape
      guard a.count == 2, b.count == 2, a.allSatisfy({ $0 > 0 }), b.allSatisfy({ $0 > 0 }),
            a[0] == b[1] else { throw AdapterError.invalid("Invalid LoRA rank or shape: \(source)") }
      guard let target = normalize(source), targets.insert(target).inserted,
            targetShapes[target] == [b[0], a[1]] else {
        throw AdapterError.invalid("LoRA target is missing, duplicated or has an incompatible destination shape: \(source)")
      }
      var multiplier = 1.0
      if let alpha = alphas[source] {
        multiplier = try Self.scalar(file, named: alpha) / Double(a[0])
      } else if let globalAlpha {
        multiplier = globalAlpha / (globalRank ?? Double(a[0]))
      }
      let scale = Float(multiplier * Double(strength))
      guard multiplier.isFinite, multiplier >= 0, scale.isFinite else {
        throw AdapterError.invalid("Invalid LoRA scaling: \(source)")
      }
      result.append(Pair(target: target, downTensor: down, upTensor: up,
        rank: a[0], shape: [b[0], a[1]], scale: scale))
    }
    self.pairs = result; self.strength = strength
  }

  private static func scaling(_ metadata: [String: String], aliases: [String], minimum: Double) throws -> Double? {
    let unspecified: Set<String> = ["", "none", "null", "dynamic", "baked", "baked_scale"]
    var found: Double?
    for alias in aliases {
      guard let raw = metadata[alias] else { continue }
      let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      if unspecified.contains(value) { continue }
      guard let number = Double(value), number.isFinite, number >= minimum,
            found == nil || found == number else {
        throw AdapterError.invalid("Invalid or conflicting LoRA scaling metadata: \(alias)")
      }
      found = number
    }
    return found
  }

  private static func scalar(_ file: SafeTensorFile, named name: String) throws -> Double {
    guard let tensor = file.tensors[name], tensor.shape.allSatisfy({ $0 == 1 }) else {
      throw AdapterError.invalid("LoRA alpha must be a scalar: \(name)")
    }
    return try file.withTensorBytes(named: name) { bytes in
      switch tensor.dtype {
      case "F32": return Double(Float(bitPattern: bytes.loadUnaligned(as: UInt32.self).littleEndian))
      case "F64": return Double(bitPattern: bytes.loadUnaligned(as: UInt64.self).littleEndian)
      case "F16": return Double(Float16(bitPattern: bytes.loadUnaligned(as: UInt16.self).littleEndian))
      case "BF16": return Double(Float(bitPattern: UInt32(bytes.loadUnaligned(as: UInt16.self).littleEndian) << 16))
      default: throw AdapterError.invalid("LoRA alpha must have a floating-point dtype: \(name)")
      }
    }
  }
}
