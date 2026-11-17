import Foundation
import MLX
import TensorIO

protocol MOSSWeightAccess: AnyObject {
  func contains(_ name: String) -> Bool
  func tensor(_ name: String) throws -> MLXArray
  func linear(_ x: MLXArray, _ name: String) throws -> MLXArray
  var profiler: MOSSProfiler? { get }
}
extension MOSSWeightAccess {
  var profiler: MOSSProfiler? { nil }
  func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
    let w = try tensor(name + ".weight")
    var output = matmul(x.asType(w.dtype), w.transposed())
    if contains(name + ".bias") { output = output + (try tensor(name + ".bias")) }
    return output
  }
}

final class MOSSWeightSource: MOSSWeightAccess {
  let index: MOSSCheckpointIndex
  private var cache: [String: MLXArray] = [:]
  let retainedBlocks: Int?
  let cacheEnabled: Bool
  let profiler: MOSSProfiler?
  let role: String
  private(set) var loads = 0, hits = 0
  init(index: MOSSCheckpointIndex, retainedBlocks: Int? = nil, cacheEnabled: Bool = true,
    profiler: MOSSProfiler? = nil, role: String = "weights") {
    self.index = index; self.retainedBlocks = retainedBlocks; self.cacheEnabled = cacheEnabled
    self.profiler = profiler; self.role = role
  }
  func contains(_ name: String) -> Bool { index.descriptors[name] != nil }
  func tensor(_ name: String) throws -> MLXArray {
    try Task.checkCancellation(); try index.checkUnchanged()
    if let value = cache[name] { hits += 1; profiler?.record(role + "/cache hit", seconds: 0); return value }
    guard let descriptor = index.descriptors[name], let url = index.owners[name], let file = index.files[url],
      let raw = index.rawNames[name], let dtype = ["BF16": DType.bfloat16, "F32": .float32, "F16": .float16, "U32": .uint32][descriptor.dtype] else {
      throw MOSSError("Unsupported native MOSS tensor: \(name)")
    }
    let started = profiler == nil ? nil : Date()
    let value = try file.withTensorBytes(named: raw, range: 0..<descriptor.byteCount,
      access: descriptor.byteCount <= 4 * 1024 * 1024 ? .buffered : .mapped) { bytes in
        MLXArray(Data(bytes), descriptor.shape.map(Int.init), dtype: dtype)
      }
    eval(value); loads += 1
    if let started, let profiler {
      Stream.defaultStream(.gpu).synchronize()
      profiler.record(role + "/tensor read + upload", seconds: Date().timeIntervalSince(started), bytes: Int(descriptor.byteCount))
    }
    let block = name.hasPrefix("layers.") ? Int(name.split(separator: ".")[1]) : nil
    if cacheEnabled && (retainedBlocks == nil || block == nil || block! < retainedBlocks!) { cache[name] = value }
    return value
  }
  func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
    let key = name + ".weight"
    var result: MLXArray
    if let quantization = index.quantization, quantization.quantizedNames.contains(key) {
      let scales = try tensor(key + ".scales")
      result = try quantizedMM(x.asType(scales.dtype), tensor(key), scales: scales,
        biases: tensor(key + ".biases"), transpose: true, groupSize: 64, bits: quantization.bits)
    } else {
      let weight = try tensor(key); result = matmul(x.asType(weight.dtype), weight.transposed())
    }
    if contains(name + ".bias") { result = result + (try tensor(name + ".bias")) }
    return profiler?.evaluate(role + "/linear projections", result) ?? result
  }
  func release() { Stream.defaultStream(.gpu).synchronize(); cache.removeAll(); Memory.clearCache() }
}
