import Foundation
import MLX

struct MOSSQuantizationManifest: Codable {
  var format = "weetodd-moss-affine-v1"
  var bits: Int
  var groupSize = 64
  var component: String
  var logicalShapes: [String: [Int]]
  var quantizedNames: [String]
  var sourceIdentity: String
  var sourceSHA256: [String: String]
  var parameterHash: String
  static func read(_ directory: URL) throws -> Self? {
    let file = directory.appendingPathComponent("moss_quantization.json")
    guard FileManager.default.fileExists(atPath: file.path) else { return nil }
    let data = try MOSSImagePreflight.bounded(file, maximum: 2 * 1024 * 1024)
    let value = try JSONDecoder().decode(Self.self, from: data)
    guard value.format == "weetodd-moss-affine-v1", [4, 8].contains(value.bits), value.groupSize == 64,
      ["transformer", "text_encoder"].contains(value.component),
      Set(value.quantizedNames).count == value.quantizedNames.count,
      !value.quantizedNames.isEmpty, value.parameterHash == MOSSQuantization.parameterHash(bits: value.bits, component: value.component, shapes: value.logicalShapes),
      value.quantizedNames.allSatisfy({ name in
        guard let shape = value.logicalShapes[name] else { return false }
        return MOSSQuantization.shouldQuantize(name, shape: shape, component: value.component)
      }) else { throw MOSSError("Invalid or incompatible native MOSS quantization manifest.") }
    return value
  }
}

public enum MOSSQuantization {
  static func shouldQuantize(_ name: String, shape: [Int], component: String) -> Bool {
    guard name.hasSuffix(".weight"), shape.count == 2, shape[1] % 64 == 0 else { return false }
    if component == "transformer" {
      return name.hasPrefix("layers.") && name.contains(".proj.") || name.hasPrefix("text_embedder.blocks.") || name == "text_embedder.proj.weight"
    }
    return name.hasPrefix("language_model.layers.") && !name.contains(".norm")
      && !name.contains("in_proj_a.") && !name.contains("in_proj_b.")
  }
  static func parameterHash(bits: Int, component: String, shapes: [String: [Int]]) -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return MOSSImagePreflight.digest(Data("affine-v1|\(bits)|64|\(component)|".utf8) + (try! encoder.encode(shapes)))
  }
  public static func convert(input: URL, output: URL, bits: Int, component: String,
    progress: (String, Double) -> Void = { _, _ in }) throws {
    let config = try MOSSImagePreflight.bounded(input.appendingPathComponent("config.json"), maximum: 1024 * 1024)
    let shapes: [String: [Int]]
    if component == "transformer" { try MOSSModelContract.validateTransformerConfig(config); shapes = MOSSModelContract.transformerShapes() }
    else if component == "text_encoder" { try MOSSModelContract.validateQwenConfig(config); shapes = MOSSModelContract.qwenShapes() }
    else { throw MOSSError("Choose transformer or text_encoder for native conversion.") }
    try convert(input: input, output: output, bits: bits, component: component, shapes: shapes, progress: progress)
  }
  static func convert(input: URL, output: URL, bits: Int, component: String, shapes: [String: [Int]],
    progress: (String, Double) -> Void = { _, _ in }) throws {
    let fm = FileManager.default
    guard [4, 8].contains(bits), ["transformer", "text_encoder"].contains(component),
      output.path.hasPrefix("/"), !fm.fileExists(atPath: output.path) else { throw MOSSError("Choose a new model-library component folder and affine Q8 or Q4.") }
    let index = try MOSSCheckpointIndex(directory: input)
    guard index.quantization == nil else { throw MOSSError("Convert from the original official floating-point weights.") }
    for (name, shape) in shapes { try index.require(name, shape: shape) }
    try fm.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    let staging = output.deletingLastPathComponent().appendingPathComponent(".moss-convert-" + UUID().uuidString)
    try fm.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? fm.removeItem(at: staging) }
    let source = MOSSWeightSource(index: index, cacheEnabled: false)
    defer { source.release() }
    var arrays: [String: MLXArray] = [:], bytes = 0, shard = 0, names: [String] = []
    func flush() throws {
      guard !arrays.isEmpty else { return }
      try MLX.save(arrays: arrays, url: staging.appendingPathComponent(String(format: "moss-%03d.safetensors", shard)))
      arrays.removeAll(); bytes = 0; shard += 1; Memory.clearCache()
    }
    for (i, name) in shapes.keys.sorted().enumerated() {
      try Task.checkCancellation()
      let value = try source.tensor(name)
      if shouldQuantize(name, shape: shapes[name]!, component: component) {
        let (q, scales, biases) = quantized(value.asType(.float32), groupSize: 64, bits: bits)
        eval(q, scales, biases!)
        arrays[name] = q; arrays[name + ".scales"] = scales; arrays[name + ".biases"] = biases!
        bytes += q.nbytes + scales.nbytes + biases!.nbytes; names.append(name)
      } else { arrays[name] = value; bytes += value.nbytes }
      if bytes >= 256 * 1024 * 1024 { try flush() }
      progress("Converting \(component) tensor \(i + 1)/\(shapes.count)", 0.9 * Double(i + 1) / Double(shapes.count))
    }
    try flush(); try index.checkUnchanged()
    var hashes: [String: String] = [:]
    for file in index.files.keys.sorted(by: { $0.path < $1.path }) {
      progress("Recording source SHA-256 · \(file.lastPathComponent)", 0.92)
      hashes[file.lastPathComponent] = try MOSSLoRA.sha256(file)
    }
    let manifest = MOSSQuantizationManifest(bits: bits, component: component, logicalShapes: shapes,
      quantizedNames: names.sorted(), sourceIdentity: try index.identity(), sourceSHA256: hashes,
      parameterHash: parameterHash(bits: bits, component: component, shapes: shapes))
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
    try encoder.encode(manifest).write(to: staging.appendingPathComponent("moss_quantization.json"))
    try fm.copyItem(at: input.appendingPathComponent("config.json"), to: staging.appendingPathComponent("config.json"))
    if let license = [input.appendingPathComponent("LICENSE"), input.deletingLastPathComponent().appendingPathComponent("LICENSE")].first(where: { fm.fileExists(atPath: $0.path) }) {
      try fm.copyItem(at: license, to: staging.appendingPathComponent("LICENSE"))
    }
    let converted = try MOSSCheckpointIndex(directory: staging)
    try MOSSImagePreflight.validate(converted, shapes: shapes, selected: { _ in true })
    try fm.moveItem(at: staging, to: output); progress("Native affine Q\(bits) component ready", 1)
  }
}
