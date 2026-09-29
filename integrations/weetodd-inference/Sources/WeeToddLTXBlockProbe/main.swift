import Darwin
import Foundation
import LTX25NNC
import TensorIO

/// Developer qualification only, not a generation provider. Tensor fixtures use
/// bounded binary safetensors; the JSON report contains scalar metrics only.
@main
struct BlockProbe {
  static func main() {
    do { try run() }
    catch {
      try? FileHandle.standardError.write(contentsOf: Data("\(error)\n".utf8))
      exit(2)
    }
  }

  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard (args.count == 6 || args.count == 7), let index = Int(args[4]),
      let precision = LTXPrecisionPolicy(rawValue: args.count == 7 ? args[6] : "float32") else {
      throw BlockError.invalid("Usage: WeeToddLTXBlockProbe CONFIG INPUTS.safetensors EXPECTED.safetensors CHECKPOINT BLOCK_INDEX REPORT.json [float32|fp16-projections|bf16-projections]")
    }
    let configURL = URL(fileURLWithPath: args[0])
    let configBytes = try configURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max
    guard configBytes <= 16384 else { throw BlockError.invalid("Configuration is too large.") }
    let config = try JSONDecoder().decode(AVBlockConfiguration.self, from: Data(contentsOf: configURL))
    try AVBlockRunner.validateAllocation(configuration: config)
    let shapes = try AVBlockRunner.expectedWeightShapes(configuration: config)
    let source = try BlockWeightSource(url: URL(fileURLWithPath: args[3]), blockIndex: index, expectedShapes: shapes)
    let inputFile = try SafeTensorFile(url: URL(fileURLWithPath: args[1]))
    let expectedFile = try SafeTensorFile(url: URL(fileURLWithPath: args[2]))
    for file in [inputFile, expectedFile] {
      guard file.tensors.values.allSatisfy({ $0.dtype == "F32" }),
            file.tensors.values.reduce(UInt64(0), { $0 + $1.byteCount }) <= 64 * 1024 * 1024 else {
        throw BlockError.invalid("Qualification fixtures require at most 64 MiB of Float32 tensors each.")
      }
    }
    let inputShapes = try AVBlockRunner.expectedInputShapes(configuration: config)
    guard Set(inputFile.tensors.keys) == Set(inputShapes.keys) else {
      throw BlockError.invalid("Input fixture keys do not match the supported block inputs.")
    }
    var inputs: [String: [Float]] = [:]
    for (name, shape) in inputShapes {
      guard inputFile.tensors[name]?.shape == shape.map(UInt64.init) else {
        throw BlockError.invalid("Input fixture shape differs: \(name)")
      }
      inputs[name] = try inputFile.readFloat32(named: name)
    }
    let start = Date()
    let runner = try AVBlockRunner(configuration: config, precision: precision)
    var maxWeightError: Float = 0
    try runner.load { name, shape in
      let values = try source.read(name, shape: shape)
      let reference = try expectedFile.readFloat32(named: "weight_sample." + name)
      guard reference.count == min(32, values.count), reference.allSatisfy(\.isFinite) else {
        throw BlockError.invalid("Missing or invalid weight sample: \(name)")
      }
      maxWeightError = max(maxWeightError, zip(values, reference).map { abs($0 - $1) }.max() ?? 0)
      return values
    }
    let loaded = Date()
    let output = try runner.evaluate(inputs)
    let cold = Date()
    let repeated = try runner.evaluate(inputs)
    let warm = Date()
    var metrics: [String: Any] = [:]
    var passed = maxWeightError <= 0.0000001
    for (name, actual, again) in [("video", output.video, repeated.video), ("audio", output.audio, repeated.audio)] {
      let expected = try expectedFile.readFloat32(named: name)
      guard expected.count == actual.count, again.count == actual.count,
            expected.allSatisfy(\.isFinite), actual.allSatisfy(\.isFinite), again.allSatisfy(\.isFinite) else {
        throw BlockError.invalid("Output shape or finite-value check failed: \(name)")
      }
      let maxError = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
      let squaredError = zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
      let squaredNorm = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
      let relativeL2 = sqrt(squaredError / max(squaredNorm, 1e-30))
      let repeatError = zip(actual, again).map { abs($0 - $1) }.max() ?? 0
      let strict = maxError <= 0.002 && relativeL2 <= 0.0001
      // Frozen exploratory component limits, separate from the Float32 gates.
      // Passing these does not qualify a trajectory or production media.
      let absoluteLimit = precision == .float32 ? 0.002 : 0.25
      let relativeLimit = precision == .float32 ? 0.0001 : 0.002
      passed = passed && Double(maxError) <= absoluteLimit && relativeL2 <= relativeLimit && repeatError <= 0.000001
      metrics[name] = ["max_absolute_error": maxError, "relative_l2_error": relativeL2,
        "repeat_max_absolute_error": repeatError, "elements": actual.count,
        "float32_gate_passed": strict, "absolute_limit": absoluteLimit, "relative_limit": relativeLimit]
    }
    let report: [String: Any] = ["scope": "one-ltx25-joint-block-precision-qualification", "precision": precision.rawValue, "passed": passed,
      "decoded_weight_bytes": source.decodedWeightBytes, "largest_decode_bytes": source.largestDecodedTensorBytes,
      "nnc_runtime_memory_bytes": runner.scratchBytes, "weight_sample_max_absolute_error": maxWeightError,
      "compile_and_load_seconds": loaded.timeIntervalSince(start), "cold_forward_seconds": cold.timeIntervalSince(loaded),
      "warm_forward_seconds": warm.timeIntervalSince(cold), "outputs": metrics]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: args[5]), options: .atomic)
    try FileHandle.standardOutput.write(contentsOf: data + Data([10]))
    if !passed { throw BlockError.invalid("Block numerical qualification failed; see report.") }
  }
}
