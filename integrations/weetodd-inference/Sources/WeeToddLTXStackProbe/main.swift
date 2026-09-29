import Darwin
import Foundation
import LTX25NNC
import TensorIO

/// Separate-process qualification of the GPU-resident stack. No model generation
/// or Studio capability is advertised by this probe.
@main
struct StackProbe {
  static func main() {
    do { try run() }
    catch {
      try? FileHandle.standardError.write(contentsOf: Data("\(error)\n".utf8))
      exit(2)
    }
  }

  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 6, let repeats = Int(args[4]), (1...10).contains(repeats) else {
      throw BlockError.invalid("Usage: WeeToddLTXStackProbe CONFIG INPUTS.safetensors EXPECTED.safetensors PAGED_ROOT REPEATS REPORT.json")
    }
    let configURL = URL(fileURLWithPath: args[0])
    guard (try configURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw BlockError.invalid("Configuration is too large.")
    }
    let config = try JSONDecoder().decode(AVBlockConfiguration.self, from: Data(contentsOf: configURL))
    try AVBlockRunner.validateAllocation(configuration: config)
    let weights = try PagedBlockWeights(root: URL(fileURLWithPath: args[3]), configuration: config)
    let inputFile = try SafeTensorFile(url: URL(fileURLWithPath: args[1]))
    let reference = try SafeTensorFile(url: URL(fileURLWithPath: args[2]))
    for file in [inputFile, reference] {
      guard file.tensors.values.allSatisfy({ $0.dtype == "F32" }),
        file.tensors.values.reduce(UInt64(0), { $0 + $1.byteCount }) <= 128 * 1024 * 1024 else {
        throw BlockError.invalid("Fixtures require at most 128 MiB of Float32 data each.")
      }
    }
    let shapes = try AVBlockRunner.expectedInputShapes(configuration: config)
    guard Set(shapes.keys) == Set(inputFile.tensors.keys) else { throw BlockError.invalid("Invalid input fixture keys.") }
    var inputs: [String: [Float]] = [:], expected: [String: [Float]] = [:]
    for (name, shape) in shapes {
      guard inputFile.tensors[name]?.shape == shape.map(UInt64.init) else { throw BlockError.invalid("Invalid input shape: \(name)") }
      inputs[name] = try inputFile.readFloat32(named: name)
    }
    for name in ["video", "audio"] {
      guard reference.tensors[name]?.shape == shapes[name]!.map(UInt64.init) else { throw BlockError.invalid("Invalid expected shape: \(name)") }
      expected[name] = try reference.readFloat32(named: name)
      guard expected[name]!.allSatisfy(\.isFinite) else { throw BlockError.invalid("Expected output is nonfinite.") }
    }
    let runner = try AVStackRunner(configuration: config, blockCount: weights.blockCount)
    let initialMetal = runner.metalAllocatedBytes
    var allPassed = true, runs: [[String: Any]] = [], first: [String: [Float]] = [:]
    for run in 0..<repeats {
      var samples: [[String: Any]] = []
      let started = Date()
      let output = try runner.evaluate(inputs, retainWeights: true, weights: { try weights.read(block: $0, name: $1, shape: $2) }) { event in
        let sample: [String: Any] = ["event": "block_progress", "run": run, "completed": event.completedBlocks,
          "total": event.totalBlocks, "resident_blocks": event.residentBlocks, "metal_bytes": event.metalAllocatedBytes,
          "runtime_bytes": event.runtimeBytes, "graph_variables": event.graphVariables,
          "weight_load_seconds": event.weightLoadSeconds, "compute_seconds": event.computeSeconds,
          "metrics": try JSONSerialization.jsonObject(with: JSONEncoder().encode(event.metrics))]
        samples.append(sample)
        try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: sample, options: [.sortedKeys]) + Data([10]))
      }
      var outputs: [String: Any] = [:]
      for (name, actual) in [("video", output.video), ("audio", output.audio)] {
        let reference = expected[name]!
        let maxError = zip(actual, reference).map { abs($0 - $1) }.max() ?? 0
        let sumError = zip(actual, reference).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
        let norm = reference.reduce(0.0) { $0 + Double($1) * Double($1) }
        let relative = sqrt(sumError / max(norm, 1e-30))
        let repeatError = zip(actual, first[name] ?? actual).map { abs($0 - $1) }.max() ?? 0
        let passed = actual.count == reference.count && maxError <= 0.05 && relative <= 0.0001 && repeatError <= 0.000001
        allPassed = allPassed && passed
        outputs[name] = ["max_absolute_error": maxError, "relative_l2_error": relative,
          "repeat_max_absolute_error": repeatError, "passed": passed]
        if run == 0 { first[name] = actual }
      }
      runs.append(["seconds": Date().timeIntervalSince(started), "samples": samples, "outputs": outputs,
        "activation_uploads": runner.lastTransferCounts.activationUploads,
        "activation_downloads": runner.lastTransferCounts.activationDownloads,
        "retained_metal_bytes": runner.metalAllocatedBytes])
    }
    try runner.release()
    let report: [String: Any] = ["scope": "48-ltx25-transformer-blocks-float32-small-token-fixture",
      "passed": allPassed, "runs": runs, "initial_metal_bytes": initialMetal,
      "released_metal_bytes": runner.metalAllocatedBytes, "resident_blocks_after_release": runner.residentBlocks,
      "decoded_slot_bytes": weights.decodedSlotBytes, "largest_decoded_matrix_bytes": weights.largestDecodedMatrixBytes]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: args[5]), options: .atomic)
    if !allPassed { throw BlockError.invalid("Stack qualification failed; see report.") }
  }
}
