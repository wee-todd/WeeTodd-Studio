import Darwin
import Foundation
import LTX25NNC
import Metal
import TensorIO

/// Packed-latent to velocity qualification; does not advertise Studio generation.
@main
struct DenoiserProbe {
  static func main() {
    do { try run() }
    catch {
      try? FileHandle.standardError.write(contentsOf: Data("\(error)\n".utf8))
      exit(2)
    }
  }
  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 7, let sigma = Float(args[4]), sigma.isFinite, (0...1).contains(sigma),
      let repeats = Int(args[5]), (1...10).contains(repeats) else {
      throw BlockError.invalid("Usage: WeeToddLTXDenoiserProbe CONFIG INPUTS.safetensors EXPECTED.safetensors PAGED_ROOT SIGMA REPEATS REPORT.json")
    }
    let configURL = URL(fileURLWithPath: args[0])
    guard (try configURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw BlockError.invalid("Configuration is too large.")
    }
    let config = try JSONDecoder().decode(AVBlockConfiguration.self, from: Data(contentsOf: configURL))
    let runner = try DenoiserRunner(configuration: config)
    let weights = try DenoiserWeights(root: URL(fileURLWithPath: args[3]), configuration: config)
    let inputFile = try SafeTensorFile(url: URL(fileURLWithPath: args[1]))
    let reference = try SafeTensorFile(url: URL(fileURLWithPath: args[2]))
    for file in [inputFile, reference] {
      guard file.tensors.values.allSatisfy({ $0.dtype == "F32" }),
        file.tensors.values.reduce(UInt64(0), { $0 + $1.byteCount }) <= 128 * 1024 * 1024 else {
        throw BlockError.invalid("Fixtures require at most 128 MiB of Float32 data each.")
      }
    }
    let shapes = runner.inputShapes
    guard Set(shapes.keys) == Set(inputFile.tensors.keys), Set(reference.tensors.keys) == ["video", "audio"] else {
      throw BlockError.invalid("Invalid fixture keys.")
    }
    var inputs: [String: [Float]] = [:], expected: [String: [Float]] = [:]
    for (name, shape) in shapes {
      guard inputFile.tensors[name]?.shape == shape.map(UInt64.init) else { throw BlockError.invalid("Invalid input shape: \(name)") }
      inputs[name] = try inputFile.readFloat32(named: name)
    }
    for (name, count) in [("video", config.videoTokens), ("audio", config.audioTokens)] {
      guard reference.tensors[name]?.shape == [UInt64(count), 128] else { throw BlockError.invalid("Invalid reference shape.") }
      expected[name] = try reference.readFloat32(named: name)
      guard expected[name]!.allSatisfy(\.isFinite) else { throw BlockError.invalid("Nonfinite reference.") }
    }
    let initialMetal = MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0
    var runs: [[String: Any]] = [], first: [String: [Float]] = [:], allPassed = true
    for run in 0..<repeats {
      var samples: [[String: Any]] = []
      let started = Date()
      let output = try runner.evaluate(inputs, sigma: sigma, fixedWeights: { try weights.readFixed($0, shape: $1) },
        blockWeights: { try weights.blocks.read(block: $0, name: $1, shape: $2) }) { event in
        let sample: [String: Any] = ["run": run, "stage": event.stage, "completed_blocks": event.completedBlocks,
          "metal_bytes": event.metalAllocatedBytes, "elapsed_seconds": Date().timeIntervalSince(started)]
        samples.append(sample)
        try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: sample, options: [.sortedKeys]) + Data([10]))
      }
      var outputs: [String: Any] = [:]
      for (name, actual) in [("video", output.videoVelocity), ("audio", output.audioVelocity)] {
        let reference = expected[name]!
        let maxError = zip(actual, reference).map { abs($0 - $1) }.max() ?? 0
        let sumError = zip(actual, reference).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
        let norm = reference.reduce(0.0) { $0 + Double($1) * Double($1) }
        let relative = sqrt(sumError / max(norm, 1e-30))
        let repeatError = zip(actual, first[name] ?? actual).map { abs($0 - $1) }.max() ?? 0
        // Full-model small-token tolerance is calibrated against the unchanged
        // MLX CPU/GPU reference spread (video relative 0.001181, absolute 0.000959).
        // Component/isolated-block tests retain their stricter 1e-4 bound.
        let strictPassed = maxError <= 0.05 && relative <= 0.0001
        let tolerance = name == "video" ? 0.002 : 0.0001
        let passed = actual.count == reference.count && Double(maxError) <= tolerance && relative <= tolerance && repeatError <= 0.000001
        allPassed = allPassed && passed
        outputs[name] = ["max_absolute_error": maxError, "relative_l2_error": relative,
          "repeat_max_absolute_error": repeatError, "strict_component_tolerance_passed": strictPassed, "passed": passed]
        if run == 0 { first[name] = actual }
      }
      runs.append(["seconds": Date().timeIntervalSince(started), "samples": samples, "outputs": outputs,
        "released_metal_bytes": MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0])
    }
    let report: [String: Any] = ["scope": "ltx25-packed-latent-to-velocity-float32-small-token-fixture",
      "passed": allPassed, "sigma": sigma, "runs": runs, "initial_metal_bytes": initialMetal,
      "qualification": "component fixture only; calibrated against MLX CPU/GPU variation; no media-quality claim",
      "max_absolute_tolerance": ["video": 0.002, "audio": 0.0001],
      "relative_l2_tolerance": ["video": 0.002, "audio": 0.0001], "repeat_absolute_tolerance": 0.000001]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: args[6]), options: .atomic)
    if !allPassed { throw BlockError.invalid("Denoiser qualification failed; see report.") }
  }
}
