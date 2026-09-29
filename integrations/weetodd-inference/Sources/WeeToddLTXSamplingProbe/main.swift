import CryptoKit
import Darwin
import Foundation
import LTX25Engine
import LTX25NNC
import Metal
import TensorIO

/// Numerical trajectories only; no Studio/media-quality claim.
@main struct SamplingProbe {
  struct Recipe: Codable, Equatable {
    let sigmas: [Double]
    let eta: Double
  }
  static func main() {
    do { try run() } catch {
      try? FileHandle.standardError.write(contentsOf: Data("\(error)\n".utf8))
      exit(2)
    }
  }
  static func json<T: Decodable>(_ path: String, _ type: T.Type) throws -> T {
    let url = URL(fileURLWithPath: path)
    guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw BlockError.invalid("Oversized configuration.")
    }
    return try JSONDecoder().decode(type, from: Data(contentsOf: url))
  }
  static func fixture(_ path: String) throws -> SafeTensorFile {
    let f = try SafeTensorFile(url: URL(fileURLWithPath: path))
    guard f.tensors.values.allSatisfy({ $0.dtype == "F32" }),
      f.tensors.values.reduce(UInt64(0), { $0 + $1.byteCount }) <= 128 * 1024 * 1024
    else { throw BlockError.invalid("Fixture exceeds Float32 memory bounds.") }
    return f
  }
  static func error(_ a: [Float], _ b: [Float]) -> (absolute: Double, relative: Double) {
    var maximum = 0.0
    var sum = 0.0
    var norm = 0.0
    for (x, y) in zip(a, b) {
      let d = Double(x) - Double(y)
      maximum = max(maximum, abs(d))
      sum += d * d
      norm += Double(y) * Double(y)
    }
    return (maximum, sqrt(sum / max(norm, 1e-30)))
  }
  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard (5...7).contains(args.count), (args.count == 5 || ["staged", "session"].contains(args[5])),
      let precision = LTXPrecisionPolicy(rawValue: args.count == 7 ? args[6] : "float32"),
      let repeats = Int(args[3]), (2...10).contains(repeats) else {
      throw BlockError.invalid(
        "Usage: WeeToddLTXSamplingProbe GPU_FIXTURE_PREFIX CPU_FIXTURE_PREFIX PAGED_ROOT REPEATS(2...10) REPORT.json [staged|session] [float32|fp16-projections|bf16-projections]"
      )
    }
    let config = try json(args[0] + ".config.json", AVBlockConfiguration.self)
    let recipe = try json(args[0] + ".schedule.json", Recipe.self)
    guard try recipe == json(args[1] + ".schedule.json", Recipe.self) else {
      throw BlockError.invalid("Reference schedules differ.")
    }
    let schedule = try SamplingSchedule(sigmas: recipe.sigmas, eta: recipe.eta)
    let shapes = try DenoiserRunner(configuration: config).inputShapes
    let inputFile = try fixture(args[0] + ".inputs.safetensors")
    let cpuInputs = try fixture(args[1] + ".inputs.safetensors")
    let gpuSteps = try fixture(args[0] + ".steps.safetensors")
    let cpuSteps = try fixture(args[1] + ".steps.safetensors")
    let noiseFile = try fixture(args[0] + ".noise.safetensors")
    let cpuNoise = try fixture(args[1] + ".noise.safetensors")
    guard Set(inputFile.tensors.keys) == Set(shapes.keys),
      Set(cpuInputs.tensors.keys) == Set(shapes.keys),
      Set(noiseFile.tensors.keys) == Set(cpuNoise.tensors.keys)
    else { throw BlockError.invalid("Fixture keys differ.") }
    var inputs: [String: [Float]] = [:]
    for (name, shape) in shapes {
      guard inputFile.tensors[name]?.shape == shape.map(UInt64.init),
        cpuInputs.tensors[name]?.shape == shape.map(UInt64.init)
      else { throw BlockError.invalid("Fixture input shape mismatch.") }
      let values = try inputFile.readFloat32(named: name)
      guard values.allSatisfy(\.isFinite), try values == cpuInputs.readFloat32(named: name) else {
        throw BlockError.invalid("Reference inputs differ.")
      }
      inputs[name] = values
    }
    for name in noiseFile.tensors.keys {
      guard try noiseFile.readFloat32(named: name) == cpuNoise.readFloat32(named: name) else {
        throw BlockError.invalid("Reference noise differs.")
      }
    }
    var expectedKeys = Set<String>()
    for step in schedule.steps.indices {
      for (name, count) in [("video", config.videoTokens), ("audio", config.audioTokens)] {
        let key = "\(step).\(name)"
        expectedKeys.insert(key)
        for file in [gpuSteps, cpuSteps] {
          guard file.tensors[key]?.shape == [UInt64(count), 128],
            try file.readFloat32(named: key).allSatisfy(\.isFinite)
          else { throw BlockError.invalid("Invalid reference trajectory.") }
        }
        if schedule.steps[step].ancestral {
          guard noiseFile.tensors[key]?.shape == [UInt64(count), 128],
            try noiseFile.readFloat32(named: key).allSatisfy(\.isFinite)
          else { throw BlockError.invalid("Invalid noise trajectory.") }
        }
      }
    }
    guard Set(gpuSteps.tensors.keys) == expectedKeys, Set(cpuSteps.tensors.keys) == expectedKeys
    else { throw BlockError.invalid("Reference trajectory keys differ.") }
    let weights = try DenoiserWeights(root: URL(fileURLWithPath: args[2]), configuration: config)
    let runner = try LTXSamplingRunner(configuration: config, experimentalPrecision: precision)
    var first: [String: [Float]] = [:]
    var runs: [[String: Any]] = []
    var passed = true
    for run in 0..<repeats {
      let start = Date()
      var metrics: [[String: Any]] = []
      var stages: [[String: Any]] = []
      _ = try runner.evaluate(
        inputs, schedule: schedule, reuseSession: args.count == 5 || args[5] == "session",
        fixedWeights: { try weights.readFixed($0, shape: $1) },
        blockWeights: { try weights.blocks.read(block: $0, name: $1, shape: $2) },
        noise: { step, modality, _ in
          try noiseFile.readFloat32(named: "\(step).\(modality.rawValue)")
        },
        stageProgress: { step, event in
          var item: [String: Any] = [
            "run": run, "step": step, "stage": event.stage,
            "completed_blocks": event.completedBlocks, "metal_bytes": event.metalAllocatedBytes,
            "elapsed_seconds": Date().timeIntervalSince(start),
          ]
          if let m = event.stackMetrics {
            item["stack_metrics"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m))
          }
          stages.append(item)
          try FileHandle.standardOutput.write(
            contentsOf: JSONSerialization.data(withJSONObject: item, options: [.sortedKeys])
              + Data([10]))
        },
        preview: { state, event in
          for (name, actual) in [("video", state.video), ("audio", state.audio)] {
            let key = "\(event.completedSteps-1).\(name)"
            let gpu = try gpuSteps.readFloat32(named: key)
            let cpu = try cpuSteps.readFloat32(named: key)
            let spread = error(cpu, gpu)
            let vsGPU = error(actual, gpu)
            let vsCPU = error(actual, cpu)
            let floor = name == "video" ? 0.002 : 0.0001
            let absoluteLimit = max(floor, spread.absolute * 2)
            let relativeLimit = max(floor, spread.relative * 2)
            let repeatError = first[key].map { error(actual, $0).absolute }
            let repeatPassed = repeatError == nil || repeatError! <= 1e-6
            let calibrated =
              vsGPU.absolute <= absoluteLimit && vsGPU.relative <= relativeLimit && repeatPassed
            // Frozen before mixed-precision trajectory holdouts. The Float32
            // reference gate remains independently reported, never replaced.
            let mixedAbsolute = name == "video" ? 0.02 : 0.005
            let mixedPassed = vsGPU.absolute <= mixedAbsolute && vsGPU.relative <= 0.01 && repeatPassed
            passed = passed && (precision == .float32 ? calibrated : mixedPassed)
            var digest = SHA256()
            actual.withUnsafeBytes { digest.update(bufferPointer: $0) }
            let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
            metrics.append([
              "native_sha256": hash,
              "step": event.completedSteps, "modality": name, "gpu_absolute": vsGPU.absolute,
              "gpu_relative": vsGPU.relative,
              "cpu_absolute": vsCPU.absolute, "cpu_relative": vsCPU.relative,
              "reference_spread_absolute": spread.absolute,
              "reference_spread_relative": spread.relative,
              "absolute_limit": absoluteLimit, "relative_limit": relativeLimit,
              "repeat_absolute": repeatError.map { $0 as Any } ?? NSNull(),
              "strict_component_passed": vsGPU.absolute <= floor && vsGPU.relative <= floor,
              "calibrated_trajectory_passed": calibrated,
              "mixed_precision_absolute_limit": mixedAbsolute,
              "mixed_precision_relative_limit": 0.01,
              "mixed_precision_trajectory_passed": mixedPassed,
            ])
            if run == 0 { first[key] = actual }
          }
        })
      runs.append([
        "seconds": Date().timeIntervalSince(start), "steps": metrics, "stages": stages,
        "released_metal_bytes": MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0,
      ])
    }
    let report: [String: Any] = [
      "scope": "\(precision.rawValue) small-token trajectory qualification, not production quality",
      "passed": passed,
      "precision": precision.rawValue,
      "sigmas": recipe.sigmas, "eta": recipe.eta,
      "governing_gate": precision == .float32 ? "calibrated_float32" : "experimental_mixed_precision",
      "float32_calibration": "frozen max(single-evaluation floor, 2x independent CPU/GPU spread) per step",
      "mixed_precision_calibration": "frozen video maxabs 0.02, audio maxabs 0.005, relative L2 0.01; does not replace Float32 acceptance",
      "repeat_absolute_tolerance": 1e-6, "runs": runs,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: args[4]), options: .atomic)
    if !passed { throw BlockError.invalid("Trajectory qualification failed; retained report.") }
  }
}
