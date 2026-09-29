import CryptoKit
import Darwin
import Foundation
import LTX25NNC
import TensorIO

/// Fixed synthetic activations at real model widths; not a media generation.
@main struct PerformanceProbe {
  static func main() {
    do { try run() } catch { fputs("\(error)\n", stderr); exit(2) }
  }
  static func run() throws {
    let a = Array(CommandLine.arguments.dropFirst())
    guard (8...9).contains(a.count), let blocks = Int(a[2]), (1...48).contains(blocks),
      let repeats = Int(a[3]), (1...5).contains(repeats), let budget = UInt64(a[4]),
      ["0", "1"].contains(a[5]), ["scalar", "simd"].contains(a[6]),
      let precision = LTXPrecisionPolicy(rawValue: a.count == 9 ? a[8] : "float32") else {
      throw BlockError.invalid("Usage: WeeToddLTXPerformanceProbe CONFIG ROOT BLOCKS REPEATS PREPARED_BYTES SEQUENCE_0_OR_1 scalar|simd REPORT [float32|fp16-projections|bf16-projections]")
    }
    let url = URL(fileURLWithPath: a[0])
    guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw BlockError.invalid("Oversized benchmark configuration.")
    }
    let c = try JSONDecoder().decode(AVBlockConfiguration.self, from: Data(contentsOf: url))
    try AVBlockRunner.validateAllocation(configuration: c)
    let weights = try PagedBlockWeights(root: URL(fileURLWithPath: a[1]), configuration: c)
    let shapes = try AVBlockRunner.expectedInputShapes(configuration: c)
    var inputs: [String: [Float]] = [:]
    for (name, shape) in shapes {
      let seed = name.utf8.reduce(0) { $0 + Int($1) }
      inputs[name] = (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 73 - 36) / 128 }
    }
    let runner = try AVStackRunner(configuration: c, blockCount: blocks, sequenceAttention: a[5] == "1", experimentalPrecision: precision)
    var runs: [[String: Any]] = []
    for index in 0..<repeats {
      let begin = Date()
      var last: AVStackRunner.Progress?
      let result = try runner.evaluate(inputs, retainWeights: true, maximumPreparationBytes: budget,
        weights: { try weights.read(block: $0, name: $1, shape: $2, decoding: a[6] == "scalar" ? .scalar : .simd) }) {
          last = $0
        }
      let elapsed = Date().timeIntervalSince(begin)
      func hash(_ values: [Float]) -> String {
        var digest = SHA256()
        values.withUnsafeBytes { digest.update(bufferPointer: $0) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
      }
      let record: [String: Any] = ["run": index, "seconds": elapsed,
        "video_sha256": hash(result.video), "audio_sha256": hash(result.audio),
        "metrics": try JSONSerialization.jsonObject(with: JSONEncoder().encode(last!.metrics)),
        "scratch_bytes": last!.runtimeBytes, "metal_bytes": last!.metalAllocatedBytes]
      runs.append(record)
      try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) + Data([10]))
    }
    try runner.release()
    let report: [String: Any] = ["scope": "synthetic-input-real-width-block-performance", "blocks": blocks,
      "video_tokens": c.videoTokens, "audio_tokens": c.audioTokens, "text_tokens": c.textTokens,
      "precision": precision.rawValue,
      "sequence_attention": a[5] == "1", "decode": a[6], "prepared_byte_budget": budget,
      "runs": runs, "released_metal_bytes": runner.metalAllocatedBytes]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: a[7]), options: .atomic)
  }
}
