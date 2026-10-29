import Drama3Contracts
import MLX
import XCTest

@testable import Drama3MLX

/// Optional independent numerical qualification. Each layer receives the same reference input;
/// this isolates implementation error from BF16 rounding and discrete MoE routing in later layers.
final class CheckpointParityTests: XCTestCase {
  func testReferenceBackboneAndTowerWhenQualified() throws {
    guard let root = ProcessInfo.processInfo.environment["DRAMA3_NUMERICAL_ROOT"] else {
      throw XCTSkip("Set DRAMA3_NUMERICAL_ROOT for independent checkpoint qualification.")
    }
    let folder = URL(fileURLWithPath: root)
    let model = folder.appendingPathComponent("balanced")
    let oracle = folder.appendingPathComponent("oracle-parity.safetensors")
    guard FileManager.default.fileExists(atPath: oracle.path) else {
      throw XCTSkip("Create the independently computed pinned reference fixture first.")
    }
    let reference = try loadArrays(url: oracle)
    let admission = try Drama3ModelLoader.inspect(at: model)
    let weights = try Drama3Weights(at: model, files: admission.files)
    let backbone = Drama3HybridBackbone(weights)
    let tower = Drama3CodebookTower(weights, backbone: backbone)
    let frames = try XCTUnwrap(reference["frames"])
    let hidden = try XCTUnwrap(reference["prefill_hidden"])
    var errors: [String: Float] = [:]
    func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray, _ tolerance: Float) {
      let a = actual.asType(.float32)
      let b = expected.asType(.float32)
      let difference = a - b
      let relative =
        sqrt((difference * difference).mean()).item(Float.self)
        / max(0.0001, sqrt((b * b).mean()).item(Float.self))
      errors[name] = relative
      XCTAssertLessThan(relative, tolerance, name)
    }
    compare(
      "embedding", try backbone.embedding(frames), try XCTUnwrap(reference["embedding"]), 0.00001)
    for layer in 0..<40 {
      let input = try XCTUnwrap(reference[layer == 0 ? "embedding" : "layer-\(layer-1)"])
      let actual = try backbone.block(layer, input, state: Drama3State())
      compare("backbone-layer-\(layer)", actual, try XCTUnwrap(reference["layer-\(layer)"]), 0.005)
    }
    compare(
      "semantic-head", try weights.linear("semantic_head", hidden),
      try XCTUnwrap(reference["prefill_logits"]), 0.0001)
    for i in 0..<13 {
      let name = i < 4 ? "tower.shared_layers.\(i)" : "tower.residual_layers.\(i-4)"
      let state = Drama3State()
      for phase in ["prefill", "decode"] {
        let prefix = name + "." + phase
        let actual = try tower.layer(
          name, hidden: XCTUnwrap(reference[prefix + ".input"]),
          codes: XCTUnwrap(reference[prefix + ".codes"]),
          mask: XCTUnwrap(reference[prefix + ".mask"]), state: state,
          extra: reference[prefix + ".extra"], enrichment: i < 4)
        compare(prefix, actual, try XCTUnwrap(reference[prefix + ".output"]), 0.005)
        state.evaluate([actual])
      }
    }
    let maximum = errors.values.max() ?? 0
    print(
      "Independent layer-local parity: \(errors.count) comparisons, maximum relative RMS \(maximum)"
    )
    try JSONSerialization.data(withJSONObject: errors, options: [.prettyPrinted, .sortedKeys])
      .write(to: folder.appendingPathComponent("native-layer-parity.json"))
  }
}
