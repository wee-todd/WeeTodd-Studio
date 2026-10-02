import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FunControlTests: XCTestCase {
  func testGuidePaddingAndFirstProjectionReplaceOnlyTargetVideoWithoutMutatingBase() throws {
    let rows = MLXArray((0..<192).map(Float.init), [1, 2, 96])
    let padded = try H3FunControlMath.paddedGuideRows(rows)
    XCTAssertEqual(padded.shape, [1, 2, 196])
    XCTAssertEqual(padded[0, 0, 0..<96].asArray(Float.self), (0..<96).map(Float.init))
    XCTAssertEqual(padded[0, 1, 96..<196].asArray(Float.self), [Float](repeating: 0, count: 100))
    let original = MLXArray([Float(1), 2, 3, 4, 5, 6], [1, 3, 2])
    let branch = try H3FunControlMath.initialize(hidden: original,
      projected: MLXArray([Float(7), 8], [1, 1, 2]), targetIndices: MLXArray([Int32(2)])) { $0 * 2 }
    XCTAssertEqual(original.asArray(Float.self), [1, 2, 3, 4, 5, 6])
    XCTAssertEqual(branch.asArray(Float.self), [3, 6, 9, 12, 19, 22])
    let masked = H3FunControlMath.suppressAudio(branch, audioIndices: MLXArray([Int32(1)]))
    XCTAssertEqual(masked.asArray(Float.self), [3, 6, 0, 0, 19, 22])
    XCTAssertEqual(branch.asArray(Float.self), [3, 6, 9, 12, 19, 22])
  }

  func testInjectionRunsAfterBaseAndKeepsIndependentBranchState() throws {
    var events: [String] = []
    let result = try H3FunControlMath.runBlocks(input: MLXArray([Float(0)], [1, 1, 1]),
      blockCount: 50, control: MLXArray([Float(100)], [1, 1, 1]),
      baseBlock: { index, hidden in
        events.append("base\(index)")
        return hidden * 2 + 1
      }, controlBlock: { index, control in
        events.append("control\(index)")
        XCTAssertEqual(control.item(Float.self), Float(100 + index))
        let next = control + 1
        return (next, next * Float(0.25))
      })
    var expected: Float = 0
    var branch = 0
    for layer in 0..<50 {
      expected = expected * 2 + 1
      if [0, 10, 20, 30, 40].contains(layer) {
        let event = try XCTUnwrap(events.firstIndex(of: "control\(branch)"))
        XCTAssertEqual(events[event - 1], "base\(layer)")
        expected += Float(101 + branch) * 0.25
        branch += 1
      }
    }
    XCTAssertEqual(result.item(Float.self), expected)
    XCTAssertEqual(events.count, 55)
    XCTAssertThrowsError(try H3FunControlMath.runBlocks(input: MLXArray([Float(0)], [1, 1, 1]),
      blockCount: 40, control: MLXArray([Float(0)], [1, 1, 1]), baseBlock: { _, x in x },
      controlBlock: { _, x in (x, x) }))
  }

  func testSharedTinyDenseBlockMatchesIndependentScalarOracle() throws {
    let input = MLXArray([Float(3), 4], [1, 1, 2])
    var table = [Float](repeating: 0, count: 36)
    // Three modality tables of shift, scale and gate for two sublayers.
    for modality in 0..<3 {
      for channel in 0..<2 {
        table[modality * 12 + 4 + channel] = 0.25
        table[modality * 12 + 10 + channel] = 0.5
      }
    }
    let actual = try H3TransformerBlock.evaluateKernel(input: input,
      modulation: MLXArray(table, [1, 36]), modulationIndices: MLXArray([Int32(2)]),
      angles: H3RotaryAngles(rows: 1, cosine: MLXArray.ones([1, 1, 1, 2]),
        sine: MLXArray.zeros([1, 1, 1, 2])),
      hiddenWidth: 2, heads: 1, headWidth: 2, feedWidth: 2, rotaryWidth: 2,
      read: { _, shape in MLXArray.ones(shape) },
      project: { x, name, _, _, _ in
        if name == "attn.qkv_proj" {
          return concatenated([MLXArray.zeros(x.shape), MLXArray.zeros(x.shape), x], axis: 2)
        }
        if name == "mlp.fc1" { return concatenated([x * 2, x * 3], axis: 2) }
        return x
      })
    let norm = sqrt(Float(12.5) + 1e-5)
    let residual = [Float(3), 4].map { $0 + 0.25 * $0 / norm }
    let secondNorm = sqrt(residual.map { $0 * $0 }.reduce(0, +) / 2 + 1e-5)
    let expected = residual.map { value -> Float in
      let x = value / secondNorm
      let gate = (2 * x) / (1 + exp(-2 * x))
      return value + 0.5 * gate * (3 * x)
    }
    for (left, right) in zip(actual.asArray(Float.self), expected) {
      XCTAssertEqual(left, right, accuracy: 0.00002)
    }
  }

  func testControlActivationsUnloadTogetherAndRepeatedUnloadIsSafe() {
    let activations = H3FunControlActivations()
    activations.guideRows = MLXArray.ones([1, 2, 196])
    activations.modulations = [MLXArray.ones([2, 36]), MLXArray.ones([2, 36])]
    XCTAssertEqual(activations.residentBytes, (392 + 144) * 4)
    activations.unload()
    XCTAssertNil(activations.guideRows)
    XCTAssertTrue(activations.modulations.isEmpty)
    XCTAssertEqual(activations.residentBytes, 0)
    activations.unload()
    XCTAssertEqual(activations.residentBytes, 0)
  }

  func testCancellationBetweenBaseAndControlStopsBranchAndReleasesStage() async {
    let report = await Task.detached { () -> (Int, Int, Bool) in
      let activations = H3FunControlActivations()
      activations.guideRows = MLXArray.ones([1, 1, 196])
      activations.modulations = [MLXArray.ones([1, 36])]
      defer { activations.unload() }
      var branchCalls = 0
      do {
        _ = try H3FunControlMath.runBlocks(input: MLXArray.zeros([1, 1, 1]),
          blockCount: 50, control: MLXArray.ones([1, 1, 1]),
          baseBlock: { _, value in
            withUnsafeCurrentTask { $0?.cancel() }
            return value + 1
          }, controlBlock: { _, value in
            branchCalls += 1
            return (value, value)
          })
        return (-1, branchCalls, false)
      } catch is CancellationError {
        activations.unload()
        return (activations.residentBytes, branchCalls, true)
      } catch { return (-1, branchCalls, false) }
    }.value
    XCTAssertEqual(report.0, 0)
    XCTAssertEqual(report.1, 0)
    XCTAssertTrue(report.2)
  }

  func testUnionTwoRunsAllTenInjectionsOnTheVerifiedLayerSchedule() throws {
    var base = -1
    var layers: [Int] = []
    _ = try H3FunControlMath.runBlocks(input: MLXArray.zeros([1, 1, 1]),
      blockCount: 50, control: MLXArray.zeros([1, 1, 1]),
      injectionLayers: H3FunControlLayout.v2InjectionLayers,
      baseBlock: { index, value in base = index; return value },
      controlBlock: { index, value in
        XCTAssertEqual(index, layers.count); layers.append(base)
        return (value, value)
      })
    XCTAssertEqual(layers, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
  }

  private func header(split: Bool = true, time: UInt64 = 2688, count: Int = 5) -> [String: H3TensorInfo] {
    var result: [String: H3TensorInfo] = [:]
    func put(_ name: String, _ shape: [UInt64]) { result[name] = H3TensorInfo(dtype: "BF16", shape: shape) }
    put("control_proj_in.weight", [5376, 196]); put("control_proj_in.bias", [5376])
    for index in 0..<count {
      let block = "control_blocks.\(index)."
      put(block + "adaln_proj.linear.weight", [96768, time]); put(block + "adaln_proj.linear.bias", [96768])
      put(block + "norm1.weight", [5376]); put(block + "norm2.weight", [5376])
      for name in split ? ["attn.norm_q.weight", "attn.norm_k.weight"] : ["attn.q_norm.weight", "attn.k_norm.weight"] { put(block + name, [128]) }
      if split { for name in ["q", "k", "v"] { put(block + "attn.to_\(name).weight", [7168, 5376]) } }
      else { put(block + "attn.qkv_proj.weight", [21504, 5376]) }
      put(block + (split ? "attn.to_out.0.weight" : "attn.out_proj.weight"), [5376, 7168])
      put(block + (split ? "ff.net.0.proj.weight" : "mlp.fc1.weight"), [28672, 5376])
      put(block + (split ? "ff.net.2.weight" : "mlp.fc2.weight"), [5376, 14336])
      put(block + "after_proj.weight", [5376, 5376]); put(block + "after_proj.bias", [5376])
    }
    put("control_blocks.0.before_proj.weight", [5376, 5376]); put("control_blocks.0.before_proj.bias", [5376])
    return result
  }

  func testInjectionMetadataMustMatchExactReleasedSchedule() throws {
    XCTAssertEqual(try H3FunControlLayout(tensors: header(),
      metadata: ["control_blocks_places": "[0,10,20,30,40]"]).blockCount, 5)
    XCTAssertEqual(try H3FunControlLayout(tensors: header(count: 10),
      metadata: ["control_blocks_places": "[0,5,10,15,20,25,30,35,40,45]"]).blockCount, 10)
    for bad in ["[0,5,10,15,20]", "[false,10,20,30,40]", "[0,10,20,30,40,45]", "0,10,20,30,40"] {
      XCTAssertThrowsError(try H3FunControlLayout(tensors: header(),
        metadata: ["control_blocks_places": bad]))
    }
  }

  func testInstalledFullControlCheckpointHeaderWhenProvided() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_FUN_CONTROL_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_FUN_CONTROL_CHECKPOINT for actual checkpoint header admission.")
    }
    let layout = try H3FunControlLayout(url: URL(fileURLWithPath: path))
    XCTAssertEqual(layout.timeWidth, 2688)
    XCTAssertTrue([5, 10].contains(layout.blockCount))
  }

  func testRawAndConvertedHeadersAndRejectIncompleteOrChangedTopology() throws {
    XCTAssertEqual(try H3FunControlLayout(tensors: header()).timeWidth, 2688)
    XCTAssertTrue(try H3FunControlLayout(tensors: header()).splitQKV)
    XCTAssertEqual(try H3FunControlLayout(tensors: header(count: 10)).injectionLayers,
      [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    XCTAssertFalse(try H3FunControlLayout(tensors: header(split: false, time: 64)).splitQKV)
    for bad in ["control_blocks.3.attn.to_k.weight", "control_blocks.4.after_proj.bias"] {
      var values = header(); values.removeValue(forKey: bad)
      XCTAssertThrowsError(try H3FunControlLayout(tensors: values))
    }
    var values = header(); values["control_blocks.5.norm1.weight"] = values["control_blocks.0.norm1.weight"]
    XCTAssertThrowsError(try H3FunControlLayout(tensors: values))
    values = header(); values["control_proj_in.weight"] = H3TensorInfo(dtype: "BF16", shape: [5376, 96])
    XCTAssertThrowsError(try H3FunControlLayout(tensors: values))
  }
}
