import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3QwenLayerTests: XCTestCase {
  func testBadContextShapeAndPositionsFailBeforeCheckpointReads() throws {
    try Device.withDefaultDevice(.cpu) {
      let missing = URL(fileURLWithPath:"/missing/weetodd-qwen-context.safetensors")
      for rows in [0,2049] {
        XCTAssertThrowsError(try H3QwenLayer.evaluate(checkpointURL:missing,index:0,
          input:MLXArray.zeros([rows,5120],dtype:.bfloat16))) {
          XCTAssertTrue(String(describing:$0).contains("rows"),String(describing:$0))
        }
      }
      let input = MLXArray.zeros([1056,5120],dtype:.bfloat16)
      for positions in [[[Int32]](repeating:[0],count:3),
                        [[Int32]](repeating:[Int32](repeating:-1,count:1056),count:3)] {
        XCTAssertThrowsError(try H3QwenLayer.evaluate(checkpointURL:missing,index:0,
          input:input,positions:positions)) {
          XCTAssertTrue(String(describing:$0).contains("positions"),String(describing:$0))
        }
      }
    }
  }

  func testInstalledLongMixedContextRetainsCausalPrefix() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"] else {
      throw XCTSkip("Set installed Qwen pages for the >1024-row language-layer check.")
    }
    let rows = 1056, prefix = 32
    let token = MLXArray((0..<rows).map(Float.init),[rows,1])
    let feature = MLXArray((0..<5120).map(Float.init),[1,5120])
    let input = sin(token * Float(0.007) + feature * Float(0.011)).asType(.bfloat16)
    let positions: [[Int32]] = (0..<3).map { axis in
      (0..<rows).map { row in
        if row >= 1024 { return Int32(row-1024+33) }
        return Int32(axis == 0 ? 1 : axis == 1 ? row/32+1 : row%32+1)
      }
    }
    let checkpoint = URL(fileURLWithPath:root).appendingPathComponent("pages/layer-000.safetensors")
    let short = try H3QwenLayer.evaluate(checkpointURL:checkpoint,index:0,
      input:input[0..<prefix,0..<5120],positions:positions.map { Array($0.prefix(prefix)) })
    let long = try H3QwenLayer.evaluate(checkpointURL:checkpoint,index:0,input:input,positions:positions)
    XCTAssertEqual(long.shape,[rows,5120])
    XCTAssertEqual(long.dtype,.bfloat16)
    XCTAssertTrue(all(isFinite(long)).item(Bool.self))
    let difference = max(abs(long[0..<prefix,0..<5120].asType(.float32) - short.asType(.float32))).item(Float.self)
    print("Long mixed Qwen causal prefix maxabs=\(difference)")
    XCTAssertLessThanOrEqual(difference,0.03125)
  }

  func testInstalledMixedVisionLayerMatchesReferenceBeforeDeepstack() throws {
    guard let paged = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_MIXED"] else {
      throw XCTSkip("Set paged Qwen and mixed-vision oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let inputData = try Data(contentsOf: root.appendingPathComponent("layer-00.f32"))
    let input = inputData.withUnsafeBytes {
      MLXArray($0, [18, 5120], type: Float.self).asType(.bfloat16)
    }
    let positionData = try Data(contentsOf: root.appendingPathComponent("position.i32"))
    let flat = positionData.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    let positions = (0..<3).map { Array(flat[($0 * 18)..<(($0 + 1) * 18)]) }
    var stageErrors: [(String, Float)] = []
    let stageRoot = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_MIXED_STAGE"]
    let actual = try H3QwenLayer.evaluate(checkpointURL: URL(fileURLWithPath: paged)
      .appendingPathComponent("pages/layer-000.safetensors"), index: 0,
      input: input, positions: positions) { name, value in
      if let stageRoot {
        let data = try Data(contentsOf: URL(fileURLWithPath: stageRoot)
          .appendingPathComponent(name + ".f32"))
        let oracle = data.withUnsafeBytes { MLXArray($0, value.shape, type: Float.self) }
        stageErrors.append((name,
          max(abs(value.asType(.float32) - oracle)).item(Float.self)))
      }
    }
    if !stageErrors.isEmpty { XCTAssertEqual(stageErrors.map(\.1).max(), 0) }
    let outputData = try Data(contentsOf: root.appendingPathComponent("raw-01.f32"))
    let expected = outputData.withUnsafeBytes { MLXArray($0, [18, 5120], type: Float.self) }
    XCTAssertEqual(max(abs(actual.asType(.float32) - expected)).item(Float.self), 0)
    let fourthInputData = try Data(contentsOf: root.appendingPathComponent("layer-03.f32"))
    let fourthInput = fourthInputData.withUnsafeBytes {
      MLXArray($0, [18, 5120], type: Float.self).asType(.bfloat16)
    }
    var fourthStageErrors: [(String, Float)] = []
    let fourth = try H3QwenLayer.evaluate(checkpointURL: URL(fileURLWithPath: paged)
      .appendingPathComponent("pages/layer-003.safetensors"), index: 3,
      input: fourthInput, positions: positions) { name, value in
      if let stageRoot = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_MIXED_STAGE3"] {
        let data = try Data(contentsOf: URL(fileURLWithPath: stageRoot)
          .appendingPathComponent(name + ".f32"))
        let oracle = data.withUnsafeBytes { MLXArray($0, value.shape, type: Float.self) }
        fourthStageErrors.append((name,
          max(abs(value.asType(.float32) - oracle)).item(Float.self)))
      }
    }
    if !fourthStageErrors.isEmpty { XCTAssertEqual(fourthStageErrors.map(\.1).max(), 0) }
    let fourthExpectedData = try Data(contentsOf: root.appendingPathComponent("layer-04.f32"))
    let fourthExpected = fourthExpectedData.withUnsafeBytes {
      MLXArray($0, [18, 5120], type: Float.self)
    }
    XCTAssertEqual(max(abs(fourth.asType(.float32) - fourthExpected)).item(Float.self), 0)
  }
  func testInstalledEmbeddingBoundaryStageTrace() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"],
      let trace = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TRACE"],
      let stage = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_STAGE"] else {
      throw XCTSkip("Set Qwen page, hidden-state trace and stage oracle paths.")
    }
    func floats(_ url: URL, count: Int) throws -> [Float] {
      let bytes = try Data(contentsOf: url)
      guard bytes.count == count * 4 else { throw H3CheckpointError.invalid("Bad trace length.") }
      return bytes.withUnsafeBytes { raw in
        (0..<count).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
      }
    }
    let initial = try floats(URL(fileURLWithPath: trace).appendingPathComponent("layer-000.f32"),
      count: 6 * 5120)
    let input = MLXArray(initial, [6, 5120]).asType(.bfloat16)
    let checkpoint = URL(fileURLWithPath: root).appendingPathComponent("pages/layer-000.safetensors")
    var gaps: [(String, Float)] = []
    _ = try H3QwenLayer.evaluate(checkpointURL: checkpoint, index: 0, input: input,
      observe: { name, value in
        let actual = value.asType(.float32).asArray(Float.self)
        let reference = try floats(URL(fileURLWithPath: stage)
          .appendingPathComponent(name + ".f32"), count: actual.count)
        gaps.append((name, zip(actual, reference).map { abs($0 - $1) }.max()!))
      })
    XCTAssertLessThan(gaps.map(\.1).max()!, 0.01)
  }

  func testInstalledFirstLayerMatchesCausalTwoTokenReference() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_PAGED"] else {
      throw XCTSkip("Set WEETODD_H3_QWEN_PAGED for the optional real layer check.")
    }
    let first = (0..<5120).map { Float(sin(Double(Float($0) * 0.01))) }
    let second = (0..<5120).map { Float(cos(Double(Float($0) * 0.007))) }
    let input = MLXArray(first + second, [2, 5120]).asType(.bfloat16)
    let checkpoint = URL(fileURLWithPath: root).appendingPathComponent("pages/layer-000.safetensors")
    let result = try H3QwenLayer.evaluate(checkpointURL: checkpoint, index: 0,
      input: input).asType(.float32).asArray(Float.self)
    let expectedFirst: [Float] = [-0.16308594, -0.12109375, 0.09472656, 0.07421875,
      0.0625, -0.20019531, 0.123046875, 0.013427734]
    let expectedSecond: [Float] = [1.0234375, 0.76953125, 1.125, 1.0390625,
      1.015625, 0.94921875, 1.1328125, 0.94921875]
    for index in 0..<8 {
      XCTAssertEqual(result[index], expectedFirst[index], accuracy: 0.01, "first \(index)")
      XCTAssertEqual(result[5120 + index], expectedSecond[index], accuracy: 0.01,
        "second \(index)")
    }
    if let oracle = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_LAYER0_ORACLE"] {
      let bytes = try Data(contentsOf: URL(fileURLWithPath: oracle))
      XCTAssertEqual(bytes.count, 2 * 5120 * 4)
      let expected = bytes.withUnsafeBytes { raw in
        (0..<(2 * 5120)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
      }
      let differences = zip(result, expected).map { abs($0 - $1) }
      XCTAssertLessThan(differences.max()!, 0.01)
    }
  }
}
