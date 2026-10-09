import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3NativeBlockTensorIOTests: XCTestCase {
  private func temporary(_ body: (URL) throws -> Void) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try body(root)
  }

  func testSignedIndicesAndBF16RequestRoundTripWithoutChangingWords() throws {
    try temporary { root in
      let words = (0..<(3 * 5376)).map { UInt16(0x3f00 + $0 % 100) }
      let x = MLXArray(words, [1, 3, 5376]).view(dtype: .bfloat16)
      let indices = MLXArray([Int32(0), 1, 2])
      let url = root.appendingPathComponent("request.safetensors")
      try H3NativeBlockTensorIO.writeRequest(x: x, indices: indices, tableRows: 3, to: url)
      let file = try SafeTensorFile(url: url)
      XCTAssertEqual(Set(file.tensors.keys), ["x", "indices"])
      XCTAssertEqual(file.tensors["x"]?.shape, [3, 5376])
      XCTAssertEqual(file.tensors["x"]?.dtype, "BF16")
      XCTAssertEqual(file.tensors["indices"]?.dtype, "I32")
      XCTAssertEqual(try file.withTensorBytes(named: "x") { Array($0.bindMemory(to: UInt16.self)) }, words)
      XCTAssertEqual(try file.withTensorBytes(named: "indices") { Array($0.bindMemory(to: Int32.self)) }, [0, 1, 2])
      XCTAssertThrowsError(try H3NativeBlockTensorIO.writeRequest(x: x, indices: indices, tableRows: 3, to: url))
    }
  }

  func testInvalidShapeIndexOrNonfiniteInputRejectsBeforeCreatingFile() throws {
    try temporary { root in
      let x = MLXArray([Float](repeating: 1, count: 5376), [1, 1, 5376]).asType(.bfloat16)
      for (label, input, indices) in [
        ("shape", x.reshaped([5376]), MLXArray([Int32(0)])),
        ("negative", x, MLXArray([Int32(-1)])),
        ("beyond", x, MLXArray([Int32(3)])),
        ("infinite", x * MLXArray(Float.infinity), MLXArray([Int32(0)]))
      ] {
        let url = root.appendingPathComponent(label + ".safetensors")
        XCTAssertThrowsError(try H3NativeBlockTensorIO.writeRequest(x: input, indices: indices, tableRows: 3, to: url))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
      }
    }
  }

  func testCompleteInitialTablesPreserveBlockAndModalityCoordinates() throws {
    try temporary { root in
      let x = MLXArray.zeros([1, 3, 5376], dtype: .bfloat16)
      let indices = MLXArray([Int32(0), 1, 2])
      let tables = (0..<50).map { block in
        MLXArray((0..<(3 * 6 * 5376)).map { Float(block) / 16 + Float(($0 / 5376) % 6) / 8 }, [1, 96768]).asType(.bfloat16)
      }
      let angles = H3RotaryAngles(rows: 3, cosine: MLXArray.ones([1, 1, 3, 96], dtype: .bfloat16), sine: MLXArray.zeros([1, 1, 3, 96], dtype: .bfloat16))
      let url = root.appendingPathComponent("initial.safetensors")
      try H3NativeBlockTensorIO.writeInitial(x: x, indices: indices, modulations: tables, angles: angles, to: url)
      let file = try SafeTensorFile(url: url)
      XCTAssertEqual(file.tensors.count, 304)
      XCTAssertEqual(file.tensors["cos"]?.shape, [3, 1, 96])
      for block in [0, 49] {
        for mod in 0..<6 {
          let descriptor = file.tensors["block\(block).mod\(mod)"]
          XCTAssertEqual(descriptor?.shape, [3, 5376])
          let words = try file.withTensorBytes(named: "block\(block).mod\(mod)") { Array($0.bindMemory(to: UInt16.self)) }
          let values = MLXArray(words).view(dtype: .bfloat16).asType(.float32).asArray(Float.self)
          XCTAssertTrue(values.allSatisfy { $0 == Float(block) / 16 + Float(mod) / 8 })
        }
      }
    }
  }

  func testRawOutputRequiresExactFiniteF32BytesAndReturnsFP32Residual() throws {
    try temporary { root in
      let url = root.appendingPathComponent("output.f32")
      let values = (0..<5376).map { Float($0) / 1024 }
      try values.withUnsafeBytes { try Data($0).write(to: url) }
      let output = try H3NativeBlockTensorIO.readOutput(url: url, rows: 1)
      XCTAssertEqual(output.shape, [1, 1, 5376])
      XCTAssertEqual(output.dtype, .float32)
      XCTAssertEqual(output.asArray(Float.self), values)
      XCTAssertThrowsError(try H3NativeBlockTensorIO.readOutput(url: url, rows: 2))
      var invalid = values; invalid[7] = .nan
      try invalid.withUnsafeBytes { try Data($0).write(to: url) }
      XCTAssertThrowsError(try H3NativeBlockTensorIO.readOutput(url: url, rows: 1))
    }
  }
}
