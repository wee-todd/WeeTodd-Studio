import XCTest
import MLX
import InferenceContracts
@testable import ReCaVSRMLX

final class ReCaCheckpointTests: XCTestCase {
  func testGenericWanOrRelayTensorSetsCannotAdmitReCaVSR() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("wrong.safetensors")
    try save(arrays: ["patch_embedding.weight": MLXArray.zeros([1])], url: url)
    let identity = try ReCaFileIdentity.capture(url)
    XCTAssertThrowsError(try ReCaCheckpoint(asset: identity, component: .transformer, precision: .bf16))
  }
  func testSignedQuantizedProjectionPreservesScaleAndRotation() throws {
    Device.withDefaultDevice(.cpu) {
      var bytes = Array(repeating: Int8(0), count: 32 * 256)
      bytes[0] = -128; bytes[256] = 127
      let scales = MLXArray(Array(repeating: Float(0.125), count: 32), [32, 1])
      let weight = MLXArray(bytes, [32, 256])
      let linear = ReCaLinear(weight: weight, bias: nil, scale: scales, computeDType: .float32)
      let x = MLXArray.ones([1, 256])
      let y = linear(x).asArray(Float.self)
      // The regular fourth Kronecker power has +1/16 in every row sum.
      XCTAssertEqual(y[0], -16, accuracy: 1e-5)
      XCTAssertEqual(y[1], 15.875, accuracy: 1e-5)
      XCTAssertEqual(y[2], 0, accuracy: 1e-5)
    }
  }
}
