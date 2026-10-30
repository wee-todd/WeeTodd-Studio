import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateAdapterTests: XCTestCase {
  func testOrdinaryLoaderStillRejectsOfficialFloat32Rank576() throws {
    // A sparse header fixture exercises current admission without allocating weights.
    let root = "transformer.transformer_blocks.0.attn.to_q"
    let aBytes = 576 * 5376 * 4, bBytes = 7168 * 576 * 4
    let header: [String: Any] = ["__metadata__": ["format":"lora", "scale":"alpha == rank for every module (strengths folded)", "dtype":"float32"],
      root + ".lora_A.weight": ["dtype":"F32", "shape":[576,5376], "data_offsets":[0,aBytes]],
      root + ".lora_B.weight": ["dtype":"F32", "shape":[7168,576], "data_offsets":[aBytes,aBytes+bBytes]]]
    let data = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(data.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }; bytes.append(data)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try bytes.write(to: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: UInt64(bytes.count+aBytes+bBytes)); try handle.close()
    XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1))
  }
}
