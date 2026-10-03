import CryptoKit
import Foundation
import MLX
import MLXNN
import XCTest
@testable import H3MLX

final class H3VideoVAEEncoderScopeTests: XCTestCase {
  func testSharedScopedBlockMatchesOriginalFP16OperationsAndShortcutOrder() throws {
    try Device.withDefaultDevice(.cpu) {
      let pixels = (0..<1440).map { Float(($0 * 7) % 23 - 11) / 16 }
      let input = MLXArray(pixels, [1, 5, 3, 3, 32]).asType(.float16)
      for out in [32, 64] {
        var calls: [String] = []
        func normalize(_ value: MLXArray, _ name: String) throws -> MLXArray {
          calls.append(name)
          return try H3VideoVAEEncoderOps.temporallyIsolatedGroupNorm(value,
            weight: .ones([value.shape[4]], dtype: .float16),
            bias: MLXArray([Float](repeating: 0.125, count: value.shape[4])).asType(.float16))
        }
        func convolve(_ value: MLXArray, _ name: String, _ channels: Int,
          _ kernel: Int, _ spatial: Int, _ temporal: Int) throws -> MLXArray {
          calls.append(name)
          let count = channels * kernel * kernel * kernel * value.shape[4]
          let values = (0..<count).map { Float(($0 * 11) % 17 - 8) / 128 }
          let weight = MLXArray(values,
            [channels, kernel, kernel, kernel, value.shape[4]]).asType(.float16)
          return try H3VideoVAEEncoderOps.causalConv(value, weight: weight,
            bias: MLXArray([Float](repeating: 0.0625, count: channels)).asType(.float16),
            spatialPadding: spatial, temporalPadding: temporal)
        }
        // Independent original operation sequence, before scope-lifetime changes.
        let first = try normalize(input, "block.norm1")
        let hidden = try convolve(silu(first), "block.conv1", out, 3, 1, 2)
        let second = try normalize(hidden, "block.norm2")
        let projected = try convolve(silu(second), "block.conv2", out, 3, 1, 2)
        let residual = out == 32 ? input
          : try convolve(input, "block.nin_shortcut", out, 1, 0, 0)
        let expected = residual + projected
        eval(expected)
        let originalCalls = calls
        calls.removeAll()
        let actual = try H3VideoVAEEncoder.residualBlock(input, name: "block",
          out: out, normalize: normalize, convolve: convolve)
        XCTAssertEqual(actual.dtype, .float16)
        XCTAssertEqual(actual.shape, [1, 5, 3, 3, out])
        XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),
          expected.asArray(Float.self).map(\.bitPattern))
        XCTAssertTrue(MLX.isFinite(actual).all().item(Bool.self))
        XCTAssertEqual(calls, originalCalls)
        XCTAssertEqual(calls, ["block.norm1", "block.conv1", "block.norm2", "block.conv2"]
          + (out == 32 ? [] : ["block.nin_shortcut"]))
      }
    }
  }

  func testSharedScopedBlockPropagatesCancellationBeforeResidualShortcut() throws {
    try Device.withDefaultDevice(.cpu) {
      let input = MLXArray.ones([1, 1, 1, 1, 32], dtype: .float16)
      var calls: [String] = []
      XCTAssertThrowsError(try H3VideoVAEEncoder.residualBlock(input,
        name: "block", out: 64, normalize: { value, name in
          calls.append(name)
          return value
        }, convolve: { value, name, _, _, _, _ in
          calls.append(name)
          if name == "block.conv2" { throw CancellationError() }
          return value
        })) { XCTAssertTrue($0 is CancellationError) }
      XCTAssertEqual(calls, ["block.norm1", "block.conv1", "block.norm2", "block.conv2"])
    }
  }

  func testInstalledControlEncoderMatchesExactFrozenLatentBytes() throws {
    guard let manifestPath = ProcessInfo.processInfo.environment["H3_VIDEO_VAE_SCOPE_ORACLE"] else {
      throw XCTSkip("Set H3_VIDEO_VAE_SCOPE_ORACLE for the frozen installed control encoder witness.")
    }
    let fixture = try JSONDecoder().decode(ScopeOracle.self,
      from: Data(contentsOf: URL(fileURLWithPath: manifestPath)))
    let rgbURL = URL(fileURLWithPath: fixture.rgb)
    let rgb = try Data(contentsOf: rgbURL)
    let expected = try Data(contentsOf: URL(fileURLWithPath: fixture.latent))
    XCTAssertEqual(digest(rgb), fixture.rgbSHA256)
    XCTAssertEqual(digest(expected), fixture.latentSHA256)
    guard digest(rgb) == fixture.rgbSHA256,
      digest(expected) == fixture.latentSHA256 else {
      throw H3CheckpointError.invalid("Frozen encoder oracle changed.")
    }
    let checkpoint = URL(fileURLWithPath: fixture.checkpoint)
    let headerBefore = try checkpointHeader(checkpoint)
    guard digest(headerBefore) == fixture.checkpointHeaderSHA256 else {
      throw H3CheckpointError.invalid("Frozen encoder checkpoint header changed.")
    }
    let actual = try H3VideoVAEEncoder.encodeControlVideo(checkpointURL: checkpoint,
      rgb8: Array(rgb), frameCount: fixture.frames, width: fixture.width, height: fixture.height)
    XCTAssertTrue(MLX.isFinite(actual).all().item(Bool.self))
    let bytes = actual.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
    XCTAssertEqual(bytes, expected)
    XCTAssertEqual(digest(bytes), fixture.latentSHA256)
    XCTAssertEqual(try checkpointHeader(checkpoint), headerBefore)
    XCTAssertEqual(try Data(contentsOf: rgbURL), rgb)
  }

  private struct ScopeOracle: Decodable {
    let checkpoint, checkpointHeaderSHA256, rgb, rgbSHA256, latent, latentSHA256: String
    let width, height, frames: Int
  }

  private func digest(_ bytes: Data) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
  }

  private func checkpointHeader(_ url: URL) throws -> Data {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
      throw H3CheckpointError.invalid("Missing frozen encoder checkpoint header length.")
    }
    let count = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
    guard count <= 4 * 1024 * 1024,
      let header = try file.read(upToCount: Int(count)), header.count == Int(count) else {
      throw H3CheckpointError.invalid("Invalid frozen encoder checkpoint header length.")
    }
    return header
  }
}
