import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3AudioVAEDecoderTests: XCTestCase {
  func testInstalledTwoFrameStereoDecoderMatchesReferenceBoundaries() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_VAE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_AUDIO_ORACLE"] else {
      throw XCTSkip("Set installed H3 audio VAE and audio oracle paths.")
    }
    let root = URL(fileURLWithPath: fixture)
    let latent = try Data(contentsOf: root.appendingPathComponent("latent.f32"))
      .withUnsafeBytes { MLXArray($0, [2, 2, 32], type: Float.self) }
    let output = try H3AudioVAEDecoder.decode(
      checkpointURL: URL(fileURLWithPath: checkpoint), latent: latent,
      observe: { name, value in
        let expectedShape: [Int]
        switch name {
        case "decin": expectedShape = [2, 2, 2048]
        case "convpre": expectedShape = [2, 2, 1024]
        case let step where step.hasPrefix("up") || step.hasPrefix("res"):
          guard let stage = Int(step.dropFirst(step.hasPrefix("up") ? 2 : 3)),
            (0..<7).contains(stage) else {
            XCTFail("Unknown audio stage \(name)"); return
          }
          let lengths = [10, 50, 100, 200, 400, 800, 1600]
          expectedShape = [2, lengths[stage], 512 >> stage]
        case "postact": expectedShape = [2, 1600, 8]
        case "postconv", "wave": expectedShape = [2, 1600, 1]
        default: XCTFail("Unknown audio observation \(name)"); return
        }
        let expected = try Data(contentsOf: root.appendingPathComponent("\(name).f32"))
          .withUnsafeBytes { MLXArray($0, expectedShape, type: Float.self) }
        XCTAssertEqual(value.shape, expectedShape)
        XCTAssertLessThan(max(abs(value - expected)).item(Float.self), 1e-4, name)
      })
    XCTAssertEqual(output.shape, [2, 1600, 1])
  }
}
