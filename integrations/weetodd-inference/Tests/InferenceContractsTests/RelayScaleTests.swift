import Foundation
import XCTest
@testable import InferenceContracts

final class RelayScaleTests: XCTestCase {
  func testScaleSelectsTheActualReleasedPair() throws {
    XCTAssertNil(NativeRelayVSRScale(rawValue: 3))
    XCTAssertEqual(NativeRelayVSRScale.x2.wanFormat, "relayvsr-wan-x2-v1")
    XCTAssertEqual(NativeRelayVSRScale.x4.wanFormat, "refvsr-wan-v1")
    XCTAssertEqual(NativeRelayVSRScale.x4.flashFormat, "refvsr-flash-v1")
    XCTAssertEqual(NativeRelayVSRScale.x4.checkpointNames, ["relayvsr_x4_int8_convrot.safetensors", "relayvsr_flash_vae_x4_fp32.safetensors"])
    let wan = ["format": "refvsr-wan-v1", "config": "{}"]
    let flash = ["format": "refvsr-flash-v1", "config": #"{"upscale_factor":4}"#]
    XCTAssertNoThrow(try NativeRelayVSRScale.x4.validateMetadata(wan: wan, flash: flash))
    XCTAssertThrowsError(try NativeRelayVSRScale.x2.validateMetadata(wan: wan, flash: flash))
    XCTAssertThrowsError(try NativeRelayVSRScale.x4.validateMetadata(wan: wan,
      flash: ["format": "refvsr-flash-v1", "config": #"{"upscale_factor":2}"#]))
  }
  func testFamilyAndExistingVariantFoldersResolveWithoutMixingPairs() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    for scale in NativeRelayVSRScale.allCases {
      let folder = root.appendingPathComponent("x\(scale.rawValue)")
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      for name in scale.checkpointNames { try Data([1]).write(to: folder.appendingPathComponent(name)) }
    }
    XCTAssertEqual(try NativeRelayVSRScale.x4.resolveDirectory(root), root.appendingPathComponent("x4"))
    XCTAssertEqual(try NativeRelayVSRScale.x4.resolveDirectory(root.appendingPathComponent("x2")), root.appendingPathComponent("x4"))
    try FileManager.default.removeItem(at: root.appendingPathComponent("x4/" + NativeRelayVSRScale.x4.checkpointNames[1]))
    XCTAssertThrowsError(try NativeRelayVSRScale.x4.resolveDirectory(root))
    XCTAssertNoThrow(try NativeRelayVSRScale.x2.resolveDirectory(root))
  }
}
