import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

final class RelayScalePreparationTests: XCTestCase {
  func testOldSelectionsKeepTwoTimesAndFourTimesRoundTripsExplicitly() throws {
    let old = try JSONDecoder().decode(GenerationSelection.self,
      from: Data(#"{"task":"video_upscale","preset":"custom","movieUpscaleMethod":"relayVSR"}"#.utf8))
    XCTAssertNil(old.relayVSRScale)
    var selection = old; selection.relayVSRScale = .x4
    XCTAssertEqual(try JSONDecoder().decode(GenerationSelection.self, from: JSONEncoder().encode(selection)).relayVSRScale, .x4)
    XCTAssertTrue(selection.isModified)
    selection.resetOverrides()
    XCTAssertNil(selection.relayVSRScale)
    XCTAssertEqual(selection.movieUpscaleMethod, .relayVSR)
  }
  func testDescriptionUsesTheSelectedCheckpointPairAndFingerprint() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mov"); try Data([1]).write(to: source)
    for scale in NativeRelayVSRScale.allCases {
      let folder = root.appendingPathComponent("x\(scale.rawValue)")
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      for name in scale.checkpointNames { try Data([2]).write(to: folder.appendingPathComponent(name)) }
    }
    var clip = Clip(name: "Relay", engine: .ltx25)
    clip.generationSelection = GenerationSelection(task: "video_upscale", preset: .custom)
    clip.generationSelection?.movieUpscaleMethod = .relayVSR
    let asset = MediaAsset(name: "Source", kind: .video, path: source.path, scope: .project)
    clip.attachments = [Attachment(assetID: asset.id, role: .reference)]
    var fingerprints: [String] = []
    for scale in NativeRelayVSRScale.allCases {
      clip.generationSelection?.relayVSRScale = scale == .x2 ? nil : scale
      var project = StudioProject(); project.clips = [clip]; project.assets = [asset]
      let request: [String: Any] = ["project": try JSONSerialization.jsonObject(with: JSONEncoder().encode(project)),
        "clipID": clip.id.uuidString, "runtime": ["relayVSRWorkerPath": "/usr/bin/true", "relayVSRModelDirectory": root.path]]
      let result = try NativeRelayVSRPreparation.describe(request: request)
      let paths = try XCTUnwrap(result["sourcePaths"] as? [String])
      XCTAssertEqual(Array(paths.suffix(2)).map { URL(fileURLWithPath: $0).lastPathComponent }, scale.checkpointNames)
      fingerprints.append(try XCTUnwrap(result["fingerprint"] as? String))
    }
    XCTAssertNotEqual(fingerprints[0], fingerprints[1])
  }
}
