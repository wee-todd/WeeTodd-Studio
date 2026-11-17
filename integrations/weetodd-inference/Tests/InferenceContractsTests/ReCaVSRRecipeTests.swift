import CryptoKit
import Foundation
import XCTest
@testable import InferenceContracts

final class ReCaVSRRecipeTests: XCTestCase {
  func testReleasedConfigurationsAdmitTheirActualFileLengths() {
    let pins = ReCaReleasedAssets.pins(.bf16)
    XCTAssertEqual(pins.first { $0.role == .modelConfig }?.bytes, 1145)
    XCTAssertEqual(pins.first { $0.role == .vaeConfig }?.bytes, 1599)
    XCTAssertEqual(pins.first { $0.role == .flashDecoderConfig }?.bytes, 88)
  }
  func testRejectsUnsupportedSettingsAndIntervalsBeforeLoading() throws {
    XCTAssertThrowsError(try ReCaVSRSettings(scale: .nan))
    XCTAssertThrowsError(try ReCaVSRSettings(scale: 3))
    XCTAssertThrowsError(try ReCaSourceInterval(startSeconds: -1, durationSeconds: 1))
    XCTAssertThrowsError(try ReCaSourceInterval(startSeconds: 0, durationSeconds: .infinity))
    XCTAssertEqual(try ReCaVSRSettings().scale, 4)
    XCTAssertEqual(try ReCaVSRSettings().seed, 42)
    XCTAssertEqual(try ReCaVSRSettings().residency, .staged)
  }

  func testFrozenIdentityRejectsChangedBytesButCanBeMoved() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let first = folder.appendingPathComponent("asset.bin"), second = folder.appendingPathComponent("moved.bin")
    try Data("original".utf8).write(to: first)
    let identity = try ReCaFileIdentity.capture(first)
    try identity.verify()
    try FileManager.default.copyItem(at: first, to: second)
    let moved = try ReCaFileIdentity.capture(second)
    XCTAssertEqual(identity.sha256, moved.sha256)
    try Data("replaced".utf8).write(to: first)
    XCTAssertThrowsError(try identity.verify())
    try moved.verify()
  }

  func testRecipeRejectsDuplicateAssetRolesAndOutputAlias() throws {
    let f = ReCaFileIdentity(path: "/source.mov", sha256: String(repeating: "a", count: 64), bytes: 10)
    let roles = ReCaAssetRole.allCases.map { ReCaAsset(role: $0, file: f) }
    let assets = try ReCaAssets(files: roles)
    XCTAssertThrowsError(try ReCaAssets(files: roles + [roles[0]]))
    XCTAssertThrowsError(try ReCaAssets(files: Array(roles.dropLast())))
    XCTAssertThrowsError(try NativeReCaVSRRecipe(source: f, interval: nil, worker: f,
      metal: f, assets: assets, settings: ReCaVSRSettings(), outputPath: "/source.mov"))
    let recipe = try NativeReCaVSRRecipe(source: f, interval: nil, worker: f, metal: f,
      assets: assets, settings: ReCaVSRSettings(), outputPath: "/result.mov")
    let data = try JSONEncoder().encode(recipe)
    XCTAssertEqual(try NativeReCaVSRRecipe.prepare(data), recipe)
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    object["unexpected"] = true
    XCTAssertThrowsError(try NativeReCaVSRRecipe.prepare(JSONSerialization.data(withJSONObject: object)))
    object.removeValue(forKey: "unexpected")
    var settings = try XCTUnwrap(object["settings"] as? [String: Any])
    settings["cfg"] = 7
    object["settings"] = settings
    XCTAssertThrowsError(try NativeReCaVSRRecipe.prepare(JSONSerialization.data(withJSONObject: object)))
  }

  func testFingerprintBindsContentAndSettingsNotModelFolder() throws {
    func recipe(_ root: String, scale: Double) throws -> NativeReCaVSRRecipe {
      let f = ReCaFileIdentity(path: "/source.mov", sha256: String(repeating: "a", count: 64), bytes: 10)
      let assets = try ReCaAssets(files: ReCaAssetRole.allCases.map {
        ReCaAsset(role: $0, file: .init(path: root + "/" + $0.rawValue,
          sha256: String(repeating: "b", count: 64), bytes: 12))
      })
      return try NativeReCaVSRRecipe(source: f, interval: nil, worker: f, metal: f,
        assets: assets, settings: ReCaVSRSettings(scale: scale), outputPath: "/result.mov")
    }
    XCTAssertEqual(try recipe("/one", scale: 4).fingerprint(), try recipe("/two", scale: 4).fingerprint())
    XCTAssertNotEqual(try recipe("/one", scale: 4).fingerprint(), try recipe("/one", scale: 2).fingerprint())
  }

  func testReceiptRejectsAnOutputFromAnotherJob() throws {
    let file = ReCaFileIdentity(path: "/source.mov", sha256: String(repeating: "a", count: 64), bytes: 10)
    let assets = try ReCaAssets(files: ReCaAssetRole.allCases.map { ReCaAsset(role: $0, file: file) })
    let recipe = try NativeReCaVSRRecipe(source: file, interval: nil, worker: file, metal: file,
      assets: assets, settings: ReCaVSRSettings(), outputPath: "/result.mov")
    let wrong = try NativeReCaVSRReceipt(recipe: recipe, video: file, frames: 1, width: 64, height: 64,
      sourceIn: 0, duration: 1, seconds: 2)
    XCTAssertThrowsError(try wrong.validate())
    let video = ReCaFileIdentity(path: "/result.mov", sha256: file.sha256, bytes: 20)
    let valid = try NativeReCaVSRReceipt(recipe: recipe, video: video, frames: 1, width: 64, height: 64,
      sourceIn: 0, duration: 1, seconds: 2)
    try valid.validate(expected: recipe, output: "/result.mov")
    XCTAssertThrowsError(try valid.validate(expected: recipe, output: "/another.mov"))
    let other = try NativeReCaVSRRecipe(source: file, interval: nil, worker: file, metal: file,
      assets: assets, settings: ReCaVSRSettings(seed: 7), outputPath: "/result.mov")
    XCTAssertThrowsError(try valid.validate(expected: other, output: "/result.mov"))
  }
}
