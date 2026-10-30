import Foundation
import XCTest
@testable import StudioCore

final class FlashVSRContractTests: XCTestCase {
  func testVisibleGeometryNeverStretchesOrRoundsDelivery() throws {
    var settings = FlashVSRSettings()
    settings.width = 1920; settings.height = 1080
    let canvas = try settings.canvas()
    XCTAssertEqual(canvas.width, 1920); XCTAssertEqual(canvas.height, 1152)
    XCTAssertEqual(canvas.visibleWidth, 1920); XCTAssertEqual(canvas.visibleHeight, 1080)
    settings.width = 1080; settings.height = 1920
    XCTAssertEqual(try settings.canvas().width, 1152)
    settings.width = 1919
    XCTAssertThrowsError(try settings.validate())
  }
  func testBothDecodersRetainIndependentSettingsAndAdmitNativeExecution() throws {
    var settings = FlashVSRSettings()
    settings.decoder = .vae; settings.seed = 4294967295
    let decoded = try JSONDecoder().decode(FlashVSRSettings.self, from: JSONEncoder().encode(settings))
    XCTAssertEqual(decoded.decoder, .vae); XCTAssertEqual(decoded.seed, 4294967295)
    settings.temporalCompression = 3
    XCTAssertThrowsError(try settings.validate())
    settings.temporalCompression = 4; settings.temporalSigma = .nan
    XCTAssertThrowsError(try settings.validate())
    settings.temporalSigma = 1; settings.backend = .swiftMLX
    XCTAssertNoThrow(try settings.validateExecution())
  }
  func testMissingSelectedDecoderDoesNotFallback() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder.appendingPathComponent("turbo"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for name in ["turbo/dit.safetensors", "turbo/lq_proj_in.safetensors", "posi_prompt.pth", "Wan2.1_VAE.pth"] {
      try Data([1]).write(to: folder.appendingPathComponent(name))
    }
    var settings = FlashVSRSettings()
    XCTAssertThrowsError(try FlashVSRModelManifest.resolve(root: folder, settings: settings))
    settings.decoder = .vae
    let files = try FlashVSRModelManifest.resolve(root: folder, settings: settings)
    XCTAssertTrue(files.decoder.path.hasSuffix("Wan2.1_VAE.pth"))
    XCTAssertEqual(files.files.count, 4)
  }
  func testUnknownControlsAreRejectedInsteadOfIgnored() throws {
    let request = VideoUpscaleRequest(sourcePath: "/movie.mp4", runtime: .init())
    var body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    var settings = body["settings"] as! [String: Any]
    settings["steps"] = 8; body["settings"] = settings
    XCTAssertThrowsError(try JSONDecoder().decode(VideoUpscaleRequest.self, from: JSONSerialization.data(withJSONObject: body)))
    body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    body["fps"] = 60
    XCTAssertThrowsError(try JSONDecoder().decode(VideoUpscaleRequest.self, from: JSONSerialization.data(withJSONObject: body)))
  }
  func testLegacyVariantLocalComponentsMatchEngineLookupAndFreezeSymlinkTarget() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let variant = root.appendingPathComponent("turbo")
    try FileManager.default.createDirectory(at: variant, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for name in ["turbo/dit.safetensors", "turbo/lq_proj_in.safetensors", "posi_prompt.pth", "TCDecoder.ckpt", "Wan2.1_VAE.pth",
      "turbo/posi_prompt.pth", "turbo/TCDecoder.ckpt", "turbo/Wan2.1_VAE.pth", "alternate-context"] {
      try Data(name.utf8).write(to: root.appendingPathComponent(name))
    }
    for decoder in [FlashVSRSettings.Decoder.tiny, .vae] {
      var settings = FlashVSRSettings(); settings.backend = .vpipe; settings.decoder = decoder
      let files = try FlashVSRModelManifest.resolve(root: root, settings: settings)
      XCTAssertEqual(files.context.path, variant.appendingPathComponent("posi_prompt.pth").path)
      XCTAssertTrue(files.decoder.path.hasPrefix(variant.path + "/"))
    }
    let link = variant.appendingPathComponent("posi_prompt.pth")
    try FileManager.default.removeItem(at: link)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root.appendingPathComponent("posi_prompt.pth"))
    var settings = FlashVSRSettings(); settings.backend = .vpipe
    let frozen = try FlashVSRModelManifest.resolve(root: root, settings: settings)
    try FileManager.default.removeItem(at: link)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root.appendingPathComponent("alternate-context"))
    XCTAssertThrowsError(try FlashVSRModelManifest.verifyLookup(frozen, settings: settings))
  }
  func testStrictCLIRejectsUnknownAndDuplicateFlags() throws {
    XCTAssertThrowsError(try FlashVSRCLIArguments(["--request", "a", "--output-directory", "b", "--wat"]))
    XCTAssertThrowsError(try FlashVSRCLIArguments(["--request", "a", "--request", "b", "--output-directory", "c"]))
    let args = try FlashVSRCLIArguments(["--request", "a", "--output-directory", "b", "--preflight-only"])
    XCTAssertTrue(args.preflightOnly); XCTAssertFalse(args.resume)
  }
}
