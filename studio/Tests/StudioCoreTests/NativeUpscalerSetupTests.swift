import Foundation
import XCTest
@testable import StudioCore

final class NativeUpscalerSetupTests: XCTestCase {
  func testCLIRejectsUnknownDuplicateOrIncompleteSetupSettings() throws {
    let options = try NativeUpscalerSetupArguments(["convert", "--profile", "tiny", "--source", "/models/TCDecoder.ckpt", "--model-library", "/models"])
    XCTAssertEqual(options.action, .convert)
    XCTAssertEqual(options.values["--profile"], "tiny")
    XCTAssertThrowsError(try NativeUpscalerSetupArguments(["convert", "--source", "/models/x"]))
    XCTAssertThrowsError(try NativeUpscalerSetupArguments(["download", "--python", "/bin/python"]))
    XCTAssertThrowsError(try NativeUpscalerSetupArguments(["list", "--catalog", "/one", "--catalog", "/two"]))
    XCTAssertThrowsError(try NativeUpscalerSetupArguments(["list", "--catalog"]))
  }
  func testConversionRejectsUnknownProfileAndInvalidSourceBeforePublication() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("untrusted.pth")
    try Data("not an archive".utf8).write(to: source)
    XCTAssertThrowsError(try NativeUpscalerSetup.convert(profile: "arbitrary", source: source, modelLibrary: root))
    XCTAssertThrowsError(try NativeUpscalerSetup.convert(profile: "tiny", source: source, modelLibrary: root))
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["untrusted.pth"])
  }
  func testSetupListsNativePinnedPackagesAndSharedCapability() throws {
    XCTAssertEqual(NativeUpscalerSetup.capabilities.map(\.id).count, 5)
    let catalog = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let packages = try NativeModelDownloads.catalog(at: catalog)
    let native = packages.filter { ["h3-regenerate", "h3-fast-vae", "h3-encoder-source", "h3-ref2va-source", "flashvsr-native"].contains($0.kind) }
    XCTAssertEqual(Set(native.map(\.descriptor.id)), ["h3-regenerate-base-source", "h3-regenerate-publisher", "h3-fast-decoder", "h3-regenerate-encoder-source", "flashvsr-native-turbo-tiny", "flashvsr-native-turbo-full", "flashvsr-native-base-tiny", "flashvsr-native-base-full"])
    XCTAssertTrue(native.allSatisfy { !$0.descriptor.licenseURL.isEmpty && $0.files.allSatisfy { $0.revision.count == 40 && $0.sha256.count == 64 } })
  }
}
