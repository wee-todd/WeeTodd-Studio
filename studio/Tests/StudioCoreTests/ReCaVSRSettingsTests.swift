import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

final class ReCaVSRSettingsTests: XCTestCase {
  func testMovieSettingsRoundTripAndResetWithoutChangingMethod() throws {
    var selection = GenerationSelection(task: "video_upscale", preset: .custom)
    selection.movieUpscaleMethod = .reCaVSR
    selection.reCaVSRSettings = try ReCaVSRSettings(scale: 1.5, seed: 1234, precision: .int8ConvRot,
      colorFix: .wavelet, residency: .resident)
    let decoded = try JSONDecoder().decode(GenerationSelection.self, from: JSONEncoder().encode(selection))
    XCTAssertEqual(decoded.reCaVSRSettings, selection.reCaVSRSettings)
    XCTAssertTrue(decoded.isModified)
    selection.resetOverrides()
    XCTAssertNil(selection.reCaVSRSettings)
    XCTAssertEqual(selection.movieUpscaleMethod, .reCaVSR)
  }
  func testSwiftCatalogRecognizesBothPinnedReCaPackages() throws {
    let catalog = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let packages = try NativeModelDownloads.catalog(at: catalog).filter { $0.kind == "recavsr-native" }
    XCTAssertEqual(Set(packages.map(\.descriptor.id)), ["recavsr-native-bf16", "recavsr-native-int8"])
    for package in packages {
      XCTAssertEqual(package.files.count, 6)
      XCTAssertTrue(package.files.allSatisfy { $0.revision.count == 40 && $0.sha256.count == 64 })
      XCTAssertEqual(package.files.filter { $0.target.hasPrefix("shared/") }.count, 5)
    }
  }
}
