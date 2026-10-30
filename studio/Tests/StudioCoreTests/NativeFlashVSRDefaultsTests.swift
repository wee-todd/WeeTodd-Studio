import Foundation
import XCTest
@testable import StudioCore

final class NativeFlashVSRDefaultsTests: XCTestCase {
  func testNewRequestsDefaultNativeAndOldMissingBackendRetainsLegacyIdentity() throws {
    XCTAssertEqual(FlashVSRSettings().backend, .swiftMLX)
    XCTAssertEqual(VideoUpscaleRequest(sourcePath: "/source.mov", runtime: .init()).settings.backend, .swiftMLX)
    let historical = try JSONDecoder().decode(FlashVSRSettings.self, from: Data("{\"version\":1}".utf8))
    XCTAssertEqual(historical.backend, .vpipe)
    let explicit = try JSONDecoder().decode(FlashVSRSettings.self, from: Data("{\"version\":1,\"backend\":\"swiftMLX\"}".utf8))
    XCTAssertEqual(explicit.backend, .swiftMLX)
  }
}
