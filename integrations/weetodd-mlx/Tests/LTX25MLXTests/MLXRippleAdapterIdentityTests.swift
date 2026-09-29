import Foundation
import XCTest
@testable import LTX25MLX

final class MLXRippleAdapterIdentityTests: XCTestCase {
  func testRejectsWrongSizeBeforeOpeningTransformer() throws {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("not-ripple-\(UUID().uuidString).safetensors")
    try Data("wrong adapter".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    XCTAssertThrowsError(try MLXRippleAdapterIdentity.verify(url))
  }

  func testInstalledPinnedAdapterWhenRequested() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_RIPPLE_ADAPTER"] else {
      throw XCTSkip("Installed Ripple adapter verification is opt-in.")
    }
    try MLXRippleAdapterIdentity.verify(URL(fileURLWithPath: path))
    try MLXRippleAdapterIdentity.verify(URL(fileURLWithPath: path))
  }
}
