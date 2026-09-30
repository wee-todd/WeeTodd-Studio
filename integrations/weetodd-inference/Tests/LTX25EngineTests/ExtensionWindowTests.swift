import XCTest
@testable import LTX25Engine

final class ExtensionWindowTests: XCTestCase {
  func testContextAndAdditionalFramesDefineOneCausalWindow() throws {
    let window = try LTX25ExtensionWindow(contextFrames: 25, additionalFrames: 96,
      width: 768, height: 448, fps: 24)
    XCTAssertEqual(window.totalFrames, 121)
    XCTAssertEqual(window.contextLatentFrames, 4)
    XCTAssertEqual(window.geometry.latentFrames, 16)
    XCTAssertEqual(window.outputRange, 25..<121)
    XCTAssertEqual(try window.sourceRange(sourceFrames: 241), 216..<241)
    XCTAssertEqual(try window.sourceRange(sourceFrames: 25), 0..<25)
    XCTAssertEqual(window.additionalDuration, 4)
  }

  func testRejectsMisalignedOrOverlongWindowsAndShortSources() throws {
    for (context, additional) in [(24, 96), (25, 95), (1, 96), (249, 96), (25, 0)] {
      XCTAssertThrowsError(try LTX25ExtensionWindow(contextFrames: context,
        additionalFrames: additional, width: 768, height: 448, fps: 24))
    }
    let window = try LTX25ExtensionWindow(contextFrames: 25, additionalFrames: 96,
      width: 768, height: 448, fps: 24)
    XCTAssertThrowsError(try window.sourceRange(sourceFrames: 24))
    XCTAssertThrowsError(try LTX25ExtensionWindow(contextFrames: 25, additionalFrames: 96,
      width: 736, height: 448, fps: 24))
    XCTAssertThrowsError(try LTX25ExtensionWindow(contextFrames: 241, additionalFrames: 96,
      width: 768, height: 448, fps: 8))
  }
}
