import Foundation
import ImageIO
import XCTest
@testable import H3MLX

final class H3ImageOutputTests: XCTestCase {
  func testPNGKeepsExactCanvasAndRejectsIncompleteRGB() throws {
    let bytes=try H3ImageOutput.png(rgb:Data([255,0,0,0,255,0]),width:2,height:1)
    let source=try XCTUnwrap(CGImageSourceCreateWithData(bytes as CFData,nil))
    let image=try XCTUnwrap(CGImageSourceCreateImageAtIndex(source,0,nil))
    XCTAssertEqual(image.width,2); XCTAssertEqual(image.height,1)
    XCTAssertEqual(CGImageSourceGetCount(source),1)
    XCTAssertThrowsError(try H3ImageOutput.png(rgb:Data([255,0,0]),width:2,height:1))
  }
}
