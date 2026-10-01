import XCTest
import LTX25Engine
@testable import LTX25MLX

final class MLXIngredientsGuideTests:XCTestCase {
  func testOneSheetFillsTheCompleteCausalGuideWithoutFrameDrift() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:121,fps:24)
    let output=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".rgb")
    defer { try? FileManager.default.removeItem(at:output) }
    let pixels=[Float](repeating:-1,count:64*64*3)
    try MLXIngredientsGuide.writeRepeatedRGB(pixels:pixels,geometry:geometry,to:output)
    let bytes=try Data(contentsOf:output)
    XCTAssertEqual(bytes.count,121*64*64*3)
    XCTAssertEqual(bytes.prefix(3),Data([0,0,0]))
    XCTAssertEqual(bytes.suffix(3),Data([0,0,0]))
    XCTAssertThrowsError(try MLXIngredientsGuide.writeRepeatedRGB(pixels:pixels,
      geometry:geometry,to:output))
  }
}
