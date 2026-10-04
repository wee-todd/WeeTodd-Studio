import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXIngredientsGuideTests:XCTestCase {
  func testStaticGuideStoresOneFrameAndRetainsExactQuantizedPixels() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:121,fps:24)
    let output=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".rgb")
    defer { try? FileManager.default.removeItem(at:output) }
    let pixels:[Float]=Array(repeating:[Float(-1),Float(0),Float(1)],count:64*64).flatMap { $0 }
    try MLXIngredientsGuide.writeStaticRGB(pixels:pixels,geometry:geometry,to:output)
    let bytes=try Data(contentsOf:output)
    XCTAssertEqual(bytes.count,64*64*3)
    XCTAssertEqual(Array(bytes.prefix(3)),[0,128,255])
    XCTAssertEqual(Array(bytes.suffix(3)),[0,128,255])
    XCTAssertThrowsError(try MLXIngredientsGuide.writeStaticRGB(pixels:pixels,geometry:geometry,to:output))
  }
  func testStaticLatentRepeatsEveryFrameWithoutTemporalMixing() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:64,height:64,frames:121,fps:24)
      let values=(0..<(2*2*128)).map { Float($0-200)/137 }
      let frame=MLXArray(values,[1,2,2,128])
      let result=try MLXIngredientsGuide.repeatStaticLatent(frame,geometry:g)
      XCTAssertEqual(result.shape,[g.videoTokens,128])
      let actual=result.reshaped([g.latentFrames,2*2*128]).asArray(Float.self)
      for index in 0..<g.latentFrames {
        XCTAssertEqual(Array(actual[(index*values.count)..<((index+1)*values.count)]),values)
      }
      XCTAssertThrowsError(try MLXIngredientsGuide.repeatStaticLatent(.zeros([2,2,2,128]),geometry:g))
      XCTAssertThrowsError(try MLXIngredientsGuide.repeatStaticLatent(frame.asType(.bfloat16),geometry:g))
    }
  }
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
