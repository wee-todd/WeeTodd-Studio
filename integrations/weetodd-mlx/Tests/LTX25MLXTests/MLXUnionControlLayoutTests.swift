import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXUnionControlLayoutTests: XCTestCase {
  func testHalfSizeReferenceAlignsCausalTimeAndSpatialCenters() throws {
    let geometry=try AVGeometry(width:256,height:128,frames:17,fps:24)
    let layout=try MLXUnionControlLayout(geometry:geometry,strength:0.75)
    XCTAssertEqual(layout.referenceTokens,geometry.latentFrames*4*2)
    XCTAssertEqual(layout.videoTokens,geometry.videoTokens+layout.referenceTokens)
    let suffix=Array(layout.positions.suffix(layout.referenceTokens*3))
    XCTAssertEqual(Array(suffix.prefix(3)),[Float(0.5/24),32,32])
    XCTAssertEqual(Array(suffix[3..<6]),[Float(0.5/24),32,96])
    XCTAssertEqual(suffix[layout.referenceTokens/geometry.latentFrames*3],geometry.videoPositions[geometry.latentHeight*geometry.latentWidth*3])
    let target=MLXArray.zeros([geometry.videoTokens,128])
    let guide=MLXArray.ones([layout.referenceTokens,128])
    let prepared=try layout.prepare(generated:target,reference:guide)
    XCTAssertEqual(prepared.latent.shape,[layout.videoTokens,128])
    XCTAssertEqual(prepared.condition.mask.last,0.25)
    XCTAssertThrowsError(try layout.prepare(generated:target,reference:target))
  }

  func testRejectsNonHalfSizeCanvasAndInvalidStrength() throws {
    let geometry=try AVGeometry(width:96,height:128,frames:9,fps:24)
    XCTAssertThrowsError(try MLXUnionControlLayout(geometry:geometry,strength:1))
    let valid=try AVGeometry(width:128,height:128,frames:9,fps:24)
    XCTAssertThrowsError(try MLXUnionControlLayout(geometry:valid,strength:.nan))
  }
}
