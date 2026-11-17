import CoreGraphics
import ImageLayout
import XCTest
@testable import MOSSImageMLX

final class ColorEvidenceTests: XCTestCase {
  func testColorFactsDistinguishGrayscaleAndUniformBorderWithoutModelGuesses() throws {
    func sample(_ r: CGFloat,_ g: CGFloat,_ b: CGFloat) throws -> CGImage {
      let c = try XCTUnwrap(CGContext(data:nil,width:64,height:64,bitsPerComponent:8,bytesPerRow:256,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue))
      c.setFillColor(CGColor(red:r,green:g,blue:b,alpha:1));c.fill(CGRect(x:0,y:0,width:64,height:64));return try XCTUnwrap(c.makeImage())
    }
    XCTAssertGreaterThan(try MOSSColorEvidence.grayscaleFraction(sample(0.5,0.5,0.5)),0.99)
    let olive = try sample(0.47,0.5,0.36)
    XCTAssertLessThan(MOSSColorEvidence.grayscaleFraction(olive),0.1)
    XCTAssertTrue(try XCTUnwrap(MOSSColorEvidence.uniformBorder(olive)).caption.contains("olive"))
  }
}
