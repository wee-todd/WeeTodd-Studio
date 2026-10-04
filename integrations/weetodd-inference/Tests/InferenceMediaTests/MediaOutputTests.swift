import Foundation
import CoreGraphics
import ImageIO
import XCTest
@testable import InferenceMedia

final class MediaOutputTests: XCTestCase {
  func testRGB24PNGPreservesQuantizedPixelsAndRejectsOverwrite() throws {
    let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:directory) }
    let url=directory.appendingPathComponent("frame.png")
    let pixels=Data([1,2,3,255,128,0])
    try MediaOutput.writeRGB8PNG(pixels,width:2,height:1,to:url)
    let source=try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL,nil))
    let image=try XCTUnwrap(CGImageSourceCreateImageAtIndex(source,0,nil))
    XCTAssertEqual(image.width,2);XCTAssertEqual(image.height,1)
    var decoded=[UInt8](repeating:0,count:8)
    try decoded.withUnsafeMutableBytes { bytes in
      let space=try XCTUnwrap(CGColorSpace(name:CGColorSpace.sRGB))
      let context=try XCTUnwrap(CGContext(data:bytes.baseAddress,width:2,height:1,
        bitsPerComponent:8,bytesPerRow:8,space:space,
        bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue|CGBitmapInfo.byteOrder32Big.rawValue))
      context.draw(image,in:CGRect(x:0,y:0,width:2,height:1))
    }
    XCTAssertEqual(decoded,[1,2,3,255,255,128,0,255])
    XCTAssertThrowsError(try MediaOutput.writeRGB8PNG(pixels,width:2,height:1,to:url))
    XCTAssertThrowsError(try MediaOutput.writeRGB8PNG(pixels,width:3,height:1,
      to:directory.appendingPathComponent("invalid.png")))
  }
  func testStereoWAVRetainsNativeH3AndLTXSampleRates() throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-wav-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    for rate in [32_000, 48_000] {
      let url = directory.appendingPathComponent("\(rate).wav")
      try MediaOutput.writeWAV(samples: [0.25, 0.5, -0.25, -0.5],
        sampleRate: rate, channels: 2, to: url)
      let bytes = try Data(contentsOf: url)
      XCTAssertEqual(bytes.count, 74)
      XCTAssertEqual(Array(bytes[24..<28]), withUnsafeBytes(of: UInt32(rate).littleEndian,
        Array.init))
      XCTAssertEqual(Array(bytes[28..<32]), withUnsafeBytes(of: UInt32(rate * 8).littleEndian,
        Array.init))
    }
  }
}
