import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import StudioCore

final class H3ImageResultTests: XCTestCase {
  func testOnlyMatchingCompletePNGInsideCapturedOutputIsAccepted() throws {
    let output=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:output) }
    let context=try XCTUnwrap(CGContext(data:nil,width:32,height:32,bitsPerComponent:8,bytesPerRow:128,
      space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue))
    let png=NSMutableData();let destination=try XCTUnwrap(CGImageDestinationCreateWithData(png,"public.png" as CFString,1,nil))
    CGImageDestinationAddImage(destination,try XCTUnwrap(context.makeImage()),nil)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    try (png as Data).write(to:output.appendingPathComponent("image.png"))
    let requestID=UUID(),fingerprint="sha256:"+String(repeating:"a",count:64)
    var result:[String:Any]=["nativeRuntime":"swift-mlx","jobID":UUID().uuidString,
      "requestID":requestID.uuidString,"fingerprint":fingerprint,
      "asset":["path":output.appendingPathComponent("image.png").path,"width":32,"height":32],
      "normalizedRequest":["configuration":["width":32,"height":32,"seed":42]]]
    func check() throws -> URL { try NativeH3ImageResult.image(result:result,output:output,
      fingerprint:fingerprint,requestID:requestID,seed:42,width:32,height:32) }
    XCTAssertEqual(try check().lastPathComponent,"image.png")
    result["requestID"]=UUID().uuidString
    XCTAssertThrowsError(try check())
    result["requestID"]=requestID.uuidString
    result["asset"]=["path":"/tmp/outside.png","width":32,"height":32]
    XCTAssertThrowsError(try check())
    result["asset"]=["path":output.appendingPathComponent("image.png").path,"width":64,"height":32]
    XCTAssertThrowsError(try check())
  }
}
