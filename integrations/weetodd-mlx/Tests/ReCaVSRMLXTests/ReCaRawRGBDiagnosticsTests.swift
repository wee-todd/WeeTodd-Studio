import XCTest
import Foundation
import CoreMedia
import InferenceMedia
@testable import ReCaVSRMLX

final class ReCaRawRGBDiagnosticsTests: XCTestCase {
  // A dropped, reordered or modified encoder input must fail this capture proof.
  func testCaptureRecordsExactOwningBytesAndRationalClock() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { if FileManager.default.fileExists(atPath:folder.path) { try? FileManager.default.removeItem(at:folder) } }
    let capture = try NativeRGBFrameCapture(directory:folder,width:2,height:2,metadata:["mode":"sync"])
    var first = Array(UInt8(0)...UInt8(11))
    try capture.appendRGB(first,at:CMTime(value:0,timescale:24000),duration:CMTime(value:1001,timescale:24000))
    first[0] = 255 // Capturing cannot retain an alias of mutable caller storage.
    try capture.appendRGB(Array(UInt8(12)...UInt8(23)),at:CMTime(value:1001,timescale:24000),duration:CMTime(value:2002,timescale:24000))
    try capture.finish()
    XCTAssertEqual(try Data(contentsOf:folder.appendingPathComponent("frames.rgb")),Data(Array(UInt8(0)...UInt8(23))))
    let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:folder.appendingPathComponent("manifest.json"))) as? [String:Any])
    XCTAssertEqual(manifest["complete"] as? Bool,true)
    XCTAssertEqual(manifest["frameCount"] as? Int,2)
    XCTAssertEqual(manifest["totalBytes"] as? Int,24)
    let frames = try XCTUnwrap(manifest["frames"] as? [[String:Any]])
    XCTAssertEqual(frames[0]["index"] as? Int,0);XCTAssertEqual(frames[1]["offset"] as? Int,12)
    XCTAssertEqual(frames[0]["sha256"] as? String,"fff3a9bcdd37363d703c1c4f9512533686157868f0d4f16a0f02d0f1da24f9a2")
    XCTAssertEqual(manifest["sha256"] as? String,"1d64add2a6388367c9bc2d1f1b384b069a6ef382cdaaa89771dd103e28613a25")
    XCTAssertEqual(frames[1]["ptsValue"] as? Int,1001);XCTAssertEqual(frames[1]["ptsTimescale"] as? Int,24000)
    XCTAssertEqual(frames[1]["durationValue"] as? Int,2002)
    XCTAssertThrowsError(try capture.appendRGB(Array(repeating:0,count:12),at:CMTime(value:3003,timescale:24000),duration:CMTime(value:1001,timescale:24000)))
    XCTAssertThrowsError(try NativeRGBFrameCapture(directory:folder,width:2,height:2))
  }
  func testCaptureRejectsBadFramesBeforeTheyEnterTheStream() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { if FileManager.default.fileExists(atPath:folder.path) { try? FileManager.default.removeItem(at:folder) } }
    let capture = try NativeRGBFrameCapture(directory:folder,width:2,height:2)
    XCTAssertThrowsError(try capture.appendRGB([0],at:.zero,duration:CMTime(value:1,timescale:24)))
    try capture.appendRGB(Array(repeating:7,count:12),at:.zero,duration:CMTime(value:1,timescale:24))
    XCTAssertThrowsError(try capture.appendRGB(Array(repeating:8,count:12),at:.zero,duration:CMTime(value:1,timescale:24)))
    XCTAssertThrowsError(try capture.appendRGB(Array(repeating:8,count:12),at:CMTime(value:1,timescale:24),duration:.zero))
    try capture.finish()
    XCTAssertEqual(try Data(contentsOf:folder.appendingPathComponent("frames.rgb")).count,12)
  }
  func testSyncControlOverridesOnlyQualifiedAsyncDispatch() {
    XCTAssertFalse(ReCaAttentionScheduling.synchronous.usesAsync(architecture:"applegpu_g15d"))
    XCTAssertTrue(ReCaAttentionScheduling.automatic.usesAsync(architecture:"applegpu_g15d"))
    XCTAssertFalse(ReCaAttentionScheduling.automatic.usesAsync(architecture:"unqualified"))
  }
}
