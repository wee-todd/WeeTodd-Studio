import Foundation
import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRDecoderCarryStoreTests:XCTestCase {
  func testTileCarryRoundTripIsAtomicAndOwnerCleanupReleasesOnlyItsFiles()throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    try Data([9]).write(to:root.appendingPathComponent("unrelated"))
    var store:FlashVSRDecoderCarryStore?=try .init(parent:root,sourceID:"clip",modelIdentity:"pinned-wan")
    let f32=MLXArray([Float(1),-2,3,4],[1,2,1,1,2]),bf16=MLXArray([Float(0.5),2],[1,1,1,1,2]).asType(.bfloat16)
    try store!.save(key:"0,0",caches:["a":f32,"b":bf16],upsampleStarted:["up3","up7"])
    let loaded=try store!.load(key:"0,0")
    XCTAssertEqual(loaded.caches["a"]?.asArray(Float.self),[1,-2,3,4])
    XCTAssertEqual(loaded.caches["b"]?.dtype,.bfloat16)
    XCTAssertEqual(loaded.caches["b"]?.asArray(Float.self),[0.5,2])
    XCTAssertEqual(loaded.upsampleStarted,["up3","up7"])
    try store!.save(key:"0,0",caches:["a":f32],upsampleStarted:[])
    XCTAssertEqual(try store!.load(key:"0,0").caches.count,1)
    XCTAssertFalse(try FileManager.default.subpathsOfDirectory(atPath:root.path).contains { $0.hasSuffix(".partial") })
    let stateFile=try XCTUnwrap(FileManager.default.subpathsOfDirectory(atPath:root.path).first { $0.hasSuffix(".state") })
    try Data([0]).write(to:root.appendingPathComponent(stateFile))
    XCTAssertThrowsError(try store!.load(key:"0,0"))
    store=nil
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["unrelated"])
  }
}
