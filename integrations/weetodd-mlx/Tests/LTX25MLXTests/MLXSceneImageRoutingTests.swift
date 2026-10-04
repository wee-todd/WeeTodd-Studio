import Foundation
import XCTest
import LTX25Engine
@testable import LTX25MLX

final class MLXSceneImageRoutingTests:XCTestCase {
  private func anchor(_ frame:Int) throws -> MLXSceneImageRouting.Anchor {
    try .init(id:"image-\(frame)",path:"/image-\(frame).png",frame:frame,strength:0.8)
  }
  func testBalancedFirstOwnerAndStrictOverlapKeepUnsnappedDeliveredLast() throws {
    let plan=try LTX25ScenePlan(durations:[2,3],fps:24)
    let anchors=try [119,49,48,0,47].map(anchor)
    let balanced=try MLXSceneImageRouting(plan:plan,anchors:anchors,policy:.balanced,width:512,height:256)
    XCTAssertEqual(balanced.anchors.map(\.frame),[0,47,48,49,119])
    XCTAssertEqual(balanced.windows.map { $0.map(\.frame) },[[0,47,48],[25,95]])
    XCTAssertEqual(balanced.windows[0][2].anchor.frame,48)
    XCTAssertEqual(balanced.windows[1][1].anchor.frame,plan.totalFrames-2)
    XCTAssertEqual(balanced.retainedConditioningBytes,5*160*131*4)
    let strict=try MLXSceneImageRouting(plan:plan,anchors:anchors,policy:.strict,width:512,height:256)
    XCTAssertEqual(strict.windows.map { $0.map(\.frame) },[[0,47,48],[23,24,25,95]])
    XCTAssertEqual(strict.retainedConditioningBytes,7*160*131*4)
  }
  func testThirtyTwoGlobalInputsBoundsDuplicatesAndCausalTailBeforeAllocation() throws {
    let plan=try LTX25ScenePlan(durations:[2,3],fps:24),anchors=try (0..<32).map(anchor)
    XCTAssertEqual(try MLXSceneImageRouting(plan:plan,anchors:anchors,policy:.balanced,width:512,height:256).anchors.count,32)
    XCTAssertThrowsError(try MLXSceneImageRouting(plan:plan,anchors:anchors+[anchor(32)],policy:.balanced,width:512,height:256))
    XCTAssertThrowsError(try MLXSceneImageRouting(plan:plan,anchors:[anchor(plan.totalFrames-1)],policy:.balanced,width:512,height:256))
    XCTAssertThrowsError(try MLXSceneImageRouting(plan:plan,anchors:[anchor(7),anchor(7)],policy:.balanced,width:512,height:256))
    XCTAssertThrowsError(try MLXSceneImageRouting(plan:plan,anchors:anchors,policy:.strict,width:4096,height:4096))
    XCTAssertThrowsError(try MLXSceneImageRouting.Anchor(id:"x",path:"/a\0.png",frame:7,strength:1))
    XCTAssertThrowsError(try MLXSceneImageRouting.Anchor(id:"x",path:"/a.png",frame:7,strength:.nan))
  }
}
