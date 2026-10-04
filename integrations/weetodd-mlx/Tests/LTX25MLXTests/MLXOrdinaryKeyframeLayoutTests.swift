import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXOrdinaryKeyframeLayoutTests:XCTestCase {
  func testOrderedAnchorsReplaceOnlyFrameZeroThenSlotsFollowWithGlobalAttention() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:64,height:32,frames:17,fps:24)
      let layout=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[
        .init(frame:7,strength:0.5),.init(frame:0,strength:1),.init(frame:16,strength:0.25)],generatedCount:3)
      XCTAssertEqual(layout.generatedFrames,[4,8,12]);XCTAssertEqual(layout.frameTokens,2)
      XCTAssertEqual(layout.anchorTokens,4);XCTAssertEqual(layout.slotTokens,6);XCTAssertEqual(layout.videoTokens,16)
      XCTAssertEqual(layout.videoAttentionGroups,[]) // Plain primary keyframes do not request isolation.
      let generated=MLXArray(Array(repeating:Float(2),count:6*128),[6,128])
      let anchors=[Float(7),Float(10),Float(16)].map { MLXArray(Array(repeating:$0,count:256),[2,128]) }
      let slots=MLXArray(Array(repeating:Float(9),count:6*128),[6,128])
      let prepared=try layout.prepare(generated:generated,anchors:anchors,initialSlots:slots)
      XCTAssertEqual(prepared.latent[0...,0].asArray(Float.self),[10,10,2,2,2,2,7,7,16,16,9,9,9,9,9,9])
      XCTAssertEqual(prepared.condition.clean[0...,0].asArray(Float.self),[10,10,0,0,0,0,7,7,16,16,0,0,0,0,0,0])
      XCTAssertEqual(prepared.condition.mask,[0,0,1,1,1,1,0.5,0.5,0.75,0.75,1,1,1,1,1,1])
      XCTAssertEqual(Array(layout.positions.prefix(g.videoTokens*3)),g.videoPositions)
      XCTAssertEqual(Array(layout.positions[(g.videoTokens*3)..<(g.videoTokens*3+6)]),[Float(7.5/24),16,16,Float(7.5/24),16,48])
      XCTAssertEqual(layout.positions[(layout.videoTokens-layout.slotTokens)*3],Float(4.5/24))
      // MLX wrapper mutation must not rewrite prepared snapshots or the supplied canvas.
      anchors[1][0,0]=MLXArray(Float(99));slots[0,0]=MLXArray(Float(88))
      XCTAssertEqual(prepared.latent[0,0].item(Float.self),10)
      XCTAssertEqual(prepared.latent[10,0].item(Float.self),9)
      let clean=prepared.condition.clean;clean[0,0]=MLXArray(Float(77))
      XCTAssertEqual(prepared.condition.clean[0,0].item(Float.self),10)
      XCTAssertEqual(generated[0,0].item(Float.self),2)
    }
  }
  func testEveryGeneratedCountUsesNearestEvenInteriorFramesAndStageTwoHasNoSlots() throws {
    let g=try AVGeometry(width:32,height:32,frames:17,fps:24)
    for count in 0...8 {
      let layout=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[],generatedCount:count)
      let expected=(1...max(1,count)).prefix(count).map { Int((Double($0)*16/Double(count+1)).rounded(.toNearestOrEven)) }
      XCTAssertEqual(layout.generatedFrames,expected);XCTAssertEqual(Set(expected).count,count)
      XCTAssertTrue(expected.allSatisfy { $0>0 && $0<16 })
    }
    XCTAssertEqual(try MLXOrdinaryKeyframeLayout.interiorFrames(count:1,frames:4),[2])
    XCTAssertEqual(try MLXOrdinaryKeyframeLayout.interiorFrames(count:1,frames:6),[2])
    let stageTwo=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[.init(frame:7,strength:1)],generatedCount:0)
    XCTAssertEqual(stageTwo.slotTokens,0);XCTAssertEqual(stageTwo.videoTokens,g.videoTokens+1)
    for count in 1...8 {
      XCTAssertNoThrow(try MLXOrdinaryKeyframeLayout(geometry:g,
        anchors:(0..<count).map { .init(frame:$0,strength:1) },generatedCount:8))
    }
  }
  func testNoFrameZeroAnchorDoesNotReplaceAnAlignedInteriorPlane() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:32,height:32,frames:17,fps:24)
      let layout=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[.init(frame:8,strength:1)],generatedCount:0)
      let result=try layout.prepare(generated:.ones([3,128]),anchors:[.ones([1,128])*4])
      XCTAssertEqual(result.latent[0...,0].asArray(Float.self),[1,1,1,4])
      XCTAssertEqual(result.condition.mask,[1,1,1,0]);XCTAssertEqual(layout.slotTokens,0)
      XCTAssertEqual(layout.positions[9],Float(8.5/24))
    }
  }
  func testMetadataAndTokenBoundsRejectBeforeArrayWork() throws {
    let g=try AVGeometry(width:32,height:32,frames:17,fps:24)
    for count in [-1,9] { XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[],generatedCount:count)) }
    for anchors:[MLXOrdinaryKeyframeLayout.Anchor] in [
      [.init(frame:-1,strength:1)],[.init(frame:17,strength:1)],[.init(frame:0,strength:.nan)],
      [.init(frame:0,strength:1.1)],[.init(frame:1,strength:1),.init(frame:1,strength:0)],
      (0...8).map { .init(frame:$0,strength:1) }] {
      XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:g,anchors:anchors,generatedCount:0))
    }
    XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:AVGeometry(width:32,height:32,frames:1,fps:24),anchors:[],generatedCount:1))
    let limit=try AVGeometry(width:2048,height:2048,frames:249,fps:24)
    XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:limit,anchors:[.init(frame:1,strength:1)],generatedCount:0))
  }
  func testMalformedLatentsAndNoisySlotCleanState() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:32,height:32,frames:9,fps:24)
      let layout=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[.init(frame:3,strength:0.25)],generatedCount:1)
      let video=MLXArray.ones([2,128]),anchor=MLXArray.ones([1,128])
      XCTAssertThrowsError(try layout.prepare(generated:video,anchors:[]))
      XCTAssertThrowsError(try layout.prepare(generated:video.asType(.bfloat16),anchors:[anchor]))
      XCTAssertThrowsError(try layout.prepare(generated:video,anchors:[anchor*Float.nan]))
      XCTAssertThrowsError(try layout.prepare(generated:video,anchors:[anchor],initialSlots:.zeros([2,128])))
      let result=try layout.prepare(generated:video,anchors:[anchor])
      XCTAssertEqual(result.latent[3,0].item(Float.self),0);XCTAssertEqual(result.condition.mask[3],1)
      XCTAssertEqual(result.condition.clean[3,0].item(Float.self),0)
    }
  }
  func testCanceledTaskStopsBeforeArrayWork() async {
    let canceled=await Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        let g=try AVGeometry(width:32,height:32,frames:9,fps:24)
        _=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[],generatedCount:1)
      } catch is CancellationError { return true } catch { return false }
      return false
    }.value
    XCTAssertTrue(canceled)
  }
}
