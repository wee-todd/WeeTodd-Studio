import Foundation
import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXSingleStageControlLayoutTests:XCTestCase {
  private func filled(_ rows:Int,_ value:Float) -> MLXArray {
    MLXArray(Array(repeating:value,count:rows*128),[rows,128])
  }
  private func control(_ family:String) throws -> MLXICControl {
    let roles=family == "motion_track" ? ["control"]:["warp","source"]
    let object:[String:Any]=["family":family,
      "adapters":[["family":family,"path":"/adapter.safetensors","strength":1]],
      "guides":roles.enumerated().map { ["path":"/guide-\($0.offset).mp4","source_sha256":String(repeating:"a",count:64),"role":$0.element,"strength":$0.offset == 0 ? 0.5:0.25] as [String:Any] },
      "publication_audio":family == "motion_track" ? NSNull() as Any:
        ["path":"/source.mp4","source_sha256":String(repeating:"b",count:64),"source_start_seconds":0,"source_duration_seconds":1] as [String:Any]]
    return try JSONDecoder().decode(MLXICControl.self,from:JSONSerialization.data(withJSONObject:object))
  }
  func testUnionOrderCleanMasksPositionsTrailingMarkerAndSnapshotIndependence() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:64,height:64,frames:17,fps:24)
      let layout=try MLXSingleStageControlLayout(geometry:g,anchors:[.init(frame:7,strength:0.75),.init(frame:0,strength:1)],generatedCount:2,unionStrength:0.5)
      XCTAssertEqual(layout.generatedFrames,[5,11]);XCTAssertEqual(layout.guideTokens,3)
      XCTAssertEqual(layout.slotTokens,8);XCTAssertEqual(layout.videoTokens,27)
      XCTAssertEqual(layout.videoAttentionGroups,[])
      let generated=filled(12,2),anchors=[filled(4,7),filled(4,10)],guides=[filled(3,20)],slots=filled(8,9)
      let result=try layout.prepare(generated:generated,anchors:anchors,guides:guides,initialSlots:slots)
      let rows=result.latent[0...,0].asArray(Float.self)
      XCTAssertEqual(rows,Array(repeating:10,count:4)+Array(repeating:2,count:8)+Array(repeating:7,count:4)+Array(repeating:20,count:3)+Array(repeating:9,count:8))
      XCTAssertEqual(result.condition.clean[0...,0].asArray(Float.self),Array(repeating:10,count:4)+Array(repeating:0,count:8)+Array(repeating:7,count:4)+Array(repeating:20,count:3)+Array(repeating:0,count:8))
      XCTAssertEqual(result.condition.mask,Array(repeating:0,count:4)+Array(repeating:1,count:8)+Array(repeating:0.25,count:4)+Array(repeating:0.5,count:3)+Array(repeating:1,count:8))
      XCTAssertEqual(Array(layout.positions[36..<39]),[Float(7.5/24),16,16])
      XCTAssertEqual(Array(layout.positions[48..<51]),[Float(0.5/24),32,32])
      XCTAssertEqual(Array(layout.positions[57..<60]),[Float(5.5/24),16,16])
      guides[0][0,0]=MLXArray(Float(99));anchors[1][0,0]=MLXArray(Float(88));slots[0,0]=MLXArray(Float(77))
      XCTAssertEqual(result.latent[16,0].item(Float.self),20)
      XCTAssertEqual(result.latent[0,0].item(Float.self),10)
      XCTAssertEqual(result.latent[19,0].item(Float.self),9)
      let exposed=result.condition.clean;exposed[16,0]=MLXArray(Float(66))
      XCTAssertEqual(result.condition.clean[16,0].item(Float.self),20)
      XCTAssertEqual(generated[0,0].item(Float.self),2)
    }
  }
  func testOrderedICGuidesRetainFullGridOrMotionDownscale() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:64,height:64,frames:17,fps:24)
      let cross=try MLXSingleStageControlLayout(geometry:g,anchors:[.init(frame:8,strength:1)],generatedCount:1,icControl:control("crossview_warp"))
      let result=try cross.prepare(generated:filled(12,2),anchors:[filled(4,8)],guides:[filled(12,30),filled(12,40)])
      XCTAssertEqual(cross.videoTokens,44);XCTAssertEqual(cross.guideTokens,24)
      XCTAssertEqual(result.latent[16..<40,0].asArray(Float.self),Array(repeating:30,count:12)+Array(repeating:40,count:12))
      XCTAssertEqual(Array(result.condition.mask[16..<40]),Array(repeating:0.5,count:12)+Array(repeating:0.75,count:12))
      XCTAssertEqual(Array(cross.positions[48..<84]),g.videoPositions)
      XCTAssertEqual(result.latent[40,0].item(Float.self),0)
      let motion=try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:0,icControl:control("motion_track"))
      XCTAssertEqual(motion.guideTokens,3);XCTAssertEqual(motion.videoTokens,15)
      XCTAssertEqual(Array(motion.positions[36..<39]),[Float(0.5/24),32,32])
      XCTAssertNoThrow(try motion.prepare(generated:filled(12,2),anchors:[],guides:[filled(3,5)]))
    }
  }
  func testPlainCompositionMatchesOrdinaryIncludingAllCounts() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:32,height:32,frames:17,fps:24)
      for count in 0...8 {
        let anchors=[MLXOrdinaryKeyframeLayout.Anchor(frame:7,strength:0.5)]
        let ordinary=try MLXOrdinaryKeyframeLayout(geometry:g,anchors:anchors,generatedCount:count)
        let composite=try MLXSingleStageControlLayout(geometry:g,anchors:anchors,generatedCount:count)
        let a=try ordinary.prepare(generated:filled(3,2),anchors:[filled(1,4)])
        let b=try composite.prepare(generated:filled(3,2),anchors:[filled(1,4)],guides:[])
        XCTAssertEqual(a.latent.asArray(Float.self),b.latent.asArray(Float.self))
        XCTAssertEqual(a.condition.clean.asArray(Float.self),b.condition.clean.asArray(Float.self))
        XCTAssertEqual(a.condition.mask,b.condition.mask);XCTAssertEqual(ordinary.positions,composite.positions)
      }
    }
  }
  func testMutuallyExclusiveControlsAndCombinedBoundsRejectBeforeArrayWork() throws {
    let g=try AVGeometry(width:64,height:64,frames:17,fps:24)
    XCTAssertThrowsError(try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:0,icControl:control("motion_track"),unionStrength:1))
    XCTAssertThrowsError(try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:0,unionStrength:.nan))
    XCTAssertThrowsError(try MLXSingleStageControlLayout(geometry:g,anchors:(0...8).map { .init(frame:$0,strength:1) },generatedCount:0))
    XCTAssertThrowsError(try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:9))
    let large=try AVGeometry(width:2048,height:2048,frames:249,fps:24)
    XCTAssertThrowsError(try MLXSingleStageControlLayout(geometry:large,anchors:[],generatedCount:0,unionStrength:1))
  }
  func testMalformedOrNonfiniteGuideAndSlotInputsReject() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:64,height:64,frames:17,fps:24)
      let layout=try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:1,unionStrength:1)
      for guides in [[],[filled(4,1)],[filled(3,.nan)],[filled(3,1).asType(.bfloat16)]] {
        XCTAssertThrowsError(try layout.prepare(generated:filled(12,2),anchors:[],guides:guides))
      }
      XCTAssertThrowsError(try layout.prepare(generated:filled(12,2),anchors:[],guides:[filled(3,1)],initialSlots:filled(3,1)))
      let plain=try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:0)
      XCTAssertThrowsError(try plain.prepare(generated:filled(12,2),anchors:[],guides:[filled(3,1)]))
    }
  }
  func testCancellationStopsBeforeArrayPreparation() async {
    let cancelled=await Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        let g=try AVGeometry(width:64,height:64,frames:17,fps:24)
        _=try MLXSingleStageControlLayout(geometry:g,anchors:[],generatedCount:0)
      } catch is CancellationError { return true } catch { return false }
      return false
    }.value
    XCTAssertTrue(cancelled)
  }
}
