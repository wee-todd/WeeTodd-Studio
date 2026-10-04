import Foundation
import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXSceneKeyframeLayoutTests:XCTestCase {
  func testDrivenFirstWindowFreezesAudioAndLaterWindowRetainsHistoryMasksOnCPU() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:128,height:128,frames:33,fps:24)
      let driver=MLXArray((0..<g.audioFrames*128).map {Float($0%17)/13},[g.audioFrames,128])
      let first=try MLXSceneKeyframeLayout(geometry:g,anchors:[.init(frame:7,strength:1)])
      let condition=try first.sourceAudioConditionForHistory(driver)
      XCTAssertNil(condition)
      let prepared=try first.prepare(generated:MLXArray.zeros([g.videoTokens,128]),
        anchors:[MLXArray.ones([g.latentHeight*g.latentWidth,128])],targetAudio:driver,
        targetAudioCondition:condition,sigma:1)
      XCTAssertNil(prepared.audioCondition)
      XCTAssertEqual(prepared.audio.asArray(Float.self),driver.asArray(Float.self))
      let history=try MLXExtensionGuideLayout(geometry:g,contextFrames:9,videoGuideLatentFrames:1,audioGuideTokens:4)
      let next=try MLXSceneKeyframeLayout(geometry:g,anchors:[],extensionGuide:history)
      let nextCondition=try next.sourceAudioConditionForHistory(driver)
      XCTAssertEqual(nextCondition?.mask,Array(repeating:0,count:g.audioFrames))
      let continued=try next.prepare(generated:MLXArray.zeros([g.videoTokens,128]),anchors:[],
        targetAudio:driver,targetAudioCondition:nextCondition,
        guides:.init(video:MLXArray.ones([history.videoGuideTokens,128]),audio:MLXArray.ones([4,128]),
          videoNoise:MLXArray.zeros([history.videoGuideTokens,128]),audioNoise:MLXArray.zeros([4,128])),sigma:1)
      XCTAssertEqual(continued.audioCondition?.mask,Array(repeating:0,count:g.audioFrames)+Array(repeating:0.5,count:4))
      XCTAssertEqual(continued.audio[0..<g.audioFrames].asArray(Float.self),driver.asArray(Float.self))
      XCTAssertNil(try next.sourceAudioConditionForHistory(nil))
      XCTAssertThrowsError(try first.sourceAudioConditionForHistory(MLXArray.zeros([1,128])))
    }
  }
  func testHistoryPrecedesExplicitImagesWithExactMasksPositionsAndBlendOnCPU() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:128,height:128,frames:33,fps:24),plane=g.latentHeight*g.latentWidth
      let history=try MLXExtensionGuideLayout(geometry:g,contextFrames:9,videoGuideLatentFrames:1,audioGuideTokens:4)
      let layout=try MLXSceneKeyframeLayout(geometry:g,anchors:[.init(frame:0,strength:1),
        .init(frame:7,strength:0.75),.init(frame:32,strength:0.5)],extensionGuide:history)
      func filled(_ rows:Int,_ value:Float) -> MLXArray { MLXArray(Array(repeating:value,count:rows*128),[rows,128]) }
      let images=[filled(plane,10),filled(plane,20),filled(plane,30)]
      let audio=filled(g.audioFrames,0)
      let result=try layout.prepare(generated:filled(g.videoTokens,2),anchors:images,targetAudio:audio,
        targetAudioCondition:MLXAudioDenoiseCondition(clean:audio,mask:Array(repeating:0,count:g.audioFrames)),
        guides:.init(video:filled(plane,3),audio:filled(4,7),videoNoise:filled(plane,5),audioNoise:filled(4,9)),sigma:0.8)
      XCTAssertEqual(result.video.shape,[g.videoTokens+plane*3,128])
      XCTAssertEqual(layout.videoAttentionGroups,[])
      let rows=result.video[0...,0].asArray(Float.self)
      XCTAssertEqual(Array(rows.prefix(plane)),Array(repeating:10,count:plane))
      XCTAssertEqual(Array(rows[plane..<g.videoTokens]),Array(repeating:2,count:g.videoTokens-plane))
      for x in rows[g.videoTokens..<g.videoTokens+plane] { XCTAssertEqual(x,3.8,accuracy:0.000001) }
      XCTAssertEqual(Array(rows[(g.videoTokens+plane)..<(g.videoTokens+plane*2)]),Array(repeating:20,count:plane))
      XCTAssertEqual(Array(rows.suffix(plane)),Array(repeating:30,count:plane))
      XCTAssertEqual(result.videoCondition.mask,Array(repeating:0,count:plane)+Array(repeating:1,count:g.videoTokens-plane)+Array(repeating:0.5,count:plane)+Array(repeating:0.25,count:plane)+Array(repeating:0.5,count:plane))
      XCTAssertEqual(result.audioCondition?.mask,Array(repeating:0,count:g.audioFrames)+Array(repeating:0.5,count:4))
      XCTAssertEqual(layout.videoPositions.count,layout.videoTokens*3)
      XCTAssertEqual(Array(layout.videoPositions[(g.videoTokens*3)..<(g.videoTokens*3+plane*3)]),Array(g.videoPositions.prefix(plane*3)))
      XCTAssertEqual(layout.videoPositions[(g.videoTokens+plane)*3],Float(7.5/24),accuracy:0.000001)
      XCTAssertEqual(layout.videoPositions[(g.videoTokens+plane*2)*3],Float(32.5/24),accuracy:0.000001)
      images[1][0..<plane] = filled(plane,99)
      XCTAssertEqual(result.video[g.videoTokens+plane,0].item(Float.self),20)
    }
  }
  func testSceneOnlyThirtyTwoAdmissionDoesNotExpandOrdinaryPublicEight() throws {
    let g=try AVGeometry(width:128,height:128,frames:33,fps:24)
    let anchors=(1...32).map { MLXOrdinaryKeyframeLayout.Anchor(frame:$0,strength:1) }
    XCTAssertNoThrow(try MLXSceneKeyframeLayout(geometry:g,anchors:anchors))
    XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:g,anchors:Array(anchors.prefix(9)),generatedCount:0))
    XCTAssertThrowsError(try MLXOrdinaryKeyframeLayout(geometry:g,anchors:[],generatedCount:0,maximumAnchors:33))
    let other=try AVGeometry(width:128,height:128,frames:41,fps:24)
    XCTAssertThrowsError(try MLXSceneKeyframeLayout(geometry:g,anchors:[],extensionGuide:MLXExtensionGuideLayout(geometry:other,contextFrames:9)))
  }
  func testMissingOrForeignHistoryAndNonfiniteInputsRejectOnCPU() throws {
    try Device.withDefaultDevice(.cpu) {
      let g=try AVGeometry(width:128,height:128,frames:33,fps:24)
      let guide=try MLXExtensionGuideLayout(geometry:g,contextFrames:9)
      let layout=try MLXSceneKeyframeLayout(geometry:g,anchors:[],extensionGuide:guide)
      let video=MLXArray.zeros([g.videoTokens,128]),audio=MLXArray.zeros([g.audioFrames,128])
      XCTAssertThrowsError(try layout.prepare(generated:video,anchors:[],targetAudio:audio,sigma:1))
      let plain=try MLXSceneKeyframeLayout(geometry:g,anchors:[])
      let result=try plain.prepare(generated:video,anchors:[],targetAudio:audio,sigma:1)
      XCTAssertEqual(result.video.shape,video.shape);XCTAssertNil(result.audioCondition)
      XCTAssertThrowsError(try plain.prepare(generated:video,anchors:[],targetAudio:MLXArray([Float.nan],[1,1]),sigma:1))
      XCTAssertThrowsError(try plain.prepare(generated:video,anchors:[],targetAudio:audio,
        guides:.init(video:video,audio:audio,videoNoise:video,audioNoise:audio),sigma:1))
    }
  }
  func testCancelledAdmissionDoesNotCreateSceneArrays() async {
    let task=Task { () throws -> Void in
      withUnsafeCurrentTask { $0?.cancel() }
      let g=try AVGeometry(width:128,height:128,frames:33,fps:24)
      _ = try MLXSceneKeyframeLayout(geometry:g,anchors:[])
    }
    do { try await task.value;XCTFail("Cancelled scene admission was accepted") }
    catch { XCTAssertTrue(error is CancellationError) }
  }
}
