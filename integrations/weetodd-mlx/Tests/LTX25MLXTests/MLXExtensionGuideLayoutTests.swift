import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXExtensionGuideLayoutTests:XCTestCase {
  func testTimedImageAndCarriedAudiovisualHistoryKeepBothMasks() throws {
    let geometry=try AVGeometry(width:64,height:32,frames:41,fps:24)
    let image=try MLXReferenceLayout(geometry:geometry,firstStrength:0.8,lastStrength:nil,firstFrame:24)
    let guide=try MLXExtensionGuideLayout(geometry:geometry,contextFrames:25,
      videoGuideLatentFrames:3,audioGuideTokens:27)
    let reference=MLXArray.ones([image.frameTokens,128])*0.7
    let anchored=try image.prepare(generated:.zeros([geometry.videoTokens,128]),first:reference,last:nil)
    let sourceVideo=MLXArray.ones([guide.videoGuideTokens,128])*0.3
    let sourceAudio=MLXArray.ones([guide.audioGuideTokens,128])*0.4
    let prepared=try guide.prepare(targetVideo:anchored.latent,
      targetVideoCondition:anchored.condition,targetAudio:.zeros([geometry.audioFrames,128]),
      sourceVideo:sourceVideo,sourceAudio:sourceAudio,
      guideVideoNoise:.zeros(sourceVideo.shape),guideAudioNoise:.zeros(sourceAudio.shape),sigma:1)
    let anchorIndex=3*image.frameTokens
    XCTAssertEqual(prepared.video[anchorIndex,0].item(Float.self),0.7)
    XCTAssertEqual(prepared.videoCondition.clean[anchorIndex,0].item(Float.self),0.7)
    XCTAssertEqual(prepared.videoCondition.mask[anchorIndex],0.2,accuracy:0.0001)
    XCTAssertEqual(prepared.videoCondition.mask[geometry.videoTokens],0.5)
    XCTAssertEqual(prepared.audioCondition.mask[geometry.audioFrames],0.5)
  }
  func testGuidedWindowKeepsExactSourceAudioWhileHistoryRemainsConditioned() throws {
    let geometry=try AVGeometry(width:64,height:32,frames:41,fps:24)
    let guide=try MLXExtensionGuideLayout(geometry:geometry,contextFrames:25,
      videoGuideLatentFrames:3,audioGuideTokens:27)
    let driver=MLXArray.ones([geometry.audioFrames,128])*0.8
    let sourceVideo=MLXArray.ones([guide.videoGuideTokens,128])*0.3
    let sourceAudio=MLXArray.ones([guide.audioGuideTokens,128])*0.4
    let frozen=try MLXAudioDenoiseCondition(clean:driver,
      mask:Array(repeating:Float(0),count:geometry.audioFrames))
    let prepared=try guide.prepare(targetVideo:.zeros([geometry.videoTokens,128]),
      targetAudio:driver,targetAudioCondition:frozen,
      sourceVideo:sourceVideo,sourceAudio:sourceAudio,
      guideVideoNoise:.zeros(sourceVideo.shape),guideAudioNoise:.zeros(sourceAudio.shape),sigma:1)
    XCTAssertEqual(prepared.audioCondition.mask[0],0)
    XCTAssertEqual(prepared.audioCondition.mask[geometry.audioFrames],0.5)
    XCTAssertEqual(prepared.audioCondition.clean[0,0].item(Float.self),0.8)
    XCTAssertEqual(prepared.audioCondition.clean[geometry.audioFrames,0].item(Float.self),0.4)
  }
  func testSceneCanGuideInteriorVideoAndExactAudioOverlap() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:97,fps:24)
    let layout=try MLXExtensionGuideLayout(geometry:geometry,contextFrames:25,
      videoGuideLatentFrames:3,audioGuideTokens:27)
    XCTAssertEqual(layout.videoGuideTokens,3*geometry.latentHeight*geometry.latentWidth)
    XCTAssertEqual(layout.audioGuideTokens,27)
    XCTAssertEqual(layout.videoPositions.count,layout.videoTokens*3)
    XCTAssertThrowsError(try MLXExtensionGuideLayout(geometry:geometry,
      contextFrames:25,videoGuideLatentFrames:5))
  }
  func testAppendsAlignedVideoAndAudioGuidesAndPreservesTargetTimeline() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:41,fps:24)
    let layout=try MLXExtensionGuideLayout(geometry:geometry,contextFrames:9)
    XCTAssertEqual(layout.videoGuideTokens,8)
    XCTAssertEqual(layout.audioGuideTokens,10)
    XCTAssertEqual(layout.videoTokens,geometry.videoTokens+8)
    XCTAssertEqual(layout.audioTokens,geometry.audioFrames+10)
    XCTAssertEqual(layout.videoPositions.count,layout.videoTokens*3)
    XCTAssertEqual(layout.audioPositions.count,layout.audioTokens)
    let targetVideo=MLXArray.ones([geometry.videoTokens,128])*2
    let targetAudio=MLXArray.ones([geometry.audioFrames,128])*3
    let sourceVideo=MLXArray.ones([layout.videoGuideTokens,128])*4
    let sourceAudio=MLXArray.ones([layout.audioGuideTokens,128])*6
    let result=try layout.prepare(targetVideo:targetVideo,targetAudio:targetAudio,
      sourceVideo:sourceVideo,sourceAudio:sourceAudio,
      guideVideoNoise:.zeros(sourceVideo.shape),guideAudioNoise:.zeros(sourceAudio.shape),sigma:1)
    XCTAssertEqual(result.video.shape,[layout.videoTokens,128])
    XCTAssertEqual(result.audio.shape,[layout.audioTokens,128])
    XCTAssertEqual(result.video[0].asArray(Float.self),targetVideo[0].asArray(Float.self))
    XCTAssertEqual(result.audio[0].asArray(Float.self),targetAudio[0].asArray(Float.self))
    XCTAssertEqual(result.video[geometry.videoTokens,0].item(Float.self),2)
    XCTAssertEqual(result.audio[geometry.audioFrames,0].item(Float.self),3)
    XCTAssertEqual(result.videoCondition.mask.last,0.5)
    XCTAssertEqual(result.audioCondition.mask.last,0.5)
  }

  func testRejectsMisalignedAndOversizedGuidesBeforeSampling() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:41,fps:24)
    XCTAssertThrowsError(try MLXExtensionGuideLayout(geometry:geometry,contextFrames:8))
    XCTAssertThrowsError(try MLXExtensionGuideLayout(geometry:geometry,contextFrames:41))
    XCTAssertThrowsError(try MLXExtensionGuideLayout(geometry:geometry,contextFrames:9,strength:Float.nan))
  }
  func testReleasedMLXNoiseDependsOnTotalShape() {
    let target=MLXNoisePolicy.seeded(1234,tokens:24).asType(.float32)
    let combined=MLXNoisePolicy.seeded(1234,tokens:32).asType(.float32)
    XCTAssertNotEqual(target.asArray(Float.self),combined[0..<24].asArray(Float.self))
  }
  func testAppendedAudiovisualGuidesPassJointSamplerAndProtectCleanRows() throws {
    let geometry=try AVGeometry(width:64,height:64,frames:41,fps:24)
    let layout=try MLXExtensionGuideLayout(geometry:geometry,contextFrames:9,strength:1)
    let sourceVideo=MLXArray.ones([layout.videoGuideTokens,128])*0.25
    let sourceAudio=MLXArray.ones([layout.audioGuideTokens,128])*0.5
    let prepared=try layout.prepare(targetVideo:.zeros([geometry.videoTokens,128]),
      targetAudio:.zeros([geometry.audioFrames,128]),sourceVideo:sourceVideo,sourceAudio:sourceAudio,
      guideVideoNoise:.ones(sourceVideo.shape),guideAudioNoise:.ones(sourceAudio.shape),sigma:1)
    let configuration=try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:layout.videoTokens,
      audioTokens:layout.audioTokens,textTokens:4)
    let runner=try MLXSamplingRunner(configuration:configuration,blockCount:1)
    func weight(_ name:String,_ shape:[Int]) throws -> MLXWeight {
      let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      return try MLXWeight(dense:MLXArray.ones(shape)*(norm == 1 ? 1 : 0.001))
    }
    let result=try runner.evaluate(["video_latent":prepared.video,"audio_latent":prepared.audio,
      "video_positions":MLXArray(layout.videoPositions,[layout.videoTokens,3]),
      "audio_positions":MLXArray(layout.audioPositions,[layout.audioTokens,1]),
      "video_text":.zeros([4,32]),"audio_text":.zeros([4,16])],
      schedule:SamplingSchedule(sigmas:[1,0]),videoConditioning:prepared.videoCondition,
      audioConditioning:prepared.audioCondition,fixedWeights:weight,
      blockWeights:{ try weight("transformer_blocks.\($0)."+$1,$2) })
    XCTAssertEqual(result["video"]!.shape,[layout.videoTokens,128])
    XCTAssertEqual(result["audio"]!.shape,[layout.audioTokens,128])
    XCTAssertEqual(result["video"]![geometry.videoTokens].asArray(Float.self),sourceVideo[0].asArray(Float.self))
    XCTAssertEqual(result["audio"]![geometry.audioFrames].asArray(Float.self),sourceAudio[0].asArray(Float.self))
  }
}
