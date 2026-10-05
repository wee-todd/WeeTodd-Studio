import XCTest
import MLX
import TensorIO
import LTX25Engine
@testable import LTX25MLX

final class MLXMSRAudioTests:XCTestCase {
  func testInstalledV2AdmitsAllTargetsAndSeparateLearnedAudioSlots() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_MSR_V2_TEST_ADAPTER"] else {
      throw XCTSkip("Installed MSR V2 qualification")
    }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:path))
    XCTAssertEqual(try LTXAdapterCompatibility.msrPlan(file:file,strength:1).pairs.count,1152)
    let state=try MLXMSRSlotEmbedding.load(URL(fileURLWithPath:path),audio:true)
    let a=try MLXMSRSlotEmbedding.embedding(slotID:1,state:state)
    let b=try MLXMSRSlotEmbedding.embedding(slotID:2,state:state)
    XCTAssertGreaterThan((a-b).abs().max().item(Float.self),0.00001)
    let fixture=try XCTUnwrap(Bundle.module.url(forResource:"msr-v2-audio-slot-witness",withExtension:"json",subdirectory:"Fixtures"))
    let golden=try JSONSerialization.jsonObject(with:Data(contentsOf:fixture)) as! [String:Any]
    let expected=golden["embeddings"] as! [[Double]]
    for (actual,row) in zip([a,b],expected) {
      let difference=zip(actual.asArray(Float.self),row).map { abs(Double($0.0)-$0.1) }.max()!
      XCTAssertLessThan(difference,0.0002,"Learned voice slot differs from independent scalar witness")
    }
    XCTAssertThrowsError(try LTXAdapterCompatibility.standardPlan(file:file,strength:1))
  }
  func testSparseSecondSpeakerUsesAbsoluteWindowAndCleanPrefix() throws {
    let plan=try MLXMSRAudioLayout(slotIDs:[2],lengths:[3],imageSlotCount:3,targetFrames:4)
    XCTAssertEqual(plan.prefixFrames,3)
    // Three causal tokens have endpoints .01/.05/.09; slot2 ends at -5.04.
    XCTAssertEqual(plan.positions[0],-5.125,accuracy:0.00001)
    XCTAssertEqual(plan.positions[2],-5.06,accuracy:0.00001)
    XCTAssertEqual(plan.positions[3],0.005,accuracy:0.00001)
    let prepared=try plan.prepare(generated:.ones([4,128]),references:[.ones([3,128])*2])
    XCTAssertEqual(prepared.condition.mask,[0,0,0,1,1,1,1])
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),2)
    XCTAssertEqual(prepared.latent[3,0].item(Float.self),1)
    XCTAssertEqual(try plan.target(prepared.latent).shape,[4,128])
    XCTAssertEqual(try plan.target(prepared.latent)[0,0].item(Float.self),1)
  }
  func testJointSamplerPreservesSpeakerPrefixAndOnlyReturnsGeneratedTargetAudio() throws {
    let layout=try MLXMSRAudioLayout(slotIDs:[1,2],lengths:[2,1],imageSlotCount:2,targetFrames:3)
    let refs=[MLXArray.ones([2,128])*0.25,MLXArray.ones([1,128])*0.5]
    let audio=try layout.prepare(generated:.ones([3,128]),references:refs)
    let video=try MLXVideoDenoiseCondition(clean:.ones([2,128]),mask:[0,1])
    let config=try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:2,audioTokens:6,textTokens:4)
    func weight(_ name:String,_ shape:[Int]) throws -> MLXWeight {
      try MLXWeight(dense:MLXArray.ones(shape)*(name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : name == "audio_proj_out.bias" ? 0.25 : 0.001))
    }
    var previews=0
    let output=try MLXSamplingRunner(configuration:config,blockCount:1).evaluate([
      "video_latent":.ones([2,128]),"audio_latent":audio.latent,
      "video_positions":MLXArray([Float](repeating:0,count:6),[2,3]),
      "audio_positions":MLXArray(layout.positions,[6,1]),"video_text":.zeros([4,32]),"audio_text":.zeros([4,16])],
      schedule:SamplingSchedule(sigmas:[1,0.5,0],eta:0),videoConditioning:video,audioConditioning:audio.condition,
      bfloat16State:["video","audio"],fixedWeights:weight,blockWeights:{ try weight("transformer_blocks.\($0)."+$1,$2) },
      preview:{ state,_ in
        previews += 1
        XCTAssertEqual(state["audio"]![0,0].item(Float.self),0.25)
        XCTAssertEqual(state["audio"]![2,0].item(Float.self),0.5)
        XCTAssertEqual(state["video"]![0,0].item(Float.self),1)
      })
    XCTAssertEqual(previews,2)
    let target=try layout.target(output["audio"]!)
    XCTAssertEqual(target.shape,[3,128]);XCTAssertTrue(MLX.isFinite(target).all().item(Bool.self))
    XCTAssertNotEqual(target.asArray(Float.self),Array(repeating:1,count:384))
  }
  func testTwoSpeakersKeepDistinctWindowsAndRejectInvalidMappingBeforeWeights() throws {
    let plan=try MLXMSRAudioLayout(slotIDs:[1,2],lengths:[125,1],imageSlotCount:2,targetFrames:4)
    XCTAssertEqual(plan.prefixFrames,126)
    XCTAssertEqual(plan.positions[124],-5.06,accuracy:0.00001)
    XCTAssertEqual(plan.positions[125],-0.045,accuracy:0.00001)
    for slots in [[0],[3],[2,1],[1,1]] {
      XCTAssertThrowsError(try MLXMSRAudioLayout(slotIDs:slots,lengths:Array(repeating:1,count:slots.count),imageSlotCount:2,targetFrames:4))
    }
    XCTAssertThrowsError(try MLXMSRAudioLayout(slotIDs:[2],lengths:[1],imageSlotCount:1,targetFrames:4))
    XCTAssertThrowsError(try MLXMSRAudioLayout(slotIDs:[1],lengths:[126],imageSlotCount:2,targetFrames:4))
    XCTAssertThrowsError(try plan.prepare(generated:.zeros([4,128]),references:[]))
  }
}
