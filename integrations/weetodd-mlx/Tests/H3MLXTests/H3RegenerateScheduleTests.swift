import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateScheduleTests: XCTestCase {
  func testPublisherFixedAudioHasExactGridAndSeparateOrdinaryAdmission() throws {
    XCTAssertThrowsError(try H3Schedule(sigmas:[0,0,0,0]))
    let schedule=H3RegenerateSchedule.publisherThreeStep
    XCTAssertEqual(schedule.sigmas,[0.909375,6.0/7,2.0/3,0])
    XCTAssertEqual(schedule.modelTimes,[Float(0.090625),Float(1.0/7),Float(1.0/3)])
    XCTAssertEqual(schedule.contract.audioSigmas,[0,0,0,0])
    XCTAssertEqual(schedule.contract.guidance,1)
    try schedule.contract.validate()
    XCTAssertEqual(H3RegenerateSchedule.publisherTwoStep.sigmas,[0.7,0.49411764705882344,0])
  }
  private final class Predictor: H3VelocityPredictor {
    let layout:H3PackedLayout
    var audio:[Float]=[]
    var indices:[[Int32]]=[]
    var calls=0
    init() throws { layout=try H3PackedLayout(geometry:H3Geometry(width:32,height:32,durationSeconds:2.5),textTags:[0],anchors:[]) }
    func predict(videoLatents:MLXArray,audioLatents:MLXArray,timestepIndices:[Int32],progress:(Int,Int)->Void) throws -> H3FinalLayer.Output {
      calls += 1; audio.append(audioLatents[0,0,0].item(Float.self)); indices.append(timestepIndices)
      XCTAssertEqual(videoLatents.dtype,.bfloat16)
      return .init(video:.ones(videoLatents.shape),audio:.full(audioLatents.shape,values:MLXArray(Float(10000))))
    }
  }
  func testRenoiseDataWardEulerAndFixedSilenceAcrossAllEvaluations() throws {
    let p=try Predictor(), schedule=H3RegenerateSchedule.publisherThreeStep
    let videoRows=p.layout.tags.count-p.layout.videoStart,audioRows=p.layout.videoStart-p.layout.audioStart
    let source=MLXArray.full([1,videoRows,96],values:MLXArray(Float(2))),noise=MLXArray.zeros([1,videoRows,96])
    let silence=MLXArray.full([1,audioRows,32],values:MLXArray(Float(3)),dtype:.bfloat16)
    let result=try H3AVSampler.runRegenerate(predictor:p,schedule:schedule,source:source,noise:noise,encodedSilence:silence)
    XCTAssertEqual(p.calls,3); XCTAssertEqual(p.audio,[3,3,3])
    XCTAssertEqual(result.video[0,0,0].item(Float.self),1.090625,accuracy:0.000001)
    XCTAssertEqual(result.audio.asArray(Float.self),silence.asArray(Float.self))
    let rows=try H3RowSchedule(regenerate:p.layout,schedule:schedule)
    for step in 0..<3 {
      XCTAssertEqual(rows.table[Int(p.indices[step][p.layout.audioStart])],1)
      XCTAssertEqual(rows.table[Int(p.indices[step][p.layout.videoStart])],schedule.modelTimes[step])
    }
  }
  func testNonfiniteSourceAndNoiseFailBeforePredictor() throws {
    let p=try Predictor(), vr=p.layout.tags.count-p.layout.videoStart, ar=p.layout.videoStart-p.layout.audioStart
    XCTAssertThrowsError(try H3AVSampler.runRegenerate(predictor:p,schedule:.publisherThreeStep,
      source:.full([1,vr,96],values:MLXArray(Float.nan)),noise:.zeros([1,vr,96]),encodedSilence:.zeros([1,ar,32],dtype:.bfloat16)))
    XCTAssertEqual(p.calls,0)
  }
}
