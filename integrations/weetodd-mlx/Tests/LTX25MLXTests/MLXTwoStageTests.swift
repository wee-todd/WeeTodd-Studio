import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXTwoStageTests:XCTestCase {
  func testStageOneObserverCapturesDenoisedTokensBeforeUpscale() throws {
    try Device.withDefaultDevice(.cpu) {
      let recipe=try DistilledTwoStageRecipe(width:64,height:64,frames:41,fps:24,seed:4)
      var observed:MLXArray?
      _=try MLXTwoStageTrajectory().evaluate(recipe:recipe,
        stageOneVideoObserver:{ observed=$0.reshaped($0.shape) },
        sample:{ stage,_,state,_,_ in
          if stage == 1 { return ["video":MLXArray.ones(state["video"]!.shape)*3,
            "audio":state["audio"]!] }
          return state
        },upscale:{ value,_ in
          XCTAssertNotNil(observed)
          XCTAssertEqual(observed![0,0].item(Float.self),3)
          return .zeros([value.shape[0]*4,128])
        })
    }
  }
  func testExtensionDrawsTargetAndGuideNoiseFromOneShapedSeedPerStage() throws {
    try Device.withDefaultDevice(.cpu) {
      let recipe=try DistilledTwoStageRecipe(width:64,height:64,frames:41,fps:24,seed:41)
      var stages:[Int]=[]
      _=try MLXTwoStageTrajectory().evaluateWithGuides(recipe:recipe,videoGuideFrames:2,
        audioGuideTokens:10,sample:{ stage,g,state,guides,_,_ in
          stages.append(stage)
          let fullVideo=MLXNoisePolicy.seeded(recipe.seed &+ (stage == 1 ? 0 : 2),
            tokens:g.videoTokens+guides["video"]!.shape[0]).asType(.float32)
          XCTAssertEqual(guides["video"]!.asArray(Float.self),
            fullVideo[g.videoTokens...].asArray(Float.self))
          if stage == 1 {
            XCTAssertEqual(state["video"]!.asArray(Float.self),
              fullVideo[0..<g.videoTokens].asArray(Float.self))
            let fullAudio=MLXNoisePolicy.seeded(recipe.seed &+ 1,
              tokens:g.audioFrames+guides["audio"]!.shape[0]).asType(.float32)
            XCTAssertEqual(state["audio"]!.asArray(Float.self),
              fullAudio[0..<g.audioFrames].asArray(Float.self))
            XCTAssertEqual(guides["audio"]!.asArray(Float.self),
              fullAudio[g.audioFrames...].asArray(Float.self))
          }
          return state
        },upscale:{ value,_ in .zeros([value.shape[0]*4,128]) })
      XCTAssertEqual(stages,[1,2])
    }
  }
  func testFrozenSourceAudioSurvivesBothStagesWithoutAudioNoise() throws {
    try Device.withDefaultDevice(.cpu) {
      let recipe=try DistilledTwoStageRecipe(width:64,height:64,frames:9,fps:24,seed:7)
      let source=MLXArray.ones([recipe.low.audioFrames,128])*0.125
      var stages:[Int]=[],noiseNames:[String]=[]
      let result=try MLXTwoStageTrajectory().evaluate(recipe:recipe,frozenAudio:source,sample:{ stage,_,state,_,noise in
        stages.append(stage)
        XCTAssertEqual(state["audio"]!.asArray(Float.self),source.asArray(Float.self))
        if stage == 1 { _ = try noise!(0,"video",state["video"]!.shape); noiseNames.append("video") }
        return state
      },upscale:{ value,_ in .zeros([value.shape[0]*4,128]) })
      XCTAssertEqual(stages,[1,2]);XCTAssertEqual(noiseNames,["video"])
      XCTAssertEqual(result["audio"]!.asArray(Float.self),source.asArray(Float.self))
      XCTAssertThrowsError(try MLXTwoStageTrajectory().evaluate(recipe:recipe,frozenAudio:.zeros([1,128]),
        sample:{ _,_,state,_,_ in state },upscale:{ value,_ in .zeros([value.shape[0]*4,128]) }))
    }
  }
  func testReleasedMLXNoiseMatchesProductionHelperFixture() throws {
    try Device.withDefaultDevice(.cpu) {
      let url=Bundle.module.url(forResource:"released-noise",withExtension:"json",subdirectory:"Fixtures")!
      let fixture=try JSONDecoder().decode([String:[Float]].self,from:Data(contentsOf:url))
      func check(_ x:MLXArray,_ name:String) {
        let expected=fixture[name]!,actual=x.asArray(Float.self)
        XCTAssertEqual(actual.count,expected.count,name)
        XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.000002,name)
      }
      let recipe=try DistilledTwoStageRecipe(width:128,height:64,frames:9,fps:24,seed:42)
      _=try MLXTwoStageTrajectory().evaluate(recipe:recipe,noisePolicy:.releasedMLX,sample:{ stage,_,state,_,noise in
        let v=state["video"]!,a=state["audio"]!
        if stage == 2 { check(v,"refined_video");check(a,"refined_audio");return state }
        check(v,"initial_video");check(a,"initial_audio")
        let vn=try noise!(0,"video",v.shape),an=try noise!(0,"audio",a.shape)
        check(vn,"ancestral_video");check(an,"ancestral_audio")
        return ["video":(v*0.25+vn*0.01).asType(.bfloat16).asType(.float32),
          "audio":(a*0.25+an*0.01).asType(.bfloat16).asType(.float32)]
      },upscale:{ value,_ in MLXArray.ones([value.shape[0]*4,128])*value[0,0] })
    }
  }

  func testStageOrderAndRNGMatchSharedNativeTrajectory() throws {
    try Device.withDefaultDevice(.cpu) {
      let recipe=try DistilledTwoStageRecipe(width:128,height:64,frames:9,fps:24,seed:42)
      let expected=try TwoStageTrajectory().evaluate(recipe:recipe,sample:{ stage,_,state,_,noise in
        if stage == 2 { return state }
        let vn=try noise!(0,.video,state.video.count), an=try noise!(0,.audio,state.audio.count)
        return AVLatents(video:zip(state.video,vn).map { $0*0.25+$1*0.01 },
          audio:zip(state.audio,an).map { $0*0.25+$1*0.01 })
      },upscale:{ values,_ in Array(repeating:values[0],count:values.count*4) })
      var order:[String]=[]
      let actual=try MLXTwoStageTrajectory().evaluate(recipe:recipe,sample:{ stage,_,state,_,noise in
        order.append("sample\(stage)")
        if stage == 2 { XCTAssertNil(noise); return state }
        let video=state["video"]!, audio=state["audio"]!
        return ["video":video*0.25+(try noise!(0,"video",video.shape))*0.01,
          "audio":audio*0.25+(try noise!(0,"audio",audio.shape))*0.01]
      },upscale:{ value,shape in
        order.append("upscale"); XCTAssertEqual(shape,[2,1,2,128])
        return MLXArray.ones([value.shape[0]*4,128])*value[0,0]
      })
      XCTAssertEqual(order,["sample1","upscale","sample2"])
      for (name,values) in [("video",expected.video),("audio",expected.audio)] {
        XCTAssertLessThan(zip(actual[name]!.asArray(Float.self),values).map { abs($0-$1) }.max()!,0.000001)
      }
    }
  }
  func testBadUpscalerAndReentryDoNotStartSecondStageAndRetryWorks() throws {
    try Device.withDefaultDevice(.cpu) {
      let r=try DistilledTwoStageRecipe(width:64,height:64,frames:1,fps:24,seed:1)
      let runner=MLXTwoStageTrajectory()
      var stages:[Int]=[]
      let sample:MLXTwoStageTrajectory.Sample={ stage,_,state,_,_ in stages.append(stage); return state }
      XCTAssertThrowsError(try runner.evaluate(recipe:r,sample:sample,upscale:{ _,_ in .zeros([1]) }))
      XCTAssertEqual(stages,[1]); stages=[]
      let up:MLXTwoStageTrajectory.Upscale={ x,_ in .zeros([x.shape[0]*4,128]) }
      XCTAssertThrowsError(try runner.evaluate(recipe:r,sample:sample,upscale:{ _,_ in
        try runner.evaluate(recipe:r,sample:sample,upscale:up)["video"]!
      }))
      XCTAssertEqual(stages,[1]); stages=[]
      XCTAssertNoThrow(try runner.evaluate(recipe:r,sample:sample,upscale:up))
      XCTAssertEqual(stages,[1,2])
    }
  }
  @MainActor func testCancellationAfterUpscaleSkipsRefinement() async throws {
    let canceled=try await Task {
      try Device.withDefaultDevice(.cpu) {
        let r=try DistilledTwoStageRecipe(width:64,height:64,frames:1,fps:24,seed:1)
        var stages:[Int]=[]
        do {
          _=try MLXTwoStageTrajectory().evaluate(recipe:r,sample:{ stage,_,state,_,_ in
            stages.append(stage); return state
          },upscale:{ x,_ in
            withUnsafeCurrentTask { $0?.cancel() }
            return .zeros([x.shape[0]*4,128])
          })
          return false
        } catch is CancellationError { return stages == [1] }
      }
    }.value
    XCTAssertTrue(canceled)
  }
}
