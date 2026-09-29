import XCTest
@testable import LTX25Engine

final class TwoStageTests: XCTestCase {
  func testRecipeGeometryAndReleasedSchedules() throws {
    let recipe = try DistilledTwoStageRecipe(width: 512,height: 256,frames: 33,fps: 24,seed: 42)
    XCTAssertEqual(recipe.low.videoTokens*4,recipe.high.videoTokens)
    XCTAssertEqual(recipe.low.audioFrames,recipe.high.audioFrames)
    XCTAssertEqual(recipe.first.sigmas,[1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0])
    XCTAssertEqual(recipe.first.eta,1)
    XCTAssertEqual(recipe.second.sigmas,[0.909375,0.725,0.421875,0]); XCTAssertEqual(recipe.second.eta,0)
    XCTAssertThrowsError(try DistilledTwoStageRecipe(width: 96,height: 64,frames: 9,fps: 24,seed: 1))
  }
  func testTwoStageOrderNoiseAndRefinementOfBothModalities() throws {
    let recipe = try DistilledTwoStageRecipe(width: 128,height: 64,frames: 9,fps: 24,seed: 42)
    var events: [String] = [], seen: [AVLatents] = []
    let runner = TwoStageTrajectory()
    var priorNoise = GaussianNoise(seed: 10042)
    let expectedNoise = try priorNoise.values(count: 2)
    let output = try runner.evaluate(recipe: recipe,sample: { stage,shape,state,schedule,noise in
      events.append("sample\(stage)"); seen.append(state)
      if stage == 1 {
        XCTAssertEqual(try noise?(0,.video,2),expectedNoise)
        XCTAssertEqual(shape.width,64)
        return AVLatents(video: [Float](repeating: 0.25,count: state.video.count),audio: [Float](repeating: -0.5,count: state.audio.count))
      }
      XCTAssertNil(noise); XCTAssertEqual(schedule.eta,0); XCTAssertEqual(shape.width,128)
      return state
    },upscale: { values,shape in
      events.append("upscale"); XCTAssertEqual(shape,[2,1,2,128]); XCTAssertTrue(values.allSatisfy { $0 == 0.25 })
      return [Float](repeating: 0.75,count: values.count*4)
    })
    XCTAssertEqual(events,["sample1","upscale","sample2"])
    var random = GaussianNoise(seed: 44)
    let vn = try random.values(count: output.video.count), an = try random.values(count: output.audio.count)
    let sigma = Float(0.909375)
    XCTAssertEqual(output.video,vn.map { (1-sigma)*0.75+sigma*$0 })
    XCTAssertEqual(output.audio,an.map { (1-sigma)*(-0.5)+sigma*$0 })
    XCTAssertEqual(seen.count,2)
  }
  func testInvalidUpscaleAndCancellationNeverStartStageTwoAndRetryIsClean() throws {
    let recipe = try DistilledTwoStageRecipe(width: 64,height: 64,frames: 1,fps: 24,seed: 1)
    let runner = TwoStageTrajectory()
    var stages: [Int] = []
    let sample: TwoStageTrajectory.Sample = { stage,_,value,_,_ in stages.append(stage);return value }
    XCTAssertThrowsError(try runner.evaluate(recipe: recipe,sample: sample,upscale: { _,_ in [0] }))
    XCTAssertEqual(stages,[1]); stages = []
    XCTAssertThrowsError(try runner.evaluate(recipe: recipe,sample: sample,upscale: { _,_ in throw CancellationError() }))
    XCTAssertEqual(stages,[1]); stages = []
    XCTAssertNoThrow(try runner.evaluate(recipe: recipe,sample: sample,upscale: { x,_ in Array(repeating: x[0],count: x.count*4) }))
    XCTAssertEqual(stages,[1,2])
  }
}

extension TwoStageTests {
  func testStageTwoFailureAndReentrantCallsReleaseGateBeforeRetry() throws {
    let recipe = try DistilledTwoStageRecipe(width: 64,height: 64,frames: 1,fps: 24,seed: 1)
    let runner = TwoStageTrajectory()
    let up: ([Float],[Int]) throws -> [Float] = { values,_ in Array(repeating: values[0],count: values.count*4) }
    var stages: [Int] = []
    XCTAssertThrowsError(try runner.evaluate(recipe: recipe,sample: { stage,_,state,_,_ in
      stages.append(stage)
      XCTAssertThrowsError(try runner.evaluate(recipe: recipe,sample: { _,_,x,_,_ in x },upscale: up))
      if stage == 2 { throw CancellationError() }
      return state
    },upscale: up))
    XCTAssertEqual(stages,[1,2])
    XCTAssertNoThrow(try runner.evaluate(recipe: recipe,sample: { _,_,x,_,_ in x },upscale: up))
  }
}
