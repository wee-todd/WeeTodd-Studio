import Foundation
import XCTest
@testable import H3MLX

final class H3TwoStageExecutionTests: XCTestCase {
  private enum Probe: Error {case failed}
  func testOrderedStagesDecodeRefinedVideoAndExactBaseAudioOnce() throws {
    let plan=try H3TwoStagePlan(width:128,height:64,durationSeconds:3)
    var events:[String]=[]
    let result=try H3TwoStageExecution.run(plan:plan,
      preflight:{events.append("preflight")},
      sampleBase:{events.append("base");return .init(video:[Float](repeating:1,count:plan.base.videoRows*96),audio:[Float](repeating:7,count:plan.base.audioRows*32))},
      upscaleVideo:{video in events.append("upscale");XCTAssertEqual(video.first,1);return [Float](repeating:10,count:plan.target.videoRows*96)},
      sampleRefinement:{initial in
        events.append("refine");XCTAssertEqual(initial.audio.first,7)
        XCTAssertEqual(initial.video.first,10)
        return .init(video:[Float](repeating:2,count:plan.target.videoRows*96),audio:[Float](repeating:99,count:plan.target.audioRows*32))
      },decode:{rows in events.append("decode");XCTAssertEqual(rows.video.first,2);XCTAssertTrue(rows.audio.allSatisfy{$0==7});return rows.video.count})
    XCTAssertEqual(result,plan.target.videoRows*96)
    XCTAssertEqual(events,["preflight","base","upscale","refine","decode"])
  }
  func testPreflightFailureAndCancellationStopDownstreamStages() throws {
    let plan=try H3TwoStagePlan(width:128,height:64,durationSeconds:3)
    var sampled=false
    XCTAssertThrowsError(try H3TwoStageExecution.run(plan:plan,
      preflight:{throw Probe.failed},sampleBase:{sampled=true;return .init(video:[],audio:[])},
      upscaleVideo:{$0},sampleRefinement:{$0},decode:{_ in true}))
    XCTAssertFalse(sampled)
    var events:[String]=[];var checks=0
    XCTAssertThrowsError(try H3TwoStageExecution.run(plan:plan,
      preflight:{events.append("preflight")},
      sampleBase:{events.append("base");return .init(video:[Float](repeating:1,count:plan.base.videoRows*96),audio:[Float](repeating:7,count:plan.base.audioRows*32))},
      upscaleVideo:{_ in XCTFail("Cancelled before upscaler");return []},
      sampleRefinement:{$0},decode:{_ in XCTFail("Cancelled before decode");return true},
      checkCancellation:{checks+=1;if checks==2 {throw CancellationError()}}))
    XCTAssertEqual(events,["preflight","base"])
  }
}
