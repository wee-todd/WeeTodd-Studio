import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3TwoStageImageTests: XCTestCase {
  private func plan() throws -> H3TwoStagePlan {
    try H3TwoStagePlan(width:1920,height:1088,durationSeconds:5.0/24,stillImageMode:.qualityFiveFrames)
  }
  func testQualityStillPlanKeepsShortGridAndIndependentRefinement() throws {
    let p=try plan()
    XCTAssertEqual(p.base.width,960);XCTAssertEqual(p.base.height,544)
    XCTAssertEqual(p.target.width,1920);XCTAssertEqual(p.target.height,1088)
    XCTAssertEqual(p.base.frames,5);XCTAssertEqual(p.target.videoLatentFrames,2)
    XCTAssertEqual(p.target.audioLatentFrames,8)
    XCTAssertEqual(p.base.videoRows,1020);XCTAssertEqual(p.target.videoRows,4080)
    XCTAssertEqual(try p.refinement.schedules(requestedSteps:8).video.timesteps.count,3)
    XCTAssertThrowsError(try H3TwoStagePlan(width:1920,height:1088,durationSeconds:5.0/24))
    XCTAssertThrowsError(try H3TwoStagePlan(width:1024,height:1024,durationSeconds:1.0/24,stillImageMode:.singleLatent))
  }
  func testLearnedGeometryAdmissionPreservesImageVideoBoundary() throws {
    let p=try plan()
    XCTAssertNoThrow(try H3LearnedLatentUpscaler.validateGeometry(source:p.base,target:p.target))
    let one=try H3Geometry(width:960,height:544,stillImageMode:.singleLatent)
    XCTAssertThrowsError(try H3LearnedLatentUpscaler.validateGeometry(source:one,target:p.target))
    let video=try H3TwoStagePlan(width:1920,height:1088,durationSeconds:3)
    XCTAssertNoThrow(try H3LearnedLatentUpscaler.validateGeometry(source:video.base,target:video.target))
    XCTAssertThrowsError(try H3LearnedLatentUpscaler.validateGeometry(source:p.base,target:video.target))
  }
  func testExplicitDraftAdapterAndTaskReplacementRetainStillMode() throws {
    let p=try plan(), file=URL(fileURLWithPath:"/draft-only.safetensors")
    let adapter=try H3LoRAAdapter(url:file,strength:1)
    let r=try H3T2VARequest(prompt:"A warrior.",width:1920,height:1088,durationSeconds:5.0/24,
      seed:42,requestedSteps:8,transformer:file,qwenPages:file,tokenizer:file,videoVAE:file,audioVAE:file,
      loRAAdapters:[adapter],stillImageMode:.qualityFiveFrames)
    let runner=try H3TwoStageRunner(task:.text(r),plan:p,resizeMethod:.bilinear,draftOnlyAdapters:[file])
    XCTAssertEqual(runner.baseTask.geometry.stillImageMode,.qualityFiveFrames)
    XCTAssertEqual(runner.finalTask.geometry.stillImageMode,.qualityFiveFrames)
    XCTAssertEqual(runner.baseTask.adapters.count,1);XCTAssertTrue(runner.finalTask.adapters.isEmpty)
    XCTAssertThrowsError(try H3TwoStageRunner(task:.text(r),plan:p,resizeMethod:.bilinear,
      draftOnlyAdapters:[URL(fileURLWithPath:"/not-in-stack")]))
    let reference=H3StillReference(rgb8:Data(count:64*64*3),width:64,height:64)
    let ref=try H3Ref2VAStillRequest(prompt:"A warrior.",references:[reference],width:1920,height:1088,
      durationSeconds:5.0/24,seed:42,requestedSteps:8,transformer:file,qwenPages:file,qwenVision:file,
      tokenizer:file,videoVAE:file,audioVAE:file,loRAAdapters:[adapter],stillImageMode:.qualityFiveFrames)
    let refRunner=try H3TwoStageRunner(task:.references(ref),plan:p,resizeMethod:.bilinear,draftOnlyAdapters:[file])
    XCTAssertEqual(refRunner.baseTask.geometry.stillImageMode,.qualityFiveFrames)
    XCTAssertEqual(refRunner.finalTask.geometry.stillImageMode,.qualityFiveFrames)
    if case .references(let replaced)=refRunner.baseTask {XCTAssertEqual(replaced.references.count,1)}
    else {XCTFail("Reference conditioning must remain active.")}
  }
  func testStillInterpolationUsesStillPixelBudgetAndRejectsMixedModes() throws {
    try Device.withDefaultDevice(.cpu) {
      let p=try plan()
      let rows=MLXArray([Float](repeating:2,count:p.base.videoRows*96),[1,p.base.videoRows,96])
      let result=try H3SpatialLatentResize.rows(rows,source:p.base,target:p.target,method:.bilinear)
      XCTAssertEqual(result.shape,[1,p.target.videoRows,96])
      XCTAssertTrue(result.asArray(Float.self).allSatisfy {$0==2})
      let single=try H3Geometry(width:p.target.width,height:p.target.height,stillImageMode:.singleLatent)
      XCTAssertThrowsError(try H3SpatialLatentResize.rows(rows,source:p.base,target:single,method:.bilinear))
    }
  }
  func testRecipeAdmitsExplicitStillDraftAdaptersOnlyForQualityImages() throws {
    let fields:[String:Any] = ["version":1,"draft_scale":2,"strength":0.35,"evaluations":3,
      "resize_method":"bilinear","refinement_adapter_policy":"standard","draft_only_loras":["/draft.safetensors"]]
    var root:[String:Any] = ["two_stage":fields,"components":["task":"ref2va"],
      "config":["width":1920,"height":1088,"duration_seconds":5.0/24],
      "conditioning":["task":"ref2va"],"still_image":["mode":"quality_five_frames"]]
    let prepared=try XCTUnwrap(H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    XCTAssertEqual(prepared.plan.target.stillImageMode,.qualityFiveFrames)
    XCTAssertEqual(prepared.draftOnlyAdapters.map(\.path),["/draft.safetensors"])
    root["still_image"]=["mode":"single_latent"]
    XCTAssertThrowsError(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    root.removeValue(forKey:"still_image")
    XCTAssertThrowsError(try H3TwoStageRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
  }
  func testStillExecutionDecodesOnceAndPreservesBaseAudio() throws {
    // Exercise the actual 2 MP target: a tiny fixture misses the ordinary-video pixel guard.
    let p=try plan()
    var events:[String]=[]
    let result=try H3TwoStageExecution.run(plan:p,preflight:{},sampleBase:{
      events.append("base")
      return .init(video:[Float](repeating:1,count:p.base.videoRows*96),audio:[Float](repeating:7,count:p.base.audioRows*32))
    },upscaleVideo:{_ in events.append("upscale");return [Float](repeating:2,count:p.target.videoRows*96)},sampleRefinement:{initial in
      events.append("refine");XCTAssertEqual(initial.video.first,2)
      return .init(video:[Float](repeating:3,count:p.target.videoRows*96),audio:[Float](repeating:99,count:p.target.audioRows*32))
    },decode:{rows in events.append("image");XCTAssertEqual(rows.audio.first,7);return rows.video.first})
    XCTAssertEqual(result,3);XCTAssertEqual(events,["base","upscale","refine","image"])
  }
}
