import XCTest
@testable import LTX25MLX
import LTX25Engine
import MLX

final class MLXStudioMemoryPlanTests:XCTestCase {
  let gib=1024*1024*1024
  func testActualHDRefinementRotaryGridsBuildWithoutWeights() throws {
    let request=try request(),geometry=try request.recipe().high
    let configuration=try AVBlockConfiguration(videoTokens:geometry.videoTokens,audioTokens:geometry.audioFrames,textTokens:1024)
    // This separate old bound failed only after the first sampling stage.
    XCTAssertThrowsError(try DenoiserMath.rotaryElementCount(axes:3,tokens:geometry.videoTokens,
      heads:32,headWidth:128))
    let plan=try MLXStudioMemoryPlan(request:request,physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    let denoiser=try MLXDenoiser(configuration:configuration,maximumActivationBytes:plan.transformerActivationBytes)
    let grids=try denoiser.rotary(["video_positions":MLXArray(geometry.videoPositions),"audio_positions":MLXArray(geometry.audioPositions)])
    XCTAssertEqual(grids.count,8)
    XCTAssertEqual(grids["video_rope_cos"]?.shape,[geometry.videoTokens*32,64])
    XCTAssertEqual(grids["video_cross_rope_sin"]?.shape,[geometry.videoTokens*32,32])
    for grid in grids.values { XCTAssertTrue(MLX.isFinite(grid).all().item(Bool.self)) }
    XCTAssertEqual(denoiser.residentWeightBytes,0)
  }
  func request(frames:Int=241,reference:Bool=false,width:Int=1344,height:Int=768) throws -> MLXDistilledRequest {
    let helper=MLXDistilledRequestTests()
    var values=helper.base();values["width"]=width;values["height"]=height;values["frames"]=frames
    if reference {
      values["version"]=2;values["task"]="fflf"
      values["reference_images"]=[["role":"first","path":"/images/first.png","strength":1,"crf":33],
        ["role":"last","path":"/images/last.png","strength":1,"crf":33]]
    }
    return try helper.decode(values)
  }
  func testUserTenSecondHDJobAdmitsBothStagesWithoutChangingGeometry() throws {
    let request=try request()
    // Reproduce the shipped worker's fixed allowance failure.
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:12*gib,
      transformerActivationBytes:12*gib,videoBackend:.mlx,audioBackend:.mlx))
    let plan=try MLXStudioMemoryPlan(request:request,physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    XCTAssertGreaterThan(plan.transformerActivationBytes,12*gib)
    XCTAssertGreaterThan(plan.videoActivationBytes,12*gib)
    XCTAssertEqual(plan.activationCeilingBytes,32*gib)
    let admission=try MLXMediaPipeline.admit(request,videoActivationBytes:plan.videoActivationBytes,
      transformerActivationBytes:plan.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx)
    XCTAssertEqual(admission.videoFrames,241)
    XCTAssertEqual(admission.videoActivationBytes,plan.videoActivationBytes)
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:plan.videoActivationBytes,
      transformerActivationBytes:plan.transformerActivationBytes-1,videoBackend:.mlx,audioBackend:.mlx))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:plan.videoActivationBytes-1,
      transformerActivationBytes:plan.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx))
    print("HD memory admission: transformer=\(plan.transformerActivationBytes), video=\(plan.videoActivationBytes)")
  }
  func testReferencesIncludeExtraTokensAndPerTokenModulation() throws {
    let t2v=try MLXStudioMemoryPlan(request:request(),physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    let request=try request(reference:true)
    let plan=try MLXStudioMemoryPlan(request:request,physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))
    XCTAssertGreaterThan(plan.transformerActivationBytes,t2v.transformerActivationBytes)
    XCTAssertEqual(plan.videoActivationBytes,t2v.videoActivationBytes)
    XCTAssertNoThrow(try MLXMediaPipeline.admit(request,videoActivationBytes:plan.videoActivationBytes,
      transformerActivationBytes:plan.transformerActivationBytes,videoBackend:.mlx,audioBackend:.mlx))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:plan.videoActivationBytes,
      transformerActivationBytes:plan.transformerActivationBytes-1,videoBackend:.mlx,audioBackend:.mlx))
  }
  func testHostReservesMetalWorkingSetAndEngineCeilingRemainEnforced() throws {
    for (physical,metal) in [(32,24),(256,16),(0,0),(8,6)] {
      XCTAssertThrowsError(try MLXStudioMemoryPlan(request:request(),physicalMemory:UInt64(physical*gib),recommendedWorkingSet:UInt64(metal*gib))) { error in
        XCTAssertTrue(String(describing:error).contains("after memory reserves"))
      }
    }
    XCTAssertThrowsError(try MLXStudioMemoryPlan(request:request(frames:481,width:1408,height:768),physicalMemory:256*UInt64(gib),recommendedWorkingSet:192*UInt64(gib))) { error in
      XCTAssertTrue(String(describing:error).contains("video decoder"),"\(error)")
      XCTAssertTrue(String(describing:error).contains("engine maximum 32 GiB"),"\(error)")
    }
    // Smaller supported work still admits on a 32 GiB machine.
    let plan=try MLXStudioMemoryPlan(request:request(frames:89),physicalMemory:32*UInt64(gib),recommendedWorkingSet:24*UInt64(gib))
    XCTAssertEqual(plan.activationCeilingBytes,12*gib)
  }
  func testEstimatorRetainsNonFusedAttentionWorkspace() throws {
    let c=try AVBlockConfiguration(videoDimension:32,audioDimension:32,heads:2,videoHeadDimension:16,audioHeadDimension:16,videoTokens:100,audioTokens:25,textTokens:50)
    let linear=(100*32+25*32+50*64)*4*24
    XCTAssertEqual(try MLXAVBlock.estimatedActivationBytes(configuration:c),linear+2*100*100*4*3)
    XCTAssertEqual(try MLXAVBlock.estimatedActivationBytes(configuration:c,perTokenVideo:true),linear+2*100*100*4*3+100*32*14*4)
  }
}
