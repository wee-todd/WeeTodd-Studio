import XCTest
@testable import LTX25MLX
import LTX25Engine

final class MLXMovieMemoryPlanTests:XCTestCase {
  private let gib:UInt64=1024*1024*1024
  private func request(mode:String="refine",frames:Int=49,chunking:Bool=false,
    anchors:String="first_last") throws -> MLXMovieUpscaleRequest {
    var components=["video_checkpoint":try MLXDecoderHeaderFixture.convolutional(test:self).path,"spatial_upscaler_checkpoint":"/model/upscaler.safetensors"]
    if mode != "latent_only" {
      components.merge(["gemma_root":"/model/gemma","connector_checkpoint":"/model/fixed.safetensors",
        "transformer_root":"/model/transformer","audio_checkpoint":"/model/audio.safetensors"]) { _,new in new }
    }
    if mode == "pixel_spatial" { components["pixel_spatial_adapter"]="/model/pixel.safetensors" }
    let body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale",
      "source":["path":"/media/source.mov","sha256":String(repeating:"a",count:64),
        "rgb_path":"/media/source.rgb24","rgb_sha256":String(repeating:"b",count:64),
        "width":384,"height":256,"frames":frames,"fps":24,"start_seconds":0,"duration_seconds":Double(frames)/24],
      "components":components,"mode":mode,"size_policy":"strict_32","output_directory":"/output/movie",
      "prompt":mode == "latent_only" ? "" : "Preserve the brass cup and hand.","seed":42,
      "refinement_strength":0.35,"anchors":mode == "latent_only" ? "none" : anchors,
      "anchor_strength":0.7,"pixel_strength":1,"reference_images":[],
      "maximum_audio_drift_seconds":0.05,"chunking":chunking,"chunk_frame_megapixel_budget":28,
      "resume":false,"keep_chunks":false]
    return try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:body,options:.sortedKeys),
      outputDirectory:URL(fileURLWithPath:"/output/movie"))
  }
  private func plan(_ request:MLXMovieUpscaleRequest,cutFrames:[Int]=[]) throws -> MLXMovieMemoryPlan {
    try MLXMovieMemoryPlan(request:request,cutFrames:cutFrames,physicalMemory:256*gib,recommendedWorkingSet:192*gib)
  }
  func testActualRepresentativeCanvasRequiresMoreThanOldTwoGiBAndAdmitsExactPerTokenReserve() throws {
    let request=try request(),plan=try plan(request)
    let layout=try MLXMovieUpscaleLayout(plan:request.plan,firstStrength:0.7,lastStrength:0.7)
    let c=try AVBlockConfiguration(videoTokens:layout.videoTokens,audioTokens:layout.geometry.audioFrames,textTokens:1024)
    let exact=try MLXAVBlock.estimatedActivationBytes(configuration:c,perTokenVideo:true)
    XCTAssertGreaterThan(exact,2*Int(gib))
    XCTAssertEqual(plan.transformerActivationBytes,exact)
    XCTAssertEqual(plan.activationCeilingBytes,32*Int(gib))
    XCTAssertNoThrow(try MLXDenoiser.admitRotary(configuration:c,maximumActivationBytes:plan.transformerActivationBytes))
    let block=try MLXAVBlock(configuration:c,maximumActivationBytes:plan.transformerActivationBytes)
    XCTAssertNoThrow(try block.admitPerTokenVideo())
    XCTAssertThrowsError(try MLXAVBlock(configuration:c,maximumActivationBytes:exact-1).admitPerTokenVideo())
    XCTAssertLessThan(plan.transformerActivationBytes,plan.activationCeilingBytes)
  }
  func testPixelReferencesAreIncludedAndLatentOnlyStillReservesFullUpscalerWorkspace() throws {
    let refine=try plan(request()),pixel=try plan(request(mode:"pixel_spatial")),latent=try plan(request(mode:"latent_only"))
    XCTAssertGreaterThan(pixel.transformerActivationBytes,refine.transformerActivationBytes)
    XCTAssertGreaterThan(latent.transformerActivationBytes,0)
    XCTAssertGreaterThan(latent.videoActivationBytes,0)
    XCTAssertLessThan(latent.transformerActivationBytes,refine.transformerActivationBytes)
  }
  func testExactHostAllowanceSucceedsButOneByteLessAndLowRamRejectBeforeFilesAreRead() throws {
    let r=try request(),p=try plan(r)
    let required=UInt64(max(p.videoActivationBytes,p.transformerActivationBytes))
    XCTAssertNoThrow(try MLXMovieMemoryPlan(request:r,physicalMemory:2*(required+4*gib),recommendedWorkingSet:required+4*gib))
    XCTAssertThrowsError(try MLXMovieMemoryPlan(request:r,physicalMemory:2*(required+4*gib-1),recommendedWorkingSet:required+4*gib-1))
    for (ram,metal) in [(0,0),(8,6),(256,4)] {
      XCTAssertThrowsError(try MLXMovieMemoryPlan(request:r,physicalMemory:UInt64(ram)*gib,recommendedWorkingSet:UInt64(metal)*gib))
    }
  }
  func testActualSceneCutsCanIncreaseLargestPaddedChunkAndMustBeAdmitted() throws {
    let r=try request(frames:180,chunking:true)
    let original=try plan(r),cut=try plan(r,cutFrames:[70])
    XCTAssertGreaterThan(cut.chunks.map(\.paddedFrames).max()!,original.chunks.map(\.paddedFrames).max()!)
    XCTAssertGreaterThan(cut.transformerActivationBytes,original.transformerActivationBytes)
    XCTAssertEqual(cut.chunks.reduce(0) { $0+$1.frames },180)
  }
}
