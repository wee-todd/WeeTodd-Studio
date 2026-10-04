import XCTest
@testable import LTX25MLX

final class MLXMovieUpscaleRequestTests:XCTestCase {
  private func fixture(mode:String="refine")->[String:Any] {
    var components=["video_checkpoint":"/model/video.safetensors","spatial_upscaler_checkpoint":"/model/upscaler.safetensors"]
    if mode != "latent_only" {
      components.merge(["gemma_root":"/model/gemma","connector_checkpoint":"/model/fixed.safetensors","transformer_root":"/model/transformer","audio_checkpoint":"/model/audio.safetensors"]) { _,new in new }
    }
    if mode == "pixel_spatial" { components["pixel_spatial_adapter"]="/model/pixel.safetensors" }
    return ["version":1,"engine":"ltx25","task":"video_upscale","source":["path":"/media/source.mov","sha256":String(repeating:"a",count:64),"rgb_path":"/media/source.rgb24","rgb_sha256":String(repeating:"b",count:64),"width":64,"height":32,"frames":98,"fps":24,"start_seconds":0,"duration_seconds":98.0/24],"components":components,"mode":mode,"size_policy":"strict_32","output_directory":"/output/movie","prompt":mode == "latent_only" ? "" : "A red robot lowers its arm.","seed":42,"refinement_strength":0.35,"anchors":mode == "latent_only" ? "none" : "first_last","anchor_strength":0.7,"pixel_strength":1,"reference_images":[],"maximum_audio_drift_seconds":0.05,"chunking":false,"chunk_frame_megapixel_budget":260,"resume":false,"keep_chunks":false]
  }
  private func parse(_ value:[String:Any]) throws -> MLXMovieUpscaleRequest {
    try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:value,options:.sortedKeys),outputDirectory:URL(fileURLWithPath:"/output/movie"))
  }
  func testAllThreeModesHaveExactComponentsAndArbitrarySourceFrameEndpoint() throws {
    for mode in ["latent_only","refine","pixel_spatial"] {
      let request=try parse(fixture(mode:mode))
      XCTAssertEqual(request.plan.frames,98);XCTAssertEqual(request.plan.paddedFrames,105)
      XCTAssertEqual(request.plan.visibleLastFrame,97)
      XCTAssertEqual(try request.chunkPlans().map(\.frames),[98])
      XCTAssertEqual(request.components.count,mode == "latent_only" ? 2 : mode == "refine" ? 6 : 7)
    }
    var body=fixture();body["mode"]="latent_only"
    XCTAssertThrowsError(try parse(body))
    body=fixture(mode:"pixel_spatial");var components=body["components"] as! [String:String];components.removeValue(forKey:"pixel_spatial_adapter");body["components"]=components
    XCTAssertThrowsError(try parse(body))
  }
  func testNestedUnknownFieldsBooleanIntegerAndUnsupportedSilentSettingsReject() throws {
    var body=fixture();body["stage_one_loras"]=[];XCTAssertThrowsError(try parse(body))
    body=fixture();var source=body["source"] as! [String:Any];source["frames"]=true;body["source"]=source;XCTAssertThrowsError(try parse(body))
    body=fixture();source=body["source"] as! [String:Any];source["unfrozen_audio_policy"]="generated";body["source"]=source;XCTAssertThrowsError(try parse(body))
    body=fixture();body["resume"]=true;XCTAssertThrowsError(try parse(body))
    body=fixture();source=body["source"] as! [String:Any];source["rgb_path"]="relative.rgb24";body["source"]=source;XCTAssertThrowsError(try parse(body))
  }
  func testLongContextIsBoundedByActualPaddedAudioClockRatherThanOldTwentySeconds() throws {
    var body=fixture();var source=body["source"] as! [String:Any]
    source["frames"]=721;source["duration_seconds"]=721.0/24;body["source"]=source
    XCTAssertNoThrow(try parse(body).chunkPlans())
    source["frames"]=1449;source["duration_seconds"]=1449.0/24;body["source"]=source
    XCTAssertThrowsError(try parse(body).chunkPlans())
  }
  func testExplicitSidecarSilenceAndExternalEndpointHashesCannotBeSilentlyIgnored() throws {
    var body=fixture();body["audio_policy"]="sidecar"
    XCTAssertThrowsError(try parse(body))
    body["audio_source"]=["path":"/audio/source.wav","sha256":String(repeating:"c",count:64),"start_seconds":0,"duration_seconds":NSNull()]
    XCTAssertEqual(try parse(body).audioPolicy,.sidecar)
    body["audio_policy"]="silence";XCTAssertThrowsError(try parse(body))
    body.removeValue(forKey:"audio_source");XCTAssertEqual(try parse(body).audioPolicy,.silence)
    body["reference_images"]=[["path":"/reference/first.png","role":"first","strength":0.7,"crf":33]]
    XCTAssertThrowsError(try parse(body))
    body["reference_image_sha256"]=["/reference/first.png":String(repeating:"d",count:64)]
    XCTAssertEqual(try parse(body).referenceImages.count,1)
    body["reference_image_sha256"]=["/reference/first.png":String(repeating:"D",count:64)]
    XCTAssertThrowsError(try parse(body))
  }
  func testLongLowResolutionChunkingObeysPaddedNativeAudioClock() throws {
    var body=fixture();var source=body["source"] as! [String:Any]
    source["frames"]=2881;source["duration_seconds"]=2881.0/24;body["source"]=source
    body["chunking"]=true
    let chunks=try parse(body).chunkPlans()
    XCTAssertEqual(chunks.reduce(0) { $0+$1.frames },2881)
    XCTAssertTrue(chunks.count>1)
    XCTAssertTrue(chunks.allSatisfy { Int(ceil(Double($0.paddedFrames)/24*25))<=1501 })
    source["width"]=Int.max;body["source"]=source
    XCTAssertThrowsError(try parse(body))
  }

}
