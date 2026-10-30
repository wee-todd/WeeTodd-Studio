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
  func testIntermediateExportIsOptionalAndBindsResumeWithoutChangingSampling() throws {
    let original=try parse(fixture(mode:"pixel_spatial"))
    XCTAssertFalse(original.intermediateOnly)
    var body=fixture(mode:"pixel_spatial");body["intermediate_only"]=true
    let intermediate=try parse(body)
    XCTAssertTrue(intermediate.intermediateOnly)
    XCTAssertNotEqual(original.contractSHA256,intermediate.contractSHA256)
    XCTAssertEqual(original.plan.sigmas,intermediate.plan.sigmas)
    XCTAssertEqual(original.plan.fps,intermediate.plan.fps)
    body["intermediate_only"]=1
    XCTAssertThrowsError(try parse(body))
  }
  func testNonChunkedFullHDEnhanceKeepsExactVisibleFrameRangeWithoutFloatingBudget() throws {
    for frames in [24,120,1] {
      var body=fixture(mode:"enhance"),components=fixture(mode:"enhance")["components"] as! [String:String]
      components.removeValue(forKey:"spatial_upscaler_checkpoint");body["components"]=components
      var source=body["source"] as! [String:Any]
      source["width"]=1920;source["height"]=1088;source["frames"]=frames
      source["duration_seconds"]=Double(frames)/24;body["source"]=source
      body["anchors"]="none";body["chunk_frame_megapixel_budget"]=0.01
      let request=try parse(body),chunks:[MLXMovieUpscalePlan.Chunk]
      do { chunks=try request.chunkPlans(cutFrames:[12]) }
      catch { XCTFail("Nonchunked \(frames)-frame source must retain one complete interval: \(error)");continue }
      XCTAssertEqual(chunks.count,1);XCTAssertEqual(chunks.first?.startFrame,0)
      XCTAssertEqual(chunks.first?.endFrame,frames)
      XCTAssertEqual(chunks.first?.paddedFrames,frames == 24 ? 25 : frames == 120 ? 121 : 1)
      XCTAssertEqual(request.plan.size.outputWidth,1920);XCTAssertEqual(request.plan.size.outputHeight,1088)
      XCTAssertEqual(request.plan.fps,24)
    }
  }
  func testNonChunkedDirectRangeStillRejectsPaddedAudioAndVideoOverflow() throws {
    var body=fixture();body["anchors"]="none"
    var source=body["source"] as! [String:Any]
    source["frames"]=1433;source["duration_seconds"]=1433.0/24;body["source"]=source
    XCTAssertEqual(try parse(body).chunkPlans().map(\.paddedFrames),[1433])
    for frames in [1434,1441,4098] {
      source["frames"]=frames;source["duration_seconds"]=Double(frames)/24;body["source"]=source
      XCTAssertThrowsError(try parse(body).chunkPlans())
    }
  }
  func testEnhanceAcceptsOnlySameSizeComponentsAndCQStack() throws {
    var body=fixture(mode:"enhance"),components=fixture(mode:"enhance")["components"] as! [String:String]
    components.removeValue(forKey:"spatial_upscaler_checkpoint");body["components"]=components
    body["refinement_strength"]=0.5
    body["refinement_loras"]=[["path":"/model/cq.safetensors","strength":1.0,"enabled":true]]
    let request=try parse(body)
    XCTAssertEqual(request.components.count,5)
    XCTAssertEqual(request.refinementAdapters.map(\.path),["/model/cq.safetensors"])
    XCTAssertEqual(request.refinementAdapters.map(\.strength),[1])
    XCTAssertEqual(request.plan.size.outputWidth,64);XCTAssertEqual(request.plan.size.outputHeight,32)
    XCTAssertEqual(request.plan.refinementStrength,0.5);XCTAssertEqual(request.plan.fps,24)
    XCTAssertEqual(request.audioPolicy,.source)
    XCTAssertEqual(request.componentIdentityPaths["refinement_lora_0"],"/model/cq.safetensors")
    components["spatial_upscaler_checkpoint"]="/model/upscaler.safetensors";body["components"]=components
    XCTAssertThrowsError(try parse(body))
    components.removeValue(forKey:"spatial_upscaler_checkpoint")
    components["pixel_spatial_adapter"]="/model/pixel.safetensors";body["components"]=components
    XCTAssertThrowsError(try parse(body))
  }
  func testPixelMovieAdmitsOrderedCQRefinementWithoutChangingSamplingOrFPS() throws {
    var body=fixture(mode:"pixel_spatial")
    body["refinement_loras"]=[["path":"/model/cq.safetensors","strength":0.5,"enabled":true]]
    let request=try parse(body)
    XCTAssertEqual(request.refinementAdapters.map(\.path),["/model/pixel.safetensors","/model/cq.safetensors"])
    XCTAssertEqual(request.refinementAdapters.map(\.strength),[1,0.5])
    XCTAssertEqual(request.componentIdentityPaths["refinement_lora_0"],"/model/cq.safetensors")
    XCTAssertEqual(request.plan.fps,24)
    XCTAssertEqual(request.plan.sigmas.count,4)
    XCTAssertEqual(request.plan.refinementStrength,0.35)
    XCTAssertNotEqual(request.contractSHA256,try parse(fixture(mode:"pixel_spatial")).contractSHA256)
    body["refinement_loras"]=[["path":"/missing/cq.safetensors","strength":0.5,"enabled":false]]
    XCTAssertEqual(try parse(body).refinementAdapters.count,1)
    XCTAssertNil(try parse(body).componentIdentityPaths["refinement_lora_0"])
  }
  func testRefinementAdaptersRejectUnusedDuplicateAndUnboundedSelections() throws {
    var body=fixture(mode:"latent_only")
    body["refinement_loras"]=[["path":"/model/cq.safetensors","strength":0.5]]
    XCTAssertThrowsError(try parse(body))
    body=fixture(mode:"pixel_spatial");body["refinement_loras"]=[["path":"/model/pixel.safetensors","strength":0.5]]
    XCTAssertThrowsError(try parse(body))
    body["refinement_loras"]=[["path":"/model/cq.safetensors","strength":4]]
    XCTAssertThrowsError(try parse(body))
    body["refinement_loras"]=[["path":"/model/cq.safetensors","strength":0.5],["path":"/model/cq.safetensors","strength":1]]
    XCTAssertThrowsError(try parse(body))
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
