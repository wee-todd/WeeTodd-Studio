import XCTest
import MLX
import LTX25Engine
import AdapterRuntime
import TensorIO
import InferenceTestSupport
@testable import LTX25MLX

final class MLXMoviePixelEnhanceTests:XCTestCase {
  private func body(mode:String="enhance_pixel_spatial",pixel:String="/model/pixel.safetensors",
    cq:String="/model/cq.safetensors",video:String="/model/video.safetensors",
    width:Int=1920,height:Int=1088,frames:Int=24)->[String:Any] {
    var components=["video_checkpoint":video,"gemma_root":"/model/gemma",
      "connector_checkpoint":"/model/fixed.safetensors","transformer_root":"/model/transformer",
      "audio_checkpoint":"/model/audio.safetensors"]
    if mode == "enhance_pixel_spatial" { components["pixel_spatial_adapter"]=pixel }
    return ["version":1,"engine":"ltx25","task":"video_upscale",
      "source":["path":"/media/source.mov","sha256":String(repeating:"a",count:64),
        "rgb_path":"/media/source.rgb24","rgb_sha256":String(repeating:"b",count:64),
        "width":width,"height":height,"frames":frames,"fps":24,"start_seconds":0,"duration_seconds":Double(frames)/24],
      "components":components,"mode":mode,"size_policy":"strict_32","output_directory":"/output/movie",
      "prompt":"Preserve the woman's face, framing and motion.","seed":42,
      "refinement_strength":0.5,"anchors":"none","anchor_strength":0.7,"pixel_strength":1,
      "refinement_loras":[["path":cq,"strength":1,"enabled":true]],"reference_images":[],
      "maximum_audio_drift_seconds":0.05,"chunking":false,"chunk_frame_megapixel_budget":150,
      "resume":false,"keep_chunks":false]
  }
  private func parse(_ body:[String:Any]) throws -> MLXMovieUpscaleRequest {
    try MLXMovieUpscaleRequest(data:JSONSerialization.data(withJSONObject:body,options:.sortedKeys),
      outputDirectory:URL(fileURLWithPath:"/output/movie"))
  }
  func testPixelEnhanceKeeps1088CanvasAndThreeDenoiseHalfUpdates() throws {
    let request=try parse(body()),plan=request.plan
    XCTAssertEqual(plan.size.outputWidth,1920);XCTAssertEqual(plan.size.outputHeight,1088)
    XCTAssertEqual(plan.size.width,1920);XCTAssertEqual(plan.size.height,1088)
    XCTAssertEqual(plan.frames,24);XCTAssertEqual(plan.fps,24)
    XCTAssertEqual(plan.sigmas.count,4)
    for (actual,expected) in zip(plan.sigmas,[0.5,0.3986254295532646,0.23195876288659792,0]) {
      XCTAssertEqual(actual,expected,accuracy:1e-15)
    }
    XCTAssertEqual(try request.chunkPlans().map(\.frames),[24])
    let mode=try XCTUnwrap(MLXMovieUpscalePlan.Mode(rawValue:"enhance_pixel_spatial"))
    for policy in [MLXMovieUpscalePlan.SizePolicy.fitNearest,.centerCrop,.strict] {
      XCTAssertThrowsError(try MLXMovieUpscalePlan(mode:mode,width:1920,height:1080,frames:24,
        fps:24,sizePolicy:policy,refinementStrength:0.5))
    }
  }
  func testPixelEnhanceAdmitsExactlyTypedPixelThenCQWithoutSpatialUpscaler() throws {
    var value=body();let request=try parse(value)
    XCTAssertEqual(Set(request.components.keys),["video_checkpoint","gemma_root","connector_checkpoint",
      "transformer_root","audio_checkpoint","pixel_spatial_adapter"])
    XCTAssertEqual(request.refinementAdapters.map(\.path),["/model/pixel.safetensors","/model/cq.safetensors"])
    XCTAssertEqual(request.refinementAdapters.map(\.strength),[1,1])
    XCTAssertEqual(request.pixelSpatialAdapter?.path,"/model/pixel.safetensors")
    let experiment=try XCTUnwrap(request.experimentalConditioningMetadata)
    XCTAssertEqual(experiment["checkpoint_reference_downscale_factor"] as? Int,2)
    XCTAssertEqual(experiment["execution_reference_downscale_factor"] as? Int,1)
    XCTAssertEqual(experiment["checkpoint_metadata_modified"] as? Bool,false)
    XCTAssertEqual(experiment["publisher_recipe_parity"] as? Bool,false)
    XCTAssertNil(try parse(body(mode:"enhance")).experimentalConditioningMetadata)
    XCTAssertEqual(request.componentIdentityPaths["pixel_spatial_adapter"],"/model/pixel.safetensors")
    XCTAssertEqual(request.componentIdentityPaths["refinement_lora_0"],"/model/cq.safetensors")
    var components=value["components"] as! [String:String]
    components["spatial_upscaler_checkpoint"]="/model/spatial.safetensors";value["components"]=components
    XCTAssertThrowsError(try parse(value))
    components.removeValue(forKey:"spatial_upscaler_checkpoint");components.removeValue(forKey:"pixel_spatial_adapter")
    value["components"]=components;XCTAssertThrowsError(try parse(value))
    value=body();value["refinement_loras"]=[["path":"/model/pixel.safetensors","strength":1]]
    XCTAssertThrowsError(try parse(value))
  }
  func testPixelEnhanceUsesFrozenFullSizeGuideAtOneToOnePositions() throws {
    let request=try parse(body(width:64,height:32,frames:2))
    let layout=try MLXMovieUpscaleLayout(plan:request.plan,firstStrength:nil,lastStrength:nil)
    XCTAssertEqual(layout.geometry.width,64);XCTAssertEqual(layout.geometry.height,32)
    XCTAssertEqual(layout.groupRows,[4,4]);XCTAssertEqual(layout.referenceTokens,4)
    XCTAssertEqual(layout.mask,[1,1,1,1,0,0,0,0])
    XCTAssertEqual(Array(layout.positions[0..<12]),Array(layout.positions[12..<24]))
    let prepared=try layout.prepare(generated:.ones([4,128])*(-1),source:.ones([4,128])*0.75,first:nil,last:nil)
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),-1)
    XCTAssertEqual(prepared.condition.clean[4,0].item(Float.self),0.75)
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([4,128]),source:nil,first:nil,last:nil))
  }
  func testPixelEnhanceReservesTheSameOneToOneWorkspaceAsCQOnly() throws {
    let video=try MLXDecoderHeaderFixture.convolutional(test:self).path
    let pixel=try parse(body(video:video,width:384,height:256,frames:49))
    let cq=try parse(body(mode:"enhance",video:video,width:384,height:256,frames:49))
    let gib:UInt64=1024*1024*1024
    let p=try MLXMovieMemoryPlan(request:pixel,physicalMemory:256*gib,recommendedWorkingSet:192*gib)
    let c=try MLXMovieMemoryPlan(request:cq,physicalMemory:256*gib,recommendedWorkingSet:192*gib)
    XCTAssertEqual(p.transformerActivationBytes,c.transformerActivationBytes)
    XCTAssertEqual(p.videoActivationBytes,c.videoActivationBytes)
    let expected=try MLXAVBlock.estimatedActivationBytes(configuration:
      AVBlockConfiguration(videoTokens:1344,audioTokens:52,textTokens:1024),perTokenVideo:true)
    XCTAssertEqual(p.transformerActivationBytes,expected)
  }
  private func withPixelFixture(referenceScale:String="2",_ run:(URL)throws->Void) throws {
    let targets=["attn1.to_k","attn1.to_out","attn1.to_q","attn1.to_v",
      "attn2.to_k","attn2.to_out","attn2.to_q","attn2.to_v","ff.proj_in","ff.proj_out"]
    var tensors:[(String,[Int],String)]=[]
    for block in 0..<48 { for target in targets {
      let name="diffusion_model.transformer_blocks.\(block).\(target)"
      tensors += [(name+".lora_A.weight",[32,target == "ff.proj_out" ? 16384 : 4096],"BF16"),
        (name+".lora_B.weight",[target == "ff.proj_in" ? 16384 : 4096,32],"BF16")]
    } }
    try withTensorFile(metadata:["model_version":"2.5","reference_downscale_factor":referenceScale,
      "reference_spatial_scale_factor":"2"],tensors:tensors,run)
  }
  func testTypedPixelRoleKeepsCheckpointFactorTwoAndOrdinaryEnhanceStillRejectsIC() throws {
    try withPixelFixture { pixel in
      try withTensorFile(tensors:[("lora_unet_proj_out.lora_down.weight",[1,4096],"F32"),
        ("lora_unet_proj_out.lora_up.weight",[128,1],"F32")]) { cq in
        let typed=try parse(body(pixel:pixel.path,cq:cq.path))
        XCTAssertNoThrow(try MLXLoRAStack(adapters:typed.refinementAdapters,
          pixelSpatialDFRAdapterPath:typed.components["pixel_spatial_adapter"]))
        let ordinary=try parse(body(mode:"enhance",cq:pixel.path))
        XCTAssertThrowsError(try MLXLoRAStack(adapters:ordinary.refinementAdapters))
        let header=try SafeTensorFile(url:pixel)
        XCTAssertEqual(header.metadata["reference_downscale_factor"],"2")
        XCTAssertEqual(header.metadata["reference_spatial_scale_factor"],"2")
      }
    }
  }
  func testTypedPixelCheckpointWithChangedReferenceMetadataStillRejects() throws {
    try withPixelFixture(referenceScale:"1") { pixel in
      XCTAssertThrowsError(try MLXLoRAStack(adapters:[LoRAAdapter(path:pixel.path,strength:1)],
        pixelSpatialDFRAdapterPath:pixel.path))
    }
  }
}
