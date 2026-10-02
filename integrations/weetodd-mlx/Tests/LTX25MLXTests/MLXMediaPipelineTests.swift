import XCTest
import Foundation
@testable import LTX25MLX
import LTX25Video

final class MLXMediaPipelineTests:XCTestCase {
  func testPaddedSpatialDFRDecodesCanvasWhileTemporalDecodesPublishedFrames() throws {
    let helper=MLXDistilledRequestTests()
    var values=helper.base()
    values["version"]=8;values["task"]="dfr";values["frames"]=41
    values["width"]=512;values["height"]=256
    values["noise_policy"]="mlx_threefry_bf16_v1"
    values["reference_images"]=[];values["audio_reference"]=NSNull()
    values["union_control_guide"]=NSNull();values["ingredients_sheet"]=NSNull()
    values["msr"]=NSNull()
    values["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":0.5]
    let spatial=try helper.decode(values)
    let budget=8*1024*1024*1024
    XCTAssertEqual(try MLXMediaPipeline.admit(spatial,videoActivationBytes:budget,
      transformerActivationBytes:budget,videoBackend:.mlx).videoFrames,49)
    values["version"]=9
    values["dfr"]=["adapter_path":"/pixel-spatial.safetensors","adapter_strength":0.5,
      "temporal_upscaler_path":"/temporal.safetensors","temporal_rounds":1]
    let temporal=try helper.decode(values)
    XCTAssertEqual(try MLXMediaPipeline.admit(temporal,videoActivationBytes:budget,
      transformerActivationBytes:budget,videoBackend:.mlx).videoFrames,81)
  }

  func testExtensionPublishesExactGeneratedAudioInterval() throws {
    XCTAssertEqual(try MLXMediaPipeline.extensionAudioRange(contextFrames:25,
      additionalFrames:24,fps:24,decodedSamples:98400),50000..<98000)
    XCTAssertThrowsError(try MLXMediaPipeline.extensionAudioRange(contextFrames:25,
      additionalFrames:24,fps:24,decodedSamples:97999))
  }
  func testStudioDurationRangeAdmitsWithinWorkerBudgetsAndStillEnforcesMemory() throws {
    let helper=MLXDistilledRequestTests()
    for frames in [121,129,137,145,241,361] {
      var values=helper.base();values["width"]=768;values["height"]=448;values["frames"]=frames
      let request=try helper.decode(values)
      let admission=try MLXMediaPipeline.admit(request,videoActivationBytes:12*1024*1024*1024,
        transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx,audioBackend:.mlx)
      XCTAssertEqual(admission.videoFrames,frames)
      // Execute-side configuration must admit the exact same shape and byte bound.
      let geometry=try request.recipe().high
      let configuration=MLXMediaPipeline.videoConfiguration(for:geometry,activationBytes:12*1024*1024*1024)
      let decodePlan=try MLXVideoDecodePlan(shape:geometry.videoShape,configuration:configuration)
      XCTAssertEqual(decodePlan.outputShape,[frames,448,768,3])
      XCTAssertEqual(decodePlan.admittedActivationBytes,admission.videoActivationBytes)
      XCTAssertGreaterThan(admission.audioSamples,0)
      XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:admission.videoActivationBytes-1,
        transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx,audioBackend:.mlx))
    }
  }
  func testTwentySecondEndpointKeepsAudioAndVideoAdmissionConsistent() throws {
    let helper=MLXDistilledRequestTests()
    var values=helper.base();values["width"]=448;values["height"]=256;values["frames"]=481
    let request=try helper.decode(values)
    let admitted=try MLXMediaPipeline.admit(request,videoActivationBytes:12*1024*1024*1024,
      transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx,audioBackend:.mlx)
    XCTAssertEqual(admitted.videoFrames,481)
    values["width"]=768;values["height"]=448
    XCTAssertThrowsError(try MLXMediaPipeline.admit(helper.decode(values),videoActivationBytes:12*1024*1024*1024,
      transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx,audioBackend:.mlx)) { error in
      XCTAssertTrue(String(describing:error).contains("activation/workspace bytes"),"Must retain the real memory limit: \(error)")
    }
  }
  func testMatchedFullSizeRequiresExplicitWorkspace() throws {
    let helper=MLXDistilledRequestTests()
    var values=helper.base();values["width"]=1344;values["height"]=768;values["frames"]=89
    let request=try helper.decode(values)
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request))
    let admitted=try MLXMediaPipeline.admit(request,videoActivationBytes:12*1024*1024*1024,transformerActivationBytes:12*1024*1024*1024)
    XCTAssertEqual(admitted.videoFrames,89)
    let mlx=try MLXMediaPipeline.admit(request,videoActivationBytes:12*1024*1024*1024,transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx)
    XCTAssertLessThan(mlx.videoActivationBytes,admitted.videoActivationBytes)
    XCTAssertNoThrow(try MLXMediaPipeline.admit(request,videoActivationBytes:mlx.videoActivationBytes,transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:mlx.videoActivationBytes,transformerActivationBytes:12*1024*1024*1024,videoBackend:.mps))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:mlx.videoActivationBytes-1,transformerActivationBytes:12*1024*1024*1024,videoBackend:.mlx))
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:12*1024*1024*1024))
  }
  func testGeometryAndOutputAdmissionRunsWithoutModelIO() throws {
    let helper=MLXDistilledRequestTests(), request=try helper.decode(helper.base())
    let admission=try MLXMediaPipeline.admit(request)
    XCTAssertEqual(admission.videoFrames,33)
    XCTAssertEqual(admission.audioSamples,65760)
    XCTAssertEqual(admission.audioEstimatedBytes,try MLXMediaPipeline.admit(request,audioBackend:.mps).audioEstimatedBytes)
    let mlx=try MLXMediaPipeline.admit(request,audioBackend:.mlx)
    XCTAssertEqual(mlx.audioSamples,admission.audioSamples)
    XCTAssertGreaterThan(mlx.audioEstimatedBytes,admission.audioEstimatedBytes)
    var tooLarge=helper.base(); tooLarge["width"]=1344; tooLarge["height"]=768; tooLarge["frames"]=145
    XCTAssertThrowsError(try MLXMediaPipeline.admit(helper.decode(tooLarge)))
  }
  func testExplicitDecoderWorkspaceAdmitsTwoSecondsWithoutChangingDefault() throws {
    let helper=MLXDistilledRequestTests()
    var values=helper.base();values["width"]=448;values["height"]=256;values["frames"]=49
    let request=try helper.decode(values)
    XCTAssertThrowsError(try MLXMediaPipeline.admit(request))
    let admitted=try MLXMediaPipeline.admit(request,videoActivationBytes:576*1024*1024)
    XCTAssertEqual(admitted.videoFrames,49)
    XCTAssertEqual(admitted.videoActivationBytes,555053056)
    for limit in [0,-1,Int.max] {
      XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:limit))
    }
    values["width"]=1344;values["height"]=768;values["frames"]=145
    XCTAssertThrowsError(try MLXMediaPipeline.admit(helper.decode(values),videoActivationBytes:576*1024*1024))
  }
  func testExplicitWorkspaceAdmitsThreeToFiveSecondReviewClips() throws {
    let helper=MLXDistilledRequestTests()
    for (frames,bytes) in [(73,819982336),(97,1084911616),(121,1349840896)] {
      var values=helper.base();values["width"]=448;values["height"]=256;values["frames"]=frames
      let request=try helper.decode(values)
      XCTAssertThrowsError(try MLXMediaPipeline.admit(request))
      let admission=try MLXMediaPipeline.admit(request,videoActivationBytes:1536*1024*1024)
      XCTAssertEqual(admission.videoFrames,frames)
      XCTAssertEqual(admission.videoActivationBytes,bytes)
      XCTAssertThrowsError(try MLXMediaPipeline.admit(request,videoActivationBytes:MLXMediaPipeline.maximumVideoActivationMiB*1024*1024+1))
    }
  }
}
