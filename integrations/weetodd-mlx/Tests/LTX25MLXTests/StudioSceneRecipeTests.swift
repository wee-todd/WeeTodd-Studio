import Foundation
import XCTest
@testable import LTX25MLX

final class StudioSceneRecipeTests: XCTestCase {
  private func fixture() -> [String: Any] {
    ["format":"weetodd-headless-v2","engine":"ltx25","prompt":"Two continuous shots.",
      "components":["transformer_path":"/models/transformer",
        "text_encoder_path":"/models/text","video_vae_path":"/models/video",
        "audio_vae_path":"/models/audio","spatial_upscaler_path":"/models/upscale"],
      "config":["pipeline_mode":"distilled","width":768,"height":448,
        "duration_seconds":5,"frame_rate":24,"seed":11,"stage1_steps":8,"stage2_steps":3],
      "conditioning":["version":1,"task":"t2v","audio_policy":"generated",
        "inputs":[]],
      "scene":["version":1,"overlap_frames":25,"boundary_image_policy":"balanced",
        "soundscape":"","music":"","segments":[
          ["clip_id":"first","prompt":"A warrior enters.","duration_seconds":2,"seed":11],
          ["clip_id":"second","prompt":"He turns.","duration_seconds":3,"seed":12]]]]
  }

  func testCompilesTwoValidatedWindowRequestsWithoutDroppingSeedsOrPrompts() throws {
    let compiled = try MLXStudioSceneRecipe.compile(
      data:JSONSerialization.data(withJSONObject:fixture()),outputDirectory:"/output")
    XCTAssertEqual(compiled.plan.totalFrames,121)
    XCTAssertEqual(compiled.requests.map(\.frames),[49,97])
    XCTAssertEqual(compiled.requests.map(\.seed),[11,12])
    XCTAssertEqual(compiled.requests.map(\.prompt),["A warrior enters.","He turns."])
    XCTAssertEqual(compiled.requests.map(\.task),["t2v","t2v"])
  }

  func testOpeningImageAppliesOnlyToFirstWindow() throws {
    var object=fixture()
    var conditioning=object["conditioning"] as! [String:Any]
    conditioning["task"]="fflf"
    conditioning["inputs"]=[["id":"first","kind":"image","role":"keyframe",
      "path":"/first.png","frame_index":0,"strength":1]]
    object["conditioning"]=conditioning
    let compiled=try MLXStudioSceneRecipe.compile(
      data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output")
    XCTAssertEqual(compiled.requests.map(\.task),["i2v","t2v"])
    XCTAssertEqual(compiled.requests[0].referenceImages.map(\.path),["/first.png"])
    XCTAssertTrue(compiled.requests[1].referenceImages.isEmpty)
  }

  func testExplicitWindowedDecodeKeepsOrdinarySceneSampling() throws {
    var object=fixture()
    var scene=object["scene"] as! [String:Any]
    scene["decode_mode"]="windowed"
    scene["decode_window_frames"]=49
    object["scene"]=scene
    let compiled=try MLXStudioSceneRecipe.compile(
      data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output")
    XCTAssertEqual(compiled.decodeMode.publicationMode,
      "windowed_decode_native_latent_chain")
    XCTAssertEqual(compiled.decodeMode.maximumWindowFrames,49)
    XCTAssertEqual(compiled.requests.map(\.frames),[49,97])
  }

  func testRejectsNonStringSceneDecodeMode() throws {
    var object=fixture()
    var scene=object["scene"] as! [String:Any]
    scene["decode_mode"]=1
    object["scene"]=scene
    XCTAssertThrowsError(try MLXStudioSceneRecipe.compile(
      data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output"))
  }

  func testRejectsLaterOrAdditionalSceneImagesBeforeWeightedWork() throws {
    var object=fixture()
    var conditioning=object["conditioning"] as! [String:Any]
    conditioning["task"]="fflf"
    conditioning["inputs"]=[["id":"first","kind":"image","role":"keyframe",
      "path":"/first.png","frame_index":0,"strength":1],
      ["id":"last","kind":"image","role":"keyframe",
      "path":"/last.png","frame_index":"last","strength":1]]
    object["conditioning"]=conditioning
    XCTAssertThrowsError(try MLXStudioSceneRecipe.compile(
      data:JSONSerialization.data(withJSONObject:object),outputDirectory:"/output"))
  }
}
