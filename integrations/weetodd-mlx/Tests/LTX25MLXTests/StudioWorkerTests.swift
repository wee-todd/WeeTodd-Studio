import XCTest
import ImageIO
@testable import LTX25MLX

final class StudioWorkerTests: XCTestCase {
  func testSceneOpeningImageReportsPreparationAndEncodingBeforeSampling() {
    var progress=MLXStudioProgress(sceneWindowCount:2)
    let preparing=progress.event(stage:"scene_reference_prepare:1:first",completed:1,total:2)
    let encoding=progress.event(stage:"scene_reference_encode:1",completed:21,total:42)
    let ready=progress.event(stage:"scene_reference_weights_released",completed:1,total:1)
    XCTAssertTrue((preparing["message"] as! String).contains("opening image"))
    XCTAssertTrue((encoding["message"] as! String).contains("opening image"))
    XCTAssertTrue((ready["message"] as! String).contains("ready"))
    XCTAssertGreaterThanOrEqual(ready["fraction"] as! Double,encoding["fraction"] as! Double)
  }
  func testSceneProgressAdvancesAcrossBothWindowsBeforeDecode() throws {
    var progress=MLXStudioProgress(sceneWindowCount:2)
    let stages:[(String,Int,Int)]=[
      ("scene_text_1:gemma",48,48),("scene_text_2:gemma",48,48),
      ("scene_text_weights_released",1,1),
      ("scene_window_1:sampling",11,11),("scene_window_released",1,2),
      ("scene_window_2:sampling",11,11),("scene_window_released",2,2),
      ("video_decode",121,121),("audio:decode",1,1),
      ("ready_to_publish",1,1)]
    let fractions=stages.map {
      progress.event(stage:$0.0,completed:$0.1,total:$0.2)["fraction"] as! Double
    }
    XCTAssertEqual(fractions,fractions.sorted())
    XCTAssertGreaterThan(fractions[5],fractions[3])
    XCTAssertLessThan(fractions.last!,1)
  }
  func testSceneProgressAdvancesWithinAnExpensiveTransformerStep() {
    var progress=MLXStudioProgress(sceneWindowCount:2)
    let start=progress.event(stage:"scene_window_1:sampling",completed:0,total:11)["fraction"] as! Double
    let middle=progress.event(stage:"scene_window_1:stage1:transformer",completed:24,total:48)["fraction"] as! Double
    let end=progress.event(stage:"scene_window_1:stage1:transformer",completed:48,total:48)["fraction"] as! Double
    XCTAssertGreaterThan(middle,start)
    XCTAssertGreaterThan(end,middle)
  }
  func testA2VEncodingStartsBeforePromptAndSamplingProgress() throws {
    var progress=MLXStudioProgress()
    let stages:[(String,Int,Int)]=[("audio_encode",1,9),("audio_encode",9,9),
      ("audio_encoder_weights_released",0,1),("text:layer",48,48),
      ("sampling",1,11),("stage1:block",48,48),("sampling",8,11),
      ("upscale:begin",0,1),("sampling",11,11),("video_decode",49,49),
      ("source_audio_published",0,1),("ready_to_publish",1,1)]
    let fractions=stages.map { progress.event(stage:$0.0,completed:$0.1,total:$0.2)["fraction"] as! Double }
    XCTAssertEqual(fractions,fractions.sorted())
    XCTAssertLessThan(fractions[1],fractions[3],"Source encoding must not consume the audio-decode end phase.")
    XCTAssertLessThan(fractions[1],fractions[4],"Sampling must advance after source encoding.")
    XCTAssertLessThan(fractions[8],fractions[9])
    XCTAssertLessThan(fractions.last!,1)
  }
  func testPublicationIncludesSidecarsAndDoesNotPublishAfterFailure() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let staging=root.appendingPathComponent("partial"),output=root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    XCTAssertThrowsError(try MLXMediaPipeline.publish(staging:staging,output:output) { throw CancellationError() })
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
    try MLXMediaPipeline.publish(staging:staging,output:output) {
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
      try Data("result".utf8).write(to:staging.appendingPathComponent("result.json"))
    }
    XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("result.json")),Data("result".utf8))
    XCTAssertFalse(FileManager.default.fileExists(atPath:staging.path))
  }

  func testProgressIsMonotonicAndCompletionReservedForPublication() throws {
    var progress=MLXStudioProgress()
    var previous=0.0
    for (stage,completed,total) in [("text:layer",1,48),("sampling",1,11),("stage1:block",48,48),
      ("sampling",8,11),("upscale:begin",0,1),("sampling",11,11),("video_decode",1,89),
      ("video_decode",89,89),("audio:decode",0,1),("ready_to_publish",1,1)] {
      let event=progress.event(stage:stage,completed:completed,total:total)
      XCTAssertEqual(event["event"] as? String,"progress")
      let fraction=try XCTUnwrap(event["fraction"] as? Double)
      XCTAssertGreaterThanOrEqual(fraction,previous);XCTAssertLessThan(fraction,1)
      previous=fraction
    }
  }
  func testRippleProgressUsesEightStepsAndReferenceEncoding() throws {
    var progress=MLXStudioProgress()
    var previous=0.0
    for (stage,done,total) in [("text:gemma",48,48),("guide_encode",1,4),
      ("guide_encode",4,4),("anchor_encode",42,42),("ripple:block",48,48),
      ("sampling",1,8),("sampling",8,8),("video_decode",1,10),
      ("video_decode",10,10),("ready_to_publish",1,1)] {
      let event=progress.event(stage:stage,completed:done,total:total)
      let fraction=try XCTUnwrap(event["fraction"] as? Double)
      XCTAssertGreaterThanOrEqual(fraction,previous)
      XCTAssertLessThan(fraction,1)
      previous=fraction
    }
  }
  func testIngredientsProgressCoversSheetEncodingAndEightSamplingSteps() throws {
    var progress=MLXStudioProgress()
    let stages:[(String,Int,Int)]=[("text:gemma",48,48),
      ("ingredients_guide_encode",1,4),("ingredients_guide_encode",4,4),
      ("ingredients:block",24,48),("sampling",1,8),
      ("sampling",8,8),("video_decode",121,121),("ready_to_publish",1,1)]
    let events=stages.map { progress.event(stage:$0.0,completed:$0.1,total:$0.2) }
    let fractions=events.map { $0["fraction"] as! Double }
    XCTAssertEqual(fractions,fractions.sorted())
    XCTAssertTrue((events[2]["message"] as! String).contains("Ingredients sheet"))
    XCTAssertTrue((events[3]["message"] as! String).contains("Ingredients"))
    XCTAssertLessThan(fractions.last!,1)
  }
  func testDecodedPreviewIsBoundedAndAtomicallyReplaced() throws {
    let file=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".png")
    defer { try? FileManager.default.removeItem(at:file) }
    for color:UInt8 in [40,200] {
      try MLXStudioPreview.write(rgb:Data(repeating:color,count:1920*1088*3),width:1920,height:1088,to:file)
      let source=try XCTUnwrap(CGImageSourceCreateWithURL(file as CFURL,nil))
      let image=try XCTUnwrap(CGImageSourceCreateImageAtIndex(source,0,nil))
      XCTAssertEqual(image.width,640);XCTAssertLessThanOrEqual(image.height,640)
    }
    XCTAssertThrowsError(try MLXStudioPreview.write(rgb:Data(),width:1920,height:1088,to:file))
  }
  func testTemporalDFRProgressAdvancesThroughRoundsAndTiles() {
    var progress=MLXStudioProgress(temporalRounds:2)
    let spatial=progress.event(stage:"sampling",completed:11,total:11)["fraction"] as! Double
    _=progress.event(stage:"temporal_upscaler_weights_released",completed:1,total:2)
    _=progress.event(stage:"temporal_tiles",completed:0,total:2)
    let first=progress.event(stage:"temporal_sampling",completed:2,total:4)
    let tile=progress.event(stage:"temporal_tile_complete",completed:1,total:2)
    _=progress.event(stage:"temporal_upscaler_weights_released",completed:2,total:2)
    _=progress.event(stage:"temporal_tiles",completed:0,total:4)
    let second=progress.event(stage:"temporal_sampling",completed:1,total:4)
    let complete=progress.event(stage:"temporal_tile_complete",completed:4,total:4)
    XCTAssertGreaterThan(first["fraction"] as! Double,spatial)
    XCTAssertGreaterThan(tile["fraction"] as! Double,first["fraction"] as! Double)
    XCTAssertGreaterThan(second["fraction"] as! Double,tile["fraction"] as! Double)
    XCTAssertGreaterThan(complete["fraction"] as! Double,second["fraction"] as! Double)
    XCTAssertTrue((second["message"] as! String).contains("round 2/2"))
    XCTAssertLessThan(complete["fraction"] as! Double,0.83)
  }
  func testTemporalDFRPreviewUsesPublishedFrameCount() {
    XCTAssertTrue(MLXStudioPreview.shouldEmit(index:0,total:193,secondsSinceLast:0))
    XCTAssertFalse(MLXStudioPreview.shouldEmit(index:48,total:193,secondsSinceLast:0))
    XCTAssertTrue(MLXStudioPreview.shouldEmit(index:192,total:193,secondsSinceLast:0))
    XCTAssertFalse(MLXStudioPreview.shouldEmit(index:193,total:193,secondsSinceLast:2))
  }
}
