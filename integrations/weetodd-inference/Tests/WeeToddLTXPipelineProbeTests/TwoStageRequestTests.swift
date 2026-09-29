import XCTest
import Foundation
@testable import WeeToddLTXPipelineProbe

final class TwoStageRequestTests: XCTestCase {
  func testOptionalUpscalerSelectsExplicitTwoStageContractAndRejectsBadPaths() throws {
    let base: [String:Any] = ["gemma_root":"/models/gemma","transformer_root":"/models/transformer",
      "connector_checkpoint":"/models/fixed.safetensors","video_checkpoint":"/models/video.safetensors",
      "audio_checkpoint":"/models/audio.safetensors","prompt":"A test.","width":128,"height":64,
      "frames":9,"fps":24,"seed":42,"output_directory":"/outputs/new"]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".json")
    defer { try? FileManager.default.removeItem(at: url) }
    for path in ["/models/upscaler.safetensors","relative","", "/invalid\u{0}"] {
      var values = base; values["spatial_upscaler_checkpoint"] = path
      try JSONSerialization.data(withJSONObject: values).write(to: url)
      if path == "/models/upscaler.safetensors" {
        XCTAssertEqual(try PipelineProbe.loadRequest(at: url).spatialUpscalerCheckpoint,path)
      } else { XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url)) }
    }
    var invalid = base; invalid["spatial_upscaler_checkpoint"] = NSNull()
    try JSONSerialization.data(withJSONObject: invalid).write(to: url)
    XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url))
  }
}

extension TwoStageRequestTests {
  func testReportDescribesActualRecipeAndEveryNoiseStream() throws {
    let legacy = PipelineProbe.recipeReport(seed: 42,twoStage: false)
    XCTAssertEqual(legacy["model_evaluations"] as? Int,8)
    XCTAssertEqual(legacy["recipe"] as? String,"legacy-stage-one")
    XCTAssertTrue((legacy["qualification"] as? String)?.contains("stage-one") == true)
    let modern = PipelineProbe.recipeReport(seed: UInt64.max,twoStage: true)
    XCTAssertEqual(modern["model_evaluations"] as? Int,11)
    let streams = try XCTUnwrap(modern["noise_streams"] as? [[String:Any]])
    XCTAssertEqual(streams.count,3)
    XCTAssertEqual(streams[0]["seed"] as? UInt64,UInt64.max)
    XCTAssertEqual(streams[1]["seed"] as? UInt64,9999)
    XCTAssertEqual(streams[2]["seed"] as? UInt64,1)
    XCTAssertEqual(streams[2]["draw_order"] as? String,"refinement_video, refinement_audio")
    XCTAssertFalse((modern["qualification"] as? String)?.contains("not the full two-stage recipe") == true)
    XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: modern))
  }
}

extension TwoStageRequestTests {
  func testOrderedLoRARequestPreservesSelectionAndRejectsUnsupportedOptions() throws {
    let base: [String:Any] = ["gemma_root":"/models/gemma","transformer_root":"/models/transformer",
      "connector_checkpoint":"/models/fixed.safetensors","video_checkpoint":"/models/video.safetensors",
      "audio_checkpoint":"/models/audio.safetensors","prompt":"A test.","width":128,"height":64,
      "frames":9,"fps":24,"seed":42,"output_directory":"/outputs/new"]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".json")
    defer { try? FileManager.default.removeItem(at: url) }
    var request = base
    request["loras"] = [["path":"/models/older-2.3.safetensors","strength":0.8],
      ["path":"/models/newer-2.5.safetensors","strength":-0.25,"enabled":false]] as [[String:Any]]
    try JSONSerialization.data(withJSONObject: request).write(to: url)
    let parsed = try PipelineProbe.loadRequest(at: url)
    XCTAssertEqual(parsed.loras?.map(\.path),["/models/older-2.3.safetensors","/models/newer-2.5.safetensors"])
    XCTAssertEqual(parsed.loras?.map(\.enabled),[true,false])
    for invalid: Any in [NSNull(),["path":"/models/a","strength":1],
      [["path":"relative","strength":1]],[["path":"/a","strength":1,"mode":"reference"]],
      [["path":"/a","strength":1,"enabled":NSNull()]],
      Array(repeating:["path":"/a","strength":1] as [String:Any],count:17)] {
      request["loras"] = invalid
      try JSONSerialization.data(withJSONObject: request).write(to: url)
      XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url))
    }
  }
}
