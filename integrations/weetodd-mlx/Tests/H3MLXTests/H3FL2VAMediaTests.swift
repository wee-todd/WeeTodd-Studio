import Foundation
import XCTest
@testable import H3MLX

final class H3FL2VAMediaTests: XCTestCase {
  func testExpandedKeyframeCompilerPassesCanvasAdmissionToMedia() throws {
    let png=Data(base64Encoded:"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII=")!
    let file=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".png")
    try png.write(to:file);defer {try? FileManager.default.removeItem(at:file)}
    let root:[String:Any]=["format":"weetodd-headless-v2","engine":"h3","prompt":"A robot waves.",
      "components":["task":"fl2va","transformer":"/base","text_encoder":"/qwen","tokenizer":"/tokenizer","video_vae":"/video","audio_vae":"/audio"],
      "config":["width":1536,"height":864,"duration_seconds":3.0,"steps":21,"seed":42],
      "conditioning":["version":1,"task":"fflf","audio_policy":"generated","inputs":[["id":"first","kind":"image","role":"first","frame_index":0,"path":file.path,"sha256":String(repeating:"a",count:64)]]]]
    XCTAssertThrowsError(try H3FL2VAMedia.load(path:file.path,width:1536,height:864,first:true))
    let request=try H3StudioRecipe.compileFL2VA(data:JSONSerialization.data(withJSONObject:root),canvasAdmission:.spatialRefinement) {path,first,geometry in
      XCTAssertEqual(geometry.canvasAdmission,.spatialRefinement)
      return try H3FL2VAMedia.load(path:path,width:geometry.width,height:geometry.height,first:first,canvasAdmission:geometry.canvasAdmission).image
    }
    XCTAssertEqual(request.images[0].width,1536);XCTAssertEqual(request.images[0].height,864)
  }
  func testFirstAndLastCanvasKeepSourceTopAtTop() throws {
    // Independent 2×2 PNG fixture: red top row, blue bottom row.
    let png = Data(base64Encoded:
      "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII=")!
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString + ".png")
    try png.write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    for (width, height, first) in [(32, 32, true), (32, 32, false),
      (1376, 768, true), (1376, 768, false)] {
      let loaded = try H3FL2VAMedia.load(path: file.path,
        width: width, height: height, first: first)
      let bytes = Array(loaded.image.rgb8)
      XCTAssertGreaterThan(Int(bytes[0]), Int(bytes[2]) + 100)
      let lower = ((height - 1) * width) * 3
      XCTAssertGreaterThan(Int(bytes[lower + 2]), Int(bytes[lower]) + 100)
    }
  }
}
