import Foundation
import XCTest
@testable import H3MLX

final class H3LearnedSpatialAdmissionTests: XCTestCase {
  func testOrdinaryRequestKeepsOneMPButExplicitSpatialFitsReleasedTwoMPAndAllRows() throws {
    let ordinary = try H3Geometry(width:1920,height:1088,durationSeconds:3)
    XCTAssertEqual(ordinary.maximumPackedRows,40_000)
    let target = try H3Geometry(width:1920,height:1088,durationSeconds:3,canvasAdmission:.spatialRefinement)
    XCTAssertEqual(target.videoRows,44_880)
    XCTAssertEqual(target.maximumPackedRows,64_000)
    XCTAssertEqual(try H3VideoVAEDecoder.preflightSpatial(geometry:target),60)
    let count = try target.packedRows(textRows:200,conditionVideoRows:512,conditionAudioRows:240)
    XCTAssertEqual(count,46_076)
    try target.canvasAdmission.validatePackedRows(count)
    XCTAssertThrowsError(try H3CanvasAdmission.ordinary.validate(width:1920,height:1088))
    XCTAssertThrowsError(try H3Geometry(width:1920,height:1088,durationSeconds:15,canvasAdmission:.spatialRefinement)
      .packedRows(textRows:1,conditionVideoRows:0,conditionAudioRows:0))
    XCTAssertThrowsError(try H3Geometry(width:1920,height:1120,durationSeconds:3,canvasAdmission:.spatialRefinement))
    XCTAssertThrowsError(try target.packedRows(textRows:20_000,conditionVideoRows:512,conditionAudioRows:240))
  }

  func testLargestConvolutionUsesRealHaloWorkspaceAndFiniteTileCountBeforeWeights() throws {
    let plan = try H3UpscalerConvolutionPlan()
    XCTAssertEqual(try plan.workspaceElements(channels:512,kernel:3),113_246_208)
    XCTAssertEqual(try plan.validate(channels:512,kernel:3,bytesPerElement:4),452_984_832)
    XCTAssertEqual(try plan.tileCount(frames:22,height:68,width:120),36)
    XCTAssertLessThan(try plan.workspaceElements(channels:512,kernel:3),UInt64(Int32.max))
    XCTAssertThrowsError(try H3UpscalerConvolutionPlan(frames:17))
    XCTAssertThrowsError(try H3UpscalerConvolutionPlan(maximumWorkspaceBytes:128*1024*1024)
      .validate(channels:512,kernel:3,bytesPerElement:4))
  }

  private func recipe(version:Int=2,mode:String="spatial", learned:Bool=true) throws -> Data {
    var fields:[String:Any] = ["version":version,"mode":mode,"source_manifest":"/tmp/full-source.json",
      "source_manifest_sha256":String(repeating:"a",count:64),"strength":0.2,"preserve_audio":true]
    if learned {
      fields["learned_upscaler_path"] = "/tmp/learned.safetensors"
      fields["learned_upscaler_header_sha256"] = String(repeating:"b",count:64)
    } else { fields["resize_method"] = "bilinear" }
    return try JSONSerialization.data(withJSONObject:["refinement":fields,
      "config":["width":1920,"height":1088,"duration_seconds":3,"steps":16]])
  }

  func testVersionTwoSpatialAndExclusiveLearnedPathAreExplicitAndLegacyCannotExpand() throws {
    let prepared = try XCTUnwrap(H3JointRefinementRecipe.prepare(data:recipe()))
    XCTAssertEqual(prepared.version,2)
    XCTAssertEqual(prepared.targetGeometry.canvasAdmission,.spatialRefinement)
    XCTAssertEqual(prepared.learnedUpscaler?.path,"/tmp/learned.safetensors")
    XCTAssertNil(prepared.resizeMethod)
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:recipe(version:1)))
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:recipe(mode:"initialized")))
    var root = try JSONSerialization.jsonObject(with:recipe()) as! [String:Any]
    var fields = root["refinement"] as! [String:Any]
    fields["resize_method"] = "bilinear"; root["refinement"] = fields
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    fields.removeValue(forKey:"resize_method"); fields.removeValue(forKey:"learned_upscaler_header_sha256")
    root["refinement"] = fields
    XCTAssertThrowsError(try H3JointRefinementRecipe.prepare(data:JSONSerialization.data(withJSONObject:root)))
    XCTAssertEqual(try H3JointRefinementRecipe.prepare(data:recipe(learned:false))?.resizeMethod,.bilinear)
  }
}
