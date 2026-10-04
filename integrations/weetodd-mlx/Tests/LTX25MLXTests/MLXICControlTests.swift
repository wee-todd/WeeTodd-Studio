import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXICControlTests:XCTestCase {
  private func object(_ family:String)->[String:Any] {
    let roles=family == "motion_track" ? ["control"] : family == "crossview_warp" ? ["warp","source"] : ["warp","source","ingredients"]
    let adapters=family == "motion_track" ? ["motion_track"] : family == "crossview_warp" ? ["crossview_warp"] : ["crossview_warp","ingredients_reference_sheet"]
    return ["family":family,"adapters":adapters.map { ["path":"/\($0).safetensors","family":$0,"strength":0.8] },
      "guides":roles.enumerated().map { ["path":"/\($0.element).rgb24","source_sha256":String(repeating:"a",count:64),"role":$0.element,"strength":Double($0.offset+1)/4] },
      "publication_audio":family == "motion_track" ? NSNull() : ["path":"/source.wav","source_sha256":String(repeating:"b",count:64),"source_start_seconds":0.0,"source_duration_seconds":5.0]]
  }
  private func decode(_ object:[String:Any]) throws->MLXICControl {
    try JSONDecoder().decode(MLXICControl.self,from:JSONSerialization.data(withJSONObject:object))
  }
  func testEncoderPreservesRequiredPublicationAudioFieldForEveryFamily() throws {
    let encoder=JSONEncoder();encoder.outputFormatting = .sortedKeys
    for family in ["motion_track","crossview_warp","crossview_ingredients"] {
      let control=try decode(object(family)),encoded=try encoder.encode(control)
      let value=try XCTUnwrap(JSONSerialization.jsonObject(with:encoded) as? [String:Any])
      XCTAssertEqual(Set(value.keys),["family","adapters","guides","publication_audio"])
      if family == "motion_track" { XCTAssertTrue(value["publication_audio"] is NSNull) }
      else { XCTAssertNotNil(value["publication_audio"] as? [String:Any]) }
      let replay=try JSONDecoder().decode(MLXICControl.self,from:encoded)
      XCTAssertEqual(try encoder.encode(replay),encoded)
      var missing=value;missing.removeValue(forKey:"publication_audio")
      XCTAssertThrowsError(try decode(missing))
    }
  }
  func testMotionGuideMatchesExistingUnionPositionsAndNumericalMask() throws {
    let control=try decode(object("motion_track")),g=try AVGeometry(width:128,height:64,frames:9,fps:24)
    let layout=try MLXICControlLayout(geometry:g,control:control)
    let union=try MLXUnionControlLayout(geometry:g,strength:0.25)
    XCTAssertEqual(layout.positions,union.positions)
    let reference=MLXArray.full([layout.referenceTokens,128],values:MLXArray(Float(2))),generated=MLXArray.zeros([g.videoTokens,128])
    let prepared=try layout.prepare(generated:generated,references:[reference])
    XCTAssertEqual(prepared.latent.shape,[layout.videoTokens,128])
    // The actual denoise contract must preserve target rows and guide blend strengths.
    let expected=try union.prepare(generated:generated,reference:reference)
    XCTAssertEqual(prepared.latent.asArray(Float.self),expected.latent.asArray(Float.self))
    XCTAssertEqual(prepared.condition.mask,expected.condition.mask)
  }
  func testCrossViewOrderedGroupsAndStrictNestedContract() throws {
    var value=object("crossview_warp"),control=try decode(value)
    let g=try AVGeometry(width:64,height:64,frames:9,fps:24),layout=try MLXICControlLayout(geometry:g,control:control)
    XCTAssertEqual(layout.videoTokens,3*g.videoTokens)
    XCTAssertEqual(layout.positions,g.videoPositions+g.videoPositions+g.videoPositions)
    let refs=[MLXArray.full([g.videoTokens,128],values:MLXArray(Float(2))),MLXArray.full([g.videoTokens,128],values:MLXArray(Float(7)))]
    let prepared=try layout.prepare(generated:MLXArray.zeros([g.videoTokens,128]),references:refs)
    XCTAssertEqual(prepared.latent[g.videoTokens..<2*g.videoTokens].asArray(Float.self),refs[0].asArray(Float.self))
    XCTAssertEqual(prepared.latent[2*g.videoTokens..<3*g.videoTokens].asArray(Float.self),refs[1].asArray(Float.self))
    XCTAssertThrowsError(try layout.prepare(generated:MLXArray.zeros([g.videoTokens,128]),references:[refs[0]]))
    var guides=value["guides"] as! [[String:Any]];guides[0]["silent_extra"]=true;value["guides"]=guides
    XCTAssertThrowsError(try decode(value))
    value=object("crossview_warp");value["publication_audio"]=NSNull();XCTAssertThrowsError(try decode(value))
    value=object("crossview_ingredients");control=try decode(value)
    XCTAssertThrowsError(try control.guideGeometry(target:g))
    let full=try AVGeometry(width:64,height:64,frames:121,fps:24)
    XCTAssertEqual(try MLXICControlLayout(geometry:full,control:control).videoTokens,4*full.videoTokens)
  }
}
