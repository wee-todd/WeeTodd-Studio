import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXIngredientsMarkerTests:XCTestCase {
  func testAuthoredIngredientsMarksGeneratedFirstFrameAndExcludesReferenceTail() throws {
    try Device.withDefaultDevice(.cpu) {
      // Two generated and two guide latent frames, two spatial rows each.
      // Primary temporal starts: [0,0,1,1,0,0,1,1]. Guide exclusion selects
      // [true,true,false,false,false,false,false,false], not all time-zero rows.
      let projection=MLXArray((0..<8).flatMap { [Float($0)/4,-Float($0)/2] },[8,2])
      let marker=MLXArray([Float(0.25),-0.5],[1,2])
      let expected: [Float] = [0.25,-0.5,0.5,-1,0.5,-1,0.75,-1.5,
        1,-2,1.25,-2.5,1.5,-3,1.75,-3.5]
      let actual=try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
        leadingRows:2,trailingRows:0)
      XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),expected.map(\.bitPattern))
      XCTAssertEqual(actual[4..<8].asArray(Float.self),projection[4..<8].asArray(Float.self))
      // Conditional and unconditional calls share one configured denoiser;
      // marker application has no text-branch argument or audio projection.
      let negative=try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
        leadingRows:2,trailingRows:0)
      XCTAssertEqual(negative.asArray(Float.self),actual.asArray(Float.self))
    }
  }

  func testLegacyDefaultAndTrailingGeneratedSlotsRetainExactArithmetic() throws {
    try Device.withDefaultDevice(.cpu) {
      let projection=MLXArray((0..<16).map { Float($0)/8 },[8,2])
      let marker=MLXArray([Float(0.25),-0.5],[1,2])
      let unchanged=try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
        leadingRows:0,trailingRows:0)
      XCTAssertEqual(unchanged.asArray(Float.self),projection.asArray(Float.self))
      let old=concatenated([projection[0..<7],projection[7..<8]+marker],axis:0)
      let actual=try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
        leadingRows:0,trailingRows:1)
      XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),old.asArray(Float.self).map(\.bitPattern))
      let both=try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
        leadingRows:2,trailingRows:1)
      XCTAssertEqual(both[0..<2].asArray(Float.self),(projection[0..<2]+marker).asArray(Float.self))
      XCTAssertEqual(both[2..<7].asArray(Float.self),projection[2..<7].asArray(Float.self))
      XCTAssertEqual(both[7..<8].asArray(Float.self),old[7..<8].asArray(Float.self))
      for (leading,trailing) in [(-1,0),(0,-1),(9,0),(5,4)] {
        XCTAssertThrowsError(try MLXDenoiser.applyKeyframeMarkers(projection,marker:marker,
          leadingRows:leading,trailingRows:trailing))
      }
    }
  }

  func testOnlyAuthoredIngredientsOptsIntoLeadingMarkers() throws {
    let geometry=try AVGeometry(width:768,height:448,frames:121,fps:24)
    XCTAssertEqual(MLXSingleStageRipple.leadingMarkerRows(geometry:geometry,
      task:.ingredients,sampling:.ancestralCFGPP),336)
    XCTAssertEqual(MLXSingleStageRipple.leadingMarkerRows(geometry:geometry,
      task:.ingredients,sampling:.deterministic),0)
    XCTAssertEqual(MLXSingleStageRipple.leadingMarkerRows(geometry:geometry,
      task:.ripple,sampling:.deterministic),0)
    XCTAssertEqual(MLXSingleStageRipple.leadingMarkerRows(geometry:geometry,
      task:.ripple,sampling:.ancestralCFGPP),0)
  }
}
