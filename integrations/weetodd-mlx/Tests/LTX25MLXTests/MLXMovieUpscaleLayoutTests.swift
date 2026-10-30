import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXMovieUpscaleLayoutTests:XCTestCase {
  func testEnhanceFullSourceGuideIsFrozenAtOneToOnePositions() throws {
    let mode=try XCTUnwrap(MLXMovieUpscalePlan.Mode(rawValue:"enhance"))
    let plan=try MLXMovieUpscalePlan(mode:mode,width:64,height:32,frames:2,fps:24,sizePolicy:.strict)
    let layout=try MLXMovieUpscaleLayout(plan:plan,firstStrength:nil,lastStrength:nil)
    XCTAssertEqual(layout.geometry.width,64);XCTAssertEqual(layout.geometry.height,32)
    XCTAssertEqual(layout.groupRows,[4,4]);XCTAssertEqual(layout.referenceTokens,4)
    XCTAssertEqual(layout.mask,[1,1,1,1,0,0,0,0])
    XCTAssertEqual(Array(layout.positions[0..<12]),Array(layout.positions[12..<24]))
    XCTAssertEqual(layout.positions[13],16);XCTAssertEqual(layout.positions[14],16)
    XCTAssertEqual(layout.positions[17],48)
    let source=MLXArray(Array(0..<512).map { Float($0)/1000 },[4,128])
    let prepared=try layout.prepare(generated:.ones([4,128])*(-1),source:source,first:nil,last:nil)
    XCTAssertEqual(prepared.latent.shape,[8,128])
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),-1)
    XCTAssertEqual(prepared.latent[4,0].item(Float.self),0)
    XCTAssertEqual(prepared.latent[7,127].item(Float.self),0.511)
    XCTAssertEqual(prepared.condition.clean[0,0].item(Float.self),0)
    XCTAssertEqual(prepared.condition.clean[7,127].item(Float.self),0.511)
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([4,128]),source:nil,first:nil,last:nil))
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([4,128]),source:.zeros([16,128]),first:nil,last:nil))
  }
  func testPixelMovieTailThenTrueLastFrameWithIndependentReferencesAndNoSlots() throws {
    let plan=try MLXMovieUpscalePlan(mode:.pixelSpatial,width:64,height:32,frames:98,fps:24,sizePolicy:.strict)
    let layout=try MLXMovieUpscaleLayout(plan:plan,firstStrength:0.7,lastStrength:0.9)
    XCTAssertEqual(layout.geometry.frames,105)
    XCTAssertEqual(layout.groupRows,[112,28,8])
    XCTAssertEqual(layout.videoTokens,148)
    XCTAssertEqual(Array(layout.mask.prefix(8)),Array(repeating:Float(0.3),count:8))
    XCTAssertTrue(layout.mask[112..<140].allSatisfy { $0 == 0 })
    XCTAssertEqual(layout.positions[112*3+1],32)
    XCTAssertEqual(layout.positions[112*3+2],32)
    XCTAssertEqual(layout.positions[140*3],Float(97.5/24),accuracy:0.000001)
    let guide=Array(layout.attentionTemplates[148..<296])
    XCTAssertTrue(guide[0..<140].allSatisfy { $0 == 1 })
    XCTAssertTrue(guide[140..<148].allSatisfy { $0 == 0 })
    let last=Array(layout.attentionTemplates[296..<444])
    XCTAssertTrue(last[112..<140].allSatisfy { $0 == 0 })
    XCTAssertTrue(last[140..<148].allSatisfy { $0 == 1 })
  }
  func testAdapterFreeRefineHasNoPixelGuideAndLatentOnlyCannotAccidentallySample() throws {
    let plan=try MLXMovieUpscalePlan(mode:.refine,width:64,height:32,frames:2,fps:24,sizePolicy:.strict)
    let layout=try MLXMovieUpscaleLayout(plan:plan,firstStrength:nil,lastStrength:nil)
    XCTAssertEqual(layout.groupRows,[16]);XCTAssertEqual(layout.referenceTokens,0)
    XCTAssertTrue(layout.attentionTemplates.isEmpty);XCTAssertTrue(layout.mask.allSatisfy { $0 == 1 })
    let latent=try MLXMovieUpscalePlan(mode:.latentOnly,width:64,height:32,frames:2,fps:24,sizePolicy:.strict)
    XCTAssertThrowsError(try MLXMovieUpscaleLayout(plan:latent,firstStrength:nil,lastStrength:nil))
    XCTAssertThrowsError(try MLXMovieUpscaleLayout(plan:plan,firstStrength:.nan,lastStrength:nil))
  }
}
