import XCTest
import MLX
import LTX25Engine
import LTX25MLX

final class MLXReferenceTests:XCTestCase {
  func testTimedSceneAnchorLandsAtTheFirstNewShotFrame() throws {
    let g=try AVGeometry(width:64,height:32,frames:41,fps:24)
    let layout=try MLXReferenceLayout(geometry:g,firstStrength:0.8,lastStrength:nil,firstFrame:24)
    let frameTokens=g.latentHeight*g.latentWidth
    let targetIndex=3*frameTokens
    let reference=MLXArray.ones([frameTokens,128])*0.7
    let prepared=try layout.prepare(generated:.ones([g.videoTokens,128])*(-1),first:reference,last:nil)
    XCTAssertEqual(prepared.latent[targetIndex,0].item(Float.self),0.7)
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),-1)
    XCTAssertEqual(prepared.condition.mask[targetIndex],0.2,accuracy:0.0001)
    XCTAssertEqual(prepared.condition.mask[0],1)
    XCTAssertEqual(prepared.condition.clean[targetIndex,0].item(Float.self),0.7)
    XCTAssertThrowsError(try MLXReferenceLayout(geometry:g,firstStrength:1,lastStrength:nil,firstFrame:25))
  }
  func testFirstLatentAndAppendedLastUsePixelFramePositionsAndStrengthMasks() throws {
    let g=try AVGeometry(width:64,height:32,frames:17,fps:24)
    let layout=try MLXReferenceLayout(geometry:g,firstStrength:1,lastStrength:0.75)
    XCTAssertEqual(layout.videoTokens,8)
    XCTAssertEqual(layout.positions.suffix(6).map { $0 },[Float(16.5/24),16,16,Float(16.5/24),16,48])
    let first=MLXArray.ones([2,128])*0.2,last=MLXArray.ones([2,128])*0.8
    let prepared=try layout.prepare(generated:.ones([6,128])*(-1),first:first,last:last)
    XCTAssertEqual(prepared.condition.mask,[0,0,1,1,1,1,0.25,0.25])
    let values=prepared.latent.asArray(Float.self)
    XCTAssertEqual(values[0],0.2);XCTAssertEqual(values[2*128],-1);XCTAssertEqual(values[6*128],0.8)
    first[.ellipsis]=MLXArray.zeros(first.shape);last[.ellipsis]=MLXArray.zeros(last.shape)
    XCTAssertEqual(prepared.condition.clean[0,0].item(Float.self),0.2)
    XCTAssertEqual(prepared.condition.clean[6,0].item(Float.self),0.8)
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([6,128]),first:.zeros([2,128]),last:nil))
    XCTAssertThrowsError(try MLXReferenceLayout(geometry:g,firstStrength:.nan,lastStrength:nil))
  }
  func testReferenceContentChangesGeneratedTokensThroughAttention() throws {
    let f=try MLXSamplingTests().fixture(),helper=MLXSamplingTests()
    let runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let shapes=DenoiserLayout.inputShapes(f.configuration)
    func sample(_ value:Float) throws -> [Float] {
      var inputs=f.inputs.mapValues { MLXArray($0) }
      let video=inputs["video_latent"]!.reshaped(shapes["video_latent"]!)
      video[0]=MLXArray.ones([128])*value;inputs["video_latent"]=video
      let condition=try MLXVideoDenoiseCondition(clean:video,mask:[0,1,1,1,1])
      return try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:[1,0],eta:0),videoConditioning:condition,
        fixedWeights:helper.weight,blockWeights:{ try helper.weight("transformer_blocks.\($0)."+$1,$2) })["video"]![1..<5].asArray(Float.self)
    }
    XCTAssertNotEqual(try sample(0.2),try sample(-0.5))
  }
}
