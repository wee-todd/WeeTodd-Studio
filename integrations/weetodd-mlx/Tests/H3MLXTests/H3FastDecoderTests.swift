import Foundation
import XCTest
import MLX
@testable import H3MLX

final class H3FastDecoderTests: XCTestCase {
  func testReleasedLayoutAndMetadataAdmission() throws {
    let shapes=H3FastDecoderLayout.expectedShapes
    XCTAssertEqual(shapes.count,297)
    XCTAssertEqual(shapes["blocks.23.attn.to_qkv.weight"],[6144,2048])
    XCTAssertEqual(shapes["post_quant_conv.weight"],[24,24,1,1,1])
    XCTAssertEqual(H3FastDecoderLayout.selectedOriginalLayers,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,31,32,35])
    var config=configuration()
    let metadata=try H3FastDecoderLayout.Configuration(json:config)
    XCTAssertEqual(metadata.width,2048); XCTAssertEqual(metadata.mean.count,24)
    config["decoder_num_attention_heads"]=16
    XCTAssertThrowsError(try H3FastDecoderLayout.Configuration(json:config))
  }
  func testShiftedWindowsAndGlobalParticipation() throws {
    let plan=try H3FastDecoderWindows.plan(frames:10,height:16,width:16,layer:1)
    XCTAssertEqual(plan.windows.count,12)
    XCTAssertEqual(plan.windows[0].globalBlocks,[0])
    XCTAssertEqual(plan.windows[4].globalBlocks,[0,1])
    XCTAssertEqual(plan.windows[8].globalBlocks,[1])
    let flattened=plan.windows.flatMap(\.tokens)
    XCTAssertEqual(Set(flattened),Set(0..<2560)); XCTAssertEqual(flattened.count,2560)
    XCTAssertEqual(plan.inversePermutation.map { flattened[$0] },Array(0..<2560))
  }
  func testFirstBlockBoundaryAndTailTrimming() throws {
    XCTAssertEqual(try H3FastDecoderWindows.retainedFrames(latentFrames:1),[3])
    let two=try H3FastDecoderWindows.retainedFrames(latentFrames:10)
    XCTAssertEqual(two.count,34); XCTAssertEqual(two[0],3)
    XCTAssertEqual(two[16],19); XCTAssertEqual(two[17],23); XCTAssertEqual(two.last,39)
    XCTAssertEqual(try H3FastDecoderWindows.retainedFrames(latentFrames:6).count,18)
    XCTAssertEqual(try H3FastDecoderWindows.retainedFrames(latentFrames:12).count,39)
    XCTAssertEqual(try H3FastDecoderWindows.retainedFrames(latentFrames:37).count,124)
  }
  func testNoncausalHaloAndIndependentChunkRejection() throws {
    let required=try H3FastDecoderWindows.requiredContext(output:10..<15,totalFrames:30,layers:2)
    XCTAssertLessThan(required.lowerBound,10); XCTAssertGreaterThan(required.upperBound,15)
    XCTAssertThrowsError(try H3FastDecoderWindows.validateContext(available:10..<15,output:10..<15,totalFrames:30,layers:24))
    try H3FastDecoderWindows.validateContext(available:0..<30,output:10..<15,totalFrames:30,layers:24)
  }
  func testOutputPatchOrderAndPixelConversion() throws {
    let head=MLXArray(Array(0..<3072).map(Float.init),[1,1,1,1,3072])
    let pixels=try H3FastVideoDecoder.reconstruct(head)
    XCTAssertEqual(pixels.shape,[1,1,16,16,3])
    let values=pixels.asArray(Float.self)
    XCTAssertEqual(Array(values[0..<6]),[768,1792,2816,769,1793,2817])
    let zero=try H3FastVideoDecoder.rgb8(MLXArray.zeros([1,1,1,1,3])).asArray(UInt8.self)
    XCTAssertEqual(zero,[124,116,104])
  }
  func testWindowAttentionExcludesPaddingAndIncludesNativeGlobals() throws {
    let plan=try H3FastDecoderWindows.plan(frames:10,height:1,width:1,layer:1)
    let q=MLXArray.zeros([1,10,1,2]), k=q
    let values=MLXArray((0..<10).flatMap { [Float($0),Float($0)] },[1,10,1,2])
    let gq=MLXArray.zeros([1,10,1,2])
    let gv=MLXArray(Array(repeating:Float(100),count:10)+Array(repeating:Float(200),count:10),[1,10,1,2])
    let actual=try H3FastVideoDecoder.attention(q:q,k:k,v:values,globalQ:gq,globalK:gq,globalV:gv,plan:plan)
    let output=actual.tokens.asArray(Float.self)
    XCTAssertEqual(output[0],62.875,accuracy:1e-5)
    XCTAssertEqual(output[6],Float(1525)/15,accuracy:2e-5)
    XCTAssertEqual(output[18],Float(1017)/7,accuracy:2e-5)
    let globals=actual.globals.asArray(Float.self)
    XCTAssertEqual(globals[0],51,accuracy:1e-5); XCTAssertEqual(globals[10],103.5,accuracy:1e-5)
  }
  func testTwoLayerHaloMatchesWholeClipAttention() throws {
    func run(_ frames:Range<Int>) throws -> MLXArray {
      var x=MLXArray(frames.flatMap { [Float($0),Float($0)] },[1,frames.count,1,2])
      var g=MLXArray.ones([1,((frames.count+4)/5)*5,1,2])*0.1
      for layer in 0..<2 {
        let plan=try H3FastDecoderWindows.plan(frames:frames.count,height:1,width:1,layer:layer)
        let result=try H3FastVideoDecoder.attention(q:MLXArray.zeros(x.shape),k:MLXArray.zeros(x.shape),v:x,globalQ:MLXArray.zeros(g.shape),globalK:MLXArray.zeros(g.shape),globalV:g,plan:plan)
        x=result.tokens; g=result.globals
      }
      return x
    }
    let needed=try H3FastDecoderWindows.requiredContext(output:10..<15,totalFrames:30,layers:2)
    XCTAssertEqual(needed,5..<20)
    let whole=try run(0..<30)[0,10..<15].asArray(Float.self)
    let bounded=try run(needed)[0,(10-needed.lowerBound)..<(15-needed.lowerBound)].asArray(Float.self)
    for (a,b) in zip(whole,bounded) { XCTAssertEqual(a,b,accuracy:2e-5) }
  }
  func testMissingWrongDtypeAndShapeRejectedBeforeLoading() throws {
    let shapes=H3FastDecoderLayout.expectedShapes,dtypes=shapes.mapValues{_ in "F32"}
    try H3FastDecoderLayout.validateSchema(shapes:shapes,dtypes:dtypes)
    var missing=shapes;missing.removeValue(forKey:"register_tokens")
    XCTAssertThrowsError(try H3FastDecoderLayout.validateSchema(shapes:missing,dtypes:dtypes))
    var wrong=dtypes;wrong["blocks.0.scale1"]="F16"
    XCTAssertThrowsError(try H3FastDecoderLayout.validateSchema(shapes:shapes,dtypes:wrong))
  }
  func testRotaryWindowCoordinatesAndF32Normalization() throws {
    let plan=try H3FastDecoderWindows.plan(frames:2,height:1,width:1,layer:1)
    let tables=H3FastVideoDecoder.rotaryTables(plan:plan,dtype:.float32)
    let vector=MLXArray(Array(repeating:Float(1),count:24)+Array(repeating:Float(0),count:24)+Array(repeating:Float(7),count:16),[1,1,1,64])
    let q=broadcast(vector,to:[1,2,1,64])
    let rotated=H3FastVideoDecoder.rotary(q,cosine:tables.0,sine:tables.1).asArray(Float.self)
    XCTAssertEqual(rotated[0],1,accuracy:1e-6);XCTAssertEqual(rotated[24],0,accuracy:1e-6)
    XCTAssertEqual(rotated[8],cos(Float.pi/8),accuracy:1e-6)
    XCTAssertEqual(Array(rotated[48..<64]),Array(repeating:Float(7),count:16))
    let config=try H3FastDecoderLayout.Configuration(mean:Array(repeating:3,count:24),standardDeviation:Array(repeating:2,count:24))
    let raw=try H3FastVideoDecoder.rawLatents(MLXArray.ones([1,1,1,1,24],dtype:.bfloat16),configuration:config)
    XCTAssertEqual(raw.dtype,.float32);XCTAssertEqual(raw.asArray(Float.self),Array(repeating:5,count:24))
  }
  private func configuration() -> [String:Any] {
    ["_class_name":"AutoencoderKLMiniMaxH3","latent_channels":24,"decoder_num_layers":36,"decoder_num_attention_heads":32,"decoder_attention_head_dim":64,"decoder_num_register_tokens":4,"decoder_ffn_mult":4,"decoder_rope_theta":100.0,"decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-5,"clip_length":17,"token_drop":3,"latents_mean":Array(repeating:0.0,count:24),"latents_std":Array(repeating:1.0,count:24)]
  }
}
