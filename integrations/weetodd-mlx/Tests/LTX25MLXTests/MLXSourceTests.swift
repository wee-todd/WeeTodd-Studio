import XCTest
import MLX
@testable import LTX25MLX
import LTX25Engine
import TensorIO
import InferenceTestSupport

final class MLXSourceTests: XCTestCase {
  func testDeferredNativeReadVerifiesIdentityWhenBlockMaterializes() throws {
    let name="transformer_blocks.0.attn1.to_q.weight"
    try withTensorFile(tensors:[(name,[2,64],"BF16")]) { url in
      let source=try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,64]])
      let weight=try source.read("attn1.to_q.weight",shape:[2,64],deferEvaluation:true)
      try MLXWeight.materialize([weight])
      XCTAssertEqual(source.pendingTensorCount,0)
      let deferred=try source.read("attn1.to_q.weight",shape:[2,64],deferEvaluation:true)
      let replacement=url.appendingPathExtension("replacement")
      try Data(contentsOf:url).write(to:replacement)
      try FileManager.default.removeItem(at:url);try FileManager.default.moveItem(at:replacement,to:url)
      XCTAssertThrowsError(try MLXWeight.materialize([deferred]))
    }
  }
  func testNativeReaderEvaluatesPackedFactorsAndReleasesPageHandles() throws {
    let stem="transformer_blocks.0.attn1.to_q"
    let words=[UInt32](repeating:0x03020100,count:32),scales:[Float]=[0.25,0.5],offsets:[Float]=[-0.5,0.25]
    try withTensorFile(tensors:[(stem+".weight",[2,16],"U32"),(stem+".scales",[2,1],"F32"),(stem+".biases",[2,1],"F32")],payloads:[
      stem+".weight":words.withUnsafeBytes { Data($0) },stem+".scales":scales.withUnsafeBytes { Data($0) },stem+".biases":offsets.withUnsafeBytes { Data($0) }]) { url in
      let source=try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,64]],nativeLoading:true)
      for _ in 0..<2 {
        let weight=try source.read("attn1.to_q.weight",shape:[2,64])
        XCTAssertEqual(weight.storageBytes,144)
        XCTAssertEqual(try weight.projected(.ones([1,64])).asArray(Float.self),[-8,64])
        XCTAssertEqual(source.pendingTensorCount,0,"No evaluated tensor may remain in a page reader.")
      }
      let replacement=url.appendingPathExtension("replacement")
      try Data(contentsOf:url).write(to:replacement)
      try FileManager.default.removeItem(at:url)
      try FileManager.default.moveItem(at:replacement,to:url)
      XCTAssertThrowsError(try source.read("attn1.to_q.weight",shape:[2,64]),"Replacing a pathname must invalidate the original checkpoint identity.")
    }
  }
  func testBlockSourceRejectsExtraAndMismatchedWeightsBeforePayloadLoad() throws {
    let original="model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    try withTensorFile(tensors:[(original,[2,64],"BF16")]) { url in
      let source=try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,64]])
      XCTAssertEqual(source.storageBytes,256)
      XCTAssertEqual(try source.read("attn1.to_q.weight",shape:[2,64]).shape,[2,64])
      XCTAssertThrowsError(try source.read("attn1.to_q.weight",shape:[1,128]))
      XCTAssertThrowsError(try MLXBlockSource(url:url,blockIndex:1,expectedShapes:["attn1.to_q.weight":[2,64]]))
      XCTAssertThrowsError(try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,64]],maximumWeightBytes:128))
      XCTAssertThrowsError(try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,32]]))
    }
    try withTensorFile(tensors:[(original,[2,64],"BF16"),("unexpected",[1],"F32")]) { url in
      XCTAssertThrowsError(try MLXBlockSource(url:url,blockIndex:0,expectedShapes:["attn1.to_q.weight":[2,64]]))
    }
  }
}
