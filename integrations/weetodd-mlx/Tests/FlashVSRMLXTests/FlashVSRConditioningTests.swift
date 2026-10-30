import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRConditioningTests: XCTestCase {
  func testPixelUnshuffleOrderAndSourceNormalization() throws {
    let x=MLXArray(Array(0..<16).map(Float.init), [1,1,1,4,4])
    let y=try LQProjectionSession.pixelUnshuffle(x, factor:2)
    XCTAssertEqual(y.shape,[1,4,1,2,2])
    XCTAssertEqual(y.asArray(Float.self),[0,2,8,10,1,3,9,11,4,6,12,14,5,7,13,15])
    XCTAssertEqual(LQProjectionSession.normalizeRGB8(MLXArray([UInt8(0),255])).asArray(Float.self),[-1,1])
  }
  func testCausalConvolutionReplicatePaddingAndCarry() throws {
    let w=MLXArray.ones([1,1,4,1,1]), bias=MLXArray.zeros([1])
    let x=MLXArray([Float(1),2,3,4],[1,1,4,1,1])
    let a=try LQProjectionSession.causalConvolution(x,weight:w,bias:bias,cache:nil,spatialPadding:0)
    XCTAssertEqual(a.asArray(Float.self),[5,10])
    let cache=MLXArray([Float(9),10],[1,1,2,1,1])
    let b=try LQProjectionSession.causalConvolution(x,weight:w,bias:bias,cache:cache,spatialPadding:0)
    XCTAssertEqual(b.asArray(Float.self),[22,10])
  }
  func testOneEvaluationFlowAndTimeEmbedding() throws {
    let result=try FlashVSRScheduler.step(noise:MLXArray([Float(2),4]),velocity:MLXArray([Float(0.5),1]))
    XCTAssertEqual(result.asArray(Float.self),[1.5,3])
    let zero=try FlashVSRTransformer.sinusoidalEmbedding(timestep:0,dimension:4)
    XCTAssertEqual(zero.asArray(Float.self),[1,1,0,0])
  }
  func testCheckpointAdmissionRejectsMixedIdentityAndMissingContext() throws {
    let id=FlashVSRCheckpoint.Identity(variant:.turbo, transformerSHA256:String(repeating:"a",count:64), projectionSHA256:String(repeating:"b",count:64), contextSHA256:String(repeating:"c",count:64))
    XCTAssertThrowsError(try FlashVSRCheckpoint.validateIdentity(id, expected:.base))
    XCTAssertThrowsError(try FlashVSRCheckpoint.validateContext(nil))
    XCTAssertThrowsError(try FlashVSRCheckpoint.validateContext(MLXArray.zeros([1,512,1536])))
  }
  func testProjectionWarmupThreeGroupsAndReset() throws {
    let weights:[String:MLXArray]=[
      "conv1.weight":MLXArray.ones([2,768,4,3,3])*0.001,"conv1.bias":MLXArray.zeros([2]),"norm1.weight":MLXArray.ones([2,1,1,1]),
      "conv2.weight":MLXArray.ones([3,2,4,3,3])*0.001,"conv2.bias":MLXArray.zeros([3]),"norm2.weight":MLXArray.ones([3,1,1,1]),
      "linear_layers.0.weight":MLXArray.ones([4,3]),"linear_layers.0.bias":MLXArray.zeros([4])]
    let session=LQProjectionSession(weights:weights,sourceID:"a")
    XCTAssertNil(try session.process(MLXArray.ones([1,3,1,16,16]),sourceID:"a"))
    var first:[Float]?
    for _ in 0..<3 {
      var group=[MLXArray]()
      for _ in 0..<4 { group.append(try XCTUnwrap(session.process(MLXArray.ones([1,3,4,16,16]),sourceID:"a"))[0]) }
      let joined=concatenated(group,axis:1)
      XCTAssertEqual(joined.shape,[1,4,4])
      let values=joined.asArray(Float.self)
      if let first { for (a,b) in zip(first,values) { XCTAssertEqual(a,b,accuracy:2e-5) } } else { first=values }
    }
    XCTAssertThrowsError(try session.process(MLXArray.ones([1,3,4,16,16]),sourceID:"b"))
    session.reset(sourceID:"b")
    XCTAssertNil(try session.process(MLXArray.ones([1,3,1,16,16]),sourceID:"b"))
    let again=try XCTUnwrap(session.process(MLXArray.ones([1,3,4,16,16]),sourceID:"b"))[0].asArray(Float.self)
    for (a,b) in zip(again,first!) { XCTAssertEqual(a,b,accuracy:2e-5) }
  }

  func testLookaheadAndExplicitTailPacking() throws {
    XCTAssertThrowsError(try LQProjectionSession.groupSourceIndices(frameCount:13,firstGroup:true,isFinal:false))
    let first=try LQProjectionSession.groupSourceIndices(frameCount:17,firstGroup:true,isFinal:false)
    XCTAssertEqual(first,[[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    let tail=try LQProjectionSession.groupSourceIndices(frameCount:13,firstGroup:true,isFinal:true)
    XCTAssertEqual(tail.last,[12,12,12,12])
    let later=try LQProjectionSession.groupSourceIndices(frameCount:2,firstGroup:false,isFinal:true)
    XCTAssertEqual(later.first,[0,1,1,1]); XCTAssertEqual(later.last,[1,1,1,1])
    XCTAssertThrowsError(try LQProjectionSession.groupSourceIndices(frameCount:0,firstGroup:false,isFinal:true))
  }

}
