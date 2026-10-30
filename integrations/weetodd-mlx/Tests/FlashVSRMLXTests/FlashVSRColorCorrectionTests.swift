import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRColorCorrectionTests:XCTestCase {
  func testOffPreservesRawPreEncodeValuesAndRangeIsExplicit()throws {
    let value=MLXArray([Float(0),0.5,1],[1,1,1,3])
    XCTAssertEqual(try FlashVSRColorCorrection.apply(value,reference:value,mode:.off,range:.zeroToOne).asArray(Float.self),[0,0.5,1])
    XCTAssertThrowsError(try FlashVSRColorCorrection.apply(value,reference:MLXArray.zeros([2,1,1,3]),mode:.wavelet(levels:1),range:.minusOneToOne))
  }
  func testWaveletReplicateBorderMatchesHandComputedGaussianAndClamps()throws {
    let ramp=MLXArray([Float(0),0,0,1,1,1],[1,1,2,3]),zero=MLXArray.zeros([1,1,2,3])
    // Replicate padding with [1,2,1]/4 leaves low frequencies [1/4,3/4].
    let corrected=try FlashVSRColorCorrection.apply(ramp,reference:zero,mode:.wavelet(levels:1),range:.minusOneToOne)
    XCTAssertEqual(corrected.asArray(Float.self),[-0.25,-0.25,-0.25,0.25,0.25,0.25])
    let white=MLXArray.ones([1,1,2,3])
    XCTAssertEqual(try FlashVSRColorCorrection.apply(ramp,reference:white,mode:.wavelet(levels:1),range:.minusOneToOne).asArray(Float.self),[0.75,0.75,0.75,1,1,1])
  }
  func testAdaINMatchesPerFrameChannelPopulationMoments()throws {
    let generated=MLXArray([Float(0),1,2,2,3,4],[1,1,2,3])
    let reference=MLXArray([Float(-1),0,1,3,4,5],[1,1,2,3])
    let result=try FlashVSRColorCorrection.apply(generated,reference:reference,mode:.adain(epsilon:1e-5),range:.minusOneToOne).asArray(Float.self)
    XCTAssertEqual(result[0],-0.99999,accuracy:2e-5)
    XCTAssertEqual(result[1],0.00001,accuracy:2e-5)
    XCTAssertEqual(result[2],1,accuracy:1e-6)
    XCTAssertEqual(Array(result[3...]),[1,1,1])
  }
  func testWaveletZeroOneRangeRoundTripAndTemporalChunksDoNotMixFrames()throws {
    let generated=MLXArray([Float](repeating:0.5,count:12),[2,1,2,3])
    let reference=MLXArray([Float](repeating:0.25,count:6)+[Float](repeating:0.75,count:6),[2,1,2,3])
    let corrected=try FlashVSRColorCorrection.apply(generated,reference:reference,mode:.wavelet(levels:5),range:.zeroToOne)
    XCTAssertEqual(corrected.asArray(Float.self),[Float](repeating:0.25,count:6)+[Float](repeating:0.75,count:6))
  }
}
