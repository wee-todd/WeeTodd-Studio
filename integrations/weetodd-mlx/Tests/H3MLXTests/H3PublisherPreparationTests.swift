import MLX
import XCTest
@testable import H3MLX

final class H3PublisherPreparationTests:XCTestCase {
  func testLosslessEncoderChannelPermutationAndSourceMapping() throws {
    XCTAssertEqual(try H3PublisherEncoderConverter.transposePlane([0,1,2,3,4,5,6,7],inputs:2,depth:1,height:2,width:2),[0,4,1,5,2,6,3,7])
    XCTAssertEqual(H3PublisherEncoderConverter.nativeName("encoder.down_blocks.2.resnets.0.conv_shortcut.weight"),"encoder.down.2.block.0.nin_shortcut.weight")
    XCTAssertEqual(H3PublisherEncoderConverter.nativeName("encoder.down_blocks.1.downsamplers.0.conv.weight"),"encoder.down.1.downsample.conv.weight")
    XCTAssertEqual(H3VideoVAEEncoder.encoderShapes.count,118)
    XCTAssertEqual(H3VideoVAEEncoder.encoderShapes["encoder.down.5.block.0.conv1.weight"],[1024,3,3,3,512])
    XCTAssertThrowsError(try H3PublisherEncoderConverter.transposePlane([1],inputs:2,depth:1,height:1,width:1))
  }
  func testF32PosteriorUsesInjectedNoiseAndClampedVarianceWithoutHalfCast() throws {
    let mean=MLXArray([Float(1.0001),2,3]),logVariance=MLXArray([Float(0),100,-100]),noise=MLXArray([Float(1),0,1])
    let result=try H3PublisherEncoderOps.posterior(mean:mean,logVariance:logVariance,noise:noise)
    XCTAssertEqual(result[0].item(Float.self),2.0001,accuracy:0.000001)
    XCTAssertEqual(result[1].item(Float.self),2)
    XCTAssertEqual(result[2].item(Float.self),3+exp(-15),accuracy:0.000001)
    XCTAssertThrowsError(try H3PublisherEncoderOps.posterior(mean:mean,logVariance:logVariance,noise:.ones([2])))
  }
  func testPerFrameStoragePreservesCompleteCausalConvolutionAtStrideTwo() throws {
    let input=sin(MLXArray(0..<(5*4*4*32)).asType(.float32)*0.01).reshaped([1,5,4,4,32])
    let weight=cos(MLXArray(0..<(32*3*3*3*32)).asType(.float32)*0.03).reshaped([32,3,3,3,32])*0.01
    let bias=MLXArray.zeros([32])
    let full=try H3VideoVAEEncoderOps.causalConv(input,weight:weight,bias:bias,spatialPadding:1,temporalPadding:2,stride:IntOrTriple((2,1,1)))
    let frames=(0..<5).map { input[0..<1,$0..<($0+1),0..<4,0..<4,0..<32] }
    let parts=try H3PublisherEncoderOps.convolution(frames:frames,weight:weight,bias:bias,spatialPadding:1,temporalPadding:2,temporalStride:2,spatialStride:1)
    XCTAssertLessThan(max(abs(full-concatenated(parts,axis:1))).item(Float.self),0.00001)
  }
  func testPublisherArithmeticScaleAndBF16NormalizationDifferExplicitly() throws {
    let publisher=H3LatentUpscalerMathProfile.publisherBF16Arithmetic,native=H3LatentUpscalerMathProfile.nativeF32Geometric
    XCTAssertEqual(publisher.scale(sourceWidth:1344,sourceHeight:768,targetWidth:2560,targetHeight:1440),1.8898809523809523,accuracy:1e-12)
    XCTAssertNotEqual(publisher.scale(sourceWidth:1344,sourceHeight:768,targetWidth:2560,targetHeight:1440),native.scale(sourceWidth:1344,sourceHeight:768,targetWidth:2560,targetHeight:1440))
    let input=MLXArray([Float(1.003)]),mean=MLXArray([Float(1)]),std=MLXArray([Float(1)])
    XCTAssertEqual(publisher.normalize(input,mean:mean,std:std).item(Float.self),0)
    XCTAssertGreaterThan(native.normalize(input,mean:mean,std:std).item(Float.self),0)
    let direct=try H3LatentUpscalePlan(sourceWidth:1344,sourceHeight:768,targetWidth:2560,targetHeight:1440,mode:.directPublisher)
    XCTAssertEqual(direct.grids.count,2)
    XCTAssertThrowsError(try H3LatentUpscalePlan(sourceWidth:1344,sourceHeight:768,targetWidth:3840,targetHeight:2176,mode:.directPublisher))
    let multi=try H3LatentUpscalePlan(sourceWidth:1344,sourceHeight:768,targetWidth:3840,targetHeight:2176,mode:.multiHopNative)
    XCTAssertEqual(multi.grids.map(\.width),[1344,2688,3840]);XCTAssertFalse(multi.isPublisherDirect)
    XCTAssertEqual(try H3UpscalerConvolutionPlan(maximumSpatial:256).tileCount(frames:37,height:90,width:160),75)
    XCTAssertThrowsError(try H3UpscalerConvolutionPlan().tileCount(frames:37,height:90,width:160))
  }
}
