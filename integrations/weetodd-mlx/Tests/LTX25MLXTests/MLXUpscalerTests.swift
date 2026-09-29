import XCTest
import Foundation
import MLX
import TensorIO
import LTX25Video
@testable import LTX25MLX

final class MLXUpscalerTests:XCTestCase {
  func testAdmissionIncludesConvolutionWorkspaceBeforePayloads() throws {
    let shape=[12,12,21,128]
    XCTAssertNoThrow(try MLXLatentUpscaler.admit(shape:shape,maximumActivationBytes:2*1024*1024*1024))
    XCTAssertThrowsError(try MLXLatentUpscaler.admit(shape:shape,maximumActivationBytes:512*1024*1024))
    XCTAssertThrowsError(try MLXLatentUpscaler.admit(shape:[Int.max,12,21,128],maximumActivationBytes:Int.max))
  }
  func testWholeVolumeNormalizationAndResidualBeforeSilu() throws {
    let values=(0..<768).map { Float($0 % 97-48)/17 }
    let weight=(0..<64).map { Float($0)/100+0.7 },bias=(0..<64).map { Float($0 % 5)/13 }
    let x=MLXArray(values,[2,2,3,64])
    let actual=MLXLatentUpscaler.normalized(x,weight:MLXArray(weight),bias:MLXArray(bias),residual:x*0.2).asArray(Float.self)
    var expected=values
    for g in 0..<32 {
      var indices:[Int]=[]
      for site in 0..<12 { indices.append(site*64+g*2);indices.append(site*64+g*2+1) }
      let mean=indices.reduce(0.0) { $0+Double(values[$1]) }/Double(indices.count)
      var variance:Double=0
      for i in indices { let delta=Double(values[i])-mean;variance += delta*delta }
      variance /= Double(indices.count)
      for i in indices {
        let normalized=Float((Double(values[i])-mean)/Foundation.sqrt(variance+1e-5))
        let v:Float=normalized*weight[i%64]+bias[i%64]+values[i]*0.2
        expected[i]=v/(1+Foundation.expf(-v))
      }
    }
    XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,3e-6)
  }
  func testZeroTemporalPaddingAndShuffleOrder() throws {
    let x=MLXArray([Float(1),2],[2,1,1,1])
    for depth in [1,3] {
      let shape=depth==1 ? [1,1,3,3] : [1,1,3,3,3]
      let result=MLXLatentUpscaler.convolution(x,weight:.ones(shape),bias:.zeros([1]))
      XCTAssertEqual(result.asArray(Float.self),depth==1 ? [1,2] : [3,3])
    }
    let shuffled=MLXLatentUpscaler.shuffle(MLXArray((0..<8).map(Float.init),[1,1,1,8]))
    XCTAssertEqual(shuffled.shape,[1,2,2,2])
    XCTAssertEqual(shuffled.asArray(Float.self),[0,4,1,5,2,6,3,7])
  }
  func testInstalledUpscalerMatchesIndependentOracleAndReleasesOnCancellation() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_UPSCALER"],let stats=env["WEETODD_VIDEO_VAE"],
      let oracle=env["WEETODD_UPSCALER_ORACLE"] else { throw XCTSkip("Installed MLX upscaler qualification is opt-in.") }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:oracle))
    let input=try MLXWeight.read(file,"input"),expected=try file.readFloat32(named:"output")
    let model=try MLXLatentUpscaler(checkpoint:URL(fileURLWithPath:checkpoint),statisticsCheckpoint:URL(fileURLWithPath:stats))
    XCTAssertThrowsError(try model.upscale(input,maximumActivationBytes:1))
    let oldCache=Memory.cacheLimit
    XCTAssertThrowsError(try model.upscale(input,progress:{ _ in throw CancellationError() }))
    XCTAssertEqual(model.residentWeightBytes,0);XCTAssertEqual(Memory.cacheLimit,oldCache)
    let start=Date()
    let actual=try model.upscale(input,progress:{ _ in XCTAssertThrowsError(try model.upscale(input)) }).asArray(Float.self)
    let error=zip(actual,expected).map { abs($0-$1) }.max()!
    let sq=zip(actual,expected).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
    let norm=expected.reduce(0.0) { $0+Double($1)*Double($1) }
    print("MLX_UPSCALER seconds=\(Date().timeIntervalSince(start)) maxabs=\(error) rel=\(sqrt(sq/norm))")
    XCTAssertLessThan(error,0.002);XCTAssertLessThan(sqrt(sq/norm),0.0001)
    XCTAssertEqual(actual.count,expected.count);XCTAssertEqual(model.residentWeightBytes,0)
  }
}
