import XCTest
import Foundation
import Darwin
import MLX
import TensorIO
@testable import LTX25MLX

final class MLXImageEncoderTests:XCTestCase {
  func testImageAdmissionRejectsInvalidGeometryAndCountsFullWorkspace() throws {
    let plan=try MLXImageEncodePlan(width:1344,height:768)
    XCTAssertEqual(plan.latentShape,[768/32*1344/32,128])
    XCTAssertGreaterThan(plan.ownedBufferBytes,1344*768*3*4)
    for size in [(0,32),(33,64),(4096,4096),(Int.max,64)] {
      XCTAssertThrowsError(try MLXImageEncodePlan(width:size.0,height:size.1))
    }
    XCTAssertThrowsError(try MLXImageEncodePlan(width:64,height:32,maximumOwnedBufferBytes:1))
  }
  func testPatchOrderMatchesChannelWidthHeightConvention() throws {
    let rgb=(0..<4*8*3).map(Float.init)
    let output=MLXImageEncoder.patch(MLXArray(rgb,[1,4,8,3])).asArray(Float.self)
    var expected:[Float]=[]
    for blockX in 0..<2 { for c in 0..<3 { for dx in 0..<4 { for dy in 0..<4 {
      expected.append(rgb[(dy*8+blockX*4+dx)*3+c])
    } } } }
    XCTAssertEqual(output,expected)
  }
  func testSingleFramePackingRetainsTemporalThenSpatialChannelOrder() throws {
    let rgb=(0..<4*6*3).map(Float.init)
    for temporal in [1,2] {
      let output=MLXImageEncoder.downpack(MLXArray(rgb,[1,4,6,3]),spatial:2,temporal:temporal).asArray(Float.self)
      var expected:[Float]=[]
      for y in 0..<2 { for x in 0..<3 { for c in 0..<3 { for _ in 0..<temporal { for dy in 0..<2 { for dx in 0..<2 {
        expected.append(rgb[((y*2+dy)*6+x*2+dx)*3+c])
      } } } } } }
      XCTAssertEqual(output,expected)
    }
  }
  func testCollapsedCausalConvolutionMatchesIndependentThreeDimensionalSum() throws {
    let h=3,w=5,ic=2,oc=3
    let rgb=(0..<h*w*ic).map { sin(Float($0)*0.7) }
    let weights=(0..<oc*ic*27).map { cos(Float($0)*0.31)*0.04 }
    let biases:[Float]=[0.1,-0.2,0.3]
    let actual=MLXImageEncoder.convolve(MLXArray(rgb,[1,h,w,ic]),
      weight:MLXArray(weights,[oc,ic,3,3,3]),bias:MLXArray(biases)).asArray(Float.self)
    var expected:[Float]=[]
    for y in 0..<h { for x in 0..<w { for o in 0..<oc {
      var value=biases[o]
      for i in 0..<ic { for t in 0..<3 { for dy in 0..<3 { for dx in 0..<3 {
        let yy=y+dy-1,xx=x+dx-1
        if yy>=0 && yy<h && xx>=0 && xx<w {
          value += rgb[(yy*w+xx)*ic+i]*weights[(((o*ic+i)*3+t)*3+dy)*3+dx]
        }
      } } } }
      expected.append(value)
    } } }
    for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:1e-5) }
  }
  func testInstalledImageEncoderAgainstIndependentFull3DOracle() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_MLX_IMAGE_ENCODER"],let reference=env["WEETODD_MLX_IMAGE_REFERENCE"] else {
      throw XCTSkip("Installed image encoder qualification is opt-in.")
    }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:reference))
    let rgb=try MLXWeight.read(file,"pixels")
    let expected=try file.readFloat32(named:"latent")
    let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:checkpoint))
    var stages=0
    Memory.peakMemory=0;let start=Date()
    let output=try encoder.encode(rgb) { _ in
      stages += 1
      XCTAssertEqual(encoder.residentWeightBytes,0)
      if stages==1 { XCTAssertThrowsError(try encoder.encode(rgb)) }
    }
    let actual=output.asArray(Float.self)
    XCTAssertEqual(actual.count,expected.count)
    let errors=zip(actual,expected).map { Double($0-$1) }
    let maxabs=errors.map { abs($0) }.max()!
    let squareError=errors.reduce(0.0) { $0+$1*$1 }
    let squareReference=expected.reduce(0.0) { $0+Double($1)*Double($1) }
    let relative=sqrt(squareError/squareReference)
    XCTAssertLessThan(maxabs,0.001);XCTAssertLessThan(relative,0.0001)
    XCTAssertEqual(stages,42);XCTAssertEqual(encoder.residentWeightBytes,0)
    print("IMAGE_ENCODER seconds=\(Date().timeIntervalSince(start)) maxabs=\(maxabs) relative=\(relative) peak_mlx=\(Memory.peakMemory)")
    XCTAssertThrowsError(try encoder.encode(rgb) { _ in throw CancellationError() })
    XCTAssertEqual(encoder.residentWeightBytes,0)
    let retry=try encoder.encode(rgb)
    XCTAssertEqual(retry.asArray(Float.self),actual)
  }
}
