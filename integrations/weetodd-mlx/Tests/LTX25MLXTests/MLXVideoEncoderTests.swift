import XCTest
import Foundation
import MLX
import TensorIO
@testable import LTX25MLX

final class MLXVideoEncoderTests:XCTestCase {
  func testAdmissionBoundsFramesAndPeakBeforeAllocation() throws {
    let plan=try MLXVideoEncodePlan(frames:17,width:64,height:32)
    XCTAssertEqual(plan.latentShape,[3,1,2,128])
    XCTAssertGreaterThan(plan.ownedBufferBytes,17*64*32*128*4/16)
    for frames in [0,2,8,16,Int.max] {
      XCTAssertThrowsError(try MLXVideoEncodePlan(frames:frames,width:64,height:32))
    }
    XCTAssertThrowsError(try MLXVideoEncodePlan(frames:17,width:64,height:32,maximumOwnedBufferBytes:1))
    XCTAssertThrowsError(try MLXVideoEncodePlan(frames:961,width:4096,height:4096))
  }
  func testPatchRetainsEveryFrameAndChannelWidthHeightOrder() {
    let pixels=(0..<3*4*8*3).map(Float.init)
    let actual=MLXVideoEncoder.patch(MLXArray(pixels,[3,4,8,3])).asArray(Float.self)
    var expected:[Float]=[]
    for f in 0..<3 { for bx in 0..<2 { for c in 0..<3 { for dx in 0..<4 { for dy in 0..<4 {
      expected.append(pixels[((f*4+dy)*8+bx*4+dx)*3+c])
    } } } } }
    XCTAssertEqual(actual,expected)
  }
  func testTemporalDownpackPrependsFirstFrameAndRetainsMotionOrder() {
    let pixels=(0..<5*4*6*3).map(Float.init)
    let input=MLXArray(pixels,[5,4,6,3])
    let padded=MLXVideoEncoder.temporalPad(input,stride:2)
    let actual=MLXVideoEncoder.downpack(padded,spatial:2,temporal:2).asArray(Float.self)
    var expected:[Float]=[]
    for f in 0..<3 { for y in 0..<2 { for x in 0..<3 { for c in 0..<3 { for dt in 0..<2 { for dy in 0..<2 { for dx in 0..<2 {
      let sourceFrame=max(0,f*2+dt-1)
      expected.append(pixels[((sourceFrame*4+y*2+dy)*6+x*2+dx)*3+c])
    } } } } } } }
    XCTAssertEqual(actual,expected)
  }
  func testInstalledVideoEncoderAgainstIndependent3DOracle() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_MLX_VIDEO_ENCODER"],let reference=env["WEETODD_MLX_VIDEO_ENCODE_REFERENCE"] else {
      throw XCTSkip("Installed video encoder numerical qualification is opt-in.")
    }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:reference))
    let pixels=try MLXWeight.read(file,"pixels"),expected=try MLXWeight.read(file,"latent")
    let encoder=try MLXVideoEncoder(checkpoint:URL(fileURLWithPath:checkpoint))
    var stages=0;Memory.peakMemory=0;let start=Date()
    let output=try encoder.encode(pixels) { _ in
      stages += 1;XCTAssertEqual(encoder.residentWeightBytes,0)
      if stages==1 { XCTAssertThrowsError(try encoder.encode(pixels)) }
    }
    XCTAssertEqual(output.shape,expected.shape)
    let delta=output-expected
    let maxabs=abs(delta).max().item(Float.self)
    let relative=sqrt((delta*delta).sum()/(expected*expected).sum()).item(Float.self)
    XCTAssertLessThan(maxabs,0.002);XCTAssertLessThan(relative,0.0002)
    XCTAssertEqual(stages,42);XCTAssertEqual(encoder.residentWeightBytes,0)
    print("VIDEO_ENCODER seconds=\(Date().timeIntervalSince(start)) maxabs=\(maxabs) relative=\(relative) peak_mlx=\(Memory.peakMemory)")
    // Causal first latent is identical whether or not future frames are present.
    let first=try encoder.encode(pixels[0..<1])
    XCTAssertLessThan(abs(first-output[0..<1]).max().item(Float.self),0.002)
    XCTAssertThrowsError(try encoder.encode(pixels) { _ in throw CancellationError() })
    XCTAssertEqual(encoder.residentWeightBytes,0)
    let retried=try encoder.encode(pixels)
    XCTAssertEqual(retried.asArray(Float.self),output.asArray(Float.self))
  }
}
