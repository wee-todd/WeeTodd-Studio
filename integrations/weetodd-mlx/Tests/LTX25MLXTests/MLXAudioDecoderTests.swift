import XCTest
import Foundation
import MLX
@testable import LTX25MLX

final class MLXAudioDecoderTests:XCTestCase {
  func testDilationAndNearestUpsampling() throws {
    let values=(0..<26).map { sin(Float($0)) },weights=(0..<18).map { cos(Float($0))*0.1 }
    for dilation in [1,3,5] {
      var expected=[Float](repeating:0,count:13*3)
      for t in 0..<13 { for o in 0..<3 { for c in 0..<2 { for k in 0..<3 {
        let s=t+(k-1)*dilation
        if s>=0 && s<13 { expected[t*3+o] += values[s*2+c]*weights[(o*2+c)*3+k] }
      } } } }
      let actual=MLXAudioMath.convolution(MLXArray(values,[13,2]),weight:MLXArray(weights,[3,2,3]),dilation:dilation).asArray(Float.self)
      XCTAssertEqual(actual.count,expected.count)
      for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-6) }
    }
    let x=MLXArray(Array(values.prefix(12)),[2,3,2])
    let y=MLXAudioMath.nearest2x(x).asArray(Float.self)
    for h in 0..<4 { for w in 0..<6 { for c in 0..<2 { XCTAssertEqual(y[(h*6+w)*2+c],values[((h/2)*3+w/2)*2+c]) } } }
  }
  func testAudioAdmissionEstimatesAreBounded() throws {
    XCTAssertThrowsError(try MLXAudioDecoder.estimatedPeakBytes(latentFrames:0))
    XCTAssertThrowsError(try MLXAudioDecoder.estimatedPeakBytes(latentFrames:1502))
    XCTAssertLessThan(try MLXAudioDecoder.estimatedPeakBytes(latentFrames:501),2*1024*1024*1024)
  }
  func testReplicateFiltersMatchDirectEndpointArithmetic() throws {
    let values:[Float]=[1,-2,0.25,0.75,3,-1],x=MLXArray(values,[3,2])
    for (ratio,pad,crop,k) in [(2,5,15,12),(3,7,42,43)] {
      let filter=(0..<k).map { Float($0+1)/Float(k*k) }
      var expected=[Float](repeating:0,count:3*ratio*2)
      for t in 0..<3*ratio { for q in 0..<k {
        let s=t+crop-q
        if s>=0 && s%ratio==0 && s/ratio<3+2*pad {
          let source=min(2,max(0,s/ratio-pad))
          for c in 0..<2 { expected[t*2+c] += Float(ratio)*filter[q]*values[source*2+c] }
        }
      } }
      let actual=MLXAudioMath.upsampleFilter(x,filter:MLXArray(filter),ratio:ratio,inputPad:pad,cropLeft:crop).asArray(Float.self)
      XCTAssertEqual(actual.count,expected.count)
      for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-6) }
    }
    let filter=(0..<12).map { Float($0+1)/144 },v=(0..<12).map { sin(Float($0)) }
    let actual=MLXAudioMath.downsampleFilter(MLXArray(v,[6,2]),filter:MLXArray(filter)).asArray(Float.self)
    var expected=[Float](repeating:0,count:6)
    for t in 0..<3 { for q in 0..<12 { for c in 0..<2 {
      expected[t*2+c] += filter[q]*v[min(5,max(0,2*t+q-5))*2+c]
    } } }
    for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-6) }
  }
  func testConvolutionAndTransposeLayoutsMatchDirectSums() throws {
    let x=(0..<24).map { Float($0)/24 },w=(0..<54).map { sin(Float($0))*0.1 }
    for causal in [false,true] {
      var expected=[Float](repeating:0,count:3*4*3)
      for y in 0..<3 { for z in 0..<4 { for o in 0..<3 { for c in 0..<2 { for dy in 0..<3 { for dx in 0..<3 {
        let yy=y+dy-(causal ? 2 : 1),xx=z+dx-1
        if yy>=0 && yy<3 && xx>=0 && xx<4 { expected[(y*4+z)*3+o] += x[(yy*4+xx)*2+c]*w[((o*2+c)*3+dy)*3+dx] }
      } } } } } }
      let actual=MLXAudioMath.convolution(MLXArray(x,[3,4,2]),weight:MLXArray(w,[3,2,3,3]),causal:causal).asArray(Float.self)
      for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-6) }
    }
    let tw=(0..<18).map { Float($0)/18 },tx=Array(x.prefix(6))
    var expected=[Float](repeating:0,count:5*3)
    for t in 0..<3 { for i in 0..<2 { for o in 0..<3 { for k in 0..<3 {
      let out=t*2-1+k
      if out>=0 && out<5 { expected[out*3+o] += tx[t*2+i]*tw[(i*3+o)*3+k] }
    } } } }
    let actual=MLXAudioMath.transpose(MLXArray(tx,[3,2]),weight:MLXArray(tw,[2,3,3]),stride:2,padding:1).asArray(Float.self)
    for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-6) }
  }
  func testInstalledAudioMatchesIndependentOracleAndRetriesCancellation() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_MLX_AUDIO_VAE"],let oracle=env["WEETODD_MLX_AUDIO_ORACLE"] else { throw XCTSkip("Installed audio oracle is opt-in.") }
    struct Oracle:Decodable { let frames:Int;let latent:[Float];let waveform:[Float] }
    let reference=try JSONDecoder().decode(Oracle.self,from:Data(contentsOf:URL(fileURLWithPath:oracle)))
    let decoder=try MLXAudioDecoder(checkpoint:URL(fileURLWithPath:checkpoint))
    var routedStart=false
    XCTAssertThrowsError(try MLXMediaPipeline.decodeAudio(latent:reference.latent,latentFrames:reference.frames,
      checkpoint:URL(fileURLWithPath:checkpoint),backend:.mlx,progress:{ stage in
        XCTAssertEqual(stage,"start");routedStart=true;throw CancellationError()
      }))
    XCTAssertTrue(routedStart)
    XCTAssertThrowsError(try decoder.decode(latent:reference.latent,latentFrames:reference.frames,progress:{ stage in
      if stage=="audio_vae" { throw CancellationError() }
    }))
    XCTAssertEqual(decoder.residentWeightBytes,0)
    let start=Date();let output=try decoder.decode(latent:reference.latent,latentFrames:reference.frames,progress:{ stage in
      if stage=="start" { XCTAssertThrowsError(try decoder.decode(latent:reference.latent,latentFrames:reference.frames)) }
    })
    XCTAssertEqual(output.samples.count,reference.waveform.count);XCTAssertEqual(output.sampleRate,48000);XCTAssertEqual(output.channels,2)
    var square=0.0,maximum:Float=0
    for (a,b) in zip(output.samples,reference.waveform) { let d=a-b;square += Double(d*d);maximum=max(maximum,abs(d)) }
    let rmse=sqrt(square/Double(output.samples.count))
    print("MLX_AUDIO seconds=\(Date().timeIntervalSince(start)) maxabs=\(maximum) rmse=\(rmse)")
    XCTAssertLessThan(maximum,0.003);XCTAssertLessThan(rmse,0.0005)
    XCTAssertEqual(decoder.residentWeightBytes,0)
    var invalid=reference.latent;invalid[invalid.count/2] = .nan
    XCTAssertThrowsError(try decoder.decode(latent:invalid,latentFrames:reference.frames,progress:{ _ in XCTFail("Nonfinite input must fail before execution.") }))
    XCTAssertThrowsError(try decoder.decode(latent:reference.latent,latentFrames:reference.frames,maximumSamples:output.frameCount+1))
    let limited=try MLXAudioDecoder(checkpoint:URL(fileURLWithPath:checkpoint),maximumResidentBytes:256*1024*1024)
    XCTAssertThrowsError(try limited.decode(latent:reference.latent,latentFrames:reference.frames,progress:{ _ in XCTFail("Admission must precede weighted execution.") }))
    let cropped=try decoder.decode(latent:reference.latent,latentFrames:reference.frames,maximumSamples:17)
    XCTAssertEqual(cropped.samples,Array(output.samples.prefix(17))+Array(output.samples[output.frameCount..<output.frameCount+17]))
  }
}
