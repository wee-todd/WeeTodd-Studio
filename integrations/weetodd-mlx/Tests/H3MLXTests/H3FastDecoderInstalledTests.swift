import Foundation
import CryptoKit
import Darwin
import XCTest
import MLX
@testable import H3MLX

final class H3FastDecoderInstalledTests:XCTestCase {
  private func assets() throws -> (H3FastDecoderLayout,URL,URL?) {
    let env=ProcessInfo.processInfo.environment
    guard let weights=env["WEETODD_H3_FAST_WEIGHTS"],let base=env["WEETODD_H3_BASE_VAE"] else { throw XCTSkip("Set existing fast/base VAE paths for explicit installed qualification.") }
    let baseURL=URL(fileURLWithPath:base)
    let layout=try H3FastDecoderLayout(weightsURL:URL(fileURLWithPath:weights),baseVAEURL:baseURL)
    if let config=env["WEETODD_H3_ORIGINAL_VAE_CONFIG"] {
      let object=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:config))) as! [String:Any]
      let original=try H3FastDecoderLayout.Configuration(json:object)
      XCTAssertEqual(original.normalizationIdentity,layout.configuration.normalizationIdentity)
    }
    let artifacts=env["WEETODD_H3_FAST_ARTIFACTS"].map { URL(fileURLWithPath:$0) }
    if let artifacts { try FileManager.default.createDirectory(at:artifacts,withIntermediateDirectories:true) }
    return(layout,baseURL,artifacts)
  }
  private func input(t:Int,h:Int,w:Int) -> MLXArray {
    MLXArray((0..<(t*h*w*24)).map { Float(($0*37)%101-50)/500 },[1,t,h,w,24])
  }
  private func digest(_ array:MLXArray) -> String {
    SHA256.hash(data:array.asType(.float32).asData(access:.noCopyIfContiguous).data).map { String(format:"%02x",$0) }.joined()
  }
  private func measure(_ name:String,_ body:() throws -> MLXArray) throws -> (MLXArray,[String:Any]) {
    Memory.clearCache();Memory.peakMemory=Memory.activeMemory
    let start=ProcessInfo.processInfo.systemUptime,result=try body();eval(result)
    let elapsed=ProcessInfo.processInfo.systemUptime-start
    var usage=rusage();getrusage(RUSAGE_SELF,&usage)
    let stats:[String:Any]=["name":name,"seconds":elapsed,"peakMLXBytes":Memory.peakMemory,"processPeakRSSBytes":usage.ru_maxrss,"shape":result.shape,"sha256":digest(result)]
    print("H3_FAST_MEASURE \(name) seconds=\(elapsed) peakMLXBytes=\(Memory.peakMemory) processPeakRSSBytes=\(usage.ru_maxrss)")
    return(result,stats)
  }
  func testOptInInstalled128PrecisionAndStandardComparison() throws {
    let (layout,base,artifacts)=try assets(),z=input(t:12,h:8,w:8)
    let fp32=try H3FastVideoDecoder(layout:layout,precision:.float32)
    let (referenceMath,f32Stats)=try measure("fast-fp32-128-synthetic") { try fp32.decode(normalizedLatents:z) }
    XCTAssertEqual(referenceMath.shape,[1,39,128,128,3])
    XCTAssertTrue(referenceMath.asArray(Float.self).allSatisfy(\.isFinite))
    let bf16=try H3FastVideoDecoder(layout:layout,precision:.bfloat16)
    let (fast,bfStats)=try measure("fast-bf16-128-synthetic") { try bf16.decode(normalizedLatents:z) }
    XCTAssertEqual(fast.shape,referenceMath.shape)
    XCTAssertTrue(fast.asArray(Float.self).allSatisfy(\.isFinite))
    let raw=z*MLXArray(layout.configuration.standardDeviation)+MLXArray(layout.configuration.mean)
    let (standard,standardStats)=try measure("standard-native-q8-fp32-128-synthetic") { try H3VideoVAEDecoder.decode(checkpointURL:base,latent:raw) }
    XCTAssertEqual(standard.shape,fast.shape)
    let precisionRMSE=sqrt(mean((fast-referenceMath)*(fast-referenceMath))).item(Float.self)
    let fastRGB=try H3FastVideoDecoder.rgb8(fast).asType(.float32),standardRGB=try H3LatentCodec.videoPixelsRGB8(standard).asType(.float32)
    let mse=mean((fastRGB-standardRGB)*(fastRGB-standardRGB)).item(Float.self)
    let psnr=10*log10(255*255/max(mse,1e-12))
    print("H3_FAST_SYNTHETIC precisionRMSE=\(precisionRMSE) standardRGB_PSNR=\(psnr); synthetic latents are not a restoration quality qualification")
    if let artifacts {
      try z.asData(access:.noCopyIfContiguous).data.write(to:artifacts.appendingPathComponent("synthetic-input-1x12x8x8x24.f32"),options:.atomic)
      try fast.asData(access:.noCopyIfContiguous).data.write(to:artifacts.appendingPathComponent("fast-bf16-pixels-1x39x128x128x3.f32"),options:.atomic)
      try referenceMath.asData(access:.noCopyIfContiguous).data.write(to:artifacts.appendingPathComponent("fast-fp32-pixels-1x39x128x128x3.f32"),options:.atomic)
      try standard.asData(access:.noCopyIfContiguous).data.write(to:artifacts.appendingPathComponent("standard-q8-pixels-1x39x128x128x3.f32"),options:.atomic)
      let record:[String:Any]=["inputKind":"synthetic deterministic normalized latents, not refined latents","inputSHA256":digest(z),"weightsSHA256":H3FastDecoderLayout.sourceSHA256,"normalizationIdentity":layout.configuration.normalizationIdentity,"stages":[f32Stats,bfStats,standardStats],"precisionRMSE":precisionRMSE,"standardRGBPSNR":psnr,"os":ProcessInfo.processInfo.operatingSystemVersionString]
      try JSONSerialization.data(withJSONObject:record,options:[.prettyPrinted,.sortedKeys]).write(to:artifacts.appendingPathComponent("installed-128-measurements.json"),options:.atomic)
    }
  }
  func testOptInInstalledNoncausalHaloMatchesWholeClip() throws {
    let (layout,_,artifacts)=try assets(),z=input(t:140,h:1,w:1)
    let decoder=try H3FastVideoDecoder(layout:layout,precision:.float32)
    let output=70..<75,needed=try H3FastDecoderWindows.requiredContext(output:output,totalFrames:140)
    XCTAssertLessThan(needed.count,140)
    let (whole,wholeStats)=try measure("fast-fp32-whole-140x16-synthetic") { try decoder.decode(normalizedLatents:z,outputRange:output) }
    let (bounded,boundedStats)=try measure("fast-fp32-halo-140x16-synthetic") { try decoder.decode(normalizedLatents:z[0...,needed],contextRange:needed,outputRange:output,totalLatentFrames:140) }
    XCTAssertEqual(whole.shape,[1,17,16,16,3]);XCTAssertEqual(bounded.shape,whole.shape)
    let error=abs(whole-bounded),maximumError=max(error).item(Float.self),rmse=sqrt(mean(error*error)).item(Float.self)
    XCTAssertLessThan(maximumError,0.003);XCTAssertLessThan(rmse,0.0001)
    print("H3_FAST_HALO needed=\(needed) maxAbs=\(maximumError) rmse=\(rmse)")
    if let artifacts {
      let record:[String:Any]=["inputKind":"synthetic temporal dependency qualification","fullLatentFrames":140,"outputLatentRange":[70,75],"requiredContext":[needed.lowerBound,needed.upperBound],"maximumAbsoluteError":maximumError,"rmse":rmse,"stages":[wholeStats,boundedStats]]
      try JSONSerialization.data(withJSONObject:record,options:[.prettyPrinted,.sortedKeys]).write(to:artifacts.appendingPathComponent("installed-halo-measurements.json"),options:.atomic)
    }
  }
  func testOptInInstalled256ShiftedSpatialEdges() throws {
    let (layout,_,artifacts)=try assets(),z=input(t:2,h:16,w:16)
    let decoder=try H3FastVideoDecoder(layout:layout,precision:.float32)
    let (pixels,stats)=try measure("fast-fp32-256-spatial-edges-synthetic") { try decoder.decode(normalizedLatents:z) }
    XCTAssertEqual(pixels.shape,[1,5,256,256,3])
    XCTAssertTrue(pixels.asArray(Float.self).allSatisfy(\.isFinite))
    if let artifacts {
      try JSONSerialization.data(withJSONObject:stats,options:[.prettyPrinted,.sortedKeys]).write(to:artifacts.appendingPathComponent("installed-256-edge-measurements.json"),options:.atomic)
    }
  }

  func testOptInInstalled2KCompleteContext() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_FAST_2K"] == "1" else { throw XCTSkip("Explicit full 2K context qualification only.") }
    let (layout,_,artifacts)=try assets(),z=input(t:37,h:90,w:160)
    let decoder=try H3FastVideoDecoder(layout:layout,precision:.bfloat16)
    let (pixels,stats)=try measure("fast-bf16-2k-37-latent-complete-context-synthetic") {
      try decoder.decode(normalizedLatents:z) { layer,total in
        print("H3_FAST_2K_LAYER \(layer)/\(total) active=\(Memory.activeMemory) peak=\(Memory.peakMemory)")
        fflush(stdout)
      }
    }
    XCTAssertEqual(pixels.shape,[1,124,1440,2560,3])
    XCTAssertTrue(all(isFinite(pixels)).item(Bool.self))
    if let artifacts {
      try JSONSerialization.data(withJSONObject:stats,options:[.prettyPrinted,.sortedKeys]).write(to:artifacts.appendingPathComponent("installed-2k-complete-context-measurements.json"),options:.atomic)
    }
  }

}
