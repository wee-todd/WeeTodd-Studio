import CryptoKit
import Darwin
import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3FastRefinedArtifactTests:XCTestCase {
  private struct Profile {
    let name:String,latentFrames:Int,visibleFrames:Int,paddedFrames:Int
    var rows:Int {latentFrames*90*160/4}
  }
  private func profile(_ name:String) throws -> Profile {
    switch name {
    case "short-22":return .init(name:name,latentFrames:7,visibleFrames:22,paddedFrames:22)
    case "full-120":return .init(name:name,latentFrames:37,visibleFrames:120,paddedFrames:124)
    default:throw H3CheckpointError.invalid("Select explicit short-22 or full-120 matched decode profile.")
    }
  }
  func testProfilesPreserveCompleteContextAndTrimOnlyRepeatedTail() throws {
    for name in ["short-22","full-120"] {
      let selected=try profile(name)
      XCTAssertEqual(try H3FastDecoderWindows.retainedFrames(latentFrames:selected.latentFrames).count,selected.paddedFrames)
      XCTAssertGreaterThanOrEqual(selected.paddedFrames,selected.visibleFrames)
      XCTAssertLessThan(selected.paddedFrames-selected.visibleFrames,17)
      XCTAssertLessThan(selected.latentFrames*90*160*2048,Int(Int32.max))
    }
    let full=try profile("full-120")
    XCTAssertEqual(full.rows,133200);XCTAssertEqual(full.visibleFrames,120)
    XCTAssertThrowsError(try profile("arbitrary"))
  }
  func testOptInMatchedRefined2KDecode() throws {
    let env=ProcessInfo.processInfo.environment
    guard let input=env["WEETODD_H3_FAST_REFINED_ROWS"],let destination=env["WEETODD_H3_FAST_REFINED_OUTPUT"],
      let weights=env["WEETODD_H3_FAST_WEIGHTS"],let base=env["WEETODD_H3_BASE_VAE"],
      let expected=env["WEETODD_H3_FAST_REFINED_SHA256"] else{throw XCTSkip("Explicit saved refined-latent qualification only.")}
    let selected=try profile(env["WEETODD_H3_FAST_REFINED_PROFILE"] ?? "short-22")
    let concurrency=env["WEETODD_H3_FAST_CONCURRENT_WORK"] ?? "unspecified; timing isolation is unverified"
    let output=URL(fileURLWithPath:destination)
    try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    let data=try Data(contentsOf:URL(fileURLWithPath:input))
    let digest=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
    XCTAssertEqual(digest,expected);guard digest==expected,data.count==selected.rows*96*4 else{throw H3CheckpointError.invalid("Refined artifact identity/shape differs.")}
    let rows=MLXArray(data,[1,selected.rows,96],dtype:.float32)
    let z=try H3LatentCodec.videoDecoderInput(rows:rows,latentFrames:selected.latentFrames,latentHeight:90,latentWidth:160,mean:[Float](repeating:0,count:24),standardDeviation:[Float](repeating:1,count:24))
    XCTAssertEqual(z.shape,[1,selected.latentFrames,90,160,24]);XCTAssertTrue(all(isFinite(z)).item(Bool.self))
    let layout=try H3FastDecoderLayout(weightsURL:URL(fileURLWithPath:weights),baseVAEURL:URL(fileURLWithPath:base))
    let decoder=try H3FastVideoDecoder(layout:layout,precision:.bfloat16,maximumActiveBytes:64*1024*1024*1024)
    Memory.clearCache();Memory.peakMemory=Memory.activeMemory
    let startUTC=ISO8601DateFormatter().string(from:Date()),start=ProcessInfo.processInfo.systemUptime
    print("FAST_REFINED_START \(startUTC) concurrent=\(concurrency)");fflush(stdout)
    let pixels=try decoder.decode(normalizedLatents:z){layer,total in
      print("FAST_REFINED_LAYER \(layer)/\(total) elapsed=\(ProcessInfo.processInfo.systemUptime-start) peak=\(Memory.peakMemory)");fflush(stdout)
    }
    eval(pixels);let seconds=ProcessInfo.processInfo.systemUptime-start
    let endUTC=ISO8601DateFormatter().string(from:Date())
    XCTAssertEqual(pixels.shape,[1,selected.paddedFrames,1440,2560,3]);XCTAssertTrue(all(isFinite(pixels)).item(Bool.self))
    let rgb=try H3FastVideoDecoder.rgb8(pixels[0...,0..<selected.visibleFrames]);eval(rgb)
    let owner=rgb.asData(access:.noCopyIfContiguous)
    try owner.data.write(to:output.appendingPathComponent("fast-video.rgb24"),options:.withoutOverwriting)
    let rgbSHA=SHA256.hash(data:owner.data).map{String(format:"%02x",$0)}.joined()
    var usage=rusage();getrusage(RUSAGE_SELF,&usage)
    var record:[String:Any]=["profile":selected.name,"visibleFrames":selected.visibleFrames,"paddedFrames":selected.paddedFrames,"inputSHA256":digest,"inputShape":[1,selected.rows,96],"canonicalShape":z.shape,"outputShape":rgb.shape,"rgbSHA256":rgbSHA,"rgbBytes":owner.data.count,"decoderSeconds":seconds,"peakMLXBytes":Memory.peakMemory,"processPeakRSSBytes":usage.ru_maxrss,"startUTC":startUTC,"endUTC":endUTC,"concurrentGPUWork":concurrency,"precision":"BF16 weights/activations, mixed FP32 normalization/attention","normalizationIdentity":layout.configuration.normalizationIdentity,"weightsSHA256":H3FastDecoderLayout.sourceSHA256]
    if let baseline=env["WEETODD_H3_RELEASED_BASELINE_SECONDS"].flatMap(Double.init) {record["releasedBaselineSeconds"]=baseline}
    try JSONSerialization.data(withJSONObject:record,options:[.prettyPrinted,.sortedKeys]).write(to:output.appendingPathComponent("measurements.json"),options:.atomic)
    print("FAST_REFINED_COMPLETE seconds=\(seconds) peak=\(Memory.peakMemory) rgbSHA=\(rgbSHA) endUTC=\(endUTC)");fflush(stdout)
  }
}
