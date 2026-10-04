import CryptoKit
import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3LearnedUpscalerTests:XCTestCase {
  func testHaloConvolutionMatchesIndependentFullIntegerOracleAtEveryTemporalSpatialBoundary() throws {
    try Device.withDefaultDevice(.cpu) {
      for spatial in [8,16] {
        let input = MLXArray((0..<(5*spatial*spatial*2)).map { Float($0%7-3) },[1,5,spatial,spatial,2])
        let weightValues:[Float] = (0..<162).map { (index:Int) -> Float in Float(index % 3 - 1) }
        let weight = MLXArray(weightValues,[3,3,3,3,2])
        let bias = MLXArray([Float(-2),0,3])
        let oracle = MLX.conv3d(input,weight,padding:1)+bias; eval(oracle)
        let tiled = try H3LearnedUpscalerOps.convolve(input,weight:weight,bias:bias,
          plan:H3UpscalerConvolutionPlan(frames:2,height:3,width:5))
        XCTAssertEqual(tiled.shape,oracle.shape)
        XCTAssertEqual(tiled.asArray(Float.self).map(\.bitPattern),oracle.asArray(Float.self).map(\.bitPattern))
      }
    }
  }

  func testGlobalNormalizationIncludesTimeAndSpaceAndDoesNotNormalizeTilesIndependently() throws {
    try Device.withDefaultDevice(.cpu) {
      let values:[Float] = [0,2,4,6, 10,12,14,16]
      let x = MLXArray(values,[1,2,1,1,4])
      let result = try H3LearnedUpscalerOps.normalized(x,weight:MLXArray([Float](repeating:2,count:4)),
        bias:MLXArray([Float](repeating:3,count:4)),groups:2).asArray(Float.self)
      // Group0 includes [0,2,10,12]; group1 [4,6,14,16]. Variance26.
      let means:[Float] = [6,6,10,10,6,6,10,10]
      for i in values.indices {
        XCTAssertEqual(result[i],2*(values[i]-means[i])/sqrt(Float(26)+Float(1e-5))+3,accuracy:1e-6)
      }
    }
  }

  func testOwnedTwentyFourChannelStatsAndVAEPatchRoundtripNeverChangeAudio() throws {
    try Device.withDefaultDevice(.cpu) {
      XCTAssertEqual(H3LearnedLatentUpscaler.latentMean.count,24)
      var stats = Data()
      for value in H3LearnedLatentUpscaler.latentMean + H3LearnedLatentUpscaler.latentStandardDeviation {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of:&bits) { stats.append(contentsOf:$0) }
      }
      XCTAssertEqual(SHA256.hash(data:stats).map { String(format:"%02x",$0) }.joined(),"d21c1b7d8728b9a826aec663d7c79d9ea2a5f1118dce073484f4a437a1286768")
      XCTAssertEqual(H3LearnedLatentUpscaler.latentMean[0],Float(0.858090341091156))
      XCTAssertEqual(H3LearnedLatentUpscaler.latentStandardDeviation[23],Float(2.6127843856811523))
      let values:[Float] = (0..<768).map { (index:Int) -> Float in Float(index % 29 - 14) }
      let raw = MLXArray(values,[1,2,4,4,24])
      let mean = [Float](repeating:2,count:24),std = [Float](repeating:4,count:24)
      let packed = try H3LatentCodec.videoEncoderRows(latents:raw,mean:mean,standardDeviation:std)
      let unpacked = try H3LatentCodec.videoDecoderInput(rows:packed,latentFrames:2,latentHeight:4,
        latentWidth:4,mean:mean,standardDeviation:std)
      XCTAssertEqual(unpacked.asArray(Float.self).map(\.bitPattern),values.map(\.bitPattern))
    }
  }


  func testLearnedInputUsesSamplerDomainWithIndependentPhaseAndNonIdentityVAEWitness() throws {
    try Device.withDefaultDevice(.cpu) {
      let geometry = try H3Geometry(width:64,height:64,durationSeconds:2.5)
      let frames = geometry.videoLatentFrames
      // Independent physical channels/phases oracle: index c*100 +y*10+x.
      let rawValues = (0..<(frames*4*4*24)).map { index -> Float in
        let c=index%24, x=(index/24)%4, y=(index/96)%4
        return Float(c*100+y*10+x)+Float(0.25)
      }
      var packed:[Float]=[]
      for t in 0..<frames { for py in 0..<2 { for px in 0..<2 {
        for c in 0..<24 { for dy in 0..<2 { for dx in 0..<2 {
          packed.append(rawValues[((t*4+py*2+dy)*4+px*2+dx)*24+c])
        } } }
      } } }
      let rows=MLXArray(packed,[1,geometry.videoRows,96])
      let raw=try H3LearnedUpscalerOps.unpackSamplerRows(rows,geometry:geometry)
      XCTAssertEqual(raw.asArray(Float.self).map(\.bitPattern),rawValues.map(\.bitPattern))
      let learnedMean=Float(0.858090341091156),learnedStd=Float(1.2223774194717407)
      let networkInput=((raw-learnedMean)/learnedStd).asArray(Float.self)
      XCTAssertEqual(networkInput[0],Float(-0.4974652826786041),accuracy:1e-7)
      // DeliberatelydifferentVAE stats establish why a decoder roundtrip is
      // insufficient: this transform must not appear at network input/output.
      let decoder=try H3LatentCodec.videoDecoderInput(rows:rows,latentFrames:frames,
        latentHeight:4,latentWidth:4,mean:[Float](repeating:2,count:24),
        standardDeviation:[Float](repeating:3,count:24))
      let wrong=((decoder-learnedMean)/learnedStd).asArray(Float.self)
      XCTAssertEqual(wrong[0],Float(1.5477294921875),accuracy:1e-7)
      XCTAssertNotEqual(networkInput[0],wrong[0])
    }
  }

  func testLearnedOutputPacksSamplerDomainWithoutRemovingVAEDecoderStats() throws {
    try Device.withDefaultDevice(.cpu) {
      let learnedMean=Float(0.858090341091156),learnedStd=Float(1.2223774194717407)
      let networkOutput=MLXArray([Float](repeating:0.5,count:2*4*4*24),[1,2,4,4,24])
      let output=networkOutput*learnedStd+learnedMean
      let packed=try H3LearnedUpscalerOps.packSamplerLatents(output).asArray(Float.self)
      XCTAssertEqual(packed.count,768)
      for value in packed { XCTAssertEqual(value,Float(1.4692790508270264),accuracy:1e-7) }
      let wrong=try H3LatentCodec.videoEncoderRows(latents:output,
        mean:[Float](repeating:2,count:24),standardDeviation:[Float](repeating:3,count:24)).asArray(Float.self)
      XCTAssertEqual(wrong[0],Float(-0.1769069880247116),accuracy:1e-7)
      XCTAssertNotEqual(packed[0],wrong[0])
    }
  }

  func testCancellationStopsBeforeAnyConvolutionOrWeights() async throws {
    let task = Task { () throws -> Void in
      while !Task.isCancelled { await Task.yield() }
      try Device.withDefaultDevice(.cpu) {
        let input = MLXArray([Float](repeating:1,count:3*4*4*2),[1,3,4,4,2])
        let weight = MLXArray([Float](repeating:1,count:2*3*3*3*2),[2,3,3,3,2])
        _ = try H3LearnedUpscalerOps.convolve(input,weight:weight,
          bias:MLXArray([Float](repeating:0,count:2)),plan:H3UpscalerConvolutionPlan())
      }
    }
    task.cancel()
    do { try await task.value; XCTFail("Cancelled learned convolution must not execute.") }
    catch is CancellationError { }
  }

  // Sparse zero payload avoids allocating or reading 690 MiB in this header-only
  // fixture. It is not an installed-model or learned-output qualification.
  private func fixture(_ directory:URL, malformed:Bool=false) throws -> (URL,String) {
    var object:[String:Any] = [:],offset:UInt64=0
    for (name,shape) in H3LearnedUpscalerLayout.expectedShapes.sorted(by:{$0.key<$1.key}) {
      let count = shape.reduce(1,*)*2
      object[name] = ["dtype":malformed && name == "norm_out.weight" ? "F16":"BF16",
        "shape":shape,"data_offsets":[offset,offset+count]]
      offset += count
    }
    let json = try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])
    var length = UInt64(json.count).littleEndian
    var header = withUnsafeBytes(of:&length) { Data($0) };header.append(json)
    let file = directory.appendingPathComponent(UUID().uuidString+".safetensors")
    try header.write(to:file)
    let handle = try FileHandle(forWritingTo:file);defer { try? handle.close() }
    try handle.truncate(atOffset:UInt64(header.count)+offset)
    return (file,SHA256.hash(data:header).map { String(format:"%02x",$0) }.joined())
  }

  func testReleasedHeaderDigestHasExplicitPrefixDomainAndRejectsDtypePinAndMutation() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:directory) }
    let (url,pin) = try fixture(directory)
    let admitted = try H3LearnedUpscalerLayout(url:url,expectedHeaderSHA256:pin)
    XCTAssertEqual(admitted.residentWeightBytes,690_560_432)
    XCTAssertEqual(admitted.headerSHA256,pin)
    XCTAssertThrowsError(try H3LearnedUpscalerLayout(url:url,expectedHeaderSHA256:String(repeating:"a",count:64)))
    let (bad,badPin) = try fixture(directory,malformed:true)
    XCTAssertThrowsError(try H3LearnedUpscalerLayout(url:bad,expectedHeaderSHA256:badPin))
    let file = try SafeTensorFile(url:url)
    let handle = try FileHandle(forWritingTo:url);try handle.truncate(atOffset:64);try handle.close()
    XCTAssertThrowsError(try H3LearnedUpscalerLayout.headerDigest(url:url,file:file))
  }
}
