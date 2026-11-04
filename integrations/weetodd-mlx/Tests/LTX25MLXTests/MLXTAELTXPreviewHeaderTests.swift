import Foundation
import XCTest
import InferenceTestSupport
@testable import LTX25MLX

final class MLXTAELTXPreviewHeaderTests:XCTestCase {
  // Changing admission to trust a filename, omit an operator, or accept a
  // malformed grouped kernel must fail these real sparse-file probes.
  func testStandardAndWideAreAdmittedFromHeadersWithoutResidentWeights() throws {
    for wide in [false,true] {
      try withTensorFile(tensors:Self.tensors(wide:wide)) { url in
        let decoder=try MLXTAELTXPreviewDecoder(checkpoint:url)
        XCTAssertEqual(decoder.variant.rawValue,wide ? "taeltx2_3_wide" : "taeltx2_3")
        XCTAssertEqual(decoder.residentWeightBytes,0)
      }
    }
  }
  func testMissingExtraWrongDtypeAndWrongGroupedShapeFailHeaderAdmission() throws {
    let standard=Self.tensors(wide:false),wide=Self.tensors(wide:true)
    var wrongType=standard;wrongType[0].2="BF16"
    var grouped=wide
    let index=try XCTUnwrap(grouped.firstIndex { $0.0 == "decoder.3.conv.2.weight" })
    grouped[index].1=[1024,1024,3,3]
    for tensors in [Array(standard.dropFirst()),standard+[("unknown.weight",[1],"F16")],wrongType,grouped] {
      try withTensorFile(tensors:tensors) { url in
        XCTAssertThrowsError(try MLXTAELTXPreviewDecoder(checkpoint:url))
      }
    }
  }
  func testPreviewAdmissionBoundsLatentGridAndCausalOutputIndex() throws {
    let landscape=try MLXTAELTXPreviewPlan(shape:[1,128,9,24,42],maximumEdge:384,maximumLatentFrames:3)
    XCTAssertEqual(landscape.latentHeight,7)
    XCTAssertEqual(landscape.latentWidth,12)
    XCTAssertEqual(landscape.latentFrames,3)
    XCTAssertEqual(landscape.frameIndex,16)
    let portrait=try MLXTAELTXPreviewPlan(shape:[1,128,2,42,24],maximumEdge:384,maximumLatentFrames:3)
    XCTAssertEqual(portrait.latentHeight,12)
    XCTAssertEqual(portrait.latentWidth,7)
    XCTAssertEqual(portrait.frameIndex,8)
    let still=try MLXTAELTXPreviewPlan(shape:[1,128,1,1,1],maximumEdge:384,maximumLatentFrames:1)
    XCTAssertEqual(still.frameIndex,0)
    XCTAssertEqual(still.latentWidth,1)
    for shape in [[2,128,3,1,1],[1,16,3,1,1],[1,128,0,1,1],[1,128,3,0,1],[1,128,3,Int.max,1]] {
      XCTAssertThrowsError(try MLXTAELTXPreviewPlan(shape:shape,maximumEdge:384,maximumLatentFrames:3))
    }
    for edge in [0,31,385,Int.max] {
      XCTAssertThrowsError(try MLXTAELTXPreviewPlan(shape:[1,128,3,2,2],maximumEdge:edge,maximumLatentFrames:3))
    }
    for frames in [0,4,Int.max] {
      XCTAssertThrowsError(try MLXTAELTXPreviewPlan(shape:[1,128,3,2,2],maximumEdge:384,maximumLatentFrames:frames))
    }
  }
  static func tensors(wide:Bool) -> [(String,[Int],String)] {
    var result:[(String,[Int],String)]=[]
    func conv(_ name:String,_ output:Int,_ input:Int,_ kernel:Int=3,_ bias:Bool=true) {
      result.append((name+".weight",[output,input,kernel,kernel],"F16"))
      if bias { result.append((name+".bias",[output],"F16")) }
    }
    conv("encoder.0",64,48)
    for start in [2,7,12] {
      conv("encoder.\(start).conv",64,128,1,false)
      conv("encoder.\(start+1)",64,64,3,false)
      for block in (start+2)...(start+4) {
        conv("encoder.\(block).conv.0",64,128)
        conv("encoder.\(block).conv.2",64,64)
        conv("encoder.\(block).conv.4",64,64)
      }
    }
    conv("encoder.17",128,64)
    let channels=wide ? [1024,512,256,64] : [256,128,64,64]
    conv("decoder.1",channels[0],128)
    for stage in 0..<3 {
      let c=channels[stage],start=3+stage*6
      for block in start..<(start+3) {
        conv("decoder.\(block).conv.0",c,2*c,wide ? 1 : 3)
        conv("decoder.\(block).conv.2",c,wide ? 64 : c)
        conv("decoder.\(block).conv.4",c,c,wide ? 1 : 3)
        if wide { conv("decoder.\(block).conv.6",c,64) }
      }
      conv("decoder.\(start+4).conv",2*c,c,1,false)
      conv("decoder.\(start+5)",channels[stage+1],c,3,false)
    }
    conv("decoder.22",48,64)
    return result
  }
}
