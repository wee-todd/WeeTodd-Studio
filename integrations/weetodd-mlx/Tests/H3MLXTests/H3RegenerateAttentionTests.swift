import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateAttentionTests:XCTestCase {
  func testQueryWindowsAndHeadGroupsKeepEveryKeyVisible() throws {
    let q=sin(MLXArray(0..<(1*4*11*8)).asType(.float32)).reshaped([1,4,11,8])
    let k=cos(MLXArray(0..<(1*4*17*8)).asType(.float32)*0.03).reshaped([1,4,17,8])
    let v=sin(MLXArray(0..<(1*4*17*8)).asType(.float32)*0.07).reshaped([1,4,17,8])
    let full=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:1/Float(8).squareRoot(),mask:nil)
    let chunked=try H3RegenerateAttention.attend(query:q,keys:k,values:v,queryChunk:3)
    XCTAssertLessThan(max(abs(full-chunked)).item(Float.self),0.000001)
    XCTAssertThrowsError(try H3RegenerateAttention.attend(query:q,keys:k,values:v,queryChunk:0))
  }
  func testTokenWindowQKVAndHeadGroupsMatchFullDenseProjectedAttention() throws {
    let input=sin(MLXArray(0..<(9*8)).asType(.float32)*0.03).reshaped([1,9,8])
    let qkvWeight=cos(MLXArray(0..<(48*8)).asType(.float32)*0.07).reshaped([48,8])*0.1
    let outputWeight=sin(MLXArray(0..<(8*16)).asType(.float32)*0.1).reshaped([8,16])*0.1
    let qkv=matmul(input,qkvWeight.T).reshaped([1,9,4,3,4]),norm=MLXArray.ones([4])
    let q=MLXFast.rmsNorm(qkv[.ellipsis,0,0..<4],weight:norm,eps:1e-5).transposed(0,2,1,3)
    let k=MLXFast.rmsNorm(qkv[.ellipsis,1,0..<4],weight:norm,eps:1e-5).transposed(0,2,1,3)
    let v=qkv[.ellipsis,2,0..<4].transposed(0,2,1,3)
    let attended=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:0.5,mask:nil)
      .transposed(0,2,1,3).reshaped([1,9,16])
    let expected=matmul(attended,outputWeight.T)
    let angles=H3RotaryAngles(rows:9,cosine:.ones([1,1,9,4]),sine:.zeros([1,1,9,4]))
    let actual=try H3RegenerateAttention.projected(first:input,angles:angles,qNorm:norm,kNorm:norm,
      heads:4,headWidth:4,rotaryWidth:4,maximumStageBytes:32*1024*1024*1024,
      tokenChunk:3,queryChunk:2,headsPerGroup:2) { activation,name,_,_,_ in
      matmul(activation,(name == "attn.qkv_proj" ? qkvWeight:outputWeight).T)
    }
    XCTAssertLessThan(max(abs(expected-actual)).item(Float.self),0.000001)
  }
  func testWholeGridIndexingWithoutAFullQKVMegatensor() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_GRID_PROBE"] == "1" else {
      throw XCTSkip("Enable the explicit full-grid MLX indexing probe before 2K/4K qualification.")
    }
    for rows in [133870,302590] {
      let started=Date()
      let q=MLXArray.zeros([1,14,256,128],dtype:.bfloat16)
      let k=MLXArray.zeros([1,14,rows,128],dtype:.bfloat16)
      let v=MLXArray.ones([1,14,rows,128],dtype:.bfloat16)
      let result=try H3RegenerateAttention.attend(query:q,keys:k,values:v,queryChunk:256)
      XCTAssertLessThan(max(abs(result.asType(.float32)-1)).item(Float.self),0.015625)
      eval(result)
      print("Regenerate full-grid indexing rows=\(rows) heads=14 queries=256 seconds=\(Date().timeIntervalSince(started)) peakMiB=\(Memory.peakMemory/1024/1024)")
      Stream.gpu.synchronize();Memory.clearCache()
    }
  }
}
