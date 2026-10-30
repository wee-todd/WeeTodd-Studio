import Foundation
import InferenceContracts
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3RegenerateAttentionPolicyTests: XCTestCase {
  func testFull2KUsesOneHeadGroupWithoutWideningQueryWindows() throws {
    let rows = 133_727
    let group = try H3RegenerateAttention.preferredHeadGroupSize(rows:rows,heads:56,headWidth:128)
    XCTAssertEqual(group,56)
    XCTAssertEqual(((rows+255)/256)*((56+group-1)/group),523)
  }

  func testSigned32BoundaryRetainsFourHeadGroupsForLargerSequences() throws {
    XCTAssertEqual(try H3RegenerateAttention.preferredHeadGroupSize(rows:149_796,heads:56,headWidth:128),56)
    for rows in [149_797,302_590,390_000] {
      XCTAssertEqual(try H3RegenerateAttention.preferredHeadGroupSize(rows:rows,heads:56,headWidth:128),14)
    }
  }

  func testSelectedGroupsAdmitCompleteKeysAndEverySigned32Product() throws {
    for rows in [1,8,256,133_727,149_796,149_797,302_590,390_000] {
      let group = try H3RegenerateAttention.preferredHeadGroupSize(rows:rows,heads:56,headWidth:128)
      let q = Int64(min(256,rows)), h = Int64(group), k = Int64(rows)
      XCTAssertLessThan(q*h*128,Int64(Int32.max))
      XCTAssertLessThan(k*h*128,Int64(Int32.max))
      XCTAssertLessThan(q*h*k,Int64(Int32.max))
    }
  }

  func testPolicyRejectsMalformedGeometryAndProductionQueryCapChanges() {
    for rows in [0,-1,390_001,Int.max] {
      XCTAssertThrowsError(try H3RegenerateAttention.preferredHeadGroupSize(rows:rows,heads:56,headWidth:128))
    }
    for heads in [0,-1,57,Int.max] {
      XCTAssertThrowsError(try H3RegenerateAttention.preferredHeadGroupSize(rows:133_727,heads:heads,headWidth:128))
    }
    for width in [0,-1,129,Int.max] {
      XCTAssertThrowsError(try H3RegenerateAttention.preferredHeadGroupSize(rows:133_727,heads:56,headWidth:width))
    }
    for chunk in [0,-1,257,Int.max] {
      XCTAssertThrowsError(try H3RegenerateAttention.preferredHeadGroupSize(rows:133_727,heads:56,headWidth:128,queryChunk:chunk))
    }
  }

  func testOptInBF16ProjectedAutomaticGroupingMatchesFourGroupsExactly() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_REGENERATE_PROJECTED_PARITY"] == "1" else {
      throw XCTSkip("Enable the bounded nonzero BF16 projected-attention parity under the GPU lease.")
    }
    let worker = try XCTUnwrap(environment["WEETODD_H3_REGENERATE_ATTENTION_WORKER"])
    GPU.metallib = try NativeMLXMetalLibrary.resolve(executable:URL(fileURLWithPath:worker))
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    let rows = 545, hidden = 8
    let input = (MLXRandom.normal([1,rows,hidden],key:MLXRandom.key(941))*0.25).asType(.bfloat16)
    let qkvWeight = (MLXRandom.normal([21504,hidden],key:MLXRandom.key(942))*0.1).asType(.bfloat16)
    let outputWeight = (MLXRandom.normal([hidden,7168],key:MLXRandom.key(943))*0.02).asType(.bfloat16)
    let phases = MLXArray(0..<(rows*96)).asType(.float32)*0.002
    let angles = H3RotaryAngles(rows:rows,
      cosine:cos(phases).asType(.bfloat16).reshaped([1,1,rows,96]),
      sine:sin(phases).asType(.bfloat16).reshaped([1,1,rows,96]))
    let norm = MLXArray.ones([128],dtype:.bfloat16)
    eval(input,qkvWeight,outputWeight,norm,angles.cosine,angles.sine)
    func project(_ activation:MLXArray,_ name:String,_ output:Int,_ columns:Int,_ qkv:Bool) throws -> MLXArray {
      matmul(activation,(name == "attn.qkv_proj" ? qkvWeight:outputWeight).T)
    }
    let reference = try H3RegenerateAttention.projected(first:input,angles:angles,qNorm:norm,kNorm:norm,
      heads:56,headWidth:128,rotaryWidth:96,maximumStageBytes:2*1024*1024*1024,
      tokenChunk:127,queryChunk:256,headsPerGroup:14,project:project)
    let plan=try H3RegenerateAttentionPlan.runtime(rows:rows,heads:56,headWidth:128,dtype:.bfloat16)
    let actual = try H3RegenerateAttention.projected(first:input,angles:angles,qNorm:norm,kNorm:norm,
      heads:56,headWidth:128,rotaryWidth:96,maximumStageBytes:2*1024*1024*1024,
      tokenChunk:127,executionPlan:plan,project:project)
    XCTAssertEqual(actual.shape,[1,rows,hidden])
    XCTAssertEqual(actual.dtype,.bfloat16)
    XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
    XCTAssertTrue(all(actual .== reference).item(Bool.self))
    print("H3_REGENERATE_PROJECTED_PARITY rows=545 heads=56 D128 automatic_q\(plan.queryChunk)_vs_h14 maxabs=\(max(abs(actual.asType(.float32)-reference.asType(.float32))).item(Float.self))")
  }
}
