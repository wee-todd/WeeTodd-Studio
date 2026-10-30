import Foundation
import InferenceContracts
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3RegenerateAttentionPlanTests: XCTestCase {
  func testQualifiedPinnedMetalRouteSelectsWideQueriesAndAuditableMetadata() throws {
    let plan=try H3RegenerateAttentionPlan.resolve(rows:133_727,heads:56,headWidth:128,
      isMetalGPU:true,isBF16:true,architecture:"applegpu_g15d")
    XCTAssertEqual(plan.queryChunk,8192);XCTAssertEqual(plan.headsPerGroup,56)
    XCTAssertEqual(plan.outputProjectionChunk,256)
    XCTAssertTrue(plan.forceFused);XCTAssertEqual(plan.policyVersion,2)
    XCTAssertEqual(plan.route,"metal-steel-bf16-d128")
    XCTAssertEqual(plan.metadata["queryChunk"] as? Int,8192)
    XCTAssertEqual(plan.metadata["outputProjectionChunk"] as? Int,256)
    XCTAssertEqual(plan.metadata["attentionQueryChunk"] as? Int,8192)
    XCTAssertEqual(plan.attentionCallsPerBlock,17)
    XCTAssertEqual(plan.outputProjectionCallsPerBlock,523)
    XCTAssertEqual(plan.metadata["forceFused"] as? Bool,true)
    XCTAssertEqual(plan.metadata["architecture"] as? String,"applegpu_g15d")
    XCTAssertEqual(plan.metadata["mlxRevision"] as? String,"1f8e74e3f12f31365464a6867c6579f0e9b29d85")
    XCTAssertTrue(JSONSerialization.isValidJSONObject(plan.metadata))
  }

  func testUnknownHardwareCPUOtherPrecisionAndDimensionsKeepLegacyQueryGuard() throws {
    for architecture in ["applegpu_g14d","applegpu_g16p","applegpu_g17p","unknown",""] {
      let plan=try H3RegenerateAttentionPlan.resolve(rows:133_727,heads:56,headWidth:128,
        isMetalGPU:true,isBF16:true,architecture:architecture)
      XCTAssertEqual(plan.queryChunk,256);XCTAssertFalse(plan.forceFused)
    }
    for (gpu,bf16,heads,width) in [(false,true,56,128),(true,false,56,128),(true,true,4,128),(true,true,56,8)] {
      let plan=try H3RegenerateAttentionPlan.resolve(rows:133_727,heads:heads,headWidth:width,
        isMetalGPU:gpu,isBF16:bf16,architecture:"applegpu_g15d")
      XCTAssertEqual(plan.queryChunk,256);XCTAssertFalse(plan.forceFused)
    }
  }

  func testUnqualifiedLargerSequencesRetainFourGroupsAnd256Queries() throws {
    for rows in [149_797,302_590,390_000] {
      let plan=try H3RegenerateAttentionPlan.resolve(rows:rows,heads:56,headWidth:128,
        isMetalGPU:true,isBF16:true,architecture:"applegpu_g15d")
      XCTAssertEqual(plan.queryChunk,256);XCTAssertEqual(plan.headsPerGroup,14)
      XCTAssertFalse(plan.forceFused)
    }
  }

  func testPlanRejectsMalformedRowsBeforeAnyGPUOperation() {
    for rows in [0,-1,390_001,Int.max] {
      XCTAssertThrowsError(try H3RegenerateAttentionPlan.resolve(rows:rows,heads:56,headWidth:128,
        isMetalGPU:true,isBF16:true,architecture:"applegpu_g15d"))
    }
  }

  func testEightOrFewerRowsDoNotSelectTheAuditedFullKernel() throws {
    for rows in [1,8] {
      let plan=try H3RegenerateAttentionPlan.resolve(rows:rows,heads:56,headWidth:128,
        isMetalGPU:true,isBF16:true,architecture:"applegpu_g15d")
      XCTAssertEqual(plan.queryChunk,256);XCTAssertFalse(plan.forceFused)
    }
  }

  func testOptInForcedOwnerAdmissionAndSmallTailFallback() throws {
    let environment=ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_REGENERATE_PROJECTED_PARITY"] == "1" else {
      throw XCTSkip("Enable the bounded forced-owner GPU admission regression.")
    }
    let worker=try XCTUnwrap(environment["WEETODD_H3_REGENERATE_ATTENTION_WORKER"])
    GPU.metallib=try NativeMLXMetalLibrary.resolve(executable:URL(fileURLWithPath:worker))
    guard GPU.deviceInfo().architecture == "applegpu_g15d" else { throw XCTSkip("Qualified non-NAX device required.") }
    defer { Stream.gpu.synchronize();Memory.clearCache() }
    let q=MLXRandom.normal([1,56,33,128],key:MLXRandom.key(1041)).asType(.bfloat16)
    let k=MLXRandom.normal([1,56,33,128],key:MLXRandom.key(1042)).asType(.bfloat16)
    let v=MLXRandom.normal([1,56,33,128],key:MLXRandom.key(1043)).asType(.bfloat16)
    let storage=MLXRandom.normal([1,56,66,128],key:MLXRandom.key(1044)).asType(.bfloat16)
    let strided=storage[0..<1,0..<56,.stride(by:2),0..<128]
    eval(q,k,v,strided)
    XCTAssertThrowsError(try H3RegenerateAttention.forcedAttend(queryOwner:strided,keysOwner:k,valuesOwner:v,range:0..<9))
    XCTAssertThrowsError(try H3RegenerateAttention.forcedAttend(queryOwner:q.asType(.float32),keysOwner:k,valuesOwner:v,range:0..<9))
    XCTAssertThrowsError(try H3RegenerateAttention.forcedAttend(queryOwner:q,keysOwner:k,valuesOwner:v,range:0..<34))
    let tail=try H3RegenerateAttention.forcedAttend(queryOwner:q,keysOwner:k,valuesOwner:v,range:32..<33)
    let legacy=try H3RegenerateAttention.attend(query:q[0..<1,0..<56,32..<33,0..<128],keys:k,values:v)
    XCTAssertTrue(all(tail .== legacy).item(Bool.self))
    let forced=try H3RegenerateAttention.forcedAttend(queryOwner:q,keysOwner:k,valuesOwner:v,range:0..<33)
    let expected=try H3RegenerateAttention.attend(query:q,keys:k,values:v)
    XCTAssertTrue(all(forced .== expected).item(Bool.self))
  }
}
