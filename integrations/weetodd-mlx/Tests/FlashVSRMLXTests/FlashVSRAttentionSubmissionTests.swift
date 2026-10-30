import XCTest
import Foundation
import MLX
@testable import FlashVSRMLX

final class FlashVSRAttentionSubmissionTests: XCTestCase {
  func testCanonicalWindowsShareSubmissionsWithoutPadding() throws {
    let plan=try FlashVSRAttentionSubmissionPlan(selectedCounts:Array(repeating:9,count:24))
    XCTAssertEqual(plan.groups.count,3)
    XCTAssertEqual(plan.groups.map(\.count),[8,8,8])
    XCTAssertEqual(plan.groups.flatMap{$0},Array(0..<24))
    XCTAssertLessThan(FlashVSRAttentionSubmissionPlan.maximumScratchBytes,64*1024*1024)
  }

  func testRaggedGroupsKeepQueryIdentityAndSelectedLength() throws {
    let counts=[3,1,3,9,1,3,1,9,3,1,3]
    let plan=try FlashVSRAttentionSubmissionPlan(selectedCounts:counts,maximumWindows:4)
    XCTAssertEqual(plan.groups.flatMap{$0}.sorted(),Array(counts.indices))
    for group in plan.groups {
      XCTAssertLessThanOrEqual(group.count,4)
      XCTAssertEqual(Set(group.map{counts[$0]}).count,1)
      XCTAssertEqual(group,group.sorted())
    }
    XCTAssertEqual(plan.groups.count,4)
    let serial=try FlashVSRAttentionSubmissionPlan(selectedCounts:counts,maximumWindows:1)
    XCTAssertEqual(serial.groups.count,counts.count)
  }

  func testInvalidEmptySelectionsAndUnboundedSubmissionAreRejected() {
    for counts in [[],[0],[1,-1]] {
      XCTAssertThrowsError(try FlashVSRAttentionSubmissionPlan(selectedCounts:counts))
    }
    for limit in [0,-1,9,Int.max] {
      XCTAssertThrowsError(try FlashVSRAttentionSubmissionPlan(selectedCounts:[1],maximumWindows:limit))
    }
  }
}


extension FlashVSRAttentionSubmissionTests {
  // Changing the selection, output order, matmul rank/reduction, or input dtype
  // must fail a whole-array raw-word comparison against the prechange algorithm.
  func testCanonicalRaggedOutputsRetainAllSerialF32AndBF16Bits() throws {
    let shape=[2,2,8,128,128],keyShape=[2,2,10,128,128]
    func values(_ count:Int,_ factor:Int)->[Float] {(0..<count).map{Float((($0*factor)%29)*2-27)/64}}
    let selection=(0..<2).map { b in (0..<2).map { h in (0..<8).map { block in
      (0..<(block%2 == 0 ? 1:9)).map{(b+h+block+3*$0)%10}
    } } }
    for dtype:DType in [.float32,.bfloat16] {
      let q=MLXArray(values(shape.reduce(1,*),7),shape).asType(dtype)
      let k=MLXArray(values(keyShape.reduce(1,*),11),keyShape).asType(dtype)
      let v=MLXArray(values(keyShape.reduce(1,*),13),keyShape).asType(dtype)
      let reference=try FlashAttentionSerialReference.attend(queries:q,keys:k,values:v,selection:selection)
      let actual=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selection)
      XCTAssertEqual(actual.shape,shape)
      XCTAssertEqual(actual.dtype,dtype)
      XCTAssertTrue(actual.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
      if dtype == .bfloat16 {
        let a=actual.view(dtype:.uint16).asArray(UInt16.self),b=reference.view(dtype:.uint16).asArray(UInt16.self)
        XCTAssertEqual(a.count,b.count)
        XCTAssertEqual(zip(a,b).filter{$0 != $1}.count,0,"Every BF16 word must match.")
      } else {
        let a=actual.asArray(Float.self),b=reference.asArray(Float.self)
        XCTAssertTrue(a.allSatisfy(\.isFinite))
        XCTAssertEqual(a.count,b.count)
        XCTAssertEqual(zip(a,b).filter{$0.bitPattern != $1.bitPattern}.count,0,"Every F32 word must match.")
      }
    }
  }

  func testCancellationBetweenKeySlabsReleasesScratch() throws {
    try MLX.withError { box in
      for windows in [1,8] {
        let q=MLXArray.ones([1,1,windows,128,128]).asType(.bfloat16)
        let k=MLXArray.ones([1,1,17,128,128]).asType(.bfloat16),v=k*0.7
        eval(q,k,v);Stream.gpu.synchronize();Memory.clearCache();try box.check()
        try withExtendedLifetime((q,k,v)) {
          let baseline=Memory.activeMemory
          // One complete slab is evaluated before cancellation interrupts the
          // next slab, including an already constructed window in the group8 case.
          let cancellationPoint=windows == 1 ? 5:13
          var calls=0,cancelled=false
          do {
            try autoreleasepool {
              _=try FlashVSRAttention.attend(queries:q,keys:k,values:v,
                selection:[[Array(repeating:Array(0..<17),count:windows)]],checkCancellation:{
                  calls+=1
                  if calls==cancellationPoint {throw CancellationError()}
                })
            }
          } catch is CancellationError {cancelled=true}
          XCTAssertTrue(cancelled,"Cancellation must be checked inside a long selected-key loop.")
          XCTAssertEqual(calls,cancellationPoint)
          Stream.gpu.synchronize();Memory.clearCache();try box.check()
          XCTAssertEqual(Memory.activeMemory,baseline,"Cancelled attention must drop owned scratch.")
        }
      }
    }
  }

  func testMalformedRoutingIsRejected() throws {
    let q=MLXArray.ones([1,1,2,128,128]),k=q,v=q
    for headSelection in [[[0,0],[1]],[[2],[1]],[[],[1]],[[0]]] {
      XCTAssertThrowsError(try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:[[headSelection]]))
    }
  }

  func testCustomSlabAndSmallGeometryKeepSerialResult() throws {
    let q=MLXArray([Float(0.2),-0.4,0.7,0.1],[1,1,2,1,2])
    let k=MLXArray([Float(0.3),0.2,-0.1,0.8,0.2,-0.3],[1,1,3,1,2])
    let v=k*0.7,selection=[[[[2,0,1],[1,0]]]]
    let actual=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selection,keySlabBlocks:2)
    let reference=try FlashAttentionSerialReference.attend(queries:q,keys:k,values:v,selection:selection,keySlabBlocks:2)
    XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),reference.asArray(Float.self).map(\.bitPattern))
  }

  func testSuccessfulGroupedCallReleasesOwnedArrays() throws {
    try MLX.withError { box in
      Stream.gpu.synchronize();Memory.clearCache();try box.check()
      let baseline=Memory.activeMemory
      try autoreleasepool {
        let q=MLXArray.ones([1,1,8,128,128]).asType(.bfloat16),k=q,v=q*0.7
        let result=try FlashVSRAttention.attend(queries:q,keys:k,values:v,
          selection:[[Array(repeating:[0,1],count:8)]])
        eval(result);try box.check()
      }
      Stream.gpu.synchronize();Memory.clearCache();try box.check()
      XCTAssertEqual(Memory.activeMemory,baseline)
    }
  }
}

// Frozen pre-optimization production algorithm from commit0b294415, test-only.
private enum FlashAttentionSerialReference {
  static func attend(queries:MLXArray, keys:MLXArray, values:MLXArray, selection:[[[[Int]]]], keySlabBlocks:Int = 8) throws -> MLXArray {
    guard queries.ndim == 5, keys.ndim == 5, values.shape == keys.shape,
      queries.shape[0..<2] == keys.shape[0..<2], queries.shape[3...] == keys.shape[3...],
      keySlabBlocks > 0, selection.count == queries.shape[0] else { throw FlashVSRError.invalid("Invalid sparse attention windows.") }
    let batch=queries.shape[0], heads=queries.shape[1], blocks=queries.shape[2], tokens=queries.shape[3], dim=queries.shape[4]
    var batches=[MLXArray]()
    for b in 0..<batch {
      guard selection[b].count == heads else { throw FlashVSRError.invalid("Missing head routing.") }
      var headOutputs=[MLXArray]()
      for h in 0..<heads {
        guard selection[b][h].count == blocks else { throw FlashVSRError.invalid("Missing block routing.") }
        var blockOutputs=[MLXArray]()
        for block in 0..<blocks {
          try Task.checkCancellation()
          let indices=selection[b][h][block]
          guard !indices.isEmpty, Set(indices).count == indices.count, indices.allSatisfy({ $0 >= 0 && $0 < keys.shape[2] }) else { throw FlashVSRError.invalid("Empty or invalid attention selection.") }
          let q=queries[b,h,block].asType(.float32)
          var peak=MLXArray.full([tokens,1],values: MLXArray(-Float.infinity))
          var denominator=MLXArray.zeros([tokens,1]), numerator=MLXArray.zeros([tokens,dim])
          for start in stride(from:0,to:indices.count,by:keySlabBlocks) {
            let ids=MLXArray(indices[start..<min(start+keySlabBlocks,indices.count)].map(Int32.init))
            let k=take(keys[b,h],ids,axis:0).reshaped([-1,dim]).asType(.float32)
            let v=take(values[b,h],ids,axis:0).reshaped([-1,dim]).asType(.float32)
            let scores=matmul(q,k.transposed()) / sqrt(Float(dim))
            let nextPeak=maximum(peak,max(scores,axis:1,keepDims:true))
            let correction=exp(peak-nextPeak), probabilities=exp(scores-nextPeak)
            numerator=numerator*correction+matmul(probabilities,v)
            denominator=denominator*correction+sum(probabilities,axis:1,keepDims:true)
            peak=nextPeak
            eval(numerator,denominator,peak)
          }
          let result=(numerator/denominator).asType(queries.dtype); eval(result); blockOutputs.append(result)
        }
        headOutputs.append(stacked(blockOutputs,axis:0))
      }
      batches.append(stacked(headOutputs,axis:0))
    }
    return stacked(batches,axis:0)
  }

}
