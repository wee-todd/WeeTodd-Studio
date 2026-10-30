import Foundation
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3UnfencedAttentionTests:XCTestCase {
  func testOptInNonzeroOffsetsTailsAndCancellationSensitiveBF16Bits() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_UNFENCED_PARITY"]=="1" else {
      throw XCTSkip("Enable the bounded, fresh-process unfenced attention regression.")
    }
    guard H3RegenerateAttentionPlan.unfencedRuntimeIsQualified(ProcessInfo.processInfo.operatingSystemVersion) else {
      throw XCTSkip("Qualified Metal compiler/OS required.")
    }
    try H3BoundedProbeGPUContext.run {error in
      guard GPU.deviceInfo().architecture=="applegpu_g15d" else {throw XCTSkip("Qualified g15d GPU required.")}
      func exact(_ q:MLXArray,_ k:MLXArray,_ v:MLXArray,_ range:Range<Int>) throws {
        eval(q,k,v);try error.check()
        let actual=try H3UnfencedAttention.evaluate(query:q,key:k,value:v,range:range,maximumStageBytes:256*1024*1024)
        // Independent pinned MLX entry point, including its <=8-row vector tail.
        let expected=try H3RegenerateAttention.forcedAttend(queryOwner:q,keysOwner:k,valuesOwner:v,range:range)
        eval(actual,expected);try error.check()
        XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
        let a=actual.reshaped([-1]).view(dtype:.uint16).asArray(UInt16.self)
        let b=expected.reshaped([-1]).view(dtype:.uint16).asArray(UInt16.self)
        XCTAssertTrue(a.elementsEqual(b),"Literal BF16 comparison failed for \(q.shape), \(range).")
        try error.check();try H3BoundedProbeGPUContext.checkBudget(error)
      }
      for (rows,range) in [(129,33..<98),(81,0..<65),(64,0..<64),(769,13..<558),(769,512..<769),(81,80..<81)] {
        try autoreleasepool {
          let q=MLXRandom.normal([1,56,rows,128],key:MLXRandom.key(1541)).asType(.bfloat16)
          let k=MLXRandom.normal([1,56,rows,128],key:MLXRandom.key(1542)).asType(.bfloat16)
          let v=MLXRandom.normal([1,56,rows,128],key:MLXRandom.key(1543)).asType(.bfloat16)
          try exact(q,k,v,range)
          XCTAssertThrowsError(try H3UnfencedAttention.evaluate(query:q.asType(.float32),key:k,value:v,range:range,maximumStageBytes:256*1024*1024))
          XCTAssertThrowsError(try H3UnfencedAttention.evaluate(query:q,key:k,value:v,range:range,maximumStageBytes:1),"Every query window, including vector tails, must honor the stage budget.")
        }
      }
      // Exact BF16 subnormals, signed zeros and cancellation-sensitive dots.
      let pattern:[Float]=[0,-Float.zero,Float(bitPattern:0x00010000),Float(bitPattern:0x00800000),0.125,-0.125,8,-8]
      let rows=129
      var q:[Float]=[],k:[Float]=[],v:[Float]=[]
      for r in 0..<rows {for d in 0..<128 {
        q.append(pattern[(r+d)%pattern.count]);k.append(pattern[(r*3+d*5)%pattern.count])
        v.append(Float((r+d)%9-4)*0.25)
      }}
      func owner(_ x:[Float])->MLXArray {
        contiguous(broadcast(MLXArray(x,[1,1,rows,128]).asType(.bfloat16),to:[1,56,rows,128]))
      }
      try exact(owner(q),owner(k),owner(v),33..<98)
      try error.check()
    }
  }
}
