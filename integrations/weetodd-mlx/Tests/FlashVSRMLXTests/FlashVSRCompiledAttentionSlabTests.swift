import Foundation
import XCTest
import MLX
import MLXRandom
@testable import FlashVSRMLX

final class FlashVSRCompiledAttentionSlabTests: XCTestCase {
  func testOnlyQualifiedCanonicalDeviceSelectsTheCompiledConsumer() {
    for dtype:DType in [.float32,.bfloat16] {
      let q=MLXArray.zeros([1,2,9,128,128]).asType(dtype)
      let k=MLXArray.zeros([1,2,19,128,128]).asType(dtype)
      XCTAssertTrue(FlashVSRCompiledAttentionSlab.supports(queries:q,keys:k,values:k,keySlabBlocks:8,architecture:"applegpu_g15d"))
      for architecture in [nil,"applegpu_g16d","other"] {
        XCTAssertFalse(FlashVSRCompiledAttentionSlab.supports(queries:q,keys:k,values:k,keySlabBlocks:8,architecture:architecture))
      }
      for slab in [1,4,9] {
        XCTAssertFalse(FlashVSRCompiledAttentionSlab.supports(queries:q,keys:k,values:k,keySlabBlocks:slab,architecture:"applegpu_g15d"))
      }
      XCTAssertFalse(FlashVSRCompiledAttentionSlab.supports(queries:q,keys:k,values:k.asType(.float16),keySlabBlocks:8,architecture:"applegpu_g15d"))
    }
    let small=MLXArray.ones([1,1,2,2,2])
    XCTAssertFalse(FlashVSRCompiledAttentionSlab.supports(queries:small,keys:small,values:small,keySlabBlocks:8,architecture:"applegpu_g15d"))
    let half=MLXArray.zeros([1,1,2,128,128]).asType(.float16)
    XCTAssertFalse(FlashVSRCompiledAttentionSlab.supports(queries:half,keys:half,values:half,keySlabBlocks:8,architecture:"applegpu_g15d"))
  }

  // Fusing a multiply/add, changing a reduction, a partial slab, key order or
  // a cast must fail at the first different accumulator word, before BF16.
  func testEveryAccumulatorWordMatchesOriginalAcrossPartialSlabs() throws {
    MLXRandom.seed(7304)
    let q=MLXRandom.normal([128,128]), k=MLXRandom.normal([19,128,128]), v=MLXRandom.normal(k.shape)
    for dtype:DType in [.float32,.bfloat16] {
      let query=q.asType(dtype).asType(.float32),key=k.asType(dtype),value=v.asType(dtype)
      let kernel=FlashVSRCompiledAttentionSlab()
      let longOrder=[17,2,12,0,15,7,3,9,1,18,11,4,8,14,5,16,13]
      for order in (1...8).map({Array(longOrder.prefix($0))})+[longOrder] {
        var p=MLXArray.full([128,1],values:MLXArray(-Float.infinity))
        var d=MLXArray.zeros([128,1]),n=MLXArray.zeros([128,128])
        var cp=p,cd=d,cn=n
        for start in stride(from:0,to:order.count,by:8) {
          let ids=MLXArray(order[start..<min(start+8,order.count)].map(Int32.init))
          let expected=original(query:query,key:key,value:value,indices:ids,peak:p,denominator:d,numerator:n)
          let actual=try kernel.evaluate(query:query,key:key,value:value,indices:ids,peak:cp,denominator:cd,numerator:cn)
          XCTAssertEqual(actual.count,3)
          for i in 0..<3 { assertWords(actual[i],expected[i],"\(dtype), keys\(order.count), start\(start), state\(i)") }
          n=expected[0];d=expected[1];p=expected[2]
          cn=actual[0];cd=actual[1];cp=actual[2]
        }
        assertWords((cn/cd).asType(dtype).asType(.float32),(n/d).asType(dtype).asType(.float32),"final \(dtype)")
      }
    }
  }

  func testZeroTiesExtremeAndNonfiniteValuesKeepOriginalWords() throws {
    let kernel=FlashVSRCompiledAttentionSlab(),ids=MLXArray([Int32(2),0,1])
    for value in [Float(0),-Float.zero,1e-30,1e20,Float.infinity,-Float.infinity,Float.nan] {
      let q=MLXArray.full([128,128],values:MLXArray(value))
      let k=MLXArray.full([3,128,128],values:MLXArray(Float(0.25)))
      let v=MLXArray.full(k.shape,values:MLXArray(value))
      let p=MLXArray.full([128,1],values:MLXArray(-Float.infinity))
      let d=MLXArray.zeros([128,1]),n=MLXArray.zeros([128,128])
      let expected=original(query:q,key:k,value:v,indices:ids,peak:p,denominator:d,numerator:n)
      let actual=try kernel.evaluate(query:q,key:k,value:v,indices:ids,peak:p,denominator:d,numerator:n)
      for i in 0..<3 {assertWords(actual[i],expected[i],"\(value), state\(i)")}
    }
  }

  func testInvalidSlabCannotBeCompiled() {
    let kernel=FlashVSRCompiledAttentionSlab()
    let q=MLXArray.ones([128,128]),k=MLXArray.ones([3,128,128])
    let p=MLXArray.zeros([128,1]),n=q
    for ids in [MLXArray([Int32]()),MLXArray(Array(repeating:Int32(0),count:9))] {
      XCTAssertThrowsError(try kernel.evaluate(query:q,key:k,value:k,indices:ids,peak:p,denominator:p,numerator:n))
    }
  }

  func testCompiledConsumerKeepsRaggedBatchHeadAndWindowIdentity() throws {
    MLXRandom.seed(7305)
    let q=MLXRandom.normal([2,2,9,128,128]),k=MLXRandom.normal([2,2,19,128,128]),v=MLXRandom.normal(k.shape)
    let lengths=[1,8,9,17,2,3,5,6,7]
    let selection=(0..<2).map{b in (0..<2).map{h in (0..<9).map{row in
      (0..<lengths[row]).map{(b+2*h+row+3*$0)%19}
    }}}
    for dtype:DType in [.float32,.bfloat16] {
      let expected=try FlashAttentionConsumerReference.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection)
      let kernel=FlashVSRCompiledAttentionSlab()
      let actual=try FlashVSRAttention.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection,
        checkCancellation:{try Task.checkCancellation()},compiledSlab:kernel)
      assertWords(actual.asType(.float32),expected.asType(.float32),"complete consumer \(dtype)")
    }
  }

  func testBoundedAsyncConsumerKeepsAllWordsWithAndWithoutCompilation() throws {
    MLXRandom.seed(7306)
    let q=MLXRandom.normal([1,2,9,128,128]),k=MLXRandom.normal([1,2,19,128,128]),v=MLXRandom.normal(k.shape)
    let lengths=[1,8,9,17,2,3,5,6,7]
    let selection=[(0..<2).map{h in (0..<9).map{row in
      (0..<lengths[row]).map{(h+row+3*$0)%19}
    }}]
    for dtype:DType in [.float32,.bfloat16] {
      let expected=try FlashAttentionConsumerReference.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection)
      for compiled in [false,true] {
        let kernel=compiled ? FlashVSRCompiledAttentionSlab():nil
        let actual=try FlashVSRAttention.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection,
          checkCancellation:{try Task.checkCancellation()},compiledSlab:kernel,asynchronousSlabs:true)
        assertWords(actual.asType(.float32),expected.asType(.float32),"async \(dtype), compiled\(compiled)")
      }
    }
  }

  func testCancelledAsyncConsumerDrainsWorkAndReleasesScratch() throws {
    try MLX.withError { box in
      for windows in [1,8] {
        let q=MLXArray.ones([1,1,windows,128,128]).asType(.bfloat16)
        let k=MLXArray.ones([1,1,17,128,128]).asType(.bfloat16),v=k*0.7
        eval(q,k,v);Stream.gpu.synchronize();Memory.clearCache();try box.check()
        try withExtendedLifetime((q,k,v)) {
          let baseline=Memory.activeMemory
          let cancellationPoint=windows == 1 ? 5:13
          var calls=0,cancelled=false
          do {
            try autoreleasepool {
              let kernel=FlashVSRCompiledAttentionSlab()
              _=try FlashVSRAttention.attend(queries:q,keys:k,values:v,
                selection:[[Array(repeating:Array(0..<17),count:windows)]],checkCancellation:{
                  calls += 1
                  if calls == cancellationPoint {throw CancellationError()}
                },compiledSlab:kernel,asynchronousSlabs:true)
            }
          } catch is CancellationError {cancelled=true}
          XCTAssertTrue(cancelled)
          XCTAssertEqual(calls,cancellationPoint)
          Stream.gpu.synchronize();Memory.clearCache();try box.check()
          XCTAssertEqual(Memory.activeMemory,baseline,"Queued kernels and compiled templates must release on cancellation.")
        }
      }
    }
  }

  func testSuccessfulAsyncConsumerStaysWithinReserveAndReleasesCompiledState() throws {
    try MLX.withError {box in
      Stream.gpu.synchronize();Memory.clearCache();try box.check()
      let baseline=Memory.activeMemory
      try autoreleasepool {
        let q=MLXArray.ones([1,1,8,128,128]).asType(.bfloat16)
        let k=MLXArray.ones([1,1,65,128,128]).asType(.bfloat16),v=k*0.7
        eval(q,k,v)
        let sourceActive=Memory.activeMemory
        Memory.peakMemory=sourceActive
        let kernel=FlashVSRCompiledAttentionSlab()
        let result=try FlashVSRAttention.attend(queries:q,keys:k,values:v,
          selection:[[Array(repeating:Array(0..<65),count:8)]],checkCancellation:{try Task.checkCancellation()},
          compiledSlab:kernel,asynchronousSlabs:true)
        eval(result);try box.check()
        XCTAssertLessThan(Memory.peakMemory-sourceActive,64*1024*1024,"Two queued slabs must fit the existing admission reserve.")
      }
      Stream.gpu.synchronize();Memory.clearCache();try box.check()
      XCTAssertEqual(Memory.activeMemory,baseline,"Templates and activations must release after success.")
    }
  }

  private func original(query:MLXArray,key:MLXArray,value:MLXArray,indices:MLXArray,
    peak:MLXArray,denominator:MLXArray,numerator:MLXArray)->[MLXArray] {
    let k=take(key,indices,axis:0).reshaped([-1,128]).asType(.float32)
    let v=take(value,indices,axis:0).reshaped([-1,128]).asType(.float32)
    let scores=matmul(query,k.transposed())/sqrt(Float(128))
    let nextPeak=maximum(peak,max(scores,axis:1,keepDims:true))
    let correction=exp(peak-nextPeak),probabilities=exp(scores-nextPeak)
    return [numerator*correction+matmul(probabilities,v),denominator*correction+sum(probabilities,axis:1,keepDims:true),nextPeak]
  }
  private func assertWords(_ actual:MLXArray,_ expected:MLXArray,_ message:String,file:StaticString=#filePath,line:UInt=#line) {
    let a=actual.asArray(Float.self).map(\.bitPattern),b=expected.asArray(Float.self).map(\.bitPattern)
    XCTAssertEqual(actual.shape,expected.shape,message,file:file,line:line)
    XCTAssertEqual(a.count,b.count,message,file:file,line:line)
    let differences=zip(a,b).enumerated().filter{$0.element.0 != $0.element.1}
    XCTAssertEqual(differences.count,0,"\(message); first words: \(Array(differences.prefix(3)))",file:file,line:line)
  }

  func testAutomaticConsumerPreservesStridedFloat32Inputs() throws {
    MLXRandom.seed(7307)
    let q=MLXRandom.normal([1,2,9,128,128]).transposed(0,1,2,4,3)
    let k=MLXRandom.normal([1,2,19,128,128]).transposed(0,1,2,4,3)
    let v=MLXRandom.normal(k.shape).transposed(0,1,2,4,3)
    let selection=[(0..<2).map{h in (0..<9).map{row in
      (0..<(row%2 == 0 ? 3:17)).map{(h+row+3*$0)%19}
    }}]
    let expected=try FlashAttentionConsumerReference.attend(queries:q,keys:k,values:v,selection:selection)
    let actual=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selection)
    assertWords(actual,expected,"public strided consumer")
  }
}
