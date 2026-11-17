// Frozen current consumer at the start of the November14 attention task.
// The profiling timers are test-only; rank2 GEMM, slab8, group8 and all arithmetic
// are copied unchanged. Do not change this reference when the consumer changes.
import Foundation
import XCTest
import MLX
import MLXRandom
import InferenceContracts
@testable import FlashVSRMLX

final class FlashAttentionProfile {
  var graphSeconds=0.0, evaluationSeconds=0.0
  var groups=0, windows=0, slabSubmissions=0, matmuls=0
  var groupFill=[Int:Int]()
  var json:[String:Any] { ["graphSeconds":graphSeconds,"evaluationSeconds":evaluationSeconds,
    "groups":groups,"windows":windows,"slabSubmissions":slabSubmissions,"rank2Matmuls":matmuls,
    "groupFill":Dictionary(uniqueKeysWithValues:groupFill.map{(String($0.key),$0.value)})] }
}

enum FlashAttentionConsumerReference {
  static func attend(queries:MLXArray,keys:MLXArray,values:MLXArray,selection:[[[[Int]]]],
    keySlabBlocks:Int=8,checkCancellation:()throws->Void = { try Task.checkCancellation() },
    profile: FlashAttentionProfile? = nil)throws->MLXArray {
    guard queries.ndim == 5, keys.ndim == 5, values.shape == keys.shape,
      queries.shape.allSatisfy({$0>0}),keys.shape.allSatisfy({$0>0}),
      keys.shape[2] <= Int(Int32.max),
      queries.shape[0..<2] == keys.shape[0..<2], queries.shape[3...] == keys.shape[3...],
      keySlabBlocks > 0, selection.count == queries.shape[0] else { throw FlashVSRError.invalid("Invalid sparse attention windows.") }
    let batch=queries.shape[0], heads=queries.shape[1], blocks=queries.shape[2], tokens=queries.shape[3], dim=queries.shape[4]
    // Group submissions only for the released geometry. Keeping every QK/PV
    // rank2 preserves MLX's split-K kernel and its original reduction order.
    // At most8 windows use <64MiB scratch, reserved by FlashVSRMemoryPlan.
    let grouped=tokens == 128 && dim == 128 && keySlabBlocks == 8
      && (queries.dtype == .bfloat16 || queries.dtype == .float32)
      && queries.dtype == keys.dtype && queries.dtype == values.dtype
    struct WindowState {
      let query:MLXArray
      var peak:MLXArray,denominator:MLXArray,numerator:MLXArray
    }
    var batches=[MLXArray]()
    for b in 0..<batch {
      guard selection[b].count == heads else { throw FlashVSRError.invalid("Missing head routing.") }
      var headOutputs=[MLXArray]()
      for h in 0..<heads {
        guard selection[b][h].count == blocks else { throw FlashVSRError.invalid("Missing block routing.") }
        for indices in selection[b][h] {
          guard !indices.isEmpty, Set(indices).count == indices.count,
            indices.allSatisfy({$0 >= 0 && $0 < keys.shape[2]}) else { throw FlashVSRError.invalid("Empty or invalid attention selection.") }
        }
        let plan=try FlashAttentionFrozenSubmissionPlan(selectedCounts:selection[b][h].map(\.count),maximumWindows:grouped ? 8:1)
        var blockOutputs=Array<MLXArray?>(repeating:nil,count:blocks)
        for group in plan.groups {
          var graphStart=ProcessInfo.processInfo.systemUptime
          profile?.groups += 1; profile?.windows += group.count
          profile?.groupFill[group.count,default:0] += 1
          try checkCancellation()
          var states=group.map { block in
            WindowState(query:queries[b,h,block].asType(.float32),
              peak:MLXArray.full([tokens,1],values:MLXArray(-Float.infinity)),
              denominator:MLXArray.zeros([tokens,1]),numerator:MLXArray.zeros([tokens,dim]))
          }
          let selectedCount=selection[b][h][group[0]].count
          for start in stride(from:0,to:selectedCount,by:keySlabBlocks) {
            try checkCancellation()
            for local in states.indices {
              try checkCancellation()
              let indices=selection[b][h][group[local]]
              let ids=MLXArray(indices[start..<min(start+keySlabBlocks,indices.count)].map(Int32.init))
              let k=take(keys[b,h],ids,axis:0).reshaped([-1,dim]).asType(.float32)
              let v=take(values[b,h],ids,axis:0).reshaped([-1,dim]).asType(.float32)
              let old=states[local]
              let scores=matmul(old.query,k.transposed()) / sqrt(Float(dim))
              let nextPeak=maximum(old.peak,max(scores,axis:1,keepDims:true))
              let correction=exp(old.peak-nextPeak),probabilities=exp(scores-nextPeak)
              states[local].numerator=old.numerator*correction+matmul(probabilities,v)
              states[local].denominator=old.denominator*correction+sum(probabilities,axis:1,keepDims:true)
              states[local].peak=nextPeak
            }
            let graphEnd=ProcessInfo.processInfo.systemUptime
            profile?.graphSeconds += graphEnd-graphStart
            eval(states.flatMap{[$0.numerator,$0.denominator,$0.peak]})
            graphStart=ProcessInfo.processInfo.systemUptime
            profile?.evaluationSeconds += graphStart-graphEnd
            profile?.slabSubmissions += 1
            profile?.matmuls += 2*states.count
          }
          let results=states.map{($0.numerator/$0.denominator).asType(queries.dtype)}
          let graphEnd=ProcessInfo.processInfo.systemUptime
          profile?.graphSeconds += graphEnd-graphStart
          eval(results)
          profile?.evaluationSeconds += ProcessInfo.processInfo.systemUptime-graphEnd
          for (local,block) in group.enumerated() { blockOutputs[block]=results[local] }
        }
        try checkCancellation()
        headOutputs.append(stacked(blockOutputs.map{$0!},axis:0))
      }
      batches.append(stacked(headOutputs,axis:0))
    }
    return stacked(batches,axis:0)
  }

}

private struct FlashAttentionFrozenSubmissionPlan {
  let groups:[[Int]]
  // F32 Q/state/output, K/V plus gather copies, scores/probabilities, online
  // state and indices. Inputs and accumulated final output are separate.
  static let maximumScratchBytes=8*(3*128*128*4+4*8*128*128*4+2*128*8*128*4+8*128*4+9*4)

  init(selectedCounts:[Int],maximumWindows:Int=8)throws {
    guard !selectedCounts.isEmpty,selectedCounts.allSatisfy({$0>0}),(1...8).contains(maximumWindows) else {
      throw FlashVSRError.invalid("Invalid FlashVSR attention submission plan.")
    }
    if maximumWindows == 1 {
      groups=selectedCounts.indices.map{[$0]}
      return
    }
    var buckets=[Int:[Int]]()
    for (index,count) in selectedCounts.enumerated() { buckets[count,default:[]].append(index) }
    groups=buckets.keys.sorted().flatMap { count in
      let windows=buckets[count]!
      return stride(from:0,to:windows.count,by:maximumWindows).map {
        Array(windows[$0..<min($0+maximumWindows,windows.count)])
      }
    }
  }
}


final class FlashVSRAttentionConsumerTests:XCTestCase {
  func testFrozenConsumerRetainsCurrentProductionWords() throws {
    let shape=[1,2,9,128,128], ks=[1,2,19,128,128]
    MLXRandom.seed(724)
    let q=MLXRandom.normal(shape), k=MLXRandom.normal(ks),v=MLXRandom.normal(ks)
    let lengths=[1,8,9,17,3,8,9,17,1]
    let selection=[(0..<2).map{h in (0..<9).map{row in
      (0..<lengths[row]).map{(h+row+3*$0)%19}
    }}]
    for dtype:DType in [.float32,.bfloat16] {
      let expected=try FlashAttentionConsumerReference.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection)
      let actual=try FlashVSRAttention.attend(queries:q.asType(dtype),keys:k.asType(dtype),values:v.asType(dtype),selection:selection)
      XCTAssertEqual(actual.asType(.float32).asArray(Float.self).map(\.bitPattern),expected.asType(.float32).asArray(Float.self).map(\.bitPattern))
    }
  }

  func testOptInCurrentConsumerProfile() throws {
    guard let output=ProcessInfo.processInfo.environment["WEETODD_FLASH_CONSUMER_PROFILE"] else {
      throw XCTSkip("Opt in with an external JSON output path.")
    }
    let lease=try NativeInferenceLease.acquire();defer{lease.release()}
    Memory.memoryLimit=8*1024*1024*1024;Memory.cacheLimit=1024*1024*1024
    var records=[[String:Any]]()
    for (range,keyBlocks) in [(3,756),(11,1260)] {
      try autoreleasepool {
        MLXRandom.seed(2026)
        let q=MLXRandom.normal([1,12,504,128,128]).asType(.bfloat16)
        let k=MLXRandom.normal([1,12,keyBlocks,128,128]).asType(.bfloat16)
        let v=MLXRandom.normal(k.shape).asType(.bfloat16)
        eval(q,k,v)
        let selection=try FlashVSRAttention.route(q:q,k:k,rows:12,columns:21,localRange:range,topK:158759)
        let counts=selection.flatMap{$0}.flatMap{$0}.map(\.count)
        let profile=FlashAttentionProfile()
        Memory.peakMemory=Memory.activeMemory
        let start=ProcessInfo.processInfo.systemUptime
        let result=try FlashAttentionConsumerReference.attend(queries:q,keys:k,values:v,selection:selection,profile:profile)
        eval(result)
        let seconds=ProcessInfo.processInfo.systemUptime-start
        XCTAssertTrue(result.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        var row=profile.json
        row["range"]=range;row["keyBlocks"]=keyBlocks;row["wallSeconds"]=seconds
        row["minimumSelected"]=counts.min()!;row["maximumSelected"]=counts.max()!
        row["meanSelected"]=Double(counts.reduce(0,+))/Double(counts.count)
        row["peakMLXBytes"]=Memory.peakMemory
        records.append(row)
        try JSONSerialization.data(withJSONObject:records,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:output))
        print("FLASH_CONSUMER_PROFILE range=\(range) wall=\(seconds) graph=\(profile.graphSeconds) eval=\(profile.evaluationSeconds) groups=\(profile.groups)")
      }
      Stream.gpu.synchronize();Memory.clearCache()
    }
  }

  func testOptInCompiledConsumerBenchmark() throws {
    guard let output=ProcessInfo.processInfo.environment["WEETODD_FLASH_COMPILED_BENCHMARK"] else {
      throw XCTSkip("Opt in with an external JSON output path.")
    }
    let lease=try NativeInferenceLease.acquire();defer{lease.release()}
    Memory.memoryLimit=8*1024*1024*1024;Memory.cacheLimit=2*1024*1024*1024
    let asynchronous=ProcessInfo.processInfo.environment["WEETODD_FLASH_COMPILED_ASYNC"] == "1"
    let controlAsynchronous=ProcessInfo.processInfo.environment["WEETODD_FLASH_COMPARE_ASYNC"] == "1"
    var records=[[String:Any]]()
    for (range,keyBlocks) in [(3,756),(11,1260)] {
      try autoreleasepool {
        MLXRandom.seed(2026)
        let q=MLXRandom.normal([1,12,504,128,128]).asType(.bfloat16)
        let k=MLXRandom.normal([1,12,keyBlocks,128,128]).asType(.bfloat16)
        let v=MLXRandom.normal(k.shape).asType(.bfloat16)
        eval(q,k,v)
        let selection=try FlashVSRAttention.route(q:q,k:k,rows:12,columns:21,localRange:range,topK:158759)
        func run(_ candidate:Bool,forceLegacy:Bool=false)throws->(Double,[UInt16],Int) {
          try autoreleasepool {
            Stream.gpu.synchronize()
            Memory.peakMemory=Memory.activeMemory
            let start=ProcessInfo.processInfo.systemUptime
            let kernel=candidate ? FlashVSRCompiledAttentionSlab():nil
            let result=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selection,
              checkCancellation:{try Task.checkCancellation()},compiledSlab:kernel,
              asynchronousSlabs:!forceLegacy && (candidate ? asynchronous:controlAsynchronous))
            eval(result)
            let seconds=ProcessInfo.processInfo.systemUptime-start
            let peak=Memory.peakMemory
            return (seconds,result.view(dtype:.uint16).asArray(UInt16.self),peak)
          }
        }
        let (_,expected,_)=try run(false,forceLegacy:true)
        let (_,warm,_)=try run(true)
        let warmDifferences=zip(warm,expected).reduce(0){$0+($1.0 == $1.1 ? 0:1)}
        XCTAssertEqual(warmDifferences,0)
        guard warm.count == expected.count,warmDifferences == 0 else {throw FlashVSRError.invalid("Warm consumer parity failed.")}
        if controlAsynchronous {
          let (_,controlWarm,_)=try run(false)
          let differences=zip(controlWarm,expected).reduce(0){$0+($1.0 == $1.1 ? 0:1)}
          guard controlWarm.count == expected.count,differences == 0 else {throw FlashVSRError.invalid("Async-only consumer parity failed.")}
        }
        for pair in 0..<4 {
          for candidate in (pair%2 == 0 ? [false,true]:[true,false]) {
            let (seconds,words,peak)=try run(candidate)
            let differences=zip(words,expected).reduce(0){$0+($1.0 == $1.1 ? 0:1)}
            XCTAssertEqual(words.count,expected.count)
            XCTAssertEqual(differences,0,"range\(range), pair\(pair), candidate\(candidate)")
            guard words.count == expected.count,differences == 0 else {throw FlashVSRError.invalid("Native consumer parity failed.")}
            records.append(["range":range,"keyBlocks":keyBlocks,"pair":pair,"candidate":candidate,
              "asynchronous":candidate ? asynchronous:controlAsynchronous,"compiled":candidate,
              "seconds":seconds,"peakMLXBytes":peak,"differentBF16Words":differences,"comparedWords":words.count])
            try JSONSerialization.data(withJSONObject:records,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:output))
            print("FLASH_COMPILED_BENCH range=\(range) pair=\(pair) candidate=\(candidate) seconds=\(seconds) differentWords=\(differences)")
          }
        }
      }
      Stream.gpu.synchronize();Memory.clearCache()
    }
  }
}
