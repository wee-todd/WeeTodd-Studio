import Foundation
import MLX

public enum FlashVSRAttention {
  /// Tiny independent FP64 oracle. selectedKeys addresses tokens, not blocks.
  public static func oracle(queries:[[Float]], keys:[[Float]], values:[[Float]], selectedKeys:[[Int]]) throws -> [[Float]] {
    guard let d=queries.first?.count, d > 0, !keys.isEmpty, keys.count == values.count,
      selectedKeys.count == queries.count, queries.allSatisfy({$0.count == d}), keys.allSatisfy({$0.count == d}),
      let outputWidth=values.first?.count, values.allSatisfy({$0.count == outputWidth}) else { throw FlashVSRError.invalid("Invalid oracle dimensions.") }
    return try queries.enumerated().map { row,q in
      let chosen=selectedKeys[row]
      guard !chosen.isEmpty, chosen.allSatisfy({keys.indices.contains($0)}) else { throw FlashVSRError.invalid("Invalid oracle selection.") }
      let scores=chosen.map { j in zip(q,keys[j]).reduce(0.0) { $0+Double($1.0)*Double($1.1) } / sqrt(Double(d)) }
      let peak=scores.max()!, exponents=scores.map { exp($0-peak) }, denominator=exponents.reduce(0,+)
      return (0..<outputWidth).map { c in Float(zip(chosen,exponents).reduce(0.0) { $0+Double(values[$1.0][c])*$1.1 }/denominator) }
    }
  }

  /// Windows [B,H,blocks,128,D]. Up to eight canonical query windows share
  /// submissions, each with at most keySlabBlocks key blocks. Other geometry
  /// stays serial. Online softmax preserves the selected mask and key order.
  public static func attend(queries:MLXArray, keys:MLXArray, values:MLXArray, selection:[[[[Int]]]], keySlabBlocks:Int = 8) throws -> MLXArray {
    try attend(queries:queries,keys:keys,values:values,selection:selection,keySlabBlocks:keySlabBlocks,
      checkCancellation:{ try Task.checkCancellation() })
  }

  static func attend(queries:MLXArray,keys:MLXArray,values:MLXArray,selection:[[[[Int]]]],
    keySlabBlocks:Int=8,checkCancellation:()throws->Void)throws->MLXArray {
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
        let plan=try FlashVSRAttentionSubmissionPlan(selectedCounts:selection[b][h].map(\.count),maximumWindows:grouped ? 8:1)
        var blockOutputs=Array<MLXArray?>(repeating:nil,count:blocks)
        for group in plan.groups {
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
            eval(states.flatMap{[$0.numerator,$0.denominator,$0.peak]})
          }
          let results=states.map{($0.numerator/$0.denominator).asType(queries.dtype)}
          eval(results)
          for (local,block) in group.enumerated() { blockOutputs[block]=results[local] }
        }
        try checkCancellation()
        headOutputs.append(stacked(blockOutputs.map{$0!},axis:0))
      }
      batches.append(stacked(headOutputs,axis:0))
    }
    return stacked(batches,axis:0)
  }

  /// Pool only the block axis into CPU routing. Two passes and a bounded top-k heap
  /// avoid constructing a block-square matrix, let alone token-square scores.
  static func route(q:MLXArray, k:MLXArray, rows:Int, columns:Int, localRange:Int, topK:Int) throws -> [[[[Int]]]] {
    let b=q.shape[0], h=q.shape[1], nq=q.shape[2], nk=k.shape[2], d=q.shape[4]
    let qp=mean(q.asType(.float32),axis:3).asArray(Float.self), kp=mean(k.asType(.float32),axis:3).asArray(Float.self)
    let candidates=try FlashVSRBlockRouting.candidates(queryBlocks:nq,keyBlocks:nk,rows:rows,columns:columns,localRange:localRange)
    func probabilities(_ batch:Int,_ head:Int,_ row:Int) -> [Float] {
      let qbase=((batch*h+head)*nq+row)*d
      let scores=candidates[row].map { key -> Float in
        let kbase=((batch*h+head)*nk+key)*d
        var score: Float=0
        for i in 0..<d { score += qp[qbase+i]*kp[kbase+i] }
        return score / sqrt(Float(d))
      }
      let peak=scores.max()!, e=scores.map { exp($0-peak) }, total=e.reduce(0,+)
      return e.map { $0/total }
    }
    var result=[[[[Int]]]]()
    for batch in 0..<b {
      var heads=[[[Int]]]()
      for head in 0..<h {
        // The released stream block size is two temporal windows, i.e. the whole four-frame group.
        var heap=ThresholdHeap(capacity:min(topK+1,nq*nk))
        var candidateCount=0
        for row in 0..<nq {
          try Task.checkCancellation()
          for p in probabilities(batch,head,row) { guard p.isFinite else { throw FlashVSRError.invalid("Nonfinite FlashVSR draft routing.") }; heap.insert(p); candidateCount += 1 }
        }
        let threshold=candidateCount < min(topK+1,nq*nk) ? -Float.infinity : heap.minimum
        var selected=[[Int]]()
        for row in 0..<nq {
          let p=probabilities(batch,head,row)
          let chosen=zip(candidates[row],p).filter { $0.1 > threshold }.map { $0.0 }
          selected.append(chosen.isEmpty ? [candidates[row][0]] : chosen)
        }
        heads.append(selected)
      }
      result.append(heads)
    }
    return result
  }

  static func partition(_ value:MLXArray, shape:[Int]) throws -> MLXArray {
    guard shape.count == 3, shape[0]%2 == 0, shape[1]%8 == 0, shape[2]%8 == 0,
      value.ndim == 4, value.shape[2] == shape.reduce(1,*) else { throw FlashVSRError.invalid("Stream window partition requires a padded 2x8x8 grid.") }
    let b=value.shape[0],h=value.shape[1],d=value.shape[3], t=shape[0]/2,y=shape[1]/8,x=shape[2]/8
    return value.reshaped([b,h,t,2,y,8,x,8,d]).transposed(0,1,2,4,6,3,5,7,8).reshaped([b,h,t*y*x,128,d])
  }
  static func unpartition(_ value:MLXArray, shape:[Int]) -> MLXArray {
    let b=value.shape[0],h=value.shape[1],d=value.shape[4],t=shape[0]/2,y=shape[1]/8,x=shape[2]/8
    return value.reshaped([b,h,t,y,x,2,8,8,d]).transposed(0,1,2,5,3,6,4,7,8).reshaped([b,h,shape.reduce(1,*),d])
  }
}

private struct ThresholdHeap {
  let capacity:Int
  var values=[Float]()
  var minimum:Float { values.first ?? -.infinity }
  mutating func insert(_ value:Float) {
    if values.count < capacity {
      values.append(value); var child=values.count-1
      while child > 0 { let parent=(child-1)/2; if values[parent] <= values[child] { break }; values.swapAt(parent,child); child=parent }
    } else if value > values[0] {
      values[0]=value; var parent=0
      while 2*parent+1 < values.count {
        var child=2*parent+1
        if child+1 < values.count && values[child+1] < values[child] { child += 1 }
        if values[parent] <= values[child] { break }; values.swapAt(parent,child); parent=child
      }
    }
  }
}
