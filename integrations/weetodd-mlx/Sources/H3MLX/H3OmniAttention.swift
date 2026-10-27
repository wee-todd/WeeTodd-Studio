import MLX

/// Trained reference-aware VSA only. No dense fallback, approximation cache,
/// prefix K/V reuse, or 64-row substitution. All arrays are [1,H,rows,128].
enum H3OmniAttention {
  static func routes(scores: MLXArray, plan: H3OmniVSAPlan) throws -> MLXArray {
    let count=plan.sizes.count,sparse=count-plan.denseTiles
    guard scores.ndim == 4, scores.shape[0] == 1,
      (1...56).contains(scores.shape[1]),scores.shape[2] == count,
      scores.shape[3] == count,scores.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid Omni pooled scores.")
    }
    let heads=scores.shape[1]
    var parts:[MLXArray]=[]
    if plan.denseTiles > 0 {
      parts.append(broadcast(MLXArray((0..<plan.denseTiles).map(Int32.init),[1,1,1,plan.denseTiles]),
        to:[1,heads,sparse,plan.denseTiles]))
    }
    for region in plan.videoRegions {
      // MLX argsort is stable: equal scores retain increasing tile index.
      let selected=argSort(-scores[.ellipsis,plan.denseTiles..<count,region],axis:-1)[.ellipsis,0..<H3OmniVSAPlan.keepCount(region.count)].asType(.int32)+Int32(region.lowerBound)
      parts.append(selected)
    }
    let result=contiguous(concatenated(parts,axis:-1))
    eval(result)
    return result
  }

  static func evaluate(query: MLXArray, key: MLXArray, value: MLXArray,
    gate: MLXArray, plan: H3OmniVSAPlan) throws -> MLXArray {
    try Task.checkCancellation()
    guard Device.defaultDevice().deviceType == .gpu,
      query.ndim == 4, query.shape == key.shape, key.shape == value.shape,
      query.shape == gate.shape,query.shape[0] == 1,(1...56).contains(query.shape[1]),
      query.shape[2] == plan.rows,query.shape[3] == 128,
      [query,key,value,gate].allSatisfy({$0.dtype == .bfloat16}) else {
      throw H3CheckpointError.invalid("Omni VSA requires matching GPU BF16 [1,heads,rows,128] Q/K/V/gate.")
    }
    _ = try plan.workspaceBytes(heads:query.shape[1])
    eval([query,key,value,gate])
    // Fused QKV head/row views remain in place. Broadcast feature axes need
    // bounded materialization for Steel's contiguous vector loads.
    func admitted(_ x:MLXArray) throws -> MLXArray {
      guard x.strides.allSatisfy({$0 >= 0}),x.strides.last == 1 || x.strides.last == 0 else {
        throw H3CheckpointError.invalid("Omni VSA requires contiguous features and nonnegative strides.")
      }
      return x.strides.last == 0 ? contiguous(x) : x
    }
    let q=try admitted(query),k=try admitted(key),v=try admitted(value)
    eval([q,k,v])
    let pooled=try H3OmniMetal.summaries(query:q,key:k,value:v,plan:plan)
    let scores=matmul(pooled[0],pooled[1].swappedAxes(-1,-2)) / Float(128).squareRoot()
    let compression=matmul(softmax(scores,axis:-1),pooled[2])
    let selected=try routes(scores:scores,plan:plan)
    eval(compression)
    try Task.checkCancellation()
    let sparse=try H3OmniMetal.attend(query:q,key:k,value:v,routes:selected,plan:plan)
    let dense: MLXArray
    if plan.denseRows.isEmpty {
      dense=MLXArray.zeros([1,q.shape[1],0,128],dtype:.bfloat16)
    } else {
      dense=MLXFast.scaledDotProductAttention(queries:take(q,MLXArray(plan.denseRows),axis:2),
        keys:k,values:v,scale:1/Float(128).squareRoot(),mask:nil)
    }
    let result=try H3OmniMetal.restore(dense:dense,sparse:sparse,compression:compression,gate:gate,plan:plan)
    try Task.checkCancellation()
    return result
  }
}
