import Foundation
import MLX

/// Dense attention with complete keys. Only the query/head execution windows
/// change; there is no sparse routing, spatial tiling or altered softmax domain.
enum H3RegenerateAttention {
  /// Batch independent heads while preserving complete keys and the existing
  /// 256-row query window. Larger grids retain the admitted 14-head fallback.
  static func preferredHeadGroupSize(rows:Int,heads:Int,headWidth:Int,queryChunk:Int=256) throws -> Int {
    guard (1...H3RegenerateAdmission.maximumPackedRows).contains(rows),
      (1...56).contains(heads),(1...128).contains(headWidth),(1...256).contains(queryChunk) else {
      throw H3CheckpointError.invalid("Invalid regenerate attention grouping geometry.")
    }
    let q=Int64(min(queryChunk,rows)),k=Int64(rows),width=Int64(headWidth),limit=Int64(Int32.max)
    func admitted(_ group:Int) -> Bool {
      let h=Int64(group)
      return q*h*width < limit && k*h*width < limit && q*h*k < limit
    }
    if admitted(heads) { return heads }
    let fallback=min(14,heads)
    guard admitted(fallback) else {
      throw H3CheckpointError.invalid("Regenerate attention groups exceed signed-32 indexing admission.")
    }
    return fallback
  }

  static func attend(query:MLXArray,keys:MLXArray,values:MLXArray,queryChunk:Int=256) throws -> MLXArray {
    guard query.ndim == 4,keys.ndim == 4,values.shape == keys.shape,
      query.shape[0] == 1,keys.shape[0] == 1,query.shape[1] == keys.shape[1],
      query.shape[3] == keys.shape[3],query.shape.allSatisfy({ $0 > 0 }),keys.shape.allSatisfy({ $0 > 0 }),
      query.dtype == keys.dtype,values.dtype == keys.dtype,query.dtype.isFloatingPoint,
      (1...8192).contains(queryChunk),query.size < Int(Int32.max),keys.size < Int(Int32.max),
      min(queryChunk,query.shape[2])*query.shape[1]*keys.shape[2] < Int(Int32.max) else {
      throw H3CheckpointError.invalid("Regenerate attention window exceeds shape or indexing admission.")
    }
    var parts:[MLXArray]=[]
    for start in stride(from:0,to:query.shape[2],by:queryChunk) {
      try Task.checkCancellation()
      let range=start..<min(start+queryChunk,query.shape[2])
      let result=MLXFast.scaledDotProductAttention(queries:query[0..<1,0..<query.shape[1],range,0..<query.shape[3]],
        keys:keys,values:values,scale:1/Float(query.shape[3]).squareRoot(),mask:nil)
      eval(result);parts.append(result)
    }
    return concatenated(parts,axis:2)
  }

  static func projected(first:MLXArray,angles:H3RotaryAngles,qNorm:MLXArray,kNorm:MLXArray,
    heads:Int,headWidth:Int,rotaryWidth:Int,maximumStageBytes:UInt64,
    tokenChunk:Int=8192,queryChunk:Int?=nil,headsPerGroup:Int?=nil,
    executionPlan:H3RegenerateAttentionPlan?=nil,
    project:(MLXArray,String,Int,Int,Bool) throws -> MLXArray) throws -> MLXArray {
    guard first.ndim == 3,first.shape[0] == 1,heads > 0,headWidth > 0,
      (1...8192).contains(tokenChunk),
      first.shape[1] == angles.rows,
      qNorm.shape == [headWidth],kNorm.shape == [headWidth] else {
      throw H3CheckpointError.invalid("Invalid regenerate bounded QKV geometry.")
    }
    let plan=try executionPlan ?? H3RegenerateAttentionPlan.runtime(rows:first.shape[1],heads:heads,headWidth:headWidth,dtype:first.dtype)
    guard plan.rows == first.shape[1],plan.heads == heads,plan.headWidth == headWidth,plan.outputProjectionChunk == 256,
      executionPlan == nil || (queryChunk == nil && headsPerGroup == nil) else {
      throw H3CheckpointError.invalid("Regenerate attention plan differs from the executing block.")
    }
    // Existing explicit test/diagnostic windows retain the conservative guard.
    let forceFused=plan.forceFused && queryChunk == nil && headsPerGroup == nil
    let queryChunk=queryChunk ?? plan.queryChunk
    let headsPerGroup=headsPerGroup ?? plan.headsPerGroup
    guard (1...(forceFused ? 8192:256)).contains(queryChunk),headsPerGroup > 0,headsPerGroup <= heads else {
      throw H3CheckpointError.invalid("Invalid regenerate bounded head group.")
    }
    let count=first.shape[1],hidden=first.shape[2],groups=(heads+headsPerGroup-1)/headsPerGroup
    var qs=Array(repeating:[MLXArray](),count:groups),ks=qs,vs=qs
    func checkBudget() throws {
      try Task.checkCancellation()
      guard UInt64(Memory.activeMemory) <= maximumStageBytes else {
        throw H3CheckpointError.invalid("Regenerate attention exceeded its explicit stage memory budget.")
      }
    }
    for start in stride(from:0,to:count,by:tokenChunk) {
      let range=start..<min(start+tokenChunk,count),n=range.count
      let qkv=try project(first[0..<1,range,0..<hidden],"attn.qkv_proj",3*heads*headWidth,hidden,true)
        .reshaped([1,n,heads,3,headWidth])
      let cosine=angles.cosine[0..<1,0..<1,range,0..<rotaryWidth]
      let sine=angles.sine[0..<1,0..<1,range,0..<rotaryWidth]
      let q=H3Rotary.apply(MLXFast.rmsNorm(qkv[.ellipsis,0,0..<headWidth],weight:qNorm,eps:1e-5)
        .transposed(0,2,1,3),cosine:cosine,sine:sine,rotaryWidth:rotaryWidth)
      let k=H3Rotary.apply(MLXFast.rmsNorm(qkv[.ellipsis,1,0..<headWidth],weight:kNorm,eps:1e-5)
        .transposed(0,2,1,3),cosine:cosine,sine:sine,rotaryWidth:rotaryWidth)
      let v=qkv[.ellipsis,2,0..<headWidth].transposed(0,2,1,3)
      for group in 0..<groups {
        let rangeHeads=(group*headsPerGroup)..<min((group+1)*headsPerGroup,heads)
        let qPart=contiguous(q[0..<1,rangeHeads,0..<n,0..<headWidth])
        let kPart=contiguous(k[0..<1,rangeHeads,0..<n,0..<headWidth])
        let vPart=contiguous(v[0..<1,rangeHeads,0..<n,0..<headWidth])
        eval(qPart,kPart,vPart)
        qs[group].append(qPart);ks[group].append(kPart);vs[group].append(vPart)
      }
      try checkBudget()
    }
    var query:[MLXArray]=[],key:[MLXArray]=[],value:[MLXArray]=[]
    for group in 0..<groups {
      let q=concatenated(qs[group],axis:2),k=concatenated(ks[group],axis:2),v=concatenated(vs[group],axis:2)
      eval(q,k,v);query.append(q);key.append(k);value.append(v)
      qs[group].removeAll();ks[group].removeAll();vs[group].removeAll()
      try checkBudget()
    }
    var projected:[MLXArray]=[]
    for start in stride(from:0,to:count,by:queryChunk) {
      let range=start..<min(start+queryChunk,count)
      var attendedGroups:[MLXArray]=[]
      for group in 0..<groups {
        let headCount=query[group].shape[1]
        let attended:MLXArray
        if forceFused {
          attended=try forcedAttend(queryOwner:query[group],keysOwner:key[group],valuesOwner:value[group],range:range)
        } else {
          attended=try attend(query:query[group][0..<1,0..<headCount,range,0..<headWidth],
            keys:key[group],values:value[group],queryChunk:queryChunk)
        }
        attendedGroups.append(attended.transposed(0,2,1,3).reshaped([1,range.count,headCount*headWidth]))
      }
      let attended=concatenated(attendedGroups,axis:2)
      // Preserve the qualified 256-row GEMM shape. Enlarging the attention
      // window must not change the output projection's BF16 accumulation path.
      for local in stride(from:0,to:range.count,by:plan.outputProjectionChunk) {
        let rows=local..<min(local+plan.outputProjectionChunk,range.count)
        let output=try project(attended[0..<1,rows,0..<(heads*headWidth)],"attn.out_proj",hidden,heads*headWidth,false)
        eval(output);projected.append(output);try checkBudget()
      }
    }
    let result=concatenated(projected,axis:1);eval(result)
    try checkBudget();return result
  }

  /// Only complete owned Q/K/V arrays enter this route. A small query slice's
  /// logical size cannot stand in for the bounds of its strided owner storage.
  static func forcedAttend(queryOwner:MLXArray,keysOwner:MLXArray,valuesOwner:MLXArray,
    range:Range<Int>) throws -> MLXArray {
    try Task.checkCancellation()
    guard queryOwner.ndim == 4,queryOwner.shape[0] == 1,queryOwner.shape[1] == 56,
      queryOwner.shape[3] == 128,keysOwner.shape == queryOwner.shape,valuesOwner.shape == queryOwner.shape,
      range.lowerBound >= 0,range.upperBound <= queryOwner.shape[2],(1...8192).contains(range.count),
      [queryOwner,keysOwner,valuesOwner].allSatisfy({ $0.dtype == .bfloat16 && $0.size < Int(Int32.max) }),
      Device.defaultDevice().deviceType == .gpu,GPU.deviceInfo().architecture == "applegpu_g15d" else {
      throw H3CheckpointError.invalid("Forced regenerate attention requires the qualified Metal BF16 D128 owner geometry.")
    }
    eval(queryOwner,keysOwner,valuesOwner)
    let maximumBufferBytes=GPU.deviceInfo().maxBufferSize
    for owner in [queryOwner,keysOwner,valuesOwner] {
      let strides=owner.strides,rows=owner.shape[2]
      guard rows > 0,owner.nbytes <= maximumBufferBytes,strides.count == 4,
        strides.allSatisfy({ $0 > 0 }),strides[3] == 1,strides[2] == 128,strides[1] == rows*128,
        55*Int64(strides[1])+Int64(rows-1)*Int64(strides[2])+127*Int64(strides[3])+1 == Int64(owner.size) else {
        throw H3CheckpointError.invalid("Forced regenerate attention owner storage is not canonical or exceeds real buffer bounds.")
      }
    }
    let query=queryOwner[0..<1,0..<56,range,0..<128]
    // The full-kernel indexing audit does not replace vector-tail admission.
    if range.count <= 8 { return try attend(query:query,keys:keysOwner,values:valuesOwner) }
    let result=try MLX.withError { error in
      let output=MLXFast.scaledDotProductAttention(queries:query,keys:keysOwner,values:valuesOwner,
        scale:1/Float(128).squareRoot(),mask:nil,forceFused:true)
      try error.check()
      eval(output)
      try error.check()
      return output
    }
    try Task.checkCancellation();return result
  }
}
