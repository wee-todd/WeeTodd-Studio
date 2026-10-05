import Foundation
import MLX
import MLXNN

/// Local/anchor softmax plus a learned complement recurrence. This core is
/// deliberately not a selectable engine until adapter/sampler routes qualify.
enum H3VDNAttention {
  static func evaluate(input:MLXArray,raw:[MLXArray],query:MLXArray,key:MLXArray,
    value:MLXArray,layout:H3VDNLayout,weights:[String:MLXArray],
    projectSoftmax:(MLXArray) throws -> MLXArray) throws -> MLXArray {
    let count=layout.sequence
    guard input.ndim == 3,input.shape[0] == 1,input.shape[1] == count,
      input.dtype == .bfloat16,raw.count == 3,query.ndim == 4,
      query.shape[0] == 1,query.shape[2] == count,
      query.shape == key.shape,query.shape == value.shape,
      query.dtype == .bfloat16,key.dtype == .bfloat16,value.dtype == .bfloat16,
      (1...56).contains(query.shape[1]),(1...128).contains(query.shape[3]),
      raw.allSatisfy({ $0.shape == [1,count,query.shape[1],query.shape[3]] && $0.dtype == .bfloat16 }) else {
      throw H3CheckpointError.invalid("Invalid VDN hybrid-attention inputs.")
    }
    let heads=query.shape[1],dim=query.shape[3],hidden=input.shape[2],channels=heads*dim
    guard let bottleneck=weights["linear_attention.alpha.down.weight"]?.shape.first,
      let gateBottleneck=weights["linear_attention.output_gate.down.weight"]?.shape.first else {
      throw H3CheckpointError.invalid("VDN branch projections are incomplete.")
    }
    var shapes=H3VDNCheckpoint.blockShapes
    shapes["linear_attention.alpha.A_log"]=[heads]
    shapes["linear_attention.alpha.down.weight"]=[bottleneck,hidden]
    shapes["linear_attention.alpha.dt_bias"]=[channels]
    shapes["linear_attention.alpha.up.weight"]=[channels,bottleneck]
    shapes["linear_attention.beta_proj.weight"]=[heads,hidden]
    shapes["linear_attention.norm.weight"]=[dim]
    shapes["linear_attention.output_gate.down.weight"]=[gateBottleneck,hidden]
    shapes["linear_attention.output_gate.up.bias"]=[channels]
    shapes["linear_attention.output_gate.up.weight"]=[channels,gateBottleneck]
    for name in ["k","v"] {
      shapes["linear_attention.short_conv.\(name)_sp.weight"]=[channels,1,5,5]
      shapes["linear_attention.short_conv.\(name)_tm.weight"]=[channels,1,5]
    }
    shapes["softmax_gate.up.bias"]=[heads];shapes["softmax_gate.up.weight"]=[heads,hidden]
    shapes["to_out_linear.weight"]=[hidden,channels]
    guard Set(weights.keys) == Set(shapes.keys),shapes.allSatisfy({ name,shape in
      weights[name]?.shape == shape && weights[name]?.dtype.isFloatingPoint == true
    }) else { throw H3CheckpointError.invalid("VDN branch dimensions differ from the packed attention.") }
    try Task.checkCancellation()
    var attended:[MLXArray]=[]
    for group in layout.attentionGroups {
      try Task.checkCancellation()
      let keys=group.keys.map { key[0..<1,0..<heads,$0,0..<dim] }
      let values=group.keys.map { value[0..<1,0..<heads,$0,0..<dim] }
      let result=MLXFast.scaledDotProductAttention(
        queries:query[0..<1,0..<heads,group.query,0..<dim],
        keys:keys.count == 1 ? keys[0] : concatenated(keys,axis:2),
        values:values.count == 1 ? values[0] : concatenated(values,axis:2),
        scale:1/Float(dim).squareRoot(),mask:nil)
      // Finish each group before constructing the next local K/V gather.
      eval(result);attended.append(result)
    }
    let soft=concatenated(attended,axis:2).transposed(0,2,1,3)
    let gate=sigmoid(project(input[0],weights,"softmax_gate.up"))
      .reshaped([1,count,heads,1])
    let output=try projectSoftmax((soft*gate).reshaped([1,count,channels]).asType(input.dtype))
    guard output.shape == input.shape,output.dtype == input.dtype else {
      throw H3CheckpointError.invalid("VDN backbone output projection changed dimensions or precision.")
    }
    let linear=try linearBranch(input:input,raw:raw,weights:weights,layout:layout)
    var parts:[MLXArray]=[]
    if layout.videoStart > 0 { parts.append(output[0..<1,0..<layout.videoStart,0..<hidden]) }
    parts.append(output[0..<1,layout.videoStart..<layout.videoEnd,0..<hidden]
      + linear.expandedDimensions(axis:0).asType(output.dtype))
    if layout.videoEnd < count { parts.append(output[0..<1,layout.videoEnd..<count,0..<hidden]) }
    let result=concatenated(parts,axis:1)
    eval(result);try Task.checkCancellation()
    return result
  }

  private static func project(_ input:MLXArray,_ weights:[String:MLXArray],_ prefix:String) -> MLXArray {
    let weight=weights[prefix+".weight"]!
    let value=matmul(input.asType(weight.dtype),weight.T)
    if let bias=weights[prefix+".bias"] { return value+bias }
    return value
  }

  private static func features(_ raw:MLXArray,projection:String,video:Bool,
    frames:Int,layout:H3VDNLayout,weights:[String:MLXArray]) throws -> MLXArray {
    var input=raw
    if video && projection != "q" {
      let channels=raw.dim(-2)*raw.dim(-1)
      let volume=raw.reshaped([frames,layout.height,layout.width,channels])
      let spatial=conv2d(volume,
        weights["linear_attention.short_conv.\(projection)_sp.weight"]!
          .transposed(0,2,3,1).asType(raw.dtype),padding:2,groups:channels)
      input=try H3VDNMath.temporal(spatial,
        weights:weights["linear_attention.short_conv.\(projection)_tm.weight"]!)
        .reshaped(raw.shape)
    }
    let activated=silu(input)
    if projection == "v" { return activated }
    let squared=activated.asType(.float32)*activated.asType(.float32)
    let scale=rsqrt(maximum(sum(squared,axis:-1,keepDims:true),MLXArray(Float(1e-12))))
    return (activated*scale.asType(activated.dtype)).asType(activated.dtype)
  }

  private static func statistics(key:MLXArray,value:MLXArray,beta:MLXArray)
    -> (a:MLXArray,b:MLXArray) {
    let scaled=key.asType(.float32)*beta.expandedDimensions(axis:-1).asType(.float32)
    let a=matmul(scaled.swappedAxes(-1,-2),key.asType(.float32))
    let b=matmul((value*beta.expandedDimensions(axis:-1).asType(value.dtype))
      .swappedAxes(-1,-2),key).asType(.float32)
    return (0.5*(a+a.swappedAxes(-1,-2)),b)
  }

  private static func linearBranch(input:MLXArray,raw:[MLXArray],weights:[String:MLXArray],
    layout:H3VDNLayout) throws -> MLXArray {
    let rows=layout.videoEnd-layout.videoStart,hidden=input.shape[2],perFrame=layout.tokensPerFrame
    guard layout.frames > 2 else { return MLXArray.zeros([rows,hidden],dtype:input.dtype) }
    let frames=layout.frames-2,heads=raw[0].shape[2],dim=raw[0].shape[3]
    let interval=(layout.videoStart+perFrame)..<(layout.videoEnd-perFrame)
    let frameShape=[frames,perFrame,heads,dim]
    let query=try features(raw[0][0,interval],projection:"q",video:true,
      frames:frames,layout:layout,weights:weights).reshaped(frameShape)
    let key=try features(raw[1][0,interval],projection:"k",video:true,
      frames:frames,layout:layout,weights:weights).reshaped(frameShape).transposed(0,2,1,3)
    let value=try features(raw[2][0,interval],projection:"v",video:true,
      frames:frames,layout:layout,weights:weights).reshaped(frameShape).transposed(0,2,1,3)
    let xv=input[0,interval]
    let beta=sigmoid(project(xv,weights,"linear_attention.beta_proj"))
      .reshaped([frames,perFrame,heads]).transposed(0,2,1)
    // Promote before the mean AND both alpha projections. BF16 rounding here
    // cannot be recovered by promoting only the output of the reduction.
    let frameMean=mean(xv.asType(.float32).reshaped([frames,perFrame,hidden]),axis:1)
    let alpha=try H3VDNMath.retention(frameMeans:frameMean,
      down:weights["linear_attention.alpha.down.weight"]!,up:weights["linear_attention.alpha.up.weight"]!,
      bias:weights["linear_attention.alpha.dt_bias"]!,logScale:weights["linear_attention.alpha.A_log"]!,
      heads:heads,dim:dim)
    let state=try recurrence(input:input,raw:raw,weights:weights,layout:layout,
      key:key,value:value,beta:beta,alpha:alpha).asType(query.dtype)
    let readout=einsum("fhvk,fshk->fshv",state,query)
    let squared=readout.asType(.float32)*readout.asType(.float32)
    let normalized=readout*rsqrt(mean(squared,axis:-1,keepDims:true)+Float(1e-6))
      .asType(readout.dtype)*weights["linear_attention.norm.weight"]!.asType(readout.dtype)
    let downGate=project(xv,weights,"linear_attention.output_gate.down")
    let gate=sigmoid(project(downGate,weights,"linear_attention.output_gate.up")).reshaped(frameShape)
    let projected=project((normalized*gate).reshaped([frames*perFrame,heads*dim]),
      weights,"to_out_linear")
    let zeros=MLXArray.zeros([perFrame,hidden],dtype:projected.dtype)
    return concatenated([zeros,projected,zeros],axis:0)
  }

  private static func textState(input:MLXArray,raw:[MLXArray],weights:[String:MLXArray],
    layout:H3VDNLayout) throws -> MLXArray {
    let textRange=layout.textStart..<(layout.textStart+layout.textLength)
    let textKey=try features(raw[1][0,textRange],projection:"k",video:false,
      frames:0,layout:layout,weights:weights).transposed(1,0,2).expandedDimensions(axis:0)
    let textValue=try features(raw[2][0,textRange],projection:"v",video:false,
      frames:0,layout:layout,weights:weights).transposed(1,0,2).expandedDimensions(axis:0)
    let textBeta=sigmoid(project(input[0,textRange],weights,"linear_attention.beta_proj"))
      .T.expandedDimensions(axis:0)
    let (textA,textB)=statistics(key:textKey,value:textValue,beta:textBeta)
    let textInverse=try H3VDNMath.inverse(textA+MLX.eye(raw[0].dim(-1)))
    let state=0.5*matmul(textB,textInverse)[0]
    eval(state);return state
  }

  private static func recurrence(input:MLXArray,raw:[MLXArray],weights:[String:MLXArray],
    layout:H3VDNLayout,key:MLXArray,value:MLXArray,beta:MLXArray,alpha:MLXArray) throws -> MLXArray {
    let text=try textState(input:input,raw:raw,weights:weights,layout:layout)
    let scan:(prefix:MLXArray,suffix:MLXArray)
    // A/B, inverse and factor banks are releasable before complement gather.
    // Explicit scope + evaluation avoids retaining all FP32 banks to readout.
    do {
      let (a,b)=statistics(key:key,value:value,beta:beta)
      let inverse=try H3VDNMath.inverse(a+MLX.eye(raw[0].dim(-1)))
      let transitions=alpha.expandedDimensions(axis:-1)*inverse
      let injections=matmul(b,inverse)
      eval(transitions,injections)
      scan=try H3VDNMath.scan(transitions:transitions,injections:injections,start:text)
      eval(scan.prefix,scan.suffix)
    }
    let bounds=(1..<(layout.frames-1)).map { frame in
      let window=layout.window(frame);return (window.lower-1,window.upper-1)
    }
    let state=try H3VDNMath.gather(prefix:scan.prefix,suffix:scan.suffix,alpha:alpha,
      text:text,bounds:bounds)
    eval(state);return state
  }
}
