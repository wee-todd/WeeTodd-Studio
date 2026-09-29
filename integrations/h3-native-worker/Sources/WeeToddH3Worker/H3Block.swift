// Independently expressed from WeeTodd's H3 block equations, not DT model source.
import Foundation
import NNC
enum RotaryLayout:String {
  case legacy,flat,complex
  case complexRounded="complex-rounded"
  var isComplex:Bool { self == .complex || self == .complexRounded }
}
struct WeightBinding { let model: Model; let key: String; let shape: [Int]; let adapter: Bool; let fp32: Bool; var rowRange:Range<Int>? = nil }
final class H3Block {
  let model: Model
  let outputNames: [String]
  var bindings: [WeightBinding] = []
  let attentionPermutation: Parameter<Int32>?
  let prefix: String
  let blockIndex: Int
  let half: Bool
  let complexRotary: Bool
  let prescaleValues: Bool
  let ffnScale:Float
  let runtimeInputScale:Parameter<Float16>?
  let runtimeOutputScale:Parameter<Float>?
  let adapterAlphas:[String:Float]
  init(rows: Int, diagnostic: Bool, adapter: SafeTensorReader, half: Bool = false, rotary:RotaryLayout = .complexRounded, blockIndex:Int = 24, ffnScale:Float = 2, refiner:Bool = false,modulationSpans:[Range<Int>]? = nil,reusable:Bool=false) throws {
    guard (0..<50).contains(blockIndex),ffnScale.isFinite,ffnScale >= 1 else { throw ProbeError.invalid("Invalid block configuration") }
    self.blockIndex=blockIndex
    self.ffnScale=ffnScale
    let prescaleValues=half && ProcessInfo.processInfo.environment["WEETODD_NNC_VALUE_SCALING"]=="projection"
    self.prescaleValues=prescaleValues
    let qkvSchedule=ProcessInfo.processInfo.environment["WEETODD_NNC_QKV_SCHEDULE"] ?? "parallel"
    guard ["parallel","serial"].contains(qkvSchedule) else { throw ProbeError.invalid("Unknown QKV schedule") }
    let inputScaled=ProcessInfo.processInfo.environment["WEETODD_NNC_PROJECTIONS"]=="input-scaled"
    guard !inputScaled || (half && !refiner && !prescaleValues) else { throw ProbeError.invalid("Input-scaled projections require FP16 without weight prescaling") }
    guard !reusable || (inputScaled && half && !refiner && !diagnostic && !prescaleValues) else { throw ProbeError.invalid("Reusable block requires qualified input scaling") }
    let runtimeInputScale = reusable ? Parameter<Float16>(.GPU(0),format:.NHWC,shape:[1],name:"runtime_ffn_input_scale") : nil
    let runtimeOutputScale = reusable ? Parameter<Float>(.GPU(0),format:.NHWC,shape:[1],name:"runtime_ffn_output_scale") : nil
    self.runtimeInputScale=runtimeInputScale;self.runtimeOutputScale=runtimeOutputScale
    let prefix=refiner ? "token_refiner.blocks.\(blockIndex)." : "blocks.\(blockIndex)."
    self.prefix=prefix
    guard !(half && refiner) else { throw ProbeError.invalid("FP16 token refinement is not qualified") }
    self.half=half
    let complexRotary=rotary.isComplex && !refiner
    self.complexRotary=complexRotary
    let computeType:DataType = half ? .Float16 : .BFloat16
    let x=Input(), indices=Input(), cosine=Input(), sine=Input()
    let mods=(0..<6).map { _ in Input() }
    var bindings: [WeightBinding]=[]
    var qkvProjectionStarts:[Int:[Model.IO]]=[:]
    var projectionOutputs:[(String,Model.IO)]=[]
    var adapterAlphas:[String:Float]=[:]
    func norm(_ name:String,_ input:ModelIOConvertible,_ axis:Int,_ width:Int) -> Model.IO {
      let layer=RMSNorm(epsilon:1e-5,axis:[axis],elementwiseAffine:false,name:name)
      let weight:Model, weightValue:Model.IO
      if half {
        let parameter=Parameter<Float16>(.GPU(0),format:.NHWC,shape:TensorShape(Array(repeating:1,count:axis)+[width]),name:name+"_scale")
        weight=parameter;weightValue=parameter.io
      } else {
        let parameter=Parameter<BFloat16>(.GPU(0),format:.NHWC,shape:TensorShape(Array(repeating:1,count:axis)+[width]),name:name+"_scale")
        weight=parameter;weightValue=parameter.io
      }
      bindings.append(WeightBinding(model:weight,key:name+".weight",shape:[width],adapter:false,fp32:false))
      // MLX rounds normalized values to BF16 before applying the learned BF16 scale.
      return layer(input.to(.Float32)).to(computeType) .* weightValue
    }
    func projection(_ name:String,_ input:ModelIOConvertible,_ inSize:Int,_ outSize:Int,rowRange:Range<Int>? = nil) throws -> Model.IO {
      let localSize=rowRange?.count ?? outSize
      let graphName=name+(rowRange.map{"_rows_\($0.lowerBound)"} ?? "")
      let base=Dense(count:localSize,noBias:true,name:graphName)
      bindings.append(WeightBinding(model:base,key:name+".weight",shape:[outSize,inSize],adapter:false,fp32:false,rowRange:rowRange))
      let prefix="diffusion_model."+prefix+name
      guard let a=adapter.records[prefix+".lora_A.weight"],let b=adapter.records[prefix+".lora_B.weight"],a.shape.count==2,b.shape==[outSize,a.shape[0]],a.shape[1]==inSize,a.dtype=="BF16",b.dtype=="BF16" else { throw ProbeError.invalid("Unexpected adapter shape: \(name)") }
      let alpha=try adapter.read(prefix+".alpha")
      guard alpha.values.count==1,alpha.values[0].isFinite,a.shape[0]>0 else { throw ProbeError.invalid("Invalid LoRA scaling") }
      let scale=alpha.values[0]/Float(a.shape[0])
      adapterAlphas[name]=alpha.values[0]
      let down=Dense(count:a.shape[0],noBias:true,name:graphName+"_down")
      let up=Dense(count:localSize,noBias:true,name:graphName+"_up")
      bindings.append(WeightBinding(model:down,key:prefix+".lora_A.weight",shape:a.shape,adapter:true,fp32:false))
      bindings.append(WeightBinding(model:up,key:prefix+".lora_B.weight",shape:b.shape,adapter:true,fp32:false,rowRange:rowRange))
      let baseValue=base(input), downValue=down(input)
      if name=="attn.qkv_proj",let range=rowRange { qkvProjectionStarts[range.lowerBound]=[baseValue,downValue] }
      let upValue=up(downValue), delta=scale * upValue
      if name=="attn.qkv_proj" && rowRange==nil { projectionOutputs += [("qkv_base",baseValue),("qkv_down",downValue),("qkv_up",upValue),("qkv_delta",delta)] }
      return baseValue + delta
    }
    func gather(_ index:Int,after dependency:Model.IO) -> Model.IO {
      let value=Functional.indexSelect(input:mods[index],index:indices)
      // Independent timestep gathers otherwise execute across all 50 blocks up front,
      // retaining >100GB at the requested resolution. Bind each to its consumer stage.
      value.add(dependencies:[dependency])
      return value
    }
    func sliceRows(_ value:Model.IO,_ span:Range<Int>) -> Model.IO {
      value.reshaped([span.count,5376],offset:[span.lowerBound,0],strides:[5376,1]).contiguous()
    }
    func modulationRow(_ index:Int,_ span:Range<Int>,after dependency:Model.IO) -> Model.IO {
      let rowIndex=indices.reshaped([1],offset:[span.lowerBound],strides:[1]).contiguous()
      let row=Functional.indexSelect(input:mods[index],index:rowIndex)
      row.add(dependencies:[dependency]);return row
    }
    func modulate(_ value:Model.IO,scale:Int,shift:Int) -> Model.IO {
      guard let spans=modulationSpans else { return value .* (1+gather(scale,after:value))+gather(shift,after:value) }
      return Concat(axis:0)(spans.map { span in
        sliceRows(value,span) .* (1+modulationRow(scale,span,after:value))+modulationRow(shift,span,after:value)
      })
    }
    func gate(_ value:Model.IO,index:Int) -> Model.IO {
      guard let spans=modulationSpans else { return half ? gather(index,after:value).to(.Float32) .* value : gather(index,after:value) .* value }
      return Concat(axis:0)(spans.map { span in
        let row=modulationRow(index,span,after:value)
        return (half ? row.to(.Float32) : row) .* sliceRows(value,span)
      })
    }
    func rope(_ q:Model.IO) -> Model.IO {
      if complexRotary {
        if rotary == .complexRounded {
          return Functional.cmul(left:q,right:cosine) + Functional.cmul(left:q,right:sine)
        }
        return Functional.cmul(left:q,right:cosine)
      }
      if rotary == .flat {
        // The ccv strided-copy fast path is rank two. Flatten independent heads
        // before slicing/concatenating; preserve every arithmetic operation.
        let count=rows*56
        let rot=q.reshaped([count,96],offset:[0,0],strides:[128,1]).contiguous().reshaped([rows,56,96])
        let a=q.reshaped([count,48],offset:[0,0],strides:[128,1]).contiguous()
        let b=q.reshaped([count,48],offset:[0,48],strides:[128,1]).contiguous()
        let pass=q.reshaped([count,32],offset:[0,96],strides:[128,1])
        let rotated=Functional.concat(axis:1,-1 * b,a).reshaped([rows,56,96])
        let transformed=(rot .* cosine + rotated .* sine).reshaped([count,96])
        return Functional.concat(axis:1,transformed,pass).reshaped([rows,56,128])
      }
      let rot=q.reshaped([rows,56,96],offset:[0,0,0],strides:[56*128,128,1])
      let a=q.reshaped([rows,56,48],offset:[0,0,0],strides:[56*128,128,1])
      let b=q.reshaped([rows,56,48],offset:[0,0,48],strides:[56*128,128,1])
      let pass=q.reshaped([rows,56,32],offset:[0,0,96],strides:[56*128,128,1])
      let rotated=Functional.concat(axis:2,-1 * b,a)
      return Functional.concat(axis:2,rot .* cosine + rotated .* sine,pass)
    }
    func mark(_ name:String,_ value:Model.IO) -> Model.IO {
      if ProcessInfo.processInfo.environment["WEETODD_NNC_RANGE_BLOCK"]==String(blockIndex) { return RangeTrace.inspect(value,name:"block\(blockIndex)-"+name) }
      guard ProcessInfo.processInfo.environment["WEETODD_NNC_STAGES"]=="1" else { return value }
      return value.debug(name:name) { _,stream in
        stream?.joined()
        StageTimer.mark(name)
      }
    }
    let n1=mark("n1",norm("norm1",x,1,5376))
    let h1=mark("h1",refiner ? n1 : modulate(n1,scale:1,shift:0))
    let projectedQKV:Model.IO
    var pieces:[Model.IO]=[]
    if inputScaled {
      let q=try projection("attn.qkv_proj",h1,5376,21504,rowRange:0..<7168)
      let k=try projection("attn.qkv_proj",h1,5376,21504,rowRange:7168..<14336)
      let v=try projection("attn.qkv_proj",0.125 * h1,5376,21504,rowRange:14336..<21504)
      pieces=[q,k,v]
      projectedQKV=diagnostic ? Concat(axis:1)(pieces) : q
    } else { projectedQKV=try projection("attn.qkv_proj",h1,5376,21504) }
    let qkv=inputScaled && !diagnostic ? projectedQKV : mark("qkv",projectedQKV)
    if !inputScaled { pieces=qkv.chunked(3,axis:1) }
    let q=mark("q",norm("attn.q_norm",pieces[0].contiguous().reshaped([rows,56,128]),2,128))
    let k=mark("k",norm("attn.k_norm",pieces[1].contiguous().reshaped([rows,56,128]),2,128))
    let v=pieces[2].contiguous().reshaped([1,rows,56,128])
    let qr=mark("qr",refiner ? q : rope(q)),kr=mark("kr",refiner ? k : rope(k))
    if qkvSchedule=="serial",inputScaled {
      // Q and K normalization/rotary temporaries need not overlap. Sequence the
      // producer branches, leaving the final Q/K/V values live for attention.
      for start in qkvProjectionStarts[7168] ?? [] { start.add(dependencies:[qr]) }
      for start in qkvProjectionStarts[14336] ?? [] { start.add(dependencies:[kr]) }
    }
    let restoreAttention=complexRotary && ProcessInfo.processInfo.environment["WEETODD_NNC_ATTENTION_ORDER"]=="original"
    let permutation=restoreAttention ? Parameter<Int32>(.GPU(0),format:.NHWC,shape:[128],name:"attention_channel_order") : nil
    self.attentionPermutation=permutation
    func attentionInput(_ input:Model.IO) -> Model.IO {
      if let permutation=permutation {
        let matrix=input.reshaped([rows*56,128]).transposed(0,1).contiguous()
        return Functional.indexSelect(input:matrix,index:permutation.io).transposed(0,1).contiguous().reshaped([1,rows,56,128])
      }
      return input.reshaped([1,rows,56,128])
    }
    let preScaleQuery=ProcessInfo.processInfo.environment["WEETODD_NNC_ATTENTION_SCALE"] != "kernel"
    let attention=mark("attention",nativeH3Attention(attentionInput(qr),attentionInput(kr),half && !prescaleValues && !inputScaled ? (0.125 * v) : v,half:half,preScaleQuery:preScaleQuery).reshaped([rows,7168]))
    let out=mark("out",try projection("attn.out_proj",attention,7168,5376))
    let x1=mark("x1",refiner ? x + out : x + gate(half ? (8 * out.to(.Float32)) : out,index:2))
    let n2=norm("norm2",x1,1,5376)
    let h2=mark("h2",refiner ? n2 : modulate(n2,scale:4,shift:3))
    let projectedFC1:Model.IO
    var halves:[Model.IO]=[]
    if inputScaled {
      let gate=try projection("mlp.fc1",h2,5376,28672,rowRange:0..<14336)
      let valueInput=runtimeInputScale.map { h2 .* $0.io } ?? ((1 / ffnScale) * h2)
      let value=try projection("mlp.fc1",valueInput,5376,28672,rowRange:14336..<28672)
      halves=[gate,value]
      projectedFC1=diagnostic ? Concat(axis:1)(halves) : gate
    } else { projectedFC1=try projection("mlp.fc1",h2,5376,28672) }
    let fc1=inputScaled && !diagnostic ? projectedFC1 : mark("fc1",projectedFC1)
    if !inputScaled { halves=fc1.chunked(2,axis:1) }
    let matchActivation = !half && ProcessInfo.processInfo.environment["WEETODD_NNC_SWISH"]=="mlx-bf16"
    let swish=matchActivation ? bf16ReferenceSwish(halves[0]) : (half && !diagnostic ? halves[0] : halves[0].swish())
    let activation=mark("activation",half ? Functional.swishMul(value:prescaleValues || inputScaled ? halves[1] : (1 / ffnScale) * halves[1],gate:halves[0]) : swish .* halves[1])
    let fc2=mark("fc2",try projection("mlp.fc2",activation,14336,5376))
    let restored=runtimeOutputScale.map { fc2.to(.Float32) .* $0.io } ?? (half ? (ffnScale * fc2.to(.Float32)) : fc2)
    let output=mark("output",refiner ? x1 + fc2 : x1 + gate(restored,index:5))
    let names=["norm1","h1","qkv","q_norm","k_norm","q_rot","k_rot","attention","attention_projection","attention_residual","h2","fc1","swish","activation","fc2","output"]
    let outputs:[Model.IO]=[n1,h1,qkv,q,k,qr,kr,attention,out,x1,h2,fc1,swish,activation,fc2,output]
    let selected=diagnostic ? outputs+projectionOutputs.map{$0.1} : [output]
    model=Model(refiner ? [x] : [x,indices]+mods+(complexRotary && rotary != .complexRounded ? [cosine] : [cosine,sine]),half ? selected.map{$0.to(.Float32)} : selected)
    model.testing=true
    outputNames=diagnostic ? names+projectionOutputs.map{$0.0} : ["output"]
    self.bindings=bindings
    self.adapterAlphas=adapterAlphas
  }
  func sourceKey(_ binding:WeightBinding,index:Int) -> String {
    if binding.adapter {
      return binding.key.replacingOccurrences(of:"diffusion_model."+prefix,with:"diffusion_model.blocks.\(index).")
    }
    return "model.diffusion_model.blocks.\(index)."+binding.key
  }
  func validateReusableSource(index:Int,checkpoint:SafeTensorReader,adapter:SafeTensorReader) throws {
    guard (0..<50).contains(index),runtimeInputScale != nil else { throw ProbeError.invalid("Invalid reusable block index") }
    for binding in bindings {
      let reader=binding.adapter ? adapter:checkpoint
      guard reader.records[sourceKey(binding,index:index)]?.shape==binding.shape else { throw ProbeError.invalid("Incompatible reusable block shape at \(index)") }
    }
    for (name,alpha) in adapterAlphas {
      let candidate=try adapter.read("diffusion_model.blocks.\(index)."+name+".alpha")
      guard candidate.values==[alpha] else { throw ProbeError.invalid("Incompatible reusable adapter alpha at \(index)") }
    }
  }
  func load(checkpoint:SafeTensorReader,adapter:SafeTensorReader,graph:DynamicGraph? = nil,owner:Model? = nil,sourceIndex:Int? = nil) throws {
    if let index=sourceIndex {
      try validateReusableSource(index:index,checkpoint:checkpoint,adapter:adapter)
      let factor=layerFFNScale(index)
      runtimeInputScale!.weight.copy(from:Tensor<Float16>([Float16(1/factor)],kind:.CPU,format:.NHWC,shape:[1]))
      runtimeOutputScale!.weight.copy(from:Tensor<Float>([factor],kind:.CPU,format:.NHWC,shape:[1]))
    }
    if ProcessInfo.processInfo.environment["WEETODD_NNC_WEIGHT_STORAGE"]=="i8x",!half { throw ProbeError.invalid("Compact weight qualification requires FP16 compute") }
    if let parameter=attentionPermutation {
      let even:[Int32]=(0..<48).map{Int32(2*$0)}
      let odd:[Int32]=(0..<48).map{Int32(2*$0+1)}
      let tail:[Int32]=(96..<128).map{Int32($0)}
      let order=even+odd+tail
      parameter.weight.copy(from:Tensor<Int32>(order,kind:.CPU,format:.NHWC,shape:[128]))
    }
    try decodeBindings(checkpoint:checkpoint,adapter:adapter,sourceIndex:sourceIndex) { binding,key,tensor in
      if binding.fp32 {
        binding.model.weight.copy(from:Tensor<Float32>(tensor.values,kind:.CPU,format:.NHWC,shape:TensorShape(tensor.shape)))
      } else if half {
        // Reproduce existing BF16 checkpoint decoding, then change compute precision.
        let values=halfWeightValues(tensor,key:key,ffnScale:ffnScale,prescale:prescaleValues)
        guard values.allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("FP16 weight overflow") }
        let storage=Tensor<Float16>(values,kind:.CPU,format:.NHWC,shape:TensorShape(tensor.shape))
        if ProcessInfo.processInfo.environment["WEETODD_NNC_WEIGHT_STORAGE"]=="i8x",!binding.adapter,tensor.shape.count==2 {
          guard graph != nil else { throw ProbeError.invalid("Compact load requires graph ownership") }
          throw ProbeError.invalid("Requantized H3 weights are not supported")
        } else { binding.model.weight.copy(from:storage) }
      } else { binding.model.weight.copy(from:makeBF16(tensor)) }
    }
  }
  private func decodeBindings(checkpoint:SafeTensorReader,adapter:SafeTensorReader,sourceIndex:Int?,consume:(WeightBinding,String,FloatTensor) throws -> Void) throws {
    var slicedSources:[String:FloatTensor]=[:]
    var remainingSlices:[String:Int]=[:]
    for binding in bindings where binding.rowRange != nil { remainingSlices[binding.key,default:0]+=1 }
    for binding in bindings {
      let reader=binding.adapter ? adapter:checkpoint
      let key=sourceIndex.map { sourceKey(binding,index:$0) } ?? (binding.adapter ? binding.key : "model.diffusion_model."+prefix+binding.key)
      guard reader.records[key]?.shape==binding.shape else { throw ProbeError.invalid("Unexpected base weight shape: \(key)") }
      var tensor:FloatTensor
      if let cached=slicedSources[binding.key] { tensor=cached }
      else {
        tensor=try reader.read(key)
        if complexRotary { tensor=try permuteRotaryWeights(tensor,key:key) }
        if binding.rowRange != nil { slicedSources[binding.key]=tensor }
      }
      if let range=binding.rowRange {
        tensor=try sliceProjectionRows(tensor,range:range)
        remainingSlices[binding.key,default:0]-=1
        if remainingSlices[binding.key]==0 { slicedSources.removeValue(forKey:binding.key) }
      }
      guard tensor.values.allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("Nonfinite weight: \(key)") }
      try consume(binding,key,tensor)
    }
  }
  func prepareReusable(index:Int,checkpoint:SafeTensorReader,adapter:SafeTensorReader) throws -> PreparedH3Weights {
    try validateReusableSource(index:index,checkpoint:checkpoint,adapter:adapter)
    guard half,!prescaleValues,attentionPermutation==nil,bindings.allSatisfy({!$0.fp32}) else { throw ProbeError.invalid("Unsupported prefetched weight policy") }
    var tensors:[PreparedH3Tensor]=[]
    try decodeBindings(checkpoint:checkpoint,adapter:adapter,sourceIndex:index) { _,key,tensor in
      let values=halfWeightValues(tensor,key:key,ffnScale:ffnScale,prescale:false)
      guard values.allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("Prefetched weight overflow") }
      tensors.append(PreparedH3Tensor(shape:tensor.shape,values:values))
    }
    return PreparedH3Weights(index:index,tensors:tensors)
  }
  func installPrepared(_ prepared:PreparedH3Weights,checkpoint:SafeTensorReader,adapter:SafeTensorReader) throws {
    try checkpoint.checkIdentity();try adapter.checkIdentity()
    guard runtimeInputScale != nil,prepared.tensors.count==bindings.count else { throw ProbeError.invalid("Invalid prepared weight layout") }
    let factor=layerFFNScale(prepared.index)
    runtimeInputScale!.weight.copy(from:Tensor<Float16>([Float16(1/factor)],kind:.CPU,format:.NHWC,shape:[1]))
    runtimeOutputScale!.weight.copy(from:Tensor<Float>([factor],kind:.CPU,format:.NHWC,shape:[1]))
    for (binding,tensor) in zip(bindings,prepared.tensors) {
      let shape=binding.rowRange.map { [$0.count,binding.shape[1]] } ?? binding.shape
      guard tensor.shape==shape else { throw ProbeError.invalid("Prepared weight shape changed") }
      binding.model.weight.copy(from:hostTensor(tensor.values,shape:TensorShape(tensor.shape)))
    }
  }

}
func makeBF16(_ source:FloatTensor) -> Tensor<BFloat16> {
  var t=Tensor<BFloat16>(.CPU,format:.NHWC,shape:TensorShape(source.shape))
  t.withUnsafeMutableBytes { data in
    let ptr=data.bindMemory(to:UInt16.self)
    for i in source.values.indices { ptr[i]=BFloat16(source.values[i]).bitPattern }
  }
  return t
}

// Pair the first and second rotary halves before projection; V remains unchanged.
// Both Q and K use the same permutation, preserving the mathematical dot product.
func permuteRotaryWeights(_ source:FloatTensor,key:String) throws -> FloatTensor {
  let channels=(0..<48).flatMap{[$0,$0+48]}+Array(96..<128)
  let qkv=key.hasSuffix("attn.qkv_proj.weight") || key.hasSuffix("attn.qkv_proj.lora_B.weight")
  let scale=key.hasSuffix("attn.q_norm.weight") || key.hasSuffix("attn.k_norm.weight")
  guard qkv || scale else { return source }
  guard (qkv && source.shape.count==2 && source.shape[0]==21504 && source.shape[1]>0) || (scale && source.shape==[128]) else { throw ProbeError.invalid("Invalid rotary weight shape") }
  var result=source
  let columns=scale ? 1 : source.shape[1]
  let heads=scale ? 1 : 112
  for head in 0..<heads {
    for (destination,origin) in channels.enumerated() {
      let dest=(head*128+destination)*columns, src=(head*128+origin)*columns
      result.values.replaceSubrange(dest..<(dest+columns),with:source.values[src..<(src+columns)])
    }
  }
  return result
}

// Scaling the value rows of a fused projection is algebraically equivalent to
// scaling that branch's input, while retaining one fused matrix multiplication.
// Scale both the base and LoRA B rows; leave Q, K, gates and LoRA A unchanged.
func halfWeightValues(_ source:FloatTensor,key:String,ffnScale:Float,prescale:Bool) -> [Float16] {
 let qkv=key.hasSuffix("attn.qkv_proj.weight") || key.hasSuffix("attn.qkv_proj.lora_B.weight")
 let fc1=key.hasSuffix("mlp.fc1.weight") || key.hasSuffix("mlp.fc1.lora_B.weight")
 let scaled=prescale && source.shape.count==2 && (qkv || fc1)
 let boundary=scaled ? 14336*source.shape[1] : Int.max
 let scale:Float=qkv ? 8 : ffnScale
 return source.values.enumerated().map { i,value in
  let rounded=Float(bitPattern:UInt32(BFloat16(value).bitPattern)<<16)
  return Float16(i>=boundary ? rounded/scale : rounded)
 }
}

func sliceProjectionRows(_ tensor:FloatTensor,range:Range<Int>) throws -> FloatTensor {
 guard tensor.shape.count==2,range.lowerBound>=0,range.upperBound<=tensor.shape[0],!range.isEmpty else { throw ProbeError.invalid("Invalid projection row slice") }
 let columns=tensor.shape[1]
 return FloatTensor(shape:[range.count,columns],values:Array(tensor.values[(range.lowerBound*columns)..<(range.upperBound*columns)]))
}
