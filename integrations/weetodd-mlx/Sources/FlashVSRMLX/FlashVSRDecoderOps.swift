import Foundation
import MLX
import TensorIO

/// Native channel-last primitives independently implemented from the released
/// FlashVSR tensor/graph contracts. Source kernels remain PyTorch OI[TH]W.
enum FlashVSRDecoderOps {
  static func load(checkpoint:URL,profile:TorchTensorImportProfile,dtype:DType,
    include:(String)->Bool)throws->[String:MLXArray] {
    guard dtype == .float32 || dtype == .bfloat16 else { throw FlashVSRError.invalid("FlashVSR decoders require FP32 or BF16 stage precision.") }
    let archive=try TorchTensorArchive.open(url:checkpoint,expectedSHA256:profile.sourceSHA256,profile:profile)
    var weights:[String:MLXArray]=[:]
    for name in archive.tensors.keys.sorted() where include(name) {
      try Task.checkCancellation()
      let descriptor=archive.tensors[name]!
      guard descriptor.dtype=="F32" else { throw FlashVSRError.invalid("Released decoder weights must be F32 source tensors.") }
      var pieces:[MLXArray]=[],start:UInt64=0
      while start<descriptor.byteCount {
        let stop=min(start+4*1024*1024,descriptor.byteCount)
        let piece=try archive.withTensorBytes(named:name,range:start..<stop) { MLXArray($0,[Int((stop-start)/4)],type:Float.self).asType(dtype) }
        eval(piece);pieces.append(piece);start=stop
      }
      guard !pieces.isEmpty else { throw FlashVSRError.invalid("Decoder contains an empty learned parameter.") }
      let value=(pieces.count==1 ? pieces[0]:concatenated(pieces,axis:0)).reshaped(descriptor.shape.map(Int.init))
      eval(value);weights[name]=value
    }
    try archive.checkUnchanged();return weights
  }
  static func weight(_ name:String,_ weights:[String:MLXArray])throws->MLXArray {
    guard let value=weights[name] else { throw FlashVSRError.invalid("Selected FlashVSR decoder is missing tensor \(name); choose its released checkpoint in model setup.") };return value
  }
  static func conv2D(_ x:MLXArray,weight:MLXArray,bias:MLXArray?=nil,padding:Int?=nil)throws->MLXArray {
    guard x.ndim==4,weight.ndim==4,x.shape[3]==weight.shape[1],
      weight.shape[2]==weight.shape[3],weight.shape[2]%2==1,
      bias==nil || bias!.shape==[weight.shape[0]] else { throw FlashVSRError.invalid("Invalid FlashVSR decoder 2D convolution geometry.") }
    let result=conv2d(x,weight.transposed(0,2,3,1),padding:IntOrPair(padding ?? weight.shape[2]/2))
    return bias.map { result+$0 } ?? result
  }
  static func conv2D(_ x:MLXArray,key:String,weights:[String:MLXArray])throws->MLXArray {
    try conv2D(x,weight:weight(key+".weight",weights),bias:weights[key+".bias"])
  }
  static func silu(_ x:MLXArray)->MLXArray { x*sigmoid(x) }
  static func rms(_ x:MLXArray,gamma:MLXArray)throws->MLXArray {
    let channels=x.shape.last!
    guard gamma.size==channels else { throw FlashVSRError.invalid("Decoder channel normalization scale differs from features.") }
    let f=x.asType(.float32),norm=maximum(sqrt(sum(f*f,axis:-1,keepDims:true)),MLXArray(Float(1e-12)))
    return (f/norm*sqrt(Float(channels))).asType(x.dtype)*gamma.reshaped([channels])
  }
  static func nearest2x(_ x:MLXArray)->MLXArray { repeated(repeated(x,count:2,axis:1),count:2,axis:2) }
  static func temporalGrow(_ x:MLXArray,weight:MLXArray)throws->MLXArray {
    let value=try conv2D(x,weight:weight),channels=x.shape[3]
    guard value.shape[3]%channels==0 else { throw FlashVSRError.invalid("Tiny temporal expansion channels do not divide features.") }
    let factor=value.shape[3]/channels
    return value.reshaped([x.shape[0],x.shape[1],x.shape[2],factor,channels]).transposed(0,3,1,2,4)
      .reshaped([x.shape[0]*factor,x.shape[1],x.shape[2],channels])
  }
  static func memoryBlock(_ input:MLXArray,previous:inout MLXArray?,weights:[String:MLXArray],prefix:String)throws->MLXArray {
    guard input.ndim==4,input.shape[0]>0 else { throw FlashVSRError.invalid("Tiny memory block needs nonempty temporal features.") }
    if let previous { guard previous.shape==[1,input.shape[1],input.shape[2],input.shape[3]] else { throw FlashVSRError.invalid("Tiny temporal memory canvas changed.") } }
    let initial=previous ?? MLXArray.zeros([1,input.shape[1],input.shape[2],input.shape[3]],dtype:input.dtype)
    let past=input.shape[0]==1 ? initial:concatenated([initial,input[0..<input.shape[0]-1]],axis:0)
    var value=concatenated([input,past],axis:3)
    value=maximum(try conv2D(value,key:prefix+".conv.0",weights:weights),0)
    value=maximum(try conv2D(value,key:prefix+".conv.2",weights:weights),0)
    value=try conv2D(value,key:prefix+".conv.4",weights:weights)
    let skip:MLXArray
    if let w=weights[prefix+".skip.weight"] { skip=try conv2D(input,weight:w,bias:weights[prefix+".skip.bias"]) } else { skip=input }
    guard skip.shape==value.shape else { throw FlashVSRError.invalid("Tiny memory residual shape differs.") }
    let output=maximum(value+skip,0)
    // take creates a compact carry instead of retaining the whole prior group.
    let next=take(input,MLXArray([Int32(input.shape[0]-1)]),axis:0)
    eval(output,next);previous=next;return output
  }
  /// Exact zero-left causal convolution with last-(kernel-1) INPUT carry.
  /// Layout is [B,T,H,W,C], source weight [O,I,Kt,Kh,Kw].
  static func causal(_ input:MLXArray,weight:MLXArray,bias:MLXArray?,cache:inout MLXArray?,spatialPadding:Int=1)throws->MLXArray {
    guard input.ndim==5,input.shape[1]>0,weight.ndim==5,input.shape[4]==weight.shape[1],
      weight.shape[2]>0,spatialPadding>=0,bias==nil || bias!.shape==[weight.shape[0]] else { throw FlashVSRError.invalid("Invalid Wan causal convolution geometry.") }
    let history=weight.shape[2]-1
    if let cache {
      guard cache.shape[0]==input.shape[0],Array(cache.shape[2...])==Array(input.shape[2...]),cache.shape[1]<=history else { throw FlashVSRError.invalid("Wan causal carry canvas changed.") }
    }
    var combined=cache.map { concatenated([$0,input],axis:1) } ?? input
    let next:MLXArray?
    if history>0 {
      let count=min(history,combined.shape[1]),indices=MLXArray(((combined.shape[1]-count)..<combined.shape[1]).map(Int32.init))
      next=take(combined,indices,axis:1)
    } else { next=nil }
    let missing=history-(cache?.shape[1] ?? 0)
    if missing>0 { combined=padded(combined,widths:[0,IntOrPair((missing,0)),0,0,0]) }
    let paddedInput=spatialPadding>0 ? padded(combined,widths:[0,0,IntOrPair(spatialPadding),IntOrPair(spatialPadding),0]):combined
    let output=conv3d(paddedInput,weight.transposed(0,2,3,4,1))
    let result=bias.map { output+$0 } ?? output
    if let next { eval(result,next) } else { eval(result) }
    cache=next;return result
  }
  static func attention(_ input:MLXArray,gamma:MLXArray,qkv:MLXArray,qkvBias:MLXArray,
    projection:MLXArray,projectionBias:MLXArray,queryChunk:Int=128)throws->MLXArray {
    guard input.ndim==5,input.shape[1]>0,queryChunk>0 else { throw FlashVSRError.invalid("Invalid Wan spatial attention canvas.") }
    let b=input.shape[0],t=input.shape[1],h=input.shape[2],w=input.shape[3],c=input.shape[4]
    let normalized=try rms(input.reshaped([b*t,h,w,c]),gamma:gamma)
    let projected=try conv2D(normalized,weight:qkv,bias:qkvBias)
    guard projected.shape[3]==c*3 else { throw FlashVSRError.invalid("Wan QKV output differs from its channel count.") }
    let tokens=projected.reshaped([b*t,h*w,3,c])
    let q=tokens[0...,0...,0,0...],k=tokens[0...,0...,1,0...],v=tokens[0...,0...,2,0...]
    var values:[MLXArray]=[]
    for start in stride(from:0,to:h*w,by:queryChunk) {
      try Task.checkCancellation()
      let stop=min(start+queryChunk,h*w)
      let scores=matmul(q[0...,start..<stop,0...].asType(.float32),k.asType(.float32).transposed(0,2,1))/sqrt(Float(c))
      let value=matmul(softmax(scores,axis:-1).asType(v.dtype),v)
      eval(value);values.append(value)
    }
    let attended=concatenated(values,axis:1).reshaped([b*t,h,w,c])
    let result=try conv2D(attended,weight:projection,bias:projectionBias).reshaped(input.shape)+input
    eval(result);return result
  }
}
