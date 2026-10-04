import Foundation
import MLX
import TensorIO
import LTX25Engine

/// Released LTX2.5 one-step pixel DiffVAE. Only the active block's parameters
/// are resident. Positive/negative text, transformer and audio remain outside
/// this decoder stage; no sampler seed or state is consumed here.
public final class MLXDiffusionVideoDecoder {
  private let checkpoint:MLXDiffusionVideoCheckpoint
  public let options:MLXDiffusionVideoOptions
  private let gate=NSLock()
  private static let allocatorGate=NSLock()
  public private(set) var residentWeightBytes=0
  public private(set) var maximumResidentWeightBytes=0
  public private(set) var stageSeconds:[String:Double]=[:]
  public init(checkpoint:URL,options:MLXDiffusionVideoOptions?=nil) throws {
    self.checkpoint=try MLXDiffusionVideoCheckpoint(checkpoint:checkpoint)
    self.options=try options ?? MLXDiffusionVideoOptions()
  }
  public func preflight(shape:[Int],dtype:DType = .float32) throws -> MLXDiffusionVideoPlan {
    guard [.float32,.bfloat16,.float16].contains(dtype),!options.isMetal || dtype == .bfloat16 else {
      throw LTXError.invalid("Diffusion VAE experimental Metal modes require BF16 latents; no implicit precision conversion is performed.")
    }
    try checkpoint.file.checkUnchanged(at:checkpoint.url)
    return try MLXDiffusionVideoPlan(shape:shape,options:options,elementBytes:dtype == .float32 ? 4:2)
  }
  /// Decodes normalized final B128FHW latents. A throwing observer/callback
  /// aborts before publication and drains/releases the current block.
  public func decode(latent:MLXArray,progress:(String,Int,Int) throws -> Void={ _,_,_ in }) throws -> MLXArray {
    let plan=try preflight(shape:latent.shape,dtype:latent.dtype)
    try Task.checkCancellation();try MLXVideoDecoder.checkFinite(latent)
    guard gate.try() else { throw LTXError.invalid("Diffusion VAE decoder is already active.") }
    defer { gate.unlock() }
    guard Self.allocatorGate.try() else { throw LTXError.invalid("Another Diffusion VAE allocator lease is active.") }
    let cache=Memory.cacheLimit;Memory.cacheLimit=128*1024*1024
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=cache;Self.allocatorGate.unlock() }
    stageSeconds=[:];maximumResidentWeightBytes=0
    let started=Date()
    var x=latent.transposed(0,2,3,4,1)
    let terminal=x[0...,(x.shape[1]-1)..<x.shape[1],0...,0...,0...]
    x=concatenated([x,terminal,terminal],axis:1)
    x=try withWeights(names:["per_channel_statistics.mean-of-means","per_channel_statistics.std-of-means","decoder.conv_in.weight","decoder.conv_in.bias"]) { weights in
      let mean=weights["per_channel_statistics.mean-of-means"]!,std=weights["per_channel_statistics.std-of-means"]!
      guard std.min().item(Float.self)>0 else { throw LTXError.invalid("Diffusion VAE normalization has nonpositive variance.") }
      return Self.linear(x*std+mean,"decoder.conv_in",weights)
    }
    for stage in 0..<3 { x=try runStage(x,index:stage,progress:progress) }
    let targetFrames=plan.outputShape[2],pixelFrames=max(targetFrames,11),stride=MLXDiffusionVideoPlan.strides[3]
    // Explicit key0, INPUT dtype and channel-first sampling are public Python
    // component behavior. Patch order must not change the random-number layout.
    let noise=MLXRandom.normal([1,3,pixelFrames,x.shape[2]*stride[1]*4,x.shape[3]*stride[2]*4],dtype:latent.dtype,key:MLXRandom.key(0))
    let patched=try MLXDiffusionVideoMath.patch(noise)
    eval(patched);try Task.checkCancellation()
    var result:MLXArray
    if options.optimization == .stage4WidthTiles && options.stage4TileWidth<x.shape[3] {
      let core=options.stage4TileWidth,width=x.shape[3],halo=plan.widthHalo
      var pieces:[MLXArray]=[]
      for start in Swift.stride(from:0,to:width,by:core) {
        try Task.checkCancellation()
        let stop=min(width,start+core),left=max(0,start-halo),right=min(width,stop+halo)
        let context=try runStage(x[0...,0...,0...,left..<right,0...],index:3,progress:progress)[0...,0..<pixelFrames,0...,0...,0...]
        let decoded=try diffusion(context:context,noise:patched[0...,0...,0...,left*2..<right*2,0...],deferred:false,progress:progress)
        let piece=decoded[0...,0...,0...,(start-left)*2..<(stop-left)*2,0...]
        eval(piece);pieces.append(piece)
      }
      result=concatenated(pieces,axis:3)
    } else if options.optimization == .deferredStage4 {
      let context=try runBlocks(x,index:3,progress:progress)
      result=try diffusion(context:context,noise:patched,deferred:true,progress:progress)
    } else {
      let context=try runStage(x,index:3,progress:progress)[0...,0..<pixelFrames,0...,0...,0...]
      result=try diffusion(context:context,noise:patched,deferred:false,progress:progress)
    }
    result=try MLXDiffusionVideoMath.unpatch(result)[0...,0...,0..<targetFrames,0...,0...].asType(latent.dtype)
    eval(result);try Task.checkCancellation();try MLXVideoDecoder.checkFinite(result)
    guard result.shape == plan.outputShape else { throw LTXError.invalid("Diffusion VAE output disagrees with frozen frame/canvas admission.") }
    try checkpoint.file.checkUnchanged(at:checkpoint.url)
    stageSeconds["total_decode"]=Date().timeIntervalSince(started)
    try progress("complete",24,24)
    return result
  }
  public func decodeRGB8(latent:MLXArray,progress:(String,Int,Int) throws -> Void={ _,_,_ in },receive:(Int,Data) throws -> Void) throws {
    let value=try decode(latent:latent,progress:progress)
    for frame in 0..<value.shape[2] {
      try Task.checkCancellation();try checkpoint.file.checkUnchanged(at:checkpoint.url)
      // Preserve trained output dtype through clipping/add/multiply; only the
      // final cast is UInt8. No extra tanh, rounding or forced FP32 boundary.
      let image=((clip(value[0,0...,frame,0...,0...],min:-1,max:1)+1)*Float(127.5)).asType(.uint8).transposed(1,2,0)
      try receive(frame,Data(image.asArray(UInt8.self)))
    }
    try checkpoint.file.checkUnchanged(at:checkpoint.url)
  }
  private func withWeights(names:[String],body:([String:MLXArray]) throws -> MLXArray) throws -> MLXArray {
    try Task.checkCancellation();try checkpoint.file.checkUnchanged(at:checkpoint.url)
    let readStart=Date()
    return try autoreleasepool {
      var weights:[String:MLXArray]=[:],bytes=0
      defer { weights.removeAll();residentWeightBytes=0 }
      for name in names.sorted() {
        guard let descriptor=checkpoint.file.tensors[name] else { throw LTXError.invalid("Diffusion VAE weight disappeared: \(name)") }
        let next=bytes.addingReportingOverflow(Int(descriptor.byteCount))
        guard !next.overflow,next.partialValue<=options.maximumWeightBytes else { throw LTXError.invalid("Diffusion VAE active block exceeds its weight allowance.") }
        bytes=next.partialValue;try Task.checkCancellation()
        weights[name]=try MLXWeight.read(checkpoint.file,name,access:.buffered)
        residentWeightBytes=bytes;maximumResidentWeightBytes=max(maximumResidentWeightBytes,bytes)
      }
      eval(Array(weights.values));try checkpoint.file.checkUnchanged(at:checkpoint.url);try Task.checkCancellation()
      for value in weights.values { try MLXVideoDecoder.checkFinite(value) }
      // Fold optional historical static gates FP32→BF16 BEFORE projection.
      for name in names where name.hasSuffix(".gate_msa") || name.hasSuffix(".gate_mlp") || name.hasSuffix(".gate_ctx") {
        let split=name.split(separator:"."),gate=String(split.last!),stem=split.dropLast().joined(separator:".")
        let target=gate == "gate_msa" ? "attn.proj":(gate == "gate_mlp" ? "mlp.w_down":"context_proj")
        for suffix in ["weight","bias"] {
          let key=stem+"."+target+"."+suffix
          if let value=weights[key] {
            let scale=weights[name]!.asType(.float32)
            weights[key]=(value.asType(.float32)*(value.ndim == 2 ? scale.expandedDimensions(axis:1):scale)).asType(value.dtype)
          }
        }
        weights.removeValue(forKey:name)
      }
      eval(Array(weights.values));stageSeconds["weight_read_evaluate",default:0]+=Date().timeIntervalSince(readStart)
      let compute=Date(),result=try body(weights)
      eval(result);try Task.checkCancellation();try checkpoint.file.checkUnchanged(at:checkpoint.url)
      try MLXVideoDecoder.checkFinite(result)
      stageSeconds["block_evaluate",default:0]+=Date().timeIntervalSince(compute)
      return result
    }
  }
  private func names(_ prefixes:[String]) -> [String] {
    checkpoint.file.tensors.keys.filter { name in prefixes.contains { name.hasPrefix($0+".") } }.sorted()
  }
  private static func linear(_ x:MLXArray,_ name:String,_ weights:[String:MLXArray]) -> MLXArray {
    let weight=weights[name+".weight"]!
    if let bias=weights[name+".bias"] { return addMM(bias,x,weight.T) }
    return matmul(x,weight.T)
  }
  private func attention(_ x:MLXArray,prefix:String,kernel:[Int],weights:[String:MLXArray]) throws -> MLXArray {
    let s=x.shape,dim=s[4],heads=dim/64,shape=[s[0],s[1],s[2],s[3],heads,64]
    if !options.isMetal {
      let qkv=Self.linear(x,prefix+".qkv",weights).reshaped([s[0],s[1],s[2],s[3],3,heads,64])
      let q=try MLXDiffusionVideoMath.rotary(MLXDiffusionVideoMath.rms(qkv[0...,0...,0...,0...,0,0...,0...],weight:weights[prefix+".q_norm.weight"]!))
      let k=try MLXDiffusionVideoMath.rotary(MLXDiffusionVideoMath.rms(qkv[0...,0...,0...,0...,1,0...,0...],weight:weights[prefix+".k_norm.weight"]!))
      let v=qkv[0...,0...,0...,0...,2,0...,0...]
      let output=try MLXDiffusionVideoMath.referenceAttention(q:q,k:k,v:v,kernel:kernel,queryChunk:options.queryChunkSize)
      return Self.linear(output.reshaped(s),prefix+".proj",weights)
    }
    let rows=shape[0]*shape[1]*shape[2]*shape[3],flat=x.reshaped([rows,dim])
    func projection(_ part:MLXArray,_ index:Int) -> MLXArray {
      let range=index*dim..<(index+1)*dim
      return addMM(weights[prefix+".qkv.bias"]![range],part,weights[prefix+".qkv.weight"]![range,0...].T)
    }
    let k=try MLXDiffusionVideoMetal.normRotary(projection(flat,1).reshaped([rows,heads,64]),weight:weights[prefix+".k_norm.weight"]!,shape:shape,queryStart:0).reshaped(shape)
    let v=projection(flat,2).reshaped(shape);eval(k,v)
    let chunk=options.optimization == .metalNA3DQueryTiled ? options.queryChunkSize:rows
    var output:[MLXArray]=[]
    for start in stride(from:0,to:rows,by:chunk) {
      try Task.checkCancellation();let stop=min(rows,start+chunk)
      let q=try MLXDiffusionVideoMetal.normRotary(projection(flat[start..<stop],0).reshaped([stop-start,heads,64]),weight:weights[prefix+".q_norm.weight"]!,shape:shape,queryStart:start)
      let attended=try MLXDiffusionVideoMetal.attend(q:q,k:k,v:v,shape:shape,kernel:kernel,queryStart:start)
      let result=Self.linear(attended.reshaped([stop-start,dim]),prefix+".proj",weights)
      eval(result);output.append(result)
    }
    return concatenated(output,axis:0).reshaped(s)
  }
  private func feedForward(_ x:MLXArray,prefix:String,weights:[String:MLXArray]) throws -> MLXArray {
    let shape=x.shape,flat=x.reshaped([-1,shape.last!]);var pieces:[MLXArray]=[]
    for start in stride(from:0,to:flat.shape[0],by:options.tokenChunkSize) {
      try Task.checkCancellation();let part=flat[start..<min(flat.shape[0],start+options.tokenChunkSize)]
      let gate=MLXDiffusionVideoMath.silu(Self.linear(part,prefix+".w_gate",weights)),up=Self.linear(part,prefix+".w_up",weights)
      let value=Self.linear(gate*up,prefix+".w_down",weights);eval(value);pieces.append(value)
    }
    return concatenated(pieces,axis:0).reshaped(shape)
  }
  private func runBlocks(_ input:MLXArray,index:Int,progress:(String,Int,Int) throws -> Void) throws -> MLXArray {
    var x=input
    for block in 0..<MLXDiffusionVideoPlan.depths[index] {
      let p="decoder.det_stages.\(index).\(block)"
      x=try withWeights(names:names([p])) { w in
        let first=MLXDiffusionVideoMath.rms(x,weight:w[p+".norm1.weight"]!)
        let residual=x+(try attention(first,prefix:p+".attn",kernel:MLXDiffusionVideoPlan.kernels[index],weights:w))
        let normalized=MLXDiffusionVideoMath.rms(residual,weight:w[p+".norm2.weight"]!)
        return residual+(try feedForward(normalized,prefix:p+".mlp",weights:w))
      }
      try progress("context_stage_\(index+1)",block+1,MLXDiffusionVideoPlan.depths[index])
    }
    return x
  }
  private func runStage(_ input:MLXArray,index:Int,progress:(String,Int,Int) throws -> Void) throws -> MLXArray {
    let x=try runBlocks(input,index:index,progress:progress),p="decoder.upsamples.\(index).proj"
    return try withWeights(names:names([p])) { w in
      try MLXDiffusionVideoMath.upsample(Self.linear(x,p,w),stride:MLXDiffusionVideoPlan.strides[index],outputChannels:MLXDiffusionVideoPlan.channels[index+1])
    }
  }
  private func diffusion(context:MLXArray,noise:MLXArray,deferred:Bool,progress:(String,Int,Int) throws -> Void) throws -> MLXArray {
    var x=try withWeights(names:names(["decoder.conv_in_x_t"])) { Self.linear(noise,"decoder.conv_in_x_t",$0) }
    let modulation=try withWeights(names:names(["decoder.t_embedder.mlp.0","decoder.t_embedder.mlp.2","decoder.shared_adaln.proj"])) { w in
      let frequency=exp(-Float(log(10000.0))*MLXArray(0..<128).asType(.float32)/Float(128))
      let angle=frequency*Float(1000)
      let embedding=concatenated([cos(angle),sin(angle)],axis:0).asType(x.dtype).reshaped([1,256])
      let time=Self.linear(MLXDiffusionVideoMath.silu(Self.linear(embedding,"decoder.t_embedder.mlp.0",w)),"decoder.t_embedder.mlp.2",w)
      return Self.linear(MLXDiffusionVideoMath.silu(time),"decoder.shared_adaln.proj",w).reshaped([1,7,256])
    }
    for block in 0..<8 {
      let p="decoder.diff_blocks.\(block)",up="decoder.upsamples.3.proj"
      x=try withWeights(names:names(deferred ? [p,up]:[p])) { w in
        let values=(modulation+w[p+".scale_shift_table"]!).reshaped([1,7,1,1,1,256])
        var combined:MLXArray
        if deferred {
          let width=context.shape[3],chunk=max(1,(width+options.contextWidthChunks-1)/options.contextWidthChunks)
          var pieces:[MLXArray]=[]
          for start in stride(from:0,to:width,by:chunk) {
            try Task.checkCancellation();let stop=min(width,start+chunk)
            let ctx=try MLXDiffusionVideoMath.upsample(Self.linear(context[0...,0...,0...,start..<stop,0...],up,w),stride:[2,2,2],outputChannels:256)
            let part=x[0...,0..<min(x.shape[1],ctx.shape[1]),0...,start*2..<stop*2,0...]
            let value=part+Self.linear(ctx[0...,0..<part.shape[1],0...,0...,0...],p+".context_proj",w)
            eval(value);pieces.append(value)
          }
          combined=concatenated(pieces,axis:3)
        } else { combined=x+Self.linear(context,p+".context_proj",w) }
        let normalized=MLXDiffusionVideoMath.rms(combined,weight:w[p+".norm1.weight"]!)*(1+values[0...,0,0...,0...,0...,0...])+values[0...,1,0...,0...,0...,0...]
        combined=combined+(try attention(normalized,prefix:p+".attn",kernel:[11,11,11],weights:w))
        let second=MLXDiffusionVideoMath.rms(combined,weight:w[p+".norm2.weight"]!)*(1+values[0...,3,0...,0...,0...,0...])+values[0...,4,0...,0...,0...,0...]
        return combined+(try feedForward(second,prefix:p+".mlp",weights:w))
      }
      try progress("pixel_diffusion",block+1,8)
    }
    let prediction=try withWeights(names:["decoder.norm_out.weight","decoder.conv_out.weight","decoder.conv_out.bias"]) { w in
      Self.linear(MLXDiffusionVideoMath.rms(x,weight:w["decoder.norm_out.weight"]!),"decoder.conv_out",w)
    }
    let result=MLXDiffusionVideoMath.oneStep(noise:noise,prediction:prediction);eval(result)
    return result
  }
}
