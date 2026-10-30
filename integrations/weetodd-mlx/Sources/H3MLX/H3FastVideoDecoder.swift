import Foundation
import MLX

public enum H3FastDecoderPrecision:Sendable {
  case float32, bfloat16
  var dtype:DType { self == .float32 ? .float32 : .bfloat16 }
}

/// Native execution of the pinned, noncausal 24-layer fast H3 decoder.
/// Input is canonical normalized [1,T,H,W,24]; output is ImageNet-normalized
/// [1,F,16H,16W,3]. Source audio and cadence belong to the surrounding media job.
public final class H3FastVideoDecoder {
  public let layout:H3FastDecoderLayout,precision:H3FastDecoderPrecision
  public let maximumContextTokens:Int
  public let maximumActiveBytes:UInt64
  public init(layout:H3FastDecoderLayout,precision:H3FastDecoderPrecision = .bfloat16,maximumContextTokens:Int=1_048_576,maximumActiveBytes:UInt64=64*1024*1024*1024) throws {
    guard (1...4_194_304).contains(maximumContextTokens) else { throw H3CheckpointError.invalid("Fast H3 context token limit exceeds supported admission.") }
    guard (UInt64(1024*1024*1024)...UInt64(256*1024*1024*1024)).contains(maximumActiveBytes) else { throw H3CheckpointError.invalid("Invalid fast H3 active-memory allowance.") }
    self.layout=layout;self.precision=precision;self.maximumContextTokens=maximumContextTokens;self.maximumActiveBytes=maximumActiveBytes
  }
  /// Each complete hidden/Q/K/V state is individually indexable by MLX's
  /// signed 32-bit kernels. QKV, FFN expansion and output head are chunked.
  public static func validateGeometry(frames:Int,height:Int,width:Int,outputFrames:Int,feedForwardChunkTokens:Int=1024) throws {
    guard (1...10_000).contains(frames),(1...512).contains(height),(1...512).contains(width),
      (1...frames).contains(outputFrames),(1...4096).contains(feedForwardChunkTokens),
      frames*height*width <= Int(Int32.max)/2048,
      outputFrames*height*width <= Int(Int32.max)/3072 else {
      throw H3CheckpointError.invalid("Fast H3 complete hidden state or selected pixel output exceeds signed 32-bit indexing; select a smaller output interval or supported resolution without dropping noncausal context.")
    }
  }
  /// Conservative simultaneous-state allowance: full hidden/Q/K/V, attention
  /// gather/reorder copies, residual/FFN slabs, head reconstruction and imported
  /// weights. This is admission, not a claim of measured peak residency.
  public static func estimatedActiveBytes(frames:Int,height:Int,width:Int,outputFrames:Int) throws -> UInt64 {
    try validateGeometry(frames:frames,height:height,width:width,outputFrames:outputFrames)
    let tokens=UInt64(frames*height*width),outputs=UInt64(outputFrames*height*width)
    return max(tokens*2048*48+2*1024*1024*1024,outputs*3072*4*4+tokens*2048*4+1024*1024*1024)
  }
  /// Project bounded token slabs, then join individual Q/K/V states. The
  /// combined N x 6144 tensor must never be formed for full-resolution clips.
  static func projectQKV(_ input:MLXArray,weights:[String:MLXArray],heads:Int=32,headDimension:Int=64,chunkTokens:Int=1024) throws -> (MLXArray,MLXArray,MLXArray) {
    guard input.ndim == 3,input.shape[0] == 1,input.shape[2] == heads*headDimension,chunkTokens > 0 else { throw H3CheckpointError.invalid("Invalid fast H3 QKV geometry.") }
    var queries=[MLXArray](),keys=[MLXArray](),values=[MLXArray]()
    for start in stride(from:0,to:input.shape[1],by:chunkTokens) {
      try Task.checkCancellation()
      let end=min(start+chunkTokens,input.shape[1])
      let normalized=rms(input[0...,start..<end],weight:weights["norm1.weight"]!)
      let packed=try linear(normalized,"attn.to_qkv",weights).reshaped([1,end-start,heads,3,headDimension])
      let q=rms(packed[.ellipsis,0,0..<headDimension]),k=rms(packed[.ellipsis,1,0..<headDimension]),v=contiguous(packed[.ellipsis,2,0..<headDimension])
      eval(q,k,v);queries.append(q);keys.append(k);values.append(v)
    }
    let q=concatenated(queries,axis:1),k=concatenated(keys,axis:1),v=concatenated(values,axis:1)
    eval(q,k,v);return(q,k,v)
  }
  private func read(_ name:String,dtype:DType = .float32) throws -> MLXArray {
    try Task.checkCancellation();try layout.checkUnchanged()
    guard let shape=H3FastDecoderLayout.expectedShapes[name] else { throw H3CheckpointError.invalid("Unknown fast H3 tensor: \(name).") }
    let value=try layout.file.withTensorBytes(named:name) { MLXArray($0,shape,type:Float.self) }.asType(dtype)
    eval(value);return value
  }
  private func blockWeights(_ index:Int) throws -> [String:MLXArray] {
    let prefix="blocks.\(index).";var result=[String:MLXArray]()
    for name in H3FastDecoderLayout.expectedShapes.keys where name.hasPrefix(prefix) {
      // Autocast affects linear projections. Source scales/norms remain FP32;
      // multiplying by a learned FP32 scale promotes the residual to FP32.
      let linear=name.contains(".attn.") || name.contains(".ff.")
      result[String(name.dropFirst(prefix.count))]=try read(name,dtype:linear ? precision.dtype : .float32)
    }
    return result
  }
  static func linear(_ x:MLXArray,_ name:String,_ weights:[String:MLXArray]) throws -> MLXArray {
    guard let w=weights[name+".weight"],let b=weights[name+".bias"] else { throw H3CheckpointError.invalid("Missing fast H3 projection: \(name).") }
    return matmul(x.asType(w.dtype),w.transposed())+b
  }
  static func rms(_ x:MLXArray,weight:MLXArray?=nil) -> MLXArray {
    let f=x.asType(.float32),normal=f*rsqrt(mean(f*f,axis:-1,keepDims:true)+1e-5)
    return (weight.map{normal*$0} ?? normal).asType(x.dtype)
  }
  static func rotaryTables(plan:H3FastDecoderWindows.Plan,dtype:DType) -> (MLXArray,MLXArray) {
    // Existing native H3 frequency constants preserve the original FP32 table.
    let frequencies=MLXArray([UInt32(0x3f800000),0x3f0ff59a,0x3ea1e89b,0x3e361887,0x3dcccccc,0x3d6655c2,0x3d0186e3,0x3c91ad39].map(Float.init(bitPattern:)))
    var positions=[Float]();positions.reserveCapacity(plan.tokenCount*3)
    for t in 0..<plan.frames { for y in 0..<plan.height { for x in 0..<plan.width {
      positions.append((Float((t+plan.temporalShift)%5)+0.5)/5*2-1)
      positions.append((Float((y+plan.spatialShift)%16)+0.5)/16*2-1)
      positions.append((Float((x+plan.spatialShift)%16)+0.5)/16*2-1)
    } } }
    let ids=MLXArray(positions,[1,plan.tokenCount,3,1])
    let angles=(Float(2*Double.pi)*ids*frequencies.reshaped([1,1,1,8])).reshaped([1,plan.tokenCount,24])
    let doubled=concatenated([angles,angles],axis:-1).expandedDimensions(axis:2)
    let c=cos(doubled).asType(dtype),s=sin(doubled).asType(dtype);eval(c,s);return(c,s)
  }
  static func rotary(_ x:MLXArray,cosine:MLXArray,sine:MLXArray) -> MLXArray {
    let first=x[.ellipsis,0..<48]
    let half=concatenated([-first[.ellipsis,24..<48],first[.ellipsis,0..<24]],axis:-1)
    return concatenated([first*cosine+half*sine,x[.ellipsis,48..<64]],axis:-1)
  }
  /// Exact attention over valid keys: padding is omitted rather than converted
  /// into unmasked zero keys. Globals are never rotary-transformed.
  static func attention(q:MLXArray,k:MLXArray,v:MLXArray,globalQ:MLXArray,globalK:MLXArray,globalV:MLXArray,plan:H3FastDecoderWindows.Plan) throws -> (tokens:MLXArray,globals:MLXArray) {
    guard q.ndim == 4,q.shape == k.shape,q.shape == v.shape,q.shape[0] == 1,q.shape[1] == plan.tokenCount,
      globalQ.shape == globalK.shape,globalQ.shape == globalV.shape,
      globalQ.shape == [1,plan.nativeBlocks*5,q.shape[2],q.shape[3]] else { throw H3CheckpointError.invalid("Fast H3 attention state geometry does not match the complete clip.") }
    let scale=1/sqrt(Float(q.shape[3]));var windowResults=[MLXArray]()
    for window in plan.windows {
      try Task.checkCancellation()
      let ids=MLXArray(window.tokens.map(Int32.init))
      let gids=MLXArray(window.globalBlocks.flatMap { Array(($0*5)..<($0*5+5)) }.map(Int32.init))
      let queries=take(q,ids,axis:1).transposed(0,2,1,3)
      let keys=concatenated([take(k,ids,axis:1),take(globalK,gids,axis:1)],axis:1).transposed(0,2,1,3)
      let values=concatenated([take(v,ids,axis:1),take(globalV,gids,axis:1)],axis:1).transposed(0,2,1,3)
      let result=MLXFast.scaledDotProductAttention(queries:queries,keys:keys,values:values,scale:scale,mask:nil).transposed(0,2,1,3)
      eval(result);windowResults.append(result)
    }
    let tokens=take(concatenated(windowResults,axis:1),MLXArray(plan.inversePermutation.map(Int32.init)),axis:1)
    eval(tokens)
    var globals=[MLXArray]();let area=plan.height*plan.width
    for block in 0..<plan.nativeBlocks {
      try Task.checkCancellation()
      let frames=(block*5*area)..<min(plan.tokenCount,(block+1)*5*area),gids=(block*5)..<(block*5+5)
      let queries=globalQ[0...,gids].transposed(0,2,1,3).asType(.float32)
      var peak=MLXArray.full([1,q.shape[2],5,1],values:MLXArray(-Float.infinity))
      var denominator=MLXArray.zeros([1,q.shape[2],5,1]),numerator=MLXArray.zeros([1,q.shape[2],5,q.shape[3]])
      func append(_ keys:MLXArray,_ values:MLXArray) {
        let keys=keys.transposed(0,2,1,3).asType(.float32),values=values.transposed(0,2,1,3).asType(.float32)
        let scores=matmul(queries,keys.transposed(0,1,3,2))*scale
        let next=maximum(peak,max(scores,axis:-1,keepDims:true)),correction=exp(peak-next),probabilities=exp(scores-next)
        numerator=numerator*correction+matmul(probabilities,values)
        denominator=denominator*correction+sum(probabilities,axis:-1,keepDims:true)
        peak=next;eval(numerator,denominator,peak)
      }
      // Each native global sees every spatial token, accumulated without a
      // whole-native-block K/V gather or unbounded global score allocation.
      for start in stride(from:frames.lowerBound,to:frames.upperBound,by:1024) {
        let slab=start..<min(start+1024,frames.upperBound);append(k[0...,slab],v[0...,slab])
      }
      append(globalK[0...,gids],globalV[0...,gids])
      let result=(numerator/denominator).asType(globalQ.dtype).transposed(0,2,1,3)
      eval(result);globals.append(result)
    }
    let g=concatenated(globals,axis:1);eval(g);return(tokens,g)
  }
  /// Pixel head ordering is C, temporal subframe, vertical subpixel, horizontal
  /// subpixel. Each native five-latent block drops its first three subframes.
  static func reconstruct(_ head:MLXArray,latentOffset:Int=0) throws -> MLXArray {
    guard head.ndim == 5,head.shape[0] == 1,head.shape[4] == 3072 else { throw H3CheckpointError.invalid("Invalid fast H3 pixel head shape.") }
    let t=head.shape[1],h=head.shape[2],w=head.shape[3]
    let all=head.asType(.float32).reshaped([1,t,h,w,3,4,16,16]).transposed(0,1,5,2,6,3,7,4).reshaped([1,t*4,h*16,w*16,3])
    return take(all,MLXArray(try H3FastDecoderWindows.retainedFrames(latentFrames:t,latentOffset:latentOffset).map(Int32.init)),axis:1)
  }
  public static func rgb8(_ pixels:MLXArray) throws -> MLXArray {
    guard pixels.ndim == 5,pixels.shape[0] == 1,pixels.shape[4] == 3 else { throw H3CheckpointError.invalid("Invalid fast H3 RGB output.") }
    let scaled=clip(pixels.asType(.float32)*MLXArray([Float(0.229),0.224,0.225])+MLXArray([Float(0.485),0.456,0.406]),min:0,max:1)*255
    // Publisher torch.round uses ties-to-even, including exactly half RGB steps.
    let lower=floor(scaled),fraction=scaled-lower
    let odd=(lower.asType(.int32)%2) .== 1
    let increment=logicalOr(fraction .> 0.5,logicalAnd(fraction .== 0.5,odd)).asType(.float32)
    let result=(lower+increment).asType(.uint8);eval(result);return result
  }
  static func rawLatents(_ normalized:MLXArray,configuration:H3FastDecoderLayout.Configuration) throws -> MLXArray {
    guard normalized.ndim == 5,normalized.shape[4] == 24,normalized.dtype.isFloatingPoint else { throw H3CheckpointError.invalid("Invalid normalized fast H3 latent channels.") }
    return normalized.asType(.float32)*MLXArray(configuration.standardDeviation)+MLXArray(configuration.mean)
  }
  public func decode(normalizedLatents:MLXArray,contextRange:Range<Int>?=nil,outputRange:Range<Int>?=nil,totalLatentFrames:Int?=nil,feedForwardChunkTokens:Int=1024,progress:(Int,Int)->Void={_,_ in}) throws -> MLXArray {
    // The inner scope releases every weight and workspace before cache cleanup,
    // on success, failure or cancellation; only result pixels escape.
    defer { Stream.gpu.synchronize();Memory.clearCache() }
    return try evaluate(normalizedLatents:normalizedLatents,contextRange:contextRange,outputRange:outputRange,totalLatentFrames:totalLatentFrames,feedForwardChunkTokens:feedForwardChunkTokens,progress:progress)
  }
  private func evaluate(normalizedLatents:MLXArray,contextRange:Range<Int>?=nil,outputRange:Range<Int>?=nil,totalLatentFrames:Int?=nil,feedForwardChunkTokens:Int=1024,progress:(Int,Int)->Void={_,_ in}) throws -> MLXArray {
    guard normalizedLatents.ndim == 5,normalizedLatents.shape[0] == 1,normalizedLatents.shape[4] == 24,normalizedLatents.dtype.isFloatingPoint,
      normalizedLatents.shape[1] > 0,normalizedLatents.shape[2] > 0,normalizedLatents.shape[3] > 0,feedForwardChunkTokens > 0 else { throw H3CheckpointError.invalid("Fast H3 requires normalized [1,T,H,W,24] latents.") }
    let t=normalizedLatents.shape[1],h=normalizedLatents.shape[2],w=normalizedLatents.shape[3],count=t*h*w
    guard count <= maximumContextTokens else { throw H3CheckpointError.invalid("Fast H3 complete noncausal context exceeds the selected token budget; use a verified halo schedule or a larger explicit budget.") }
    let available=contextRange ?? 0..<t,output=outputRange ?? available,total=totalLatentFrames ?? available.upperBound
    guard available.count == t else { throw H3CheckpointError.invalid("Fast H3 context interval differs from its tensor.") }
    try H3FastDecoderWindows.validateContext(available:available,output:output,totalFrames:total)
    try Self.validateGeometry(frames:t,height:h,width:w,outputFrames:output.count,feedForwardChunkTokens:feedForwardChunkTokens)
    let estimate=try Self.estimatedActiveBytes(frames:t,height:h,width:w,outputFrames:output.count)
    let physical=ProcessInfo.processInfo.physicalMemory
    guard estimate <= maximumActiveBytes,estimate <= physical-physical/4 else { throw H3CheckpointError.invalid("Fast H3 complete context needs an estimated \(estimate) active bytes; the selected memory allowance retains a physical reserve.") }
    try layout.checkUnchanged();try Task.checkCancellation()
    let configuration=layout.configuration
    let raw=try Self.rawLatents(normalizedLatents,configuration:configuration)
    // The reference's outer BF16 autocast includes post-quant Conv3d;
    // its following embedding explicitly disables autocast and returns to F32.
    let postWeight=try read("post_quant_conv.weight",dtype:precision.dtype).transposed(0,2,3,4,1),postBias=try read("post_quant_conv.bias",dtype:precision.dtype)
    let post=conv3d(raw.asType(precision.dtype),postWeight)+postBias
    let embedWeight=try read("x_embedder.weight"),embedBias=try read("x_embedder.bias")
    var x=(matmul(post.asType(.float32),embedWeight.transposed())+embedBias).asType(precision.dtype).reshaped([1,count,2048])
    let registers=try read("register_tokens").asType(precision.dtype)
    let globals=concatenated([registers,MLXArray.zeros([1,1,2048],dtype:precision.dtype)],axis:1)
    var g=broadcast(globals.expandedDimensions(axis:1),to:[1,(t+4)/5,5,2048]).reshaped([1,((t+4)/5)*5,2048])
    eval(x,g)
    let even=try H3FastDecoderWindows.plan(frames:t,height:h,width:w,layer:0),odd=try H3FastDecoderWindows.plan(frames:t,height:h,width:w,layer:1)
    let evenRotary=Self.rotaryTables(plan:even,dtype:precision.dtype),oddRotary=Self.rotaryTables(plan:odd,dtype:precision.dtype)
    for layer in 0..<24 {
      try Task.checkCancellation()
      let weights=try blockWeights(layer),plan=layer%2 == 0 ? even : odd,rotary=layer%2 == 0 ? evenRotary : oddRotary
      func qkv(_ input:MLXArray,rotate:Bool) throws -> (MLXArray,MLXArray,MLXArray) {
        let projected=try Self.projectQKV(input,weights:weights)
        var q=projected.0,k=projected.1
        if rotate { q=Self.rotary(q,cosine:rotary.0,sine:rotary.1);k=Self.rotary(k,cosine:rotary.0,sine:rotary.1) }
        eval(q,k);return(q,k,projected.2)
      }
      let local=try qkv(x,rotate:true),global=try qkv(g,rotate:false)
      let attended=try Self.attention(q:local.0,k:local.1,v:local.2,globalQ:global.0,globalK:global.1,globalV:global.2,plan:plan)
      x=x+(try Self.linear(attended.tokens.reshaped([1,count,2048]),"attn.to_out",weights))*weights["scale1"]!
      g=g+(try Self.linear(attended.globals.reshaped(g.shape),"attn.to_out",weights))*weights["scale1"]!
      eval(x,g)
      func feed(_ input:MLXArray) throws -> MLXArray {
        let fused=try Self.linear(Self.rms(input,weight:weights["norm2.weight"]!),"ff.w1",weights)
        let gate=fused[.ellipsis,0..<8192],up=fused[.ellipsis,8192..<16384]
        return input+(try Self.linear((gate*sigmoid(gate))*up,"ff.w2",weights))*weights["scale2"]!
      }
      var slabs=[MLXArray]()
      for start in stride(from:0,to:count,by:feedForwardChunkTokens) {
        let result=try feed(x[0...,start..<min(count,start+feedForwardChunkTokens)]);eval(result);slabs.append(result)
      }
      x=concatenated(slabs,axis:1);g=try feed(g);eval(x,g)
      try layout.checkUnchanged();progress(layer+1,24)
    }
    // Select token rows before normalization/projection/reconstruction: output
    // intervals bound the pixel allocation while retaining exact context math.
    let rows=((output.lowerBound-available.lowerBound)*h*w)..<((output.upperBound-available.lowerBound)*h*w)
    let norm=MLXFast.layerNorm(x[0...,rows],weight:try read("norm_out.weight"),bias:try read("norm_out.bias"),eps:1e-5)
    let head=(matmul(norm.asType(.float32),try read("proj_out.weight").transposed())+(try read("proj_out.bias"))).reshaped([1,output.count,h,w,3072])
    let pixels=try Self.reconstruct(head,latentOffset:output.lowerBound);eval(pixels)
    try layout.checkUnchanged();try Task.checkCancellation()
    return pixels
  }
}
