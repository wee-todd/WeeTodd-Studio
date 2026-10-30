import Foundation
import MLX

/// Released 30-block Wan-derived DiT. Each call evaluates exactly one four-latent
/// group. The caller owns noise and the stream state; failures invalidate that state.
public final class FlashVSRTransformer {
  private let checkpoint:FlashVSRCheckpoint
  private let fixedContext:MLXArray
  public let config:FlashVSRConfig
  public init(checkpoint:FlashVSRCheckpoint, fixedContextURL:URL, identity:FlashVSRCheckpoint.Identity) throws {
    let fixedContext=try FlashVSRCheckpoint.loadFixedContext(url:fixedContextURL)
    try FlashVSRCheckpoint.validateIdentity(identity,expected:checkpoint.config.variant)
    try FlashVSRCheckpoint.validateContext(fixedContext)
    self.checkpoint=checkpoint; self.fixedContext=fixedContext.asType(checkpoint.dtype); config=checkpoint.config
  }
  static func gelu(_ x:MLXArray) -> MLXArray { x*0.5*(1+tanh(sqrt(Float(2/Double.pi))*(x+0.044715*x*x*x))) }
  static func silu(_ x:MLXArray) -> MLXArray { x*sigmoid(x) }
  static func norm(_ x:MLXArray) -> MLXArray {
    MLXFast.layerNorm(x,weight:nil,bias:nil,eps:1e-6)
  }
  static func linear(_ x:MLXArray,_ name:String,_ weights:[String:MLXArray]) throws -> MLXArray {
    guard let w=weights[name+".weight"],let b=weights[name+".bias"] else { throw FlashVSRError.invalid("Missing FlashVSR linear \(name).") }
    return matmul(x,w.transposed())+b
  }
  public static func sinusoidalEmbedding(timestep:Float,dimension:Int=256) throws -> MLXArray {
    guard dimension > 0, dimension%2 == 0, timestep.isFinite else { throw FlashVSRError.invalid("Invalid FlashVSR timestep embedding.") }
    let angles=(0..<dimension/2).map { Double(timestep)*pow(10000,-Double($0)/Double(dimension/2)) }
    return MLXArray(angles.map { Float(cos($0)) }+angles.map { Float(sin($0)) },[1,dimension])
  }
  public static func rotary(_ x:MLXArray,shape:[Int],temporalOffset:Int) throws -> MLXArray {
    guard x.ndim == 4, shape.count == 3, shape.allSatisfy({$0 > 0}), shape.reduce(1,*) == x.shape[2], x.shape[3]%2 == 0,temporalOffset >= 0 else { throw FlashVSRError.invalid("Invalid rotary geometry.") }
    let pairs=x.shape[3]/2, spatialPairs=pairs/3, counts=[pairs-2*spatialPairs,spatialPairs,spatialPairs]
    var cosines=[Float](),sines=[Float]()
    cosines.reserveCapacity(x.shape[2]*pairs); sines.reserveCapacity(x.shape[2]*pairs)
    for t in 0..<shape[0] { for y in 0..<shape[1] { for z in 0..<shape[2] {
      for (position,count) in zip([t+temporalOffset,y,z],counts) {
        for i in 0..<count {
          // Upstream forms angles in FP32, evaluates rotations in FP64, then casts.
          let angle=Float(position)*Float(pow(10000,-Double(i)/Double(count)))
          cosines.append(Float(cos(Double(angle)))); sines.append(Float(sin(Double(angle))))
        }
      }
    } } }
    let cosine=MLXArray(cosines,[1,1,x.shape[2],pairs]), sine=MLXArray(sines,[1,1,x.shape[2],pairs])
    let paired=x.asType(.float32).reshaped([x.shape[0],x.shape[1],x.shape[2],pairs,2])
    let a=paired[.ellipsis,0], b=paired[.ellipsis,1]
    return stacked([a*cosine-b*sine,b*cosine+a*sine],axis:-1).reshaped(x.shape).asType(x.dtype)
  }
  private func load(_ prefixes:[String]) throws -> [String:MLXArray] {
    var weights=[String:MLXArray]()
    for key in FlashVSRCheckpoint.expectedShapes(component:.transformer).keys where prefixes.contains(where:{key.hasPrefix($0)}) { weights[key]=try checkpoint.tensor(key) }
    return weights
  }
  private func heads(_ x:MLXArray) -> MLXArray { x.reshaped([x.shape[0],x.shape[1],config.heads,config.width/config.heads]).transposed(0,2,1,3) }
  private func projected(_ x:MLXArray,_ name:String,_ weights:[String:MLXArray],normalize:Bool) throws -> MLXArray {
    let p=try Self.linear(x,name,weights)
    guard normalize else { return heads(p) }
    let normName=String(name.dropLast())+"norm_"+String(name.suffix(1))+".weight"
    guard let w=weights[normName] else { throw FlashVSRError.invalid("Missing attention RMS normalization.") }
    return heads(MLXFast.rmsNorm(p,weight:w,eps:config.epsilon))
  }
  private func selfAttention(_ x:MLXArray,layer:Int,shape:[Int],state:FlashVSRStreamState,weights:[String:MLXArray]) throws -> MLXArray {
    let p="blocks.\(layer).self_attn."
    let q=try FlashVSRAttention.partition(Self.rotary(projected(x,p+"q",weights,normalize:true),shape:shape,temporalOffset:state.groupIndex*4),shape:shape)
    var k=try FlashVSRAttention.partition(Self.rotary(projected(x,p+"k",weights,normalize:true),shape:shape,temporalOffset:state.groupIndex*4),shape:shape)
    var v=try FlashVSRAttention.partition(projected(x,p+"v",weights,normalize:false),shape:shape)
    if let old=state.caches.removeValue(forKey:layer) { k=concatenated([old.0,k],axis:2); v=concatenated([old.1,v],axis:2) }
    eval(q,k,v)
    let rows=shape[1]/8,cols=shape[2]/8
    let selection=try FlashVSRAttention.route(q:q,k:k,rows:rows,columns:cols,localRange:config.localRange(layer:layer),topK:config.topK(spatialBlocks:rows*cols))
    let output=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selection)
    let ids=try FlashVSRBlockRouting.cacheIndices(keyBlocks:k.shape[2],spatialBlocks:rows*cols,windows:config.cacheWindows(layer:layer))
    let index=MLXArray(ids.map(Int32.init))
    let nextK=take(k,index,axis:2),nextV=take(v,index,axis:2); eval(nextK,nextV)
    state.caches[layer]=(nextK,nextV)
    let tokens=FlashVSRAttention.unpartition(output,shape:shape).transposed(0,2,1,3).reshaped(x.shape)
    return try Self.linear(tokens,p+"o",weights)
  }
  private func crossAttention(_ x:MLXArray,context:MLXArray,layer:Int,weights:[String:MLXArray]) throws -> MLXArray {
    let p="blocks.\(layer).cross_attn."
    let q=try projected(x,p+"q",weights,normalize:true),k=try projected(context,p+"k",weights,normalize:true),v=try projected(context,p+"v",weights,normalize:false)
    let attended=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:1/sqrt(Float(config.width/config.heads)),mask:nil)
    return try Self.linear(attended.transposed(0,2,1,3).reshaped(x.shape),p+"o",weights)
  }
  public func predict(noise:MLXArray,conditions:[MLXArray],sourceID:String,state:FlashVSRStreamState,feedForwardChunkTokens:Int=1024) throws -> MLXArray {
    guard noise.ndim == 5, noise.shape[0] == 1, noise.shape[1] == 16, noise.shape[2] == 4, noise.shape[3]%16 == 0, noise.shape[4]%16 == 0,
      feedForwardChunkTokens > 0, conditions.count == 1 else { throw FlashVSRError.invalid("FlashVSR DiT requires [1,16,4,H,W] and one released LQ condition.") }
    let shape=[4,noise.shape[3]/2,noise.shape[4]/2]
    try state.validate(sourceID:sourceID,tokenShape:shape)
    try state.bindModel(FlashVSRCheckpoint.releasedDigests(config.variant).transformer)
    guard conditions[0].shape == [1,shape.reduce(1,*),1536] else { throw FlashVSRError.invalid("LQ condition geometry does not match DiT tokens.") }
    do {
      let shared=try load(["patch_embedding.","text_embedding.","time_embedding.","time_projection.","head."])
      let input=noise.asType(checkpoint.dtype)
      let embedded=try Self.linear(Self.silu(Self.linear(Self.sinusoidalEmbedding(timestep:FlashVSRScheduler.timestep).asType(input.dtype),"time_embedding.0",shared)),"time_embedding.2",shared)
      let modulation=try Self.linear(Self.silu(embedded),"time_projection.1",shared).reshaped([1,6,1536])
      let context=try Self.linear(Self.gelu(Self.linear(fixedContext,"text_embedding.0",shared)),"text_embedding.2",shared)
      guard let patch=shared["patch_embedding.weight"],let patchBias=shared["patch_embedding.bias"] else { throw FlashVSRError.invalid("Missing patch embedding.") }
      var x=(conv3d(input.transposed(0,2,3,4,1),patch.transposed(0,2,3,4,1),stride:[1,2,2])+patchBias).reshaped([1,-1,1536])
      eval(context,embedded,modulation,x)
      for layer in 0..<config.layers {
        try Task.checkCancellation()
        let p="blocks.\(layer).",weights=try load([p])
        if layer == 0 { x=x+conditions[0].asType(x.dtype) }
        let m=weights[p+"modulation"]!+modulation
        let shiftA=m[0...,0..<1],scaleA=m[0...,1..<2],gateA=m[0...,2..<3]
        let shiftF=m[0...,3..<4],scaleF=m[0...,4..<5],gateF=m[0...,5..<6]
        x=x+gateA*(try selfAttention(Self.norm(x)*(1+scaleA)+shiftA,layer:layer,shape:shape,state:state,weights:weights))
        let crossInput=MLXFast.layerNorm(x,weight:weights[p+"norm3.weight"]!,bias:weights[p+"norm3.bias"]!,eps:config.epsilon)
        x=x+(try crossAttention(crossInput,context:context,layer:layer,weights:weights))
        eval(x)
        var parts=[MLXArray]()
        for start in stride(from:0,to:x.shape[1],by:feedForwardChunkTokens) {
          let part=x[0...,start..<min(start+feedForwardChunkTokens,x.shape[1])]
          let f=try Self.linear(Self.gelu(Self.linear(Self.norm(part)*(1+scaleF)+shiftF,p+"ffn.0",weights)),p+"ffn.2",weights)
          let output=part+gateF*f; eval(output); parts.append(output)
        }
        x=concatenated(parts,axis:1); eval(x)
      }
      let m=shared["head.modulation"]!+embedded.expandedDimensions(axis:1)
      let output=try Self.linear(Self.norm(x)*(1+m[0...,1..<2])+m[0...,0..<1],"head.proj",shared)
      let reconstructed=output.reshaped([1,4,shape[1],shape[2],1,2,2,16]).transposed(0,7,1,4,2,5,3,6).reshaped(noise.shape)
      eval(reconstructed); state.finishGroup(); return reconstructed
    } catch { state.reset(sourceID:sourceID); throw error }
  }
}
