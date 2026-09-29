import Foundation
import MLX
import LTX25Engine

/// Joint AV block expressed through the same MLX primitives as the working
/// reference engine. Initial qualification is batch-one, uniform timestep,
/// split RoPE and no attention masks. Unsupported input fields are rejected.
public final class MLXAVBlock {
  public let configuration: AVBlockConfiguration
  public let inputShapes: [String:[Int]]
  public let weightShapes: [String:[Int]]
  public private(set) var loadedBytes = 0
  private var weights: [String:MLXWeight] = [:]
  private var adapters: [String:[MLXLoRA]] = [:]
  private let maximumWeightBytes: Int
  private let baseActivationBytes:Int
  private let maximumActivationBytes:Int
  private let compileGraph:Bool
  private var compiledSignature:[Int]?
  private var compiledGraph:(@Sendable ([MLXArray]) -> [MLXArray])?
  private(set) var compiledGraphBuildCount=0

  public init(configuration c: AVBlockConfiguration,
    maximumWeightBytes: Int = 2*1024*1024*1024,
    maximumActivationBytes: Int = 2*1024*1024*1024,compileGraph:Bool=true) throws {
    try c.validate()
    guard (1...32*1024*1024*1024).contains(maximumActivationBytes) else {
      throw LTXError.invalid("Transformer workspace must be positive and at most 32 GiB.")
    }
    let vd=c.videoDimension, ad=c.audioDimension, nv=c.videoTokens, na=c.audioTokens, nt=c.textTokens
    var inputs = ["video":[nv,vd],"audio":[na,ad],
      "video_modulation":[1,9*vd],"audio_modulation":[1,9*ad],
      "video_prompt_modulation":[1,2*vd],"audio_prompt_modulation":[1,2*ad],
      "video_av_modulation":[1,4*vd],"audio_av_modulation":[1,4*ad],
      "video_av_gate":[1,vd],"audio_av_gate":[1,ad],
      "video_text":[nt,vd],"audio_text":[nt,ad]]
    for (name,n,h) in [("video",nv,c.videoHeadDimension),("audio",na,c.audioHeadDimension),
      ("video_cross",nv,c.audioHeadDimension),("audio_cross",na,c.audioHeadDimension)] {
      for suffix in ["cos","sin"] { inputs[name+"_rope_"+suffix] = [n*c.heads,h/2] }
    }
    let activationEstimate=try Self.estimatedActivationBytes(configuration:c)
    guard Device.defaultDevice().deviceType == .gpu, maximumWeightBytes > 0 else {
      throw LTXError.invalid("MLX block requires a GPU and a positive weight allowance.")
    }
    guard activationEstimate <= maximumActivationBytes else {
      throw LTXError.invalid("MLX block needs an estimated \(activationEstimate) activation bytes; admitted allowance is \(maximumActivationBytes) bytes.")
    }
    var shapes: [String:[Int]] = ["scale_shift_table":[9,vd],"audio_scale_shift_table":[9,ad],
      "prompt_scale_shift_table":[2,vd],"audio_prompt_scale_shift_table":[2,ad],
      "scale_shift_table_a2v_ca_video":[5,vd],"scale_shift_table_a2v_ca_audio":[5,ad]]
    func dense(_ name:String,_ input:Int,_ output:Int,_ bias:Bool = true) {
      shapes[name+".weight"] = [output,input]
      if bias { shapes[name+".bias"] = [output] }
    }
    for (name,q,k,inner) in [("attn1",vd,vd,vd),("audio_attn1",ad,ad,ad),
      ("attn2",vd,vd,vd),("audio_attn2",ad,ad,ad),
      ("audio_to_video_attn",vd,ad,ad),("video_to_audio_attn",ad,vd,ad)] {
      dense(name+".to_q",q,inner); dense(name+".to_k",k,inner); dense(name+".to_v",k,inner)
      dense(name+".to_out",inner,q); dense(name+".to_gate_logits",q,c.heads)
      shapes[name+".q_norm.weight"] = [inner]; shapes[name+".k_norm.weight"] = [inner]
    }
    for (name,d,bias) in [("ff",vd,false),("audio_ff",ad,true)] {
      dense(name+".proj_in",d,d*4,bias); dense(name+".proj_out",d*4,d,bias)
    }
    baseActivationBytes=activationEstimate; self.maximumActivationBytes=maximumActivationBytes
    configuration=c; inputShapes=inputs; weightShapes=shapes; self.maximumWeightBytes=maximumWeightBytes
    self.compileGraph=compileGraph
  }

  /// Pure admission estimate, shared by automatic Studio sizing and execution.
  /// Includes the existing conservative reference-modulation reserve when needed.
  public static func estimatedActivationBytes(configuration c:AVBlockConfiguration,perTokenVideo:Bool=false) throws -> Int {
    try c.validate()
    let vd=c.videoDimension,ad=c.audioDimension,nv=c.videoTokens,na=c.audioTokens,nt=c.textTokens
    // Conservative admission for a single block's linear/FF intermediates.
    // This is an input guard, not a process-memory cap or production-size qualification.
    let linearBytes = (nv*vd+na*ad+nt*(vd+ad))*4*24
    // Match the pinned MLX Metal dispatch boundary. Non-fused attention
    // materializes QKᵀ and softmax storage, even for narrow hidden states.
    let attentions=[(nv,nv,c.videoHeadDimension),(na,na,c.audioHeadDimension),
      (nv,nt,c.videoHeadDimension),(na,nt,c.audioHeadDimension),
      (nv,na,c.audioHeadDimension),(na,nv,c.audioHeadDimension)]
    let fallbackBytes=attentions.map { q,k,h in
      let fused = q > 8 ? [64,80,128].contains(h) : q <= k && [64,96,128,256].contains(h)
      return fused ? 0 : c.heads*q*k*4*3
    }.max()!
    return linearBytes+fallbackBytes+(perTokenVideo ? nv*vd*14*4 : 0)
  }

  public func load(_ provider:(String,[Int]) throws -> MLXWeight) throws {
    release()
    do {
      for name in weightShapes.keys.sorted() {
        try Task.checkCancellation()
        let weight = try provider(name,weightShapes[name]!)
        guard weight.shape == weightShapes[name], weight.storageBytes <= maximumWeightBytes-loadedBytes else {
          throw LTXError.invalid("MLX weight shape or admitted storage budget differs: \(name)")
        }
        weights[name]=weight; loadedBytes += weight.storageBytes
      }
      try MLXWeight.materialize(Array(weights.values))
      try Task.checkCancellation()
    } catch { release(); throw error }
  }

  public func setAdapters(_ stack: [String:[MLXLoRA]]) throws {
    guard weights.count == weightShapes.count else { throw LTXError.invalid("Load block before adapters.") }
    for (name,items) in stack {
      guard name.hasSuffix(".weight"), let shape=weightShapes[name], shape.count == 2,
        items.allSatisfy({ $0.down.shape[1] == shape[1] && $0.up.shape[0] == shape[0] }) else {
        throw LTXError.invalid("Unsupported adapter target or shape: \(name)")
      }
    }
    let baseBytes=weights.values.reduce(0) { $0+$1.storageBytes }
    let factorBytes=stack.values.flatMap { $0 }.reduce(0) { $0+$1.down.nbytes+$1.up.nbytes }
    guard factorBytes <= maximumWeightBytes-baseBytes else { throw LTXError.invalid("Adapter factors exceed block storage budget.") }
    adapters=stack; loadedBytes=baseBytes+factorBytes
  }

  public func release() {
    Stream.gpu.synchronize()
    weights.removeAll(); adapters.removeAll(); loadedBytes=0
  }

  public func evaluate(_ inputs:[String:MLXArray]) throws -> [String:MLXArray] {
    do {
      try Task.checkCancellation()
      try validateInputs(inputs)
      guard weights.count == weightShapes.count else {
        throw LTXError.invalid("MLX block is unloaded or inputs are missing/unsupported.")
      }
      var x: [String:MLXArray] = [:]
      for (name,shape) in inputShapes {
        let value=inputs[name]!
        let perToken=["video_modulation","video_av_modulation"].contains(name) && value.ndim == 2 && (value.shape[0] == configuration.videoTokens || inputs["video_modulation_indices"] != nil)
        x[name]=value.reshaped(perToken ? value.shape : shape)
      }
      x["video_modulation_indices"]=inputs["video_modulation_indices"]
      let (graph,arguments,signature)=MLXBlockGraph.bind(configuration:configuration,inputs:x,weights:weights,adapters:adapters)
      let outputs:[MLXArray]
      if compileGraph {
        if compiledSignature != signature {
          compiledGraph=compile { arrays in graph.call(arrays) }
          compiledSignature=signature;compiledGraphBuildCount += 1
        }
        outputs=compiledGraph!(arguments)
      } else { outputs=graph.call(arguments) }
      let result=["video":outputs[0],"audio":outputs[1]]
      eval(Array(result.values))
      try Task.checkCancellation()
      guard result.values.allSatisfy({ MLX.isFinite($0).all().item(Bool.self) }) else {
        throw LTXError.invalid("Non-finite audiovisual MLX block output.")
      }
      return result
    } catch { release(); throw error }
  }

  func admitPerTokenVideo() throws {
    guard baseActivationBytes+configuration.videoTokens*configuration.videoDimension*14*4 <= maximumActivationBytes else {
      throw LTXError.invalid("Per-token modulation exceeds the activation budget.")
    }
  }
  public func validateInputs(_ inputs:[String:MLXArray]) throws {
    let indices=inputs["video_modulation_indices"]
    let allowed=Set(inputShapes.keys).union(indices == nil ? [] : ["video_modulation_indices"])
    guard Set(inputs.keys) == allowed else { throw LTXError.invalid("Missing or unsupported MLX block input.") }
    var uniqueRows:Int?
    for (name,shape) in inputShapes {
      let value=inputs[name]!
      let modulation=["video_modulation","video_av_modulation"].contains(name)
      let compact=indices != nil && modulation && value.ndim==2 && (1...configuration.videoTokens).contains(value.shape[0]) && value.shape[1]==shape[1]
      let perToken=modulation && value.shape == [configuration.videoTokens,shape[1]]
      guard value.dtype == .float32, compact || perToken || (indices == nil || !modulation) && (value.shape == shape || value.shape == [shape.reduce(1,*)]) else {
        throw LTXError.invalid("MLX input shape/dtype differs: \(name)")
      }
      if compact {
        if let rows=uniqueRows,rows != value.shape[0] { throw LTXError.invalid("Compact modulation row counts differ.") }
        uniqueRows=value.shape[0]
      }
      if perToken || compact { try admitPerTokenVideo() }
    }
    if let indices {
      guard indices.dtype == .int32,indices.shape == [configuration.videoTokens],let rows=uniqueRows,
        logicalAnd(greaterEqual(indices,0),less(indices,rows)).all().item(Bool.self) else {
        throw LTXError.invalid("Invalid compact modulation indices.")
      }
    }
  }

  static func forward(configuration c:AVBlockConfiguration,_ x:[String:MLXArray],
    parameter:(String) -> MLXArray,linear:(String,MLXArray) -> MLXArray) -> [String:MLXArray] {
    let vd=c.videoDimension, ad=c.audioDimension, heads=c.heads
    func norm(_ value:MLXArray) -> MLXArray {
      // The reference normalizes across the complete projected width, before
      // reshaping into heads (not independently within each head).
      MLXFast.rmsNorm(value,weight:.ones([value.shape.last!]),eps:1e-6)
    }
    func modulations(_ input:String,_ table:String,_ count:Int,_ width:Int) -> [MLXArray] {
      let rows=x[input]!.shape[0]
      let values=x[input]!.reshaped([rows,count,width]); let learned=parameter(table)
      return (0..<count).map {
        let value=values[0...,$0,0...] + learned[$0]
        if ["video_modulation","video_av_modulation"].contains(input),let indices=x["video_modulation_indices"] {
          return take(value,indices,axis:0)
        }
        return value
      }
    }
    let vm=modulations("video_modulation","scale_shift_table",9,vd)
    let am=modulations("audio_modulation","audio_scale_shift_table",9,ad)
    let vp=modulations("video_prompt_modulation","prompt_scale_shift_table",2,vd)
    let ap=modulations("audio_prompt_modulation","audio_prompt_scale_shift_table",2,ad)
    let vc=modulations("video_av_modulation","scale_shift_table_a2v_ca_video",4,vd)
    let ac=modulations("audio_av_modulation","scale_shift_table_a2v_ca_audio",4,ad)
    let vg=x["video_av_gate"]! + (parameter("scale_shift_table_a2v_ca_video"))[4]
    let ag=x["audio_av_gate"]! + (parameter("scale_shift_table_a2v_ca_audio"))[4]
    func rotary(_ value:MLXArray,_ key:String,_ width:Int) -> MLXArray {
      let rows=value.shape[0], flat=value.reshaped([rows*heads,width])
      let first=flat[0...,0..<(width/2)], second=flat[0...,(width/2)..<width]
      let cos=x[key+"_rope_cos"]!, sin=x[key+"_rope_sin"]!
      return concatenated([first*cos-second*sin,first*sin+second*cos],axis:1).reshaped([1,rows,heads,width])
    }
    func attention(_ name:String,_ query:MLXArray,_ context:MLXArray,_ headWidth:Int,
      queryRoPE:String? = nil,keyRoPE:String? = nil) -> MLXArray {
      let nq=query.shape[0], nk=context.shape[0]
      let qp=linear(name+".to_q",query), kp=linear(name+".to_k",context)
      let value=linear(name+".to_v",context).reshaped([1,nk,heads,headWidth]).transposed(0,2,1,3)
      var q=norm(qp) * (parameter(name+".q_norm.weight"))
      var k=norm(kp) * (parameter(name+".k_norm.weight"))
      q=queryRoPE.map { rotary(q,$0,headWidth) } ?? q.reshaped([1,nq,heads,headWidth])
      k=keyRoPE.map { rotary(k,$0,headWidth) } ?? k.reshaped([1,nk,heads,headWidth])
      let attended=MLXFast.scaledDotProductAttention(queries:q.transposed(0,2,1,3),
        keys:k.transposed(0,2,1,3),values:value,scale:1/Float(headWidth).squareRoot(),mask:nil)
      let gate=2*sigmoid(linear(name+".to_gate_logits",query))
      let gated=attended.transposed(0,2,1,3)*gate.reshaped([1,nq,heads,1])
      return linear(name+".to_out",gated.reshaped([nq,heads*headWidth]))
    }
    func normalized(_ value:MLXArray,_ mods:[MLXArray],_ index:Int) -> MLXArray {
      norm(value)*(1+mods[index+1])+mods[index]
    }
    let v=x["video"]!, a=x["audio"]!
    let vn=normalized(v,vm,0), an=normalized(a,am,0)
    var video=v + (attention("attn1",vn,vn,c.videoHeadDimension,queryRoPE:"video",keyRoPE:"video"))*vm[2]
    var audio=a + (attention("audio_attn1",an,an,c.audioHeadDimension,queryRoPE:"audio",keyRoPE:"audio"))*am[2]
    video=video + (attention("attn2",normalized(video,vm,6),x["video_text"]!*(1+vp[1])+vp[0],c.videoHeadDimension))*vm[8]
    audio=audio + (attention("audio_attn2",normalized(audio,am,6),x["audio_text"]!*(1+ap[1])+ap[0],c.audioHeadDimension))*am[8]
    let sharedVideo=norm(video), sharedAudio=norm(audio)
    video=video + (attention("audio_to_video_attn",sharedVideo*(1+vc[0])+vc[1],sharedAudio*(1+ac[0])+ac[1],c.audioHeadDimension,
      queryRoPE:"video_cross",keyRoPE:"audio_cross"))*vg
    audio=audio + (attention("video_to_audio_attn",sharedAudio*(1+ac[2])+ac[3],sharedVideo*(1+vc[2])+vc[3],c.audioHeadDimension,
      queryRoPE:"audio_cross",keyRoPE:"video_cross"))*ag
    func feedForward(_ name:String,_ value:MLXArray) -> MLXArray {
      let h=linear(name+".proj_in",value)
      let gelu=0.5*h*(1+tanh(Float(sqrt(2/Double.pi))*(h+0.044715*h*h*h)))
      return linear(name+".proj_out",gelu)
    }
    video=video + (feedForward("ff",normalized(video,vm,3)))*vm[5]
    audio=audio + (feedForward("audio_ff",normalized(audio,am,3)))*am[5]
    return ["video":video,"audio":audio]
  }
}
