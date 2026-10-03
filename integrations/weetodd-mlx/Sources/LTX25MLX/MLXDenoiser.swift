import Foundation
import MLX
import LTX25Engine

/// Packed audiovisual latents to velocities with optional reference token times. The owning worker
/// serializes calls. Fixed projections, block weights and adapters have disjoint
/// staged lifetimes; evaluated activations remain in MLX throughout the call.
public final class MLXDenoiser {
  public typealias FixedProvider = (String,[Int]) throws -> MLXWeight
  public typealias BlockProvider = (Int,String,[Int]) throws -> MLXWeight
  public typealias FixedAdapters = (String) throws -> [MLXLoRA]
  public struct Progress {
    public let stage:String
    public let completedBlocks:Int
    public let activeBytes:Int
    public let cacheBytes:Int
    public let stack:MLXAVStack.Progress?
  }
  public let configuration:AVBlockConfiguration
  public let inputShapes:[String:[Int]]
  public var residentWeightBytes:Int { fixedResidentBytes+stack.residentWeightBytes }
  private let shapes:[String:[Int]]
  private let stack:MLXAVStack
  private let blockCount:Int
  private let cacheBytes:Int
  private let maximumRotaryBytes:Int
  private let keyframeMarkerRows:Int
  private let leadingKeyframeMarkerRows:Int
  private var fixedResidentBytes=0
  private var active=false

  struct Preparation {
    let rotary:[String:MLXArray]
    let modulations:[UInt32:[String:MLXArray]]
    let embeddings:[UInt32:[String:MLXArray]]
    let videoTimes:[UInt32:[Float]]
    let audioTimes:[UInt32:[Float]]
    let referenceModulations:[UInt32:[String:MLXArray]]
    let referenceEmbeddings:[UInt32:MLXArray]
    let referenceAudioEmbeddings:[UInt32:MLXArray]
    let frozenAudio:Bool
    let rawGlobalTimesteps:Bool
  }

  public init(configuration:AVBlockConfiguration,blockCount:Int=48,cacheBytes:Int=128*1024*1024,
    maximumActivationBytes:Int=2*1024*1024*1024,videoAttentionGroups:[Int]=[],
    keyframeMarkerRows:Int=0,leadingKeyframeMarkerRows:Int=0) throws {
    guard keyframeMarkerRows>=0,keyframeMarkerRows<=configuration.videoTokens,
      leadingKeyframeMarkerRows>=0,
      leadingKeyframeMarkerRows<=configuration.videoTokens-keyframeMarkerRows else {
      throw LTXError.invalid("Generated keyframe marker rows exceed the video token layout.")
    }
    self.keyframeMarkerRows=keyframeMarkerRows
    self.leadingKeyframeMarkerRows=leadingKeyframeMarkerRows
    maximumRotaryBytes=try Self.admitRotary(configuration:configuration,maximumActivationBytes:maximumActivationBytes)
    stack=try MLXAVStack(configuration:configuration,blockCount:blockCount,cacheBytes:cacheBytes,
      maximumActivationBytes:maximumActivationBytes,videoAttentionGroups:videoAttentionGroups)
    self.configuration=configuration; self.blockCount=blockCount; self.cacheBytes=cacheBytes
    var admitted=DenoiserLayout.inputShapes(configuration)
    if !videoAttentionGroups.isEmpty {
      admitted["video_attention_templates"]=[videoAttentionGroups.count,configuration.videoTokens]
    }
    inputShapes=admitted;shapes=DenoiserLayout.weightShapes(configuration)
  }

  /// Leading rows represent the first generated latent frame. Trailing rows
  /// retain the separate generated-keyframe slot contract; IC guide rows are
  /// never selected merely because their temporal positions also start at zero.
  static func applyKeyframeMarkers(_ projection:MLXArray,marker:MLXArray,
    leadingRows:Int,trailingRows:Int) throws -> MLXArray {
    guard projection.ndim == 2,marker.shape == [1,projection.shape[1]],
      leadingRows>=0,trailingRows>=0,trailingRows<=projection.shape[0],
      leadingRows<=projection.shape[0]-trailingRows else {
      throw LTXError.invalid("Generated keyframe marker spans overlap or exceed video rows.")
    }
    if leadingRows == 0 && trailingRows == 0 { return projection }
    let split=projection.shape[0]-trailingRows
    if leadingRows == 0 {
      // Preserve the existing trailing-slot arithmetic and concatenation.
      return concatenated([projection[0..<split],projection[split..<projection.shape[0]]+marker],axis:0)
    }
    var parts=[projection[0..<leadingRows]+marker]
    if leadingRows<split { parts.append(projection[leadingRows..<split]) }
    if trailingRows>0 { parts.append(projection[split..<projection.shape[0]]+marker) }
    return parts.count == 1 ? parts[0] : concatenated(parts,axis:0)
  }

  /// The four grids live together. Account for all of them within the already
  /// conservative block estimate, and return the largest grid-pair allowance.
  static func admitRotary(configuration c:AVBlockConfiguration,maximumActivationBytes:Int) throws -> Int {
    try c.validate()
    var total=0,largest=0
    for (axes,tokens,width) in [(3,c.videoTokens,c.videoHeadDimension),(1,c.audioTokens,c.audioHeadDimension),
      (1,c.videoTokens,c.audioHeadDimension),(1,c.audioTokens,c.audioHeadDimension)] {
      let bytes=try DenoiserMath.rotaryElementCount(axes:axes,tokens:tokens,heads:c.heads,
        headWidth:width,maximumBytes:maximumActivationBytes)*8
      total += bytes;largest=max(largest,bytes)
    }
    guard total <= maximumActivationBytes else { throw LTXError.invalid("Rotary grids together exceed the admitted transformer workspace.") }
    return largest
  }

  public func evaluate(_ inputs:[String:MLXArray],sigma:Float,videoDenoiseMask:[Float]?=nil,audioDenoiseMask:[Float]?=nil,
    fixedWeights:FixedProvider,blockWeights:BlockProvider,
    fixedAdapters:FixedAdapters = { _ in [] },
    blockAdapters:(Int) throws -> [String:[MLXLoRA]] = { _ in [:] },
    progress:(Progress) throws -> Void = { _ in }) throws -> [String:MLXArray] {
    _ = try DenoiserMath.timestep(sigma)
    let snapshot=inputs.mapValues { $0.reshaped($0.shape) }
    let preparation:Preparation?
    let videoMask=videoDenoiseMask?.count == configuration.videoTokens && videoDenoiseMask?.allSatisfy({ $0 == 1 }) == true ? nil : videoDenoiseMask
    let audioMask=audioDenoiseMask?.count == configuration.audioTokens && audioDenoiseMask?.allSatisfy({ $0 == 1 }) == true ? nil : audioDenoiseMask
    if videoMask != nil || audioMask != nil {
      preparation=try prepare(snapshot,sigmas:[sigma],videoDenoiseMask:videoMask,audioDenoiseMask:audioMask,
        weights:fixedWeights,adapters:fixedAdapters,progress:progress)
    } else { preparation=nil }
    return try evaluatePrepared(snapshot,sigma:sigma,preparation:preparation,fixedWeights:fixedWeights,blockWeights:blockWeights,
      fixedAdapters:fixedAdapters,blockAdapters:blockAdapters,progress:progress)
  }

  /// Internal schedule cache is owned by a single sampling call with immutable
  /// providers. It cannot be reused with a different prompt, position or adapter.
  func evaluatePrepared(_ inputs:[String:MLXArray],sigma:Float,preparation:Preparation?,
    fixedWeights:FixedProvider,blockWeights:BlockProvider,fixedAdapters:FixedAdapters,
    blockAdapters:(Int) throws -> [String:[MLXLoRA]],progress:(Progress) throws -> Void) throws -> [String:MLXArray] {
    guard !active else { throw LTXError.invalid("MLX denoiser is already evaluating.") }
    let time=try DenoiserMath.timestep(sigma)
    let current=try validatedInputs(inputs)
    active=true
    let previousLimit=Memory.cacheLimit
    Memory.cacheLimit=cacheBytes
    defer {
      Stream.gpu.synchronize(); fixedResidentBytes=0
      Memory.clearCache(); Memory.cacheLimit=previousLimit; active=false
    }
    func report(_ stage:String,_ completed:Int=0,_ detail:MLXAVStack.Progress?=nil) throws {
      trimCache()
      try Task.checkCancellation()
      try progress(Progress(stage:stage,completedBlocks:completed,activeBytes:Memory.activeMemory,
        cacheBytes:Memory.cacheMemory,stack:detail))
      try Task.checkCancellation()
    }
    // Validate rotary arithmetic before invoking any weighted provider.
    var prepared:[String:MLXArray], embedded:[String:MLXArray]
    if let preparation {
      let scalarKey=(preparation.rawGlobalTimesteps ? sigma : DenoiserMath.bfloat16(sigma)).bitPattern
      guard let mods=preparation.modulations[scalarKey], let embeds=preparation.embeddings[scalarKey] else {
        throw LTXError.invalid("Sigma was not admitted by the sampling session.")
      }
      prepared=preparation.rotary.merging(mods) { _,new in new }; embedded=embeds
      if let tokenTimes=preparation.videoTimes[sigma.bitPattern] {
        var unique:[Float]=[],lookup:[UInt32:Int32]=[:],indices:[Int32]=[]
        for time in tokenTimes {
          if lookup[time.bitPattern] == nil { lookup[time.bitPattern]=Int32(unique.count);unique.append(time) }
          indices.append(lookup[time.bitPattern]!)
        }
        let index=MLXArray(indices)
        prepared["video_modulation_indices"]=index
        for name in ["video_modulation","video_av_modulation"] {
          // Frozen source audio routes AV modulation through sigma zero. It is
          // constant across video token times, but the compact index table
          // still needs one row per distinct video time for both projections.
          let rows=preparation.frozenAudio && name == "video_av_modulation"
            ? Array(repeating:mods[name]!,count:unique.count)
            : unique.map { preparation.referenceModulations[$0.bitPattern]![name]! }
          let table=concatenated(rows,axis:0)
          prepared[name]=table
        }
        embedded["video"]=take(concatenated(unique.map { preparation.referenceEmbeddings[$0.bitPattern]! },axis:0),index,axis:0)
      }
      if let tokenTimes=preparation.audioTimes[sigma.bitPattern] {
        var unique:[Float]=[],lookup:[UInt32:Int32]=[:],indices:[Int32]=[]
        for time in tokenTimes {
          if lookup[time.bitPattern] == nil { lookup[time.bitPattern]=Int32(unique.count);unique.append(time) }
          indices.append(lookup[time.bitPattern]!)
        }
        let index=MLXArray(indices)
        prepared["audio_modulation_indices"]=index
        for name in ["audio_modulation","audio_av_modulation"] {
          prepared[name]=concatenated(unique.map { preparation.referenceModulations[$0.bitPattern]![name]! },axis:0)
        }
        embedded["audio"]=take(concatenated(unique.map { preparation.referenceAudioEmbeddings[$0.bitPattern]! },axis:0),index,axis:0)
      }
    } else {
      prepared=try rotary(current); embedded=[:]
      for head in DenoiserLayout.heads(configuration) {
        let values=try adaptive(head,time:MLXArray(time,[1,256]),weights:fixedWeights,adapters:fixedAdapters)
        prepared[head.input]=values.parameters
        if head.name == "adaln_single" { embedded["video"]=values.embedded }
        if head.name == "audio_adaln_single" { embedded["audio"]=values.embedded }
        try report(head.name)
      }
    }
    for (name,prefix) in [("video",""),("audio","audio_")] {
      let rounded=current[name+"_latent"]!.asType(.bfloat16).asType(.float32)
      var projection=try linear(prefix+"patchify_proj",rounded,weights:fixedWeights,adapters:fixedAdapters)
      if name == "video" && (keyframeMarkerRows > 0 || leadingKeyframeMarkerRows > 0) {
        let marker=try fixedWeights("keyframes_abs_pos_embedding",[1,configuration.videoDimension])
        guard marker.shape == [1,configuration.videoDimension] else {
          throw LTXError.invalid("Generated keyframe marker shape differs from the checkpoint.")
        }
        let value=try marker.tensor().asType(.float32)
        projection=try Self.applyKeyframeMarkers(projection,marker:value,
          leadingRows:leadingKeyframeMarkerRows,trailingRows:keyframeMarkerRows)
        eval(projection)
        try report("keyframe_marker")
      }
      prepared[name]=projection
      prepared[name+"_text"]=current[name+"_text"]!.asType(.bfloat16).asType(.float32)
      try report(prefix+"patchify_proj")
    }
    prepared["video_attention_templates"]=current["video_attention_templates"]
    let hidden=try stack.evaluate(prepared,weights:blockWeights,adapters:blockAdapters) {
      try report("transformer",$0.completedBlocks,$0)
    }
    prepared.removeAll()
    try report("transformer_released",blockCount)
    var output:[String:MLXArray]=[:]
    for (name,prefix) in [("video",""),("audio","audio_")] {
      let key=prefix+"scale_shift_table"
      let table=try parameter(key,weights:fixedWeights)
      let shift=table[0]+embedded[name]!, scale=table[1]+embedded[name]!
      let normalized=MLXFast.layerNorm(hidden[name]!,eps:1e-6)*(1+scale)+shift
      eval(normalized)
      output[name]=try linear(prefix+"proj_out",normalized,weights:fixedWeights,adapters:fixedAdapters)
      try report(prefix+"proj_out",blockCount)
    }
    return output
  }

  func prepare(_ inputs:[String:MLXArray],sigmas:[Float],videoDenoiseMask:[Float]?=nil,audioDenoiseMask:[Float]?=nil,
    frozenAudio:Bool=false,rawGlobalTimesteps:Bool=false,weights:FixedProvider,adapters:FixedAdapters,
    progress:(Progress) throws -> Void) throws -> Preparation {
    guard !active, (1...256).contains(sigmas.count) else { throw LTXError.invalid("Invalid or active denoising session.") }
    for sigma in sigmas { _ = try DenoiserMath.timestep(sigma) }
    var videoTimes:[UInt32:[Float]]=[:],audioTimes:[UInt32:[Float]]=[:],uniqueSigmas:[Float]=[],seen=Set<UInt32>()
    if let mask=videoDenoiseMask {
      try stack.admitPerTokenVideo()
      guard mask.count == configuration.videoTokens,mask.allSatisfy({ $0.isFinite && $0>=0 && $0<=1 }) else {
        throw LTXError.invalid("Video denoise mask must contain one finite strength per token.")
      }
      // Fail before the first weighted head. Expansion is limited to the active
      // step; the schedule cache retains only distinct scalar modulations.
      for sigma in sigmas { videoTimes[sigma.bitPattern]=mask.map { sigma*$0 } }
    }
    if let mask=audioDenoiseMask {
      guard !frozenAudio else { throw LTXError.invalid("Frozen audio cannot also use a per-token denoise mask.") }
      try stack.admitPerTokenAudio()
      guard mask.count == configuration.audioTokens,mask.allSatisfy({ $0.isFinite && $0>=0 && $0<=1 }) else {
        throw LTXError.invalid("Audio denoise mask must contain one finite strength per token.")
      }
      for sigma in sigmas { audioTimes[sigma.bitPattern]=mask.map { sigma*$0 } }
    }
    for sigma in sigmas {
      let rounded=rawGlobalTimesteps ? sigma : DenoiserMath.bfloat16(sigma)
      if seen.insert(rounded.bitPattern).inserted { uniqueSigmas.append(rounded) }
    }
    if frozenAudio, seen.insert(Float(0).bitPattern).inserted { uniqueSigmas.append(0) }
    let zeroIndex=frozenAudio ? uniqueSigmas.firstIndex(of:0)! : 0
    var referenceSigmas:[Float]=[],referenceSeen=Set<UInt32>()
    for key in videoTimes.keys.sorted() { for sigma in videoTimes[key]! {
      if referenceSeen.insert(sigma.bitPattern).inserted { referenceSigmas.append(sigma) }
    } }
    for key in audioTimes.keys.sorted() { for sigma in audioTimes[key]! {
      if referenceSeen.insert(sigma.bitPattern).inserted { referenceSigmas.append(sigma) }
    } }
    let heads=DenoiserLayout.heads(configuration)
    let elements=heads.reduce(0) { $0+$1.dimension*$1.parameters }+configuration.videoDimension+configuration.audioDimension
    guard elements*(uniqueSigmas.count+referenceSigmas.count)*4 <= 128*1024*1024 else { throw LTXError.invalid("Timestep cache exceeds 128 MiB.") }
    // Global timesteps cross the model's BF16 scalar boundary. Per-token
    // timesteps remain Float32 in the qualified renderer; do not merge them.
    func rawTimestep(_ sigma:Float) -> [Float] {
      let scaled=sigma*1000
      let angles=(0..<128).map { scaled*expf(-Float(log(10000.0))*Float($0)/128) }
      return angles.map(cosf)+angles.map(sinf)
    }
    let times=try uniqueSigmas.flatMap { try rawGlobalTimesteps ? DenoiserMath.authoredTimestep($0) : DenoiserMath.timestep($0) }
      + referenceSigmas.flatMap { try rawGlobalTimesteps ? DenoiserMath.authoredTimestep($0) : rawTimestep($0) }
    let inputs=try validatedInputs(inputs)
    active=true
    let previousLimit=Memory.cacheLimit; Memory.cacheLimit=cacheBytes
    defer {
      Stream.gpu.synchronize(); fixedResidentBytes=0
      Memory.clearCache(); Memory.cacheLimit=previousLimit; active=false
    }
    let positions=try rotary(inputs)
    let time=MLXArray(times,[uniqueSigmas.count+referenceSigmas.count,256])
    var modulations:[UInt32:[String:MLXArray]]=[:], embeddings:[UInt32:[String:MLXArray]]=[:]
    var referenceModulations:[UInt32:[String:MLXArray]]=[:],referenceEmbeddings:[UInt32:MLXArray]=[:]
    var referenceAudioEmbeddings:[UInt32:MLXArray]=[:]
    for head in heads {
      let output=try adaptive(head,time:time,weights:weights,adapters:adapters)
      for (index,sigma) in uniqueSigmas.enumerated() {
        let routed=frozenAudio && ["audio_prompt_modulation","video_av_modulation","video_av_gate"].contains(head.input)
          ? zeroIndex : index
        modulations[sigma.bitPattern,default:[:]][head.input]=output.parameters[routed..<(routed+1)]
        if head.name == "adaln_single" { embeddings[sigma.bitPattern,default:[:]]["video"]=output.embedded[index..<(index+1)] }
        if head.name == "audio_adaln_single" { embeddings[sigma.bitPattern,default:[:]]["audio"]=output.embedded[index..<(index+1)] }
      }
      for (offset,sigma) in referenceSigmas.enumerated() {
        let index=uniqueSigmas.count+offset
        if ["video_modulation","video_av_modulation","audio_modulation","audio_av_modulation"].contains(head.input) {
          referenceModulations[sigma.bitPattern,default:[:]][head.input]=output.parameters[index..<(index+1)]
        }
        if head.name == "adaln_single" { referenceEmbeddings[sigma.bitPattern]=output.embedded[index..<(index+1)] }
        if head.name == "audio_adaln_single" { referenceAudioEmbeddings[sigma.bitPattern]=output.embedded[index..<(index+1)] }
      }
      try Task.checkCancellation(); trimCache()
      try progress(Progress(stage:"schedule_"+head.name,completedBlocks:0,activeBytes:Memory.activeMemory,
        cacheBytes:Memory.cacheMemory,stack:nil))
      try Task.checkCancellation()
    }
    return Preparation(rotary:positions,modulations:modulations,embeddings:embeddings,videoTimes:videoTimes,audioTimes:audioTimes,
      referenceModulations:referenceModulations,referenceEmbeddings:referenceEmbeddings,
      referenceAudioEmbeddings:referenceAudioEmbeddings,frozenAudio:frozenAudio,
      rawGlobalTimesteps:rawGlobalTimesteps)
  }

  func preflightWithoutText(_ inputs:[String:MLXArray],schedule:SamplingSchedule) throws -> [String:MLXArray] {
    let heads=DenoiserLayout.heads(configuration)
    let elements=heads.reduce(0) { $0+$1.dimension*$1.parameters }+configuration.videoDimension+configuration.audioDimension
    guard elements*schedule.steps.count*4 <= 128*1024*1024 else { throw LTXError.invalid("Timestep cache exceeds 128 MiB.") }
    for sigma in schedule.sigmas.dropLast() { _ = try DenoiserMath.timestep(DenoiserMath.bfloat16(Float(sigma))) }
    let result=try validatedInputs(inputs,excludingText:true)
    _ = try rotary(result)
    return result
  }

  private func validatedInputs(_ inputs:[String:MLXArray],excludingText:Bool=false) throws -> [String:MLXArray] {
    try Task.checkCancellation()
    let admitted=excludingText ? inputShapes.filter { !$0.key.hasSuffix("_text") } : inputShapes
    guard Set(inputs.keys) == Set(admitted.keys) else { throw LTXError.invalid("Missing or unsupported denoiser inputs.") }
    var result:[String:MLXArray]=[:]
    for (name,shape) in admitted {
      let value=inputs[name]!
      guard value.dtype == .float32, value.shape == shape || value.shape == [shape.reduce(1,*)] else {
        throw LTXError.invalid("Invalid denoiser input shape or dtype: \(name)")
      }
      guard MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite denoiser input: \(name)") }
      result[name]=value.reshaped(shape)
    }
    return result
  }

  private func trimCache() {
    if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
  }

  /// One dense/Q8 projection, with no CPU expansion of weights or adapter deltas.
  /// Evaluation completes before the locals release their stored parameter data.
  private func linear(_ name:String,_ input:MLXArray,weights:FixedProvider,adapters:FixedAdapters) throws -> MLXArray {
    try Task.checkCancellation()
    defer { fixedResidentBytes=0; trimCache() }
    let key=name+".weight", biasKey=name+".bias"
    guard let shape=shapes[key], let biasShape=shapes[biasKey] else { throw LTXError.invalid("Unknown fixed projection.") }
    let weight=try weights(key,shape)
    try Task.checkCancellation()
    let bias=try weights(biasKey,biasShape)
    try Task.checkCancellation()
    let factors=try adapters(key)
    try Task.checkCancellation()
    guard weight.shape == shape, bias.shape == biasShape else { throw LTXError.invalid("Fixed weight shape mismatch: \(name)") }
    let factorBytes=factors.reduce(0) { $0+$1.down.nbytes+$1.up.nbytes }
    guard factorBytes <= 512*1024*1024, weight.storageBytes+bias.storageBytes+factorBytes <= 1024*1024*1024 else {
      throw LTXError.invalid("Fixed projection and factors exceed admitted storage.")
    }
    fixedResidentBytes=weight.storageBytes+bias.storageBytes+factorBytes
    let output=try weight.projected(input,adapters:factors)+bias.tensor().asType(.float32)
    eval(output)
    try Task.checkCancellation()
    guard MLX.isFinite(output).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite fixed projection output.") }
    return output
  }

  private func parameter(_ name:String,weights:FixedProvider) throws -> MLXArray {
    try Task.checkCancellation()
    let weight=try weights(name,shapes[name]!)
    guard weight.shape == shapes[name] else { throw LTXError.invalid("Fixed table shape mismatch.") }
    let value=try weight.tensor().asType(.float32)
    eval(value)
    return value
  }

  private func adaptive(_ head:DenoiserLayout.Head,time:MLXArray,weights:FixedProvider,adapters:FixedAdapters)
    throws -> (parameters:MLXArray,embedded:MLXArray) {
    let first=try linear(head.name+".emb.timestep_embedder.linear1",time,weights:weights,adapters:adapters)
    let embedded=try linear(head.name+".emb.timestep_embedder.linear2",first*sigmoid(first),weights:weights,adapters:adapters)
    let parameters=try linear(head.name+".linear",embedded*sigmoid(embedded),weights:weights,adapters:adapters)
    return (parameters,embedded)
  }

  func rotary(_ inputs:[String:MLXArray]) throws -> [String:MLXArray] {
    let c=configuration
    let video=inputs["video_positions"]!.asArray(Float.self), audio=inputs["audio_positions"]!.asArray(Float.self)
    let temporal=stride(from:0,to:video.count,by:3).map { video[$0] }
    var result:[String:MLXArray]=[:]
    for (name,positions,axes,tokens,width,maximum):(String,[Float],Int,Int,Int,[Float]) in [
      ("video",video,3,c.videoTokens,c.videoHeadDimension,[20,2048,2048]),
      ("audio",audio,1,c.audioTokens,c.audioHeadDimension,[20]),
      ("video_cross",temporal,1,c.videoTokens,c.audioHeadDimension,[20]),
      ("audio_cross",audio,1,c.audioTokens,c.audioHeadDimension,[20])] {
      let values=try DenoiserMath.rotary(positions:positions,axes:axes,tokens:tokens,heads:c.heads,
        headWidth:width,maximumPositions:maximum,maximumBytes:maximumRotaryBytes)
      let shape=[tokens*c.heads,width/2]
      result[name+"_rope_cos"]=MLXArray(values.cos,shape)
      result[name+"_rope_sin"]=MLXArray(values.sin,shape)
    }
    return result
  }
}
