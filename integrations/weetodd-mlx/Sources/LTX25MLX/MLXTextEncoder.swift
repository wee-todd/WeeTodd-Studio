import Foundation
import MLX
import LTX25Text
import TensorIO

/// Conservative owned-array admission, not a cap on driver/VM process footprint.
public struct MLXTextEncodingPlan:Sendable {
  public let interleavedBytes:Int
  public let ownedBufferBytes:Int
  public init(promptTokens n:Int,maximumOwnedBufferBytes:Int=3*1024*1024*1024) throws {
    guard (1...1024).contains(n), maximumOwnedBufferBytes > 0 else { throw TextEncodingError.invalid("Invalid text memory request.") }
    interleavedBytes=n*3840*49*4
    let metadata=512*1024*1024, slab=64*1024*1024
    let gemmaActivations=(n*3840*12+n*15360*5+n*16*512*8+n*8*256*8+16*n*n*3)*4
    let gemma=interleavedBytes+gemmaActivations+384*1024*1024
    // Both normalized source states and the stacked/interleaved destination may
    // coexist during assembly. Aggregate weights stage one packed row slab.
    let aggregate=2*interleavedBytes+3*slab+n*6144*4*3
    let connector=(1024*4096*48+32*1024*1024*3)*4+512*1024*1024+1024*6144*4
    ownedBufferBytes=metadata+max(gemma,aggregate,connector)
    guard ownedBufferBytes <= maximumOwnedBufferBytes else {
      throw TextEncodingError.invalid("MLX text needs an owned-buffer budget of at least \(ownedBufferBytes) bytes.")
    }
  }
}

/// Trained Gemma4-12B and LTX2.5 audiovisual connectors. No Python calls, new
/// checkpoints, CPU hidden-state downloads or full aggregation-matrix loads.
public final class MLXTextEncoder {
  public struct Progress {
    public let stage:String
    public let completed:Int
    public let total:Int
    public let activeBytes:Int
    public let cacheBytes:Int
  }
  public struct Output {
    public let video:MLXArray
    public let audio:MLXArray
    public let tokenIDs:[Int]
  }
  private let checkpoint:TextCheckpoint
  private let connectorURL:URL
  private let nativeWeightLoading:Bool
  private let gate=NSLock()
  private let cacheBytes=128*1024*1024
  public init(gemmaRoot:URL,connectorURL:URL,nativeWeightLoading:Bool=true) throws {
    checkpoint=try TextCheckpoint(gemmaRoot:gemmaRoot,connectorURL:connectorURL)
    self.connectorURL=connectorURL;self.nativeWeightLoading=nativeWeightLoading
  }
  public func tokenize(_ prompt:String,maxLength:Int=1024) throws -> [Int] {
    try checkpoint.tokenizer.encode(prompt,maxLength:maxLength)
  }
  public func encode(prompt:String,maxLength:Int=1024,maximumOwnedBufferBytes:Int=3*1024*1024*1024,
    progress:(Progress) throws -> Void = { _ in }) throws -> Output {
    guard gate.try() else { throw TextEncodingError.invalid("MLX text encoder is already active.") }
    defer { gate.unlock() }
    let ids=try tokenize(prompt,maxLength:maxLength)
    _ = try MLXTextEncodingPlan(promptTokens:ids.count,maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    guard Device.defaultDevice().deviceType == .gpu else { throw TextEncodingError.invalid("MLX text requires a Metal GPU.") }
    try Task.checkCancellation()
    let oldLimit=Memory.cacheLimit; Memory.cacheLimit=cacheBytes
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit=oldLimit }
    func report(_ stage:String,_ completed:Int,_ total:Int) throws {
      if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
      try Task.checkCancellation()
      try progress(Progress(stage:stage,completed:completed,total:total,activeBytes:Memory.activeMemory,cacheBytes:Memory.cacheMemory))
      try Task.checkCancellation()
    }
    let projections=try project(ids:ids,progress:report)
    let video=try connect(projections.0,promptTokens:ids.count,modality:"video",width:4096,progress:report)
    let audio=try connect(projections.1,promptTokens:ids.count,modality:"audio",width:2048,progress:report)
    try Task.checkCancellation()
    return Output(video:video,audio:audio,tokenIDs:ids)
  }

  private func project(ids:[Int],progress:(String,Int,Int) throws -> Void) throws -> (MLXArray,MLXArray) {
    var rows:[MLXArray]=[]
    for id in ids {
      try Task.checkCancellation()
      guard (0..<262144).contains(id) else { throw TextEncodingError.invalid("Token ID outside Gemma vocabulary.") }
      let row=try MLXWeight(file:checkpoint.fixed,name:"model.embed_tokens.weight",shape:[262144,3840],rows:id..<(id+1))
      rows.append(try row.tensor().asType(.float32))
    }
    var hidden:MLXArray?=concatenated(rows,axis:0)*Float(3840).squareRoot()
    try MLXTextMath.finish(hidden!); rows.removeAll()
    var states:[MLXArray]=[]
    func append(_ value:MLXArray) throws {
      let normalized=MLXTextMath.rms(value)
      try MLXTextMath.finish(normalized); states.append(normalized)
    }
    try append(hidden!)
    for index in checkpoint.layers.indices {
      let file=checkpoint.layers[index], prefix="model.layers.\(index)."
      let source=nativeWeightLoading ? MLXNativeWeightSource(file:file,url:checkpoint.layerURLs[index]) : nil
      hidden=try MLXGemmaLayer.evaluate(hidden!,configuration:checkpoint.configuration(forLayer:index)) {
        if let source { return try source.read(prefix+$0,$1) }
        return try MLXWeight(file:file,name:prefix+$0,shape:$1)
      }
      try append(hidden!)
      try progress("gemma",index+1,48)
    }
    hidden=nil
    // Feature-major, then layer index, exactly as the trained aggregate expects.
    let interleaved=stacked(states,axis:-1).reshaped([ids.count,188160])
    try MLXTextMath.finish(interleaved); states.removeAll(); Memory.clearCache()
    let video=try aggregate(interleaved,modality:"video",width:4096)
    try progress("aggregation",1,2)
    let audio=try aggregate(interleaved,modality:"audio",width:2048)
    try progress("aggregation",2,2)
    return (video,audio)
  }

  private func aggregate(_ input:MLXArray,modality:String,width:Int) throws -> MLXArray {
    let name="text_embedding_projection.\(modality)_aggregate_embed"
    let record=checkpoint.fixed.tensors[name+".weight"]!
    // Include the packed Q8 companion tensors when choosing row counts.
    var rowBytes=record.byteCount/UInt64(width)
    if record.dtype == "U32" {
      for tail in [".scales",".biases"] { rowBytes += checkpoint.fixed.tensors[name+tail]!.byteCount/UInt64(width) }
    }
    let chunk=max(1,Int(64*1024*1024/rowBytes))
    var parts:[MLXArray]=[]
    for start in stride(from:0,to:width,by:chunk) {
      try Task.checkCancellation()
      let output=try autoreleasepool {
        let weight=try MLXWeight(file:checkpoint.fixed,name:name+".weight",shape:[width,188160],rows:start..<min(start+chunk,width))
        let projected=try weight.projected(input)
        try MLXTextMath.finish(projected)
        return projected
      }
      parts.append(output)
      if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
    }
    let bias=try MLXWeight.read(checkpoint.fixed,name+".bias").asType(.float32)
    let output=concatenated(parts,axis:1)*sqrt(Float(width)/3840)+bias
    try MLXTextMath.finish(output)
    return output
  }

  private func connect(_ projection:MLXArray,promptTokens:Int,modality:String,width:Int,
    progress:(String,Int,Int) throws -> Void) throws -> MLXArray {
    let stem=checkpoint.connectorPrefix+modality+"_embeddings_connector."
    let source=nativeWeightLoading ? MLXNativeWeightSource(file:checkpoint.connector,url:connectorURL) : nil
    defer { source?.clear() }
    let registers=try MLXWeight.read(checkpoint.connector,stem+"learnable_registers").asType(.float32)
    let indices=MLXArray((promptTokens..<1024).map { Int32($0%128) })
    var hidden=promptTokens < 1024 ? concatenated([projection,registers[indices]],axis:0) : projection.reshaped(projection.shape)
    try MLXTextMath.finish(hidden)
    for index in 0..<8 {
      hidden=try MLXTextConnector.evaluate(hidden,width:width) {
        let name=stem+"transformer_1d_blocks.\(index)."+$0
        if let source { return try source.read(name,$1) }
        return try MLXWeight(file:self.checkpoint.connector,name:name,shape:$1)
      }
      try progress(modality+"_connector",index+1,8)
    }
    let result=MLXTextMath.rms(hidden)
    try MLXTextMath.finish(result)
    return result
  }
}
