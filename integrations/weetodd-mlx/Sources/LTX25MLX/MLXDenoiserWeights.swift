import Foundation
import LTX25Engine
import AdapterRuntime

/// Header-only preflight of the complete installed paged transformer. Original
/// model files are reused in place. SHA strings are syntax checked, not rehashed;
/// SafeTensorFile checks source identity again before reading each tensor.
public final class MLXDenoiserWeights {
  private struct Page:Decodable {
    let file:String
    let tensorCount:Int
    let tensorBytes:UInt64
    let sha256:String?
  }
  private struct Architecture: Decodable {
    let timestepScaleMultiplier: Double
    let avCaTimestepScaleMultiplier: Double
    let positionalEmbeddingTheta: Double
    let frequenciesPrecision: String
    let positionalEmbeddingMaxPos: [Int]
    let audioPositionalEmbeddingMaxPos: [Int]
    let inChannels: Int
    let outChannels: Int
    let audioOutChannels: Int
    let numLayers: Int
    let numAttentionHeads: Int
    let audioNumAttentionHeads: Int
    let attentionHeadDim: Int
    let audioAttentionHeadDim: Int
    let crossAttentionDim: Int
    let audioCrossAttentionDim: Int
    let ffBias: Bool
    let audioFfBias: Bool?
    let applyGatedAttention: Bool
    let crossAttentionAdaln: Bool
    let useAudioVideoCrossAttention: Bool
    let ropeType: String
    let qkNorm: String
    let normEps: Double
    let activationFn: String
    let attentionBias: Bool
    let doubleSelfAttention: Bool
    let onlyCrossAttention: Bool
    let shareFf: Bool
    let avCrossAdaNorm: Bool
    let normElementwiseAffine: Bool
    let attentionType: String?
    let dropout: Double?

    func validate(_ c: AVBlockConfiguration) throws {
      guard timestepScaleMultiplier == 1000, avCaTimestepScaleMultiplier == 1000,
        positionalEmbeddingTheta == 10000, frequenciesPrecision == "float64",
        positionalEmbeddingMaxPos == [20,2048,2048], audioPositionalEmbeddingMaxPos == [20],
        inChannels == 128, outChannels == 128, audioOutChannels == 128,
        numLayers == 48, numAttentionHeads == c.heads, audioNumAttentionHeads == c.heads,
        attentionHeadDim == c.videoHeadDimension, audioAttentionHeadDim == c.audioHeadDimension,
        crossAttentionDim == c.videoDimension, audioCrossAttentionDim == c.audioDimension,
        !ffBias, audioFfBias ?? true, applyGatedAttention, crossAttentionAdaln,
        useAudioVideoCrossAttention, ropeType == "split", qkNorm == "rms_norm", normEps == 1e-6,
        activationFn == "gelu-approximate", attentionBias, !doubleSelfAttention, !onlyCrossAttention,
        !shareFf, avCrossAdaNorm, !normElementwiseAffine,
        (attentionType ?? "default") == "default", (dropout ?? 0) == 0 else {
        throw LTXError.invalid("Paged LTX architecture differs from the qualified audiovisual block.")
      }
    }
  }
  private struct Metadata:Decodable {
    struct Config:Decodable { let transformer:Architecture }
    let config:Config
  }
  private struct Manifest:Decodable {
    let source:String?
    let format:String
    let kind:String
    let numLayers:Int
    let bits:Int
    let groupSize:Int
    let fixed:Page
    let layers:[Page]
    let metadata:Metadata
  }
  private let fixed:MLXFixedSource
  private let pages:[MLXBlockSource]
  private let adapters:MLXLoRAStack
  public let sourceCheckpoint:String?
  public let requiresKeyframeMarker:Bool
  public var blockCount:Int { pages.count }
  public var largestBlockBytes:Int { pages.map(\.storageBytes).max() ?? 0 }
  public init(root:URL,configuration:AVBlockConfiguration,adapters:[LoRAAdapter]=[],
    unionControlAdapterPath:String?=nil,
    ingredientsAdapterPath:String?=nil,
    msrAdapterPath:String?=nil,
    maximumActivationBytes:Int=2*1024*1024*1024,requireKeyframeMarker:Bool=false,
    pixelSpatialDFRAdapterPath:String?=nil,icControlFamilies:[String:String]=[:]) throws {
    try configuration.validate()
    guard root.isFileURL else { throw LTXError.invalid("Paged transformer must be local.") }
    let root=root.resolvingSymlinksInPath().standardizedFileURL
    let manifestURL=root.appendingPathComponent("paged_manifest.json")
    let resource=try manifestURL.resourceValues(forKeys:[.fileSizeKey,.isRegularFileKey])
    guard resource.isRegularFile == true, let size=resource.fileSize, size <= 1024*1024 else {
      throw LTXError.invalid("Paged manifest must be a regular file of at most 1 MiB.")
    }
    let decoder=JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let manifest=try decoder.decode(Manifest.self,from:Data(contentsOf:manifestURL))
    guard manifest.format == "weetodd-ltx25-transformer-paged-q8-v1", manifest.kind == "transformer",
      manifest.numLayers == 48, manifest.layers.count == 48, manifest.bits == 8, manifest.groupSize == 64 else {
      throw LTXError.invalid("Expected all 48 group-64 Q8 transformer pages.")
    }
    try manifest.metadata.config.transformer.validate(configuration)
    var visited:Set<String>=[]
    func location(_ record:Page,requireHash:Bool) throws -> URL {
      let components=record.file.split(separator:"/",omittingEmptySubsequences:false)
      guard !record.file.hasPrefix("/"), !components.isEmpty,
        components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
        (1...1024).contains(record.tensorCount), record.tensorBytes > 0,
        !requireHash || record.sha256 != nil,
        record.sha256.map({ $0.count == 64 && $0.allSatisfy(\.isHexDigit) }) ?? true else {
        throw LTXError.invalid("Invalid checkpoint manifest record.")
      }
      let url=root.appendingPathComponent(record.file).resolvingSymlinksInPath().standardizedFileURL
      guard url.path.hasPrefix(root.path+"/"), visited.insert(url.path).inserted else {
        throw LTXError.invalid("Checkpoint repeats a file or escapes its root.")
      }
      return url
    }
    let fixed=try MLXFixedSource(url:location(manifest.fixed,requireHash:false),configuration:configuration,
      requireKeyframeMarker:requireKeyframeMarker)
    guard fixed.tensorCount == manifest.fixed.tensorCount, fixed.tensorBytes == manifest.fixed.tensorBytes else {
      throw LTXError.invalid("Fixed header differs from manifest.")
    }
    let shapes=try MLXAVBlock(configuration:configuration,maximumActivationBytes:maximumActivationBytes).weightShapes
    var pages:[MLXBlockSource]=[]
    for (index,record) in manifest.layers.enumerated() {
      try Task.checkCancellation()
      let page=try MLXBlockSource(url:location(record,requireHash:true),blockIndex:index,expectedShapes:shapes)
      guard page.tensorCount == record.tensorCount, page.tensorBytes == record.tensorBytes else {
        throw LTXError.invalid("Block header differs from manifest at index \(index).")
      }
      pages.append(page)
    }
    let stack=try MLXLoRAStack(adapters:adapters,
      unionControlAdapterPath:unionControlAdapterPath,ingredientsAdapterPath:ingredientsAdapterPath,
      msrAdapterPath:msrAdapterPath,pixelSpatialDFRAdapterPath:pixelSpatialDFRAdapterPath,
      icControlFamilies:icControlFamilies)
    var targets=Dictionary(uniqueKeysWithValues:DenoiserLayout.weightShapes(configuration)
      .filter { $0.key.hasSuffix(".weight") }.map { (String($0.key.dropLast(7)),$0.value) })
    for index in 0..<48 {
      for (name,shape) in shapes where name.hasSuffix(".weight") {
        targets["transformer_blocks.\(index)."+String(name.dropLast(7))]=shape
      }
    }
    try stack.validateTargets(targets)
    self.fixed=fixed; self.pages=pages; self.adapters=stack; sourceCheckpoint=manifest.source
    self.requiresKeyframeMarker=requireKeyframeMarker
  }
  public func readFixed(_ name:String,_ shape:[Int]) throws -> MLXWeight {
    try fixed.read(name,shape:shape)
  }
  public func readBlock(_ index:Int,_ name:String,_ shape:[Int]) throws -> MLXWeight {
    guard pages.indices.contains(index) else { throw LTXError.invalid("Invalid block index.") }
    return try pages[index].read(name,shape:shape,deferEvaluation:true)
  }
  public func fixedAdapters(_ name:String) throws -> [MLXLoRA] { try adapters.loadFixed(name) }
  public func blockAdapters(_ index:Int) throws -> [String:[MLXLoRA]] { try adapters.load(block:index) }
}
