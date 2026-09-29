import Foundation
import TensorIO

/// Preflights all 48 transformer block pages before allocating the GPU graph.
/// Keeps headers/file handles only. Fixed/top-level components are outside this
/// stack contract. Declared SHA-256 values are not verified by header inspection.
public final class PagedBlockWeights {
  private struct Page: Decodable {
    let file: String
    let tensorCount: Int
    let tensorBytes: UInt64
    let sha256: String
  }
  private struct Architecture: Decodable {
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
      guard numLayers == 48, numAttentionHeads == c.heads, audioNumAttentionHeads == c.heads,
        attentionHeadDim == c.videoHeadDimension, audioAttentionHeadDim == c.audioHeadDimension,
        crossAttentionDim == c.videoDimension, audioCrossAttentionDim == c.audioDimension,
        !ffBias, audioFfBias ?? true, applyGatedAttention, crossAttentionAdaln,
        useAudioVideoCrossAttention, ropeType == "split", qkNorm == "rms_norm", normEps == 1e-6,
        activationFn == "gelu-approximate", attentionBias, !doubleSelfAttention, !onlyCrossAttention,
        !shareFf, avCrossAdaNorm, !normElementwiseAffine,
        (attentionType ?? "default") == "default", (dropout ?? 0) == 0 else {
        throw BlockError.invalid("Paged LTX architecture differs from the qualified audiovisual block.")
      }
    }
  }
  private struct Metadata: Decodable {
    struct Config: Decodable { let transformer: Architecture }
    let config: Config
  }
  private struct Manifest: Decodable {
    let format: String
    let kind: String
    let numLayers: Int
    let bits: Int
    let groupSize: Int
    let layers: [Page]
    let metadata: Metadata
  }
  private let sources: [BlockWeightSource]
  public var blockCount: Int { sources.count }
  public let decodedSlotBytes: UInt64
  public let largestDecodedMatrixBytes: UInt64

  public init(root: URL, configuration: AVBlockConfiguration) throws {
    guard root.isFileURL else { throw BlockError.invalid("Paged checkpoints must be local.") }
    try configuration.validate()
    let root = root.resolvingSymlinksInPath().standardizedFileURL
    let manifestURL = root.appendingPathComponent("paged_manifest.json")
    let size = try manifestURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard size.isRegularFile == true, let bytes = size.fileSize, bytes <= 1024 * 1024 else {
      throw BlockError.invalid("Paged manifest must be a regular file of at most 1 MiB.")
    }
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let manifest = try decoder.decode(Manifest.self, from: Data(contentsOf: manifestURL))
    guard manifest.format == "weetodd-ltx25-transformer-paged-q8-v1", manifest.kind == "transformer",
      manifest.numLayers == 48, manifest.layers.count == 48, manifest.bits == 8, manifest.groupSize == 64 else {
      throw BlockError.invalid("Expected the complete 48-block group-64 Q8 transformer manifest.")
    }
    try manifest.metadata.config.transformer.validate(configuration)
    let shapes = try AVBlockRunner.expectedWeightShapes(configuration: configuration)
    var sources: [BlockWeightSource] = [], paths: Set<String> = []
    for (index, record) in manifest.layers.enumerated() {
      try Task.checkCancellation()
      let components = record.file.split(separator: "/", omittingEmptySubsequences: false)
      guard !record.file.hasPrefix("/"), !components.isEmpty,
        components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
        record.sha256.count == 64, record.sha256.allSatisfy(\.isHexDigit),
        (1...256).contains(record.tensorCount), record.tensorBytes > 0 else {
        throw BlockError.invalid("Invalid page record at block \(index).")
      }
      let url = root.appendingPathComponent(record.file).resolvingSymlinksInPath().standardizedFileURL
      guard url.path.hasPrefix(root.path + "/"), paths.insert(url.path).inserted else {
        throw BlockError.invalid("Paged checkpoint repeats a file or escapes its root.")
      }
      let source = try BlockWeightSource(url: url, blockIndex: index, expectedShapes: shapes, requireSingleBlock: true)
      guard source.tensorCount == record.tensorCount, source.tensorPayloadBytes == record.tensorBytes else {
        throw BlockError.invalid("Page \(index) header differs from its manifest.")
      }
      sources.append(source)
    }
    self.sources = sources
    decodedSlotBytes = sources.map(\.decodedWeightBytes).max()!
    largestDecodedMatrixBytes = sources.map(\.largestDecodedTensorBytes).max()!
  }

  public func read(block: Int, name: String, shape: [Int], decoding: Q8Decoding = .simd) throws -> [Float] {
    guard sources.indices.contains(block) else { throw BlockError.invalid("Invalid transformer block index.") }
    return try sources[block].read(name, shape: shape, decoding: decoding)
  }
}
