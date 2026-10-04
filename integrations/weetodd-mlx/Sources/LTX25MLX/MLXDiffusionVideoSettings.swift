import Foundation
import LTX25Engine

/// Optional orthogonal decode controls. Absence preserves old serialized
/// requests and convolutional defaults; checkpoint class selects the decoder.
public struct MLXDiffusionVideoSettings:Codable,Sendable,Equatable {
  public let optimization:MLXDiffusionVideoOptimization
  public let queryChunkSize:Int
  public let contextWidthChunks:Int
  public let stage4TileWidth:Int
  public init(optimization:MLXDiffusionVideoOptimization = .combined,queryChunkSize:Int=512,
    contextWidthChunks:Int=4,stage4TileWidth:Int=0) throws {
    _=try MLXDiffusionVideoOptions(optimization:optimization,queryChunkSize:queryChunkSize,
      contextWidthChunks:contextWidthChunks,stage4TileWidth:stage4TileWidth)
    self.optimization=optimization;self.queryChunkSize=queryChunkSize
    self.contextWidthChunks=contextWidthChunks;self.stage4TileWidth=stage4TileWidth
  }
  private struct Key:CodingKey {
    let stringValue:String
    var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { return nil }
  }
  enum CodingKeys:String,CodingKey,CaseIterable {
    case optimization,queryChunkSize="query_chunk_size",contextWidthChunks="context_width_chunks",stage4TileWidth="stage4_tile_width"
  }
  public init(from decoder:Decoder) throws {
    let all=try decoder.container(keyedBy:Key.self)
    guard Set(all.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Diffusion VAE settings require exactly four recognized fields.")
    }
    let c=try decoder.container(keyedBy:CodingKeys.self)
    try self.init(optimization:c.decode(MLXDiffusionVideoOptimization.self,forKey:.optimization),
      queryChunkSize:c.decode(Int.self,forKey:.queryChunkSize),contextWidthChunks:c.decode(Int.self,forKey:.contextWidthChunks),
      stage4TileWidth:c.decode(Int.self,forKey:.stage4TileWidth))
  }
  public var isDefault:Bool { optimization == .combined && queryChunkSize == 512 && contextWidthChunks == 4 && stage4TileWidth == 0 }
  public func options(maximumWorkspaceBytes:Int) throws -> MLXDiffusionVideoOptions {
    try MLXDiffusionVideoOptions(optimization:optimization,queryChunkSize:queryChunkSize,
      contextWidthChunks:contextWidthChunks,stage4TileWidth:stage4TileWidth,maximumWorkspaceBytes:maximumWorkspaceBytes)
  }
}
