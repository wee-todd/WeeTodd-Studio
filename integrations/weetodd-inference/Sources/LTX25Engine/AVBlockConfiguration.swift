import Foundation

/// One batch, uniform timestep, split-RoPE joint audiovisual block. Other modes
/// require separate qualification rather than silently falling back to this graph.
public struct AVBlockConfiguration: Codable, Sendable {
  public let videoDimension: Int
  public let audioDimension: Int
  public let heads: Int
  public let videoHeadDimension: Int
  public let audioHeadDimension: Int
  public let videoTokens: Int
  public let audioTokens: Int
  public let textTokens: Int

  private enum CodingKeys: String, CodingKey, CaseIterable {
    case videoDimension, audioDimension, heads, videoHeadDimension, audioHeadDimension
    case videoTokens, audioTokens, textTokens
  }
  private struct AnyKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(intValue: Int) { return nil }
    init(stringValue: String) { self.stringValue = stringValue }
  }

  public init(from decoder: Decoder) throws {
    let all = try decoder.container(keyedBy: AnyKey.self)
    guard Set(all.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Block configuration has missing or unsupported execution controls.")
    }
    let values = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(videoDimension: values.decode(Int.self, forKey: .videoDimension),
      audioDimension: values.decode(Int.self, forKey: .audioDimension),
      heads: values.decode(Int.self, forKey: .heads),
      videoHeadDimension: values.decode(Int.self, forKey: .videoHeadDimension),
      audioHeadDimension: values.decode(Int.self, forKey: .audioHeadDimension),
      videoTokens: values.decode(Int.self, forKey: .videoTokens),
      audioTokens: values.decode(Int.self, forKey: .audioTokens),
      textTokens: values.decode(Int.self, forKey: .textTokens))
  }

  public init(videoDimension: Int = 4096, audioDimension: Int = 2048, heads: Int = 32,
    videoHeadDimension: Int = 128, audioHeadDimension: Int = 64,
    videoTokens: Int, audioTokens: Int, textTokens: Int) throws {
    self.videoDimension = videoDimension; self.audioDimension = audioDimension
    self.heads = heads; self.videoHeadDimension = videoHeadDimension
    self.audioHeadDimension = audioHeadDimension; self.videoTokens = videoTokens
    self.audioTokens = audioTokens; self.textTokens = textTokens
    try validate()
  }

  public func validate() throws {
    guard (1...128).contains(heads), (2...256).contains(videoHeadDimension),
          (2...256).contains(audioHeadDimension), videoHeadDimension % 2 == 0,
          audioHeadDimension % 2 == 0, videoDimension == heads * videoHeadDimension,
          audioDimension == heads * audioHeadDimension, videoDimension <= 8192,
          audioDimension <= 8192, (1...131072).contains(videoTokens),
          (1...131072).contains(audioTokens), (1...8192).contains(textTokens) else {
      throw LTXError.invalid("LTX block dimensions, token counts or head layout are unsupported.")
    }
  }
}
