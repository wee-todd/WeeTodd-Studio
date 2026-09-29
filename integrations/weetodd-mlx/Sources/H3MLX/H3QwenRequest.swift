import Foundation

/// The Qwen presentation H3 consumes. Visual features replace image-pad rows
/// later; the vision start/end rows retain learned token embeddings. The tags
/// are H3 modality tags, not Qwen tokenizer type IDs.
public struct H3QwenRequest: Sendable {
  public struct Grid: Sendable {
    public let temporal: Int
    public let height: Int
    public let width: Int

    public init(temporal: Int, height: Int, width: Int) {
      self.temporal = temporal
      self.height = height
      self.width = width
    }
  }

  public struct VideoBlock: Sendable {
    public let timestampSeconds: Double
    public let grid: Grid

    public init(timestampSeconds: Double, grid: Grid) {
      self.timestampSeconds = timestampSeconds
      self.grid = grid
    }
  }

  /// Qwen presentation metadata in the same order as Ref2VA latent blocks.
  /// Media bytes are prepared and encoded separately from this token contract.
  public enum Reference: Sendable {
    case image(grid: Grid)
    case video(blocks: [VideoBlock], hasAudio: Bool)
    case audio
  }

  public let tokenIDs: [Int32]
  public let tags: [Int32]
  public let visualRanges: [Range<Int>]

  public static func text(_ prompt: String,
    tokenizer: H3QwenTokenizer) throws -> Self {
    try keyframes(prompt: prompt, grids: [], tokenizer: tokenizer)
  }

  public static func references(prompt: String, references: [Reference],
    tokenizer: H3QwenTokenizer) throws -> Self {
    guard (1...12).contains(references.count) else {
      throw H3CheckpointError.invalid("H3 Qwen requires one to twelve ordered references.")
    }
    var ids: [Int32] = []
    var tags: [Int32] = []
    var ranges: [Range<Int>] = []
    var imageCount = 0
    var videoCount = 0
    var audioCount = 0
    var visualPads = 0
    func appendText(_ value: String) throws {
      let encoded = try tokenizer.encode(value)
      ids.append(contentsOf: encoded)
      tags.append(contentsOf: repeatElement(1, count: encoded.count))
    }
    func appendVision(_ grid: Grid, padToken: String) throws {
      guard (1...16).contains(grid.temporal), (2...128).contains(grid.height),
        (2...128).contains(grid.width), grid.height.isMultiple(of: 2),
        grid.width.isMultiple(of: 2) else {
        throw H3CheckpointError.invalid("Invalid Ref2VA Qwen visual grid.")
      }
      let pads = grid.temporal * (grid.height / 2) * (grid.width / 2)
      visualPads += pads
      guard pads > 0, visualPads <= 1024 else {
        throw H3CheckpointError.invalid("Ref2VA Qwen visual pads exceed the token window.")
      }
      let start = try tokenizer.encode("<|vision_start|>")
      let pad = try tokenizer.encode(padToken)
      let end = try tokenizer.encode("<|vision_end|>")
      guard start.count == 1, pad.count == 1, end.count == 1 else {
        throw H3CheckpointError.invalid("H3 Qwen tokenizer lacks a reference vision token.")
      }
      let lower = ids.count
      ids.append(start[0])
      ids.append(contentsOf: repeatElement(pad[0], count: pads))
      ids.append(end[0])
      tags.append(contentsOf: repeatElement(0, count: pads + 2))
      ranges.append(lower..<ids.count)
    }
    for reference in references {
      switch reference {
      case .image(let grid):
        imageCount += 1
        try appendText("<Picture \(imageCount)>: ")
        try appendVision(grid, padToken: "<|image_pad|>")
      case .video(let blocks, let hasAudio):
        videoCount += 1
        if hasAudio {
          audioCount += 1
          try appendText("<Audio \(audioCount)>: ")
        }
        guard (1...16).contains(blocks.count),
          blocks.allSatisfy({ $0.timestampSeconds.isFinite && $0.timestampSeconds >= 0 }),
          zip(blocks, blocks.dropFirst()).allSatisfy({
            $0.timestampSeconds < $1.timestampSeconds
          }) else {
          throw H3CheckpointError.invalid("Ref2VA video blocks require ordered finite timestamps.")
        }
        try appendText("<Video \(videoCount)>: ")
        for block in blocks {
          guard block.grid.temporal == 1 else {
            throw H3CheckpointError.invalid("Ref2VA Qwen video block must contain one frame pair.")
          }
          let timestamp = String(format: "<%.1f seconds>",
            locale: Locale(identifier: "en_US_POSIX"), block.timestampSeconds)
          try appendText(timestamp)
          try appendVision(block.grid, padToken: "<|video_pad|>")
        }
      case .audio:
        audioCount += 1
        try appendText("<Audio \(audioCount)>: ")
      }
    }
    guard imageCount <= 9, videoCount <= 3, audioCount <= 3 else {
      throw H3CheckpointError.invalid("Ref2VA Qwen reference counts exceed model limits.")
    }
    try appendText(prompt)
    guard (1...1024).contains(ids.count), ids.count == tags.count,
      ids.allSatisfy({ (0..<151936).contains($0) }) else {
      throw H3CheckpointError.invalid("H3 Qwen reference request exceeds 1024 valid token rows.")
    }
    return Self(tokenIDs: ids, tags: tags, visualRanges: ranges)
  }

  /// Build the FL2VA keyframe presentation from already validated processor
  /// grids. Each 2×2 patch merge produces one image-pad row.
  public static func keyframes(prompt: String, grids: [Grid],
    tokenizer: H3QwenTokenizer) throws -> Self {
    var ids: [Int32] = []
    var tags: [Int32] = []
    var ranges: [Range<Int>] = []
    func appendText(_ text: String) throws {
      let encoded = try tokenizer.encode(text)
      ids.append(contentsOf: encoded)
      tags.append(contentsOf: repeatElement(1, count: encoded.count))
    }
    for (index, grid) in grids.enumerated() {
      guard (1...16).contains(grid.temporal), (2...128).contains(grid.height),
        (2...128).contains(grid.width), grid.height.isMultiple(of: 2),
        grid.width.isMultiple(of: 2) else {
        throw H3CheckpointError.invalid("H3 Qwen visual grid must have positive bounded temporal and even spatial dimensions.")
      }
      let pads = grid.temporal * (grid.height / 2) * (grid.width / 2)
      guard pads > 0, pads <= 1024 else {
        throw H3CheckpointError.invalid("H3 Qwen visual grid exceeds its token window.")
      }
      try appendText("<Picture \(index + 1)>: ")
      let start = try tokenizer.encode("<|vision_start|>")
      let pad = try tokenizer.encode("<|image_pad|>")
      let end = try tokenizer.encode("<|vision_end|>")
      guard start.count == 1, pad.count == 1, end.count == 1 else {
        throw H3CheckpointError.invalid("H3 Qwen tokenizer lacks required vision tokens.")
      }
      let lower = ids.count
      ids.append(start[0])
      ids.append(contentsOf: repeatElement(pad[0], count: pads))
      ids.append(end[0])
      tags.append(contentsOf: repeatElement(0, count: pads + 2))
      ranges.append(lower..<ids.count)
    }
    try appendText(prompt)
    guard (1...1024).contains(ids.count), ids.count == tags.count,
      ids.allSatisfy({ (0..<151936).contains($0) }) else {
      throw H3CheckpointError.invalid("H3 Qwen request exceeds 1024 valid token rows.")
    }
    return Self(tokenIDs: ids, tags: tags, visualRanges: ranges)
  }
}
