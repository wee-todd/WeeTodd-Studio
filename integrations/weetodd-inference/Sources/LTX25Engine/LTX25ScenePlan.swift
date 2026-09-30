import Foundation

/// Exact delivered-frame and audiovisual overlap contract for a native scene.
/// Cumulative quantization keeps rounding errors from accumulating per shot.
public struct LTX25ScenePlan: Sendable, Equatable {
  public let segmentFrames: [Int]
  public let segmentStarts: [Int]
  public let windowFrames: [Int]
  public let windowStarts: [Int]
  public let totalFrames: Int
  public let overlapFrames: Int
  public let fps: Double
  public let windowAudioTokens: [Int]
  public let joinAudioTokens: [Int]

  public var videoOverlapLatentFrames: Int { (overlapFrames - 1) / 8 + 1 }
  public var expectedAudioTokens: Int { Int(ceil(Double(totalFrames) / fps * 25)) }

  public init(durations: [Double], fps: Double, overlapFrames: Int = 25) throws {
    guard (2...6).contains(durations.count), fps.isFinite, fps > 0,
      overlapFrames >= 9, (overlapFrames - 1) % 8 == 0,
      durations.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw LTXError.invalid("LTX 2.5 scene needs two to six positive finite shots and an aligned overlap.")
    }
    var cumulative = 0.0
    var boundaries = [0]
    for duration in durations {
      cumulative += duration
      guard cumulative.isFinite, cumulative <= 30,
        cumulative * fps / 8 < Double(Int.max / 8) else {
        throw LTXError.invalid("LTX 2.5 scene exceeds its 30-second frame window.")
      }
      boundaries.append(Int((cumulative * fps / 8).rounded(.toNearestOrEven)) * 8)
    }
    let segments = zip(boundaries.dropLast(), boundaries.dropFirst()).map { $1 - $0 }
    let windows = [segments[0] + 1] + segments.dropFirst().map { $0 + overlapFrames }
    guard windows.allSatisfy({ $0 > overlapFrames && ($0 - 1) % 8 == 0 }) else {
      throw LTXError.invalid("LTX 2.5 scene has a shot too short for its causal overlap.")
    }
    let starts = [0] + boundaries.dropFirst().dropLast().map { $0 - overlapFrames + 1 }
    let audio = windows.map { Int(ceil(Double($0) / fps * 25)) }
    let cumulativeAudio = boundaries.dropFirst().map {
      Int(ceil(Double($0 + 1) / fps * 25))
    }
    let joins = (1..<windows.count).map { index in
      audio[index] - (cumulativeAudio[index] - cumulativeAudio[index - 1])
    }
    guard joins.enumerated().allSatisfy({ index, count in
      count > 0 && count < min(audio[index], audio[index + 1])
    }) else {
      throw LTXError.invalid("LTX 2.5 scene audio overlap is invalid.")
    }
    self.segmentFrames = segments
    self.segmentStarts = Array(boundaries.dropLast())
    self.windowFrames = windows
    self.windowStarts = starts
    self.totalFrames = boundaries.last! + 1
    self.overlapFrames = overlapFrames
    self.fps = fps
    self.windowAudioTokens = audio
    self.joinAudioTokens = joins
  }
}
