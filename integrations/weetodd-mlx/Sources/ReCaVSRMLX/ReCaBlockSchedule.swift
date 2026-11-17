import Foundation

public struct ReCaBlock: Equatable, Sendable {
  public let sourceStart: Int, shotStart: Int, validFrames: Int, rgbFrames: Int
  public var latentFrames: Int { rgbFrames == 21 ? 6 : 2 }
}
public enum ReCaBlockSchedule {
  public static func blocks(frameCount: Int, resetFrames: [Int] = []) throws -> [ReCaBlock] {
    guard (1...1_000_000).contains(frameCount), resetFrames == Array(Set(resetFrames)).sorted(),
      resetFrames.allSatisfy({ $0 > 0 && $0 < frameCount }) else {
      throw ReCaError.invalid("Invalid ReCaVSR frame count or shot boundaries.")
    }
    var result: [ReCaBlock] = []
    let starts = [0] + resetFrames
    for (index, start) in starts.enumerated() {
      let end = index + 1 < starts.count ? starts[index + 1] : frameCount
      var offset = start
      while offset < end {
        let block = offset == start ? 21 : 8
        let count = min(block, end - offset)
        result.append(.init(sourceStart: offset, shotStart: start, validFrames: count, rgbFrames: block))
        offset += count
      }
    }
    return result
  }
}
