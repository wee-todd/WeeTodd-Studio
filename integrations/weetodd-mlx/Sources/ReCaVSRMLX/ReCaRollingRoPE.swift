import Foundation

public enum ReCaRollingRoPE {
  public static func positions(start: Int, frames: Int) throws -> (origin: Int, positions: [Int]) {
    guard start >= 0, (1...22).contains(frames), start <= Int.max - frames else {
      throw ReCaError.invalid("Invalid ReCaVSR rolling temporal interval.")
    }
    let origin = max(0, start + frames - 22)
    return (origin, (start..<(start + frames)).map { $0 - origin })
  }
}
