import Foundation
import LTX25Engine

/// Streams decoded RGB frames across causal decode windows. Only the 25-frame
/// overlap is retained; the final causal frame is never delivered.
struct MLXSceneRGBJoiner {
  let windowCount: Int
  let frameBytes: Int
  let overlapFrames: Int
  private(set) var delivered = 0
  private var pending: [Data] = []

  init(windowCount: Int, frameBytes: Int, overlapFrames: Int = 25) throws {
    guard windowCount > 0, frameBytes > 0, overlapFrames >= 2 else {
      throw LTXError.invalid("Invalid scene RGB decode timeline.")
    }
    self.windowCount = windowCount
    self.frameBytes = frameBytes
    self.overlapFrames = overlapFrames
  }

  mutating func receive(window: Int, frame: Int, count: Int, rgb: Data,
    deliver: (Int, Data) throws -> Void) throws {
    guard (0..<windowCount).contains(window), (0..<count).contains(frame),
      rgb.count == frameBytes, count > overlapFrames else {
      throw LTXError.invalid("Scene RGB window differs from its admitted timeline.")
    }
    let last = window == windowCount - 1
    func emit(_ bytes: Data) throws {
      try deliver(delivered, bytes)
      delivered += 1
    }
    if window > 0 && frame < overlapFrames {
      guard pending.count == overlapFrames else {
        throw LTXError.invalid("Scene RGB overlap is missing its prior decoded frames.")
      }
      try emit(Self.blend(pending[frame], rgb,
        alpha: Float(frame) / Float(overlapFrames - 1)))
      if frame == overlapFrames - 1 { pending.removeAll(keepingCapacity: true) }
    } else if last {
      if frame < count - 1 { try emit(rgb) }
    } else {
      pending.append(rgb)
      if pending.count > overlapFrames { try emit(pending.removeFirst()) }
    }
  }

  func finishWindow(window: Int) throws {
    guard (0..<windowCount).contains(window),
      pending.count == (window == windowCount - 1 ? 0 : overlapFrames) else {
      throw LTXError.invalid("Scene RGB decode window cannot supply the next overlap.")
    }
  }

  func finish(expectedFrames: Int) throws {
    guard pending.isEmpty, delivered == expectedFrames else {
      throw LTXError.invalid("Scene RGB decode produced an incomplete editorial timeline.")
    }
  }

  static func blend(_ previous: Data, _ next: Data, alpha: Float) -> Data {
    if alpha <= 0 { return previous }
    if alpha >= 1 { return next }
    var result = Data(count: previous.count)
    previous.withUnsafeBytes { a in
      next.withUnsafeBytes { b in
        result.withUnsafeMutableBytes { o in
          let p = a.bindMemory(to: UInt8.self)
          let n = b.bindMemory(to: UInt8.self)
          let d = o.bindMemory(to: UInt8.self)
          for index in d.indices {
            d[index] = UInt8((Float(p[index]) * (1 - alpha)
              + Float(n[index]) * alpha).rounded())
          }
        }
      }
    }
    return result
  }
}
