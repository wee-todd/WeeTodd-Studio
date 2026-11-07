import Foundation
import MLX

public enum RelayVSRError: LocalizedError {
  case invalid(String)
  public var errorDescription: String? {
    switch self { case .invalid(let message): return message }
  }
}

struct RelayGeometry {
  let width: Int
  let height: Int
  let paddedWidth: Int
  let paddedHeight: Int
  var outputWidth: Int { width * 2 }
  var outputHeight: Int { height * 2 }

  init(width: Int, height: Int) throws {
    guard (1...4096).contains(width), (1...4096).contains(height) else {
      throw RelayVSRError.invalid("RelayVSR expects source dimensions between 1 and 4096 pixels per axis.")
    }
    self.width = width
    self.height = height
    paddedWidth = (width + 15) / 16 * 16
    paddedHeight = (height + 15) / 16 * 16
  }

  func prepare(_ rgb: MLXArray) -> MLXArray {
    let rows = MLXArray((0..<paddedHeight).map { Int32(min($0, height - 1)) })
    let columns = MLXArray((0..<paddedWidth).map { Int32(min($0, width - 1)) })
    return rgb.take(rows, axis: 0).take(columns, axis: 1).asType(.float32) / 127.5 - 1
  }
}

enum RelaySchedule {
  static func sigmas(steps: Int) throws -> [Float] {
    guard steps == 1 || steps == 4 else {
      throw RelayVSRError.invalid("RelayVSR 2× supports 1 or 4 sampling steps.")
    }
    return (0..<steps).map { step -> Float in
      let t = Float(steps - step) / Float(steps)
      return 8 * t / (1 + 7 * t)
    } + [0]
  }
}

enum RelayMemoryPlan {
  static func estimatedBytes(geometry: RelayGeometry, options: RelayMovieOptions) -> UInt64 {
    let pixels = UInt64(geometry.paddedWidth * geometry.paddedHeight)
    let wanTokens = pixels / 256
    let flashTokens = pixels / 64
    let interval = options.keyframeInterval
    let span = min((options.frameLimit ?? Int.max) - 1, interval * 2)
    let history = 1 + (span + interval - 1) / interval
    // K and V are two BF16 arrays, with independent history for each solver step.
    let wanCache = wanTokens * 3072 * 4 * 30 * UInt64(options.steps * history)
    // Upper bounds for the retained boolean masks of all tiled attention plans.
    let masks = wanTokens * 16_000 + flashTokens * 21_000
    let lookahead = min(options.keyframeInterval + 1, options.frameLimit ?? Int.max)
    let movieBuffers = pixels * 4 * UInt64(lookahead + 64)
    return 8 * 1024 * 1024 * 1024 + wanCache + masks + movieBuffers
  }

  static func validate(geometry: RelayGeometry, options: RelayMovieOptions, budget: UInt64) throws {
    let estimate = estimatedBytes(geometry: geometry, options: options)
    guard budget == 0 || estimate <= budget else {
      let gib: UInt64 = 1024 * 1024 * 1024
      let needed = (estimate + gib - 1) / gib
      throw RelayVSRError.invalid("Estimated RelayVSR working set is \(needed) GiB. Metal recommends "
        + "\(budget / gib) GiB for this GPU. Reduce the source dimensions or use fewer steps.")
    }
  }
}

struct RelayWindowBuffer<Frame> {
  let interval: Int
  private var frames: [Frame] = []
  private var emitted = false

  init(interval: Int) { self.interval = interval }

  mutating func append(_ frame: Frame) -> [Frame]? {
    frames.append(frame)
    guard frames.count == interval + 1 else { return nil }
    let result = frames
    frames = [frame]
    emitted = true
    return result
  }

  mutating func finish() -> [Frame]? {
    defer { frames.removeAll() }
    return frames.count > (emitted ? 1 : 0) ? frames : nil
  }
}

struct RelayWindow: Equatable {
  let height: Int
  let width: Int
  let slidesAtEdge: Bool

  func range(at position: Int, extent: Int, size: Int) -> Range<Int> {
    if slidesAtEdge {
      let count = min(size, extent)
      let start = min(max(position - count / 2, 0), extent - count)
      return start..<(start + count)
    }
    return max(position - size / 2, 0)..<min(position + size / 2, extent)
  }
}
