import Foundation
import MLX
import InferenceContracts

public enum RelayVSRError: LocalizedError {
  case invalid(String)
  public var errorDescription: String? {
    switch self { case .invalid(let message): return message }
  }
}

struct RelayGeometry {
  let scale: NativeRelayVSRScale
  let width: Int
  let height: Int
  let paddedWidth: Int
  let paddedHeight: Int
  var outputWidth: Int { width * scale.rawValue }
  var outputHeight: Int { height * scale.rawValue }
  var wanWidth: Int { paddedWidth / scale.wanGridDivisor }
  var wanHeight: Int { paddedHeight / scale.wanGridDivisor }
  var flashWidth: Int { paddedWidth / scale.flashGridDivisor }
  var flashHeight: Int { paddedHeight / scale.flashGridDivisor }

  init(width: Int, height: Int, scale: NativeRelayVSRScale = .x2) throws {
    guard (1...4096).contains(width), (1...4096).contains(height) else {
      throw RelayVSRError.invalid("RelayVSR expects source dimensions between 1 and 4096 pixels per axis.")
    }
    self.width = width
    self.height = height
    self.scale = scale
    let multiple = scale.paddingMultiple
    paddedWidth = (width + multiple - 1) / multiple * multiple
    paddedHeight = (height + multiple - 1) / multiple * multiple
  }

  func prepare(_ rgb: MLXArray) -> MLXArray {
    let rows = MLXArray((0..<paddedHeight).map { Int32(min($0, height - 1)) })
    let columns = MLXArray((0..<paddedWidth).map { Int32(min($0, width - 1)) })
    let padded = rgb.take(rows, axis: 0).take(columns, axis: 1)
    if scale == .x4 {
      // Released prepare_lr casts before div/sub, each retaining BF16 output.
      let divided = (padded.asType(.bfloat16).asType(.float32) / 127.5).asType(.bfloat16)
      return (divided.asType(.float32) - 1).asType(.bfloat16)
    }
    return padded.asType(.float32) / 127.5 - 1
  }
}

enum RelaySchedule {
  static func sigmas(steps: Int) throws -> [Float] {
    guard steps == 1 || steps == 4 else {
      throw RelayVSRError.invalid("RelayVSR supports 1 or 4 sampling steps.")
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
    let wanTokens = UInt64(geometry.wanWidth * geometry.wanHeight)
    let flashTokens = UInt64(geometry.flashWidth * geometry.flashHeight)
    let interval = options.keyframeInterval
    let span = min((options.frameLimit ?? Int.max) - 1, interval * 2)
    let history = 1 + (span + interval - 1) / interval
    // K and V are two BF16 arrays, with independent history for each solver step.
    let wanCache = wanTokens * 3072 * 4 * 30 * UInt64(options.steps * history)
    // Upper bounds for the retained boolean masks of all tiled attention plans.
    let masks = wanTokens * 16_000 + flashTokens * 21_000
    let lookahead = min(options.keyframeInterval + 1, options.frameLimit ?? Int.max)
    let movieBuffers = pixels * 4 * UInt64(lookahead + 64) * (geometry.scale == .x4 ? 4 : 1)
    // Cached byte classifications and the fused safety scan's compact flags.
    let blockMetadata = (wanTokens + flashTokens) * 64 + 1024 * 1024
    var queuedWorkspace: UInt64 = 0
    if RelaySkippedAttention.runtimeQualified {
      let wan = RelayWindow(height:22,width:40,slidesAtEdge:true)
      let reference = RelayWindow(height:48,width:48,slidesAtEdge:false)
      let video = RelayWindow(height:24,width:24,slidesAtEdge:false)
      queuedWorkspace = max(
        RelayAttentionPlan.queuedWorkspaceBytes(height:geometry.wanHeight,width:geometry.wanWidth,
          windows:Array(repeating:wan,count:4),queryChannels:3072,keyChannels:3072,scalarBytes:4),
        RelayAttentionPlan.queuedWorkspaceBytes(height:geometry.flashHeight,width:geometry.flashWidth,
          windows:[reference,reference,video,video],queryChannels:512,keyChannels:128,scalarBytes:4))
    }
    return 8 * 1024 * 1024 * 1024 + wanCache + masks + movieBuffers + blockMetadata + queuedWorkspace
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
