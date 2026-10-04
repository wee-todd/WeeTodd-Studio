import Foundation
import LTX25Engine

/// Model-specific source movie geometry; never creates generated DFR slots.
public struct MLXMovieUpscalePlan: Sendable {
  public enum Mode: String, Codable, Sendable { case latentOnly = "latent_only", refine, pixelSpatial = "pixel_spatial" }
  public enum SizePolicy: String, Codable, Sendable { case fitNearest = "fit_nearest_32", centerCrop = "center_crop_32", strict = "strict_32" }
  public struct Size: Sendable, Equatable {
    public let sourceWidth: Int, sourceHeight: Int, width: Int, height: Int
    public let cropLeft: Int, cropTop: Int, cropRight: Int, cropBottom: Int
    public let resized: Bool
    public var outputWidth: Int { width * 2 }
    public var outputHeight: Int { height * 2 }
  }
  public struct Chunk: Sendable, Equatable {
    public let startFrame: Int, endFrame: Int, paddedFrames: Int, reason: String
    public var frames: Int { endFrame - startFrame }
    public var visibleLastFrame: Int { frames - 1 }
  }
  public let mode: Mode, size: Size, frames: Int, paddedFrames: Int
  public let fps: Double, refinementStrength: Double
  public var visibleLastFrame: Int { frames - 1 }
  public var seconds: Double { Double(frames) / fps }
  public var outputFrameMegapixels: Double { Double(frames) * Double(size.outputWidth) * Double(size.outputHeight) / 1_000_000 }
  public var sigmas: [Double] {
    guard mode != .latentOnly else { return [] }
    return [0.909375, 0.725, 0.421875, 0].map { $0 * refinementStrength / 0.909375 }
  }
  public init(mode: Mode, width: Int, height: Int, frames: Int, fps: Double,
    sizePolicy: SizePolicy, refinementStrength: Double = 0.35,
    maximumOutputFrameMegapixels: Double = 0) throws {
    guard frames > 0, frames <= Int.max - 8, fps.isFinite, (1...60).contains(fps),
      refinementStrength.isFinite, (0.05...0.909375).contains(refinementStrength),
      maximumOutputFrameMegapixels.isFinite, maximumOutputFrameMegapixels >= 0 else {
      throw LTXError.invalid("Movie upscale requires positive frames, 1–60 fps and finite refinement/workload settings.")
    }
    self.mode = mode; self.frames = frames; self.fps = fps
    self.refinementStrength = refinementStrength
    size = try Self.prepareSize(width: width, height: height, policy: sizePolicy)
    paddedFrames = try Self.paddedFrameCount(frames)
    if maximumOutputFrameMegapixels > 0, outputFrameMegapixels > maximumOutputFrameMegapixels {
      throw LTXError.invalid("Movie upscale exceeds its explicit output frame-megapixel limit; this is a workload guard, not a memory estimate.")
    }
  }
  public static func paddedFrameCount(_ frames: Int) throws -> Int {
    guard frames > 0, frames <= Int.max - 8 else { throw LTXError.invalid("Invalid movie frame count.") }
    return 1 + ((frames - 1 + 7) / 8) * 8
  }
  public static func prepareSize(width: Int, height: Int, policy: SizePolicy) throws -> Size {
    guard width >= 32, height >= 32, width <= Int.max / 4, height <= Int.max / 4 else {
      throw LTXError.invalid("Movie input requires at least 32 pixels per axis and safe spatial arithmetic.")
    }
    if width % 32 == 0, height % 32 == 0 {
      return Size(sourceWidth: width, sourceHeight: height, width: width, height: height,
        cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0, resized: false)
    }
    guard policy != .strict else { throw LTXError.invalid("Movie dimensions must be divisible by 32 under strict-grid policy.") }
    if policy == .fitNearest, let fitted = try nearestGrid(width: width, height: height) {
      return Size(sourceWidth: width, sourceHeight: height, width: fitted.0, height: fitted.1,
        cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0, resized: true)
    }
    let w = width / 32 * 32, h = height / 32 * 32
    let left = (width - w) / 2, top = (height - h) / 2
    return Size(sourceWidth: width, sourceHeight: height, width: w, height: h,
      cropLeft: left, cropTop: top, cropRight: width - w - left, cropBottom: height - h - top, resized: false)
  }
  private static func nearestGrid(width: Int, height: Int) throws -> (Int, Int)? {
    let low = max(32, Int(floor(Double(height) * 0.65 / 32)) * 32)
    let upper = max(32, Int(ceil(Double(height) * 1.35 / 32)) * 32)
    // The loop bound protects malformed media dimensions before any pixel allocation.
    guard (upper - low) / 32 <= 1_000_000 else { throw LTXError.invalid("Movie grid search exceeds its bounded metadata work.") }
    let aspect = Double(width) / Double(height)
    var best: (scaleDistance: Double, prefersDown: Int, error: Double, width: Int, height: Int)?
    for h in stride(from: low, through: upper, by: 32) {
      let projected = (Double(h) * aspect / 32).rounded(.toNearestOrEven)
      guard projected.isFinite, projected <= Double(Int.max / 64) else { throw LTXError.invalid("Movie fitted dimensions overflow.") }
      let w = max(32, Int(projected) * 32)
      let error = abs((Double(w) / Double(h)) / aspect - 1)
      if error > 0.005 { continue }
      let scale = Double(h) / Double(height)
      let candidate = (abs(log(scale)), scale >= 1 ? 0 : 1, error, w, h)
      if let old = best {
        if candidate.0 < old.scaleDistance || (candidate.0 == old.scaleDistance &&
          (candidate.1 < old.prefersDown || (candidate.1 == old.prefersDown &&
            (candidate.2 < old.error || (candidate.2 == old.error &&
              (candidate.3 < old.width || (candidate.3 == old.width && candidate.4 < old.height))))))) {
          best = candidate
        }
      } else { best = candidate }
    }
    return best.map { ($0.width, $0.height) }
  }
  public static func sceneCuts(adjacentLuminanceDifferences values: [Float]) throws -> [Int] {
    guard values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
      throw LTXError.invalid("Scene-cut samples must contain finite normalized adjacent luminance differences.")
    }
    guard !values.isEmpty else { return [] }
    func median(_ values: [Float]) -> Float {
      let sorted = values.sorted(), mid = sorted.count / 2
      return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }
    let center = median(values), deviation = median(values.map { abs($0 - center) })
    let threshold = max(Float(0.08), center + 6 * max(deviation, Float(0.0001)))
    return values.enumerated().compactMap { $0.element >= threshold ? $0.offset + 1 : nil }
  }
  public func chunks(frameMegapixelBudget: Double = 260, cutFrames: [Int] = []) throws -> [Chunk] {
    guard frameMegapixelBudget.isFinite, frameMegapixelBudget > 0 else {
      throw LTXError.invalid("Movie chunk workload budget must be positive and finite.")
    }
    let perFrame = Double(size.outputWidth) * Double(size.outputHeight) / 1_000_000
    let rawMaximum = floor(frameMegapixelBudget / perFrame)
    guard rawMaximum.isFinite, rawMaximum >= 9 else {
      throw LTXError.invalid("Movie chunk budget permits fewer than nine frames.")
    }
    let maximum = rawMaximum >= Double(frames) ? frames : Int(rawMaximum)
    if frames <= maximum { return [Chunk(startFrame: 0, endFrame: frames, paddedFrames: paddedFrames, reason: "complete clip")] }
    let count = (frames - 1) / maximum + 1
    guard maximum >= 49, count <= frames / 49 else {
      throw LTXError.invalid("Movie chunk budget cannot partition this clip into at least 49-frame windows.")
    }
    let cuts = Array(Set(cutFrames.filter { $0 > 0 && $0 < frames })).sorted()
    var boundaries = [0], reasons: [String] = []
    for index in 1..<count {
      let start = boundaries.last!, remaining = count - index
      let ideal = Int((Double(index) * Double(frames) / Double(count)).rounded(.toNearestOrEven))
      let candidates = cuts.filter {
        $0 - start >= 49 && $0 - start <= maximum && frames - $0 >= 49 * remaining &&
        Double(frames - $0) <= Double(maximum) * Double(remaining) &&
        Double(abs($0 - ideal)) <= Double(maximum) * 0.35
      }
      if let cut = candidates.min(by: { abs($0 - ideal) == abs($1 - ideal) ? $0 < $1 : abs($0 - ideal) < abs($1 - ideal) }) {
        boundaries.append(cut); reasons.append("scene cut")
      } else {
        let low = start + 49, high = min(start + maximum, frames - 49 * remaining)
        boundaries.append(min(max(ideal, low), high)); reasons.append("workload boundary")
      }
    }
    boundaries.append(frames); reasons.append("complete clip")
    return try zip(boundaries, boundaries.dropFirst()).enumerated().map { index, pair in
      let length = pair.1 - pair.0
      guard length >= 49, length <= maximum else { throw LTXError.invalid("Movie chunk plan violates its quality/workload bounds.") }
      return Chunk(startFrame: pair.0, endFrame: pair.1,
        paddedFrames: try Self.paddedFrameCount(length), reason: reasons[index])
    }
  }
}
