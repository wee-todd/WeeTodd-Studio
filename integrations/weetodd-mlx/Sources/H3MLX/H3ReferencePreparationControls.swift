import CoreFoundation
import Foundation

/// Per-reference media policy. A missing value preserves historical native
/// preparation; an explicit value is validated and applied before weighted work.
public struct H3ReferencePreparationControls: Sendable, Equatable {
  public enum VideoSizePolicy: String, Sendable { case matchOutput = "match_output", nativeH3 = "native_h3" }
  public enum TemporalDensity: String, Sendable { case full, half, quarter, automatic }
  public let imagePixelBudgetPercent: Int?
  public let videoSizePolicy: VideoSizePolicy?
  public let temporalDensity: TemporalDensity?
  public init(imagePixelBudgetPercent: Int? = nil,
    videoSizePolicy: VideoSizePolicy? = nil, temporalDensity: TemporalDensity? = nil) throws {
    guard imagePixelBudgetPercent == nil || ((50...400).contains(imagePixelBudgetPercent!) &&
      videoSizePolicy == nil && temporalDensity == nil) else {
      throw H3CheckpointError.invalid("H3 image pixel budget requires 50–400 percent and cannot carry movie policies.")
    }
    self.imagePixelBudgetPercent = imagePixelBudgetPercent
    let hasVideoPolicy = videoSizePolicy != nil || temporalDensity != nil
    self.videoSizePolicy = hasVideoPolicy ? videoSizePolicy ?? .matchOutput : nil
    self.temporalDensity = hasVideoPolicy ? temporalDensity ?? .full : nil
  }
  public static func parse(_ input: [String: Any], kind: String) throws -> Self? {
    let budget = input["image_pixel_budget_percent"], size = input["size_policy"], density = input["temporal_density"]
    guard budget != nil || size != nil || density != nil else { return nil }
    guard (kind == "image" && size == nil && density == nil) ||
      (kind == "video" && budget == nil) else {
      throw H3CheckpointError.invalid("H3 reference preparation controls must match their image or movie kind.")
    }
    var percent: Int?
    if let budget {
      guard let n = budget as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
        n.doubleValue.isFinite, n.doubleValue.rounded(.towardZero) == n.doubleValue,
        (50...400).contains(n.doubleValue) else {
        throw H3CheckpointError.invalid("H3 image pixel budget must be an integer 50–400 percent.")
      }
      percent = n.intValue
    }
    let policy: VideoSizePolicy?
    if let size {
      guard let text = size as? String, let parsed = VideoSizePolicy(rawValue: text) else {
        throw H3CheckpointError.invalid("Unknown H3 reference movie size policy.")
      }
      policy = parsed
    } else { policy = nil }
    let temporal: TemporalDensity?
    if let density {
      guard let text = density as? String, let parsed = TemporalDensity(rawValue: text) else {
        throw H3CheckpointError.invalid("Unknown H3 reference movie temporal density.")
      }
      temporal = parsed
    } else { temporal = nil }
    return try Self(imagePixelBudgetPercent: percent, videoSizePolicy: kind == "video" ? policy ?? .matchOutput : nil,
      temporalDensity: kind == "video" ? temporal ?? .full : nil)
  }
}

public enum H3ReferenceCanvasPolicy {
  public struct Canvas: Sendable, Equatable { public let width: Int, height: Int }
  /// Owned down-only scaling, rounded with ties-to-even on the32pixel grid.
  public static func image(sourceWidth: Int, sourceHeight: Int,
    outputWidth: Int, outputHeight: Int, percent: Int) throws -> Canvas {
    try validateSource(sourceWidth, sourceHeight, outputWidth, outputHeight)
    guard (50...400).contains(percent) else {
      throw H3CheckpointError.invalid("Invalid H3 image reference pixel budget.")
    }
    let budget = Double(outputWidth) * Double(outputHeight) * Double(percent) / 100
    return scaled(sourceWidth, sourceHeight, scale: min(1, sqrt(budget / Double(sourceWidth * sourceHeight))))
  }
  public static func video(sourceWidth: Int, sourceHeight: Int,
    outputWidth: Int, outputHeight: Int, policy: H3ReferencePreparationControls.VideoSizePolicy) throws -> Canvas {
    try validateSource(sourceWidth, sourceHeight, outputWidth, outputHeight)
    if policy == .matchOutput {
      let budget = Double(outputWidth * outputHeight)
      return scaled(sourceWidth, sourceHeight, scale: min(1, sqrt(budget / Double(sourceWidth * sourceHeight))))
    }
    let aspect = Double(sourceWidth) / Double(sourceHeight)
    var width = aspect >= 1 ? 768 * aspect : 768
    var height = aspect >= 1 ? 768 : 768 / aspect
    let scale = min(1, sqrt(Double(768 * 1344) / (width * height)))
    width *= scale; height *= scale
    let canvas = Canvas(width: grid(width), height: grid(height))
    if sourceWidth * sourceHeight < canvas.width * canvas.height {
      return Canvas(width: grid(Double(sourceWidth)), height: grid(Double(sourceHeight)))
    }
    return canvas
  }
  private static func validateSource(_ width: Int, _ height: Int, _ outputWidth: Int, _ outputHeight: Int) throws {
    guard (32...20_000).contains(width), (32...20_000).contains(height),
      width * height <= 100_000_000, (32...4096).contains(outputWidth), (32...4096).contains(outputHeight),
      (0.25...4).contains(Double(width) / Double(height)) else {
      throw H3CheckpointError.invalid("Invalid H3 reference source aspect or output geometry.")
    }
  }
  private static func scaled(_ width: Int, _ height: Int, scale: Double) -> Canvas {
    Canvas(width: grid(Double(width) * scale), height: grid(Double(height) * scale))
  }
  private static func grid(_ value: Double) -> Int { max(32, Int((value / 32).rounded(.toNearestOrEven)) * 32) }
}

public enum H3ReferenceTemporalPolicy {
  public struct Decision: Sendable {
    public let policy: H3ReferencePreparationControls.TemporalDensity
    public let density: Double
    public let indices: [Int]
    public let sourceLatentFrames: Int
    public let activityMean: Double?
    public let activityP95: Double?
    public let reason: String
    public var latentFrames: Int { (indices.count - 5) / 17 * 5 + 2 }
  }
  public static func resolve(rgb8: Data, frames: Int, width: Int, height: Int,
    policy: H3ReferencePreparationControls.TemporalDensity) throws -> Decision {
    guard (5...362).contains(frames), (frames - 5).isMultiple(of: 17),
      (32...2048).contains(width), (32...2048).contains(height),
      width * height <= H3Geometry.maximumCanvasPixels,
      rgb8.count == frames * width * height * 3,
      rgb8.count <= 1024 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Reference density requires complete bounded aligned RGB frames.")
    }
    try Task.checkCancellation()
    let density: Double, reason: String
    var activityMean: Double?, activityP95: Double?
    switch policy {
    case .full: density = 1; reason = "explicit_full"
    case .half: density = 0.5; reason = "explicit_half"
    case .quarter: density = 0.25; reason = "explicit_quarter"
    case .automatic:
      if frames <= 22 { density = 1; reason = "short_reference_kept_full" }
      else {
        let rowStride = max(1, (height + 63) / 64), columnStride = max(1, (width + 63) / 64)
        var deltas: [Float] = []; deltas.reserveCapacity(frames - 1)
        try rgb8.withUnsafeBytes { bytes in
          let pixels = bytes.bindMemory(to: UInt8.self)
          for frame in 1..<frames {
            try Task.checkCancellation()
            var sum: Float = 0, count = 0
            for row in stride(from: 0, to: height, by: rowStride) {
              for column in stride(from: 0, to: width, by: columnStride) {
                for channel in 0..<3 {
                  let offset = (row * width + column) * 3 + channel
                  sum += Float(abs(Int(pixels[frame * width * height * 3 + offset]) -
                    Int(pixels[(frame - 1) * width * height * 3 + offset])))
                  count += 1
                }
              }
            }
            deltas.append((sum / Float(count)) / Float(255))
          }
        }
        let mean = deltas.reduce(Float(0), +) / Float(deltas.count)
        let sorted = deltas.sorted(), coordinate = Double(deltas.count - 1) * 0.95
        let lower = Int(floor(coordinate)), upper = Int(ceil(coordinate))
        let fraction = coordinate - Double(lower)
        let difference = Double(sorted[upper] - sorted[lower])
        let percentile = fraction >= 0.5 ? Double(sorted[upper]) - difference * (1 - fraction) :
          Double(sorted[lower]) + difference * fraction
        activityMean = Double(mean); activityP95 = percentile
        if Double(mean) >= 0.02 || percentile >= 0.06 { density = 1; reason = "high_motion_or_cut_kept_full" }
        else if Double(mean) >= 0.004 || percentile >= 0.015 { density = 0.5; reason = "moderate_redundancy_selected_half" }
        else { density = 0.25; reason = "near_static_reference_selected_quarter" }
      }
    }
    let sourceLatents = (frames - 5) / 17 * 5 + 2
    let chunks = min((frames - 5) / 17, max(0, Int(((Double(sourceLatents) * density - 2) / 5).rounded(.toNearestOrEven))))
    let count = 5 + 17 * chunks
    let step = Double(frames - 1) / Double(count - 1)
    let indices = count == frames ? Array(0..<frames) : (0..<count).map {
      $0 == count - 1 ? frames - 1 : Int((Double($0) * step).rounded(.toNearestOrEven))
    }
    return Decision(policy: policy, density: density, indices: indices,
      sourceLatentFrames: sourceLatents, activityMean: activityMean,
      activityP95: activityP95, reason: reason)
  }
}
