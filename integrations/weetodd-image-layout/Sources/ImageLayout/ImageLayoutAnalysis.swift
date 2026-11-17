import Foundation

public struct ImageLayoutTargetSize: Codable, Equatable, Sendable {
  public let width, height: Int
  public init(width: Int, height: Int) { self.width = width; self.height = height }
}

public enum ImageLayoutAnalysisGeometry {
  public static func targetSize(sourceWidth: Int, sourceHeight: Int, targetPixels: Int = 2_000_000) throws -> ImageLayoutTargetSize {
    guard (1...131_072).contains(sourceWidth), (1...131_072).contains(sourceHeight),
      (256...4_194_304).contains(targetPixels) else { throw ImageLayoutError.invalid("Invalid source dimensions or target area.") }
    let aspect = Double(sourceWidth) / Double(sourceHeight)
    guard (1.0 / 256...256).contains(aspect) else { throw ImageLayoutError.invalid("The source aspect ratio exceeds the MOSS canvas limits.") }
    let effectiveArea = min(Double(targetPixels), 4096 * 4096 / max(aspect, 1 / aspect))
    let sideLimited = effectiveArea < Double(targetPixels)
    var best: (width: Int, height: Int, ratio: Double, area: Double)?
    for width in stride(from: 16, through: 4096, by: 16) {
      let lower = sideLimited ? 16 : max(16, Int(ceil(effectiveArea * 0.98 / Double(width) / 16)) * 16)
      let upper = sideLimited ? 4096 : min(4096, Int(floor(effectiveArea * 1.02 / Double(width) / 16)) * 16)
      guard lower <= upper else { continue }
      for height in stride(from: lower, through: upper, by: 16) {
        let area = width * height
        guard area <= 4_194_304 else { continue }
        let error = abs(log(Double(width) / Double(height) / aspect))
        let areaError = abs(Double(area) - Double(targetPixels))
        if best == nil || error < best!.ratio - 1e-12 || (abs(error - best!.ratio) < 1e-12 && areaError < best!.area) {
          best = (width, height, error, areaError)
        }
      }
    }
    guard let best else { throw ImageLayoutError.invalid("No valid 16-pixel canvas fits the requested area.") }
    return .init(width: best.width, height: best.height)
  }
  public static func visionBounds(x: Double, y: Double, width: Double, height: Double) throws -> NormalizedLayoutRect {
    let rect = NormalizedLayoutRect(x: x, y: 1 - y - height, width: width, height: height)
    try validate(rect)
    let left = max(0, rect.x), top = max(0, rect.y)
    return .init(x: left, y: top, width: min(rect.width, 1 - left), height: min(rect.height, 1 - top))
  }
  static func validate(_ rect: NormalizedLayoutRect) throws {
    guard [rect.x, rect.y, rect.width, rect.height].allSatisfy(\.isFinite),
      rect.x >= -1e-9, rect.y >= -1e-9, rect.width > 0, rect.height > 0,
      rect.x + rect.width <= 1 + 1e-9, rect.y + rect.height <= 1 + 1e-9 else {
      throw ImageLayoutError.invalid("Analysis produced invalid image bounds.")
    }
  }
}

public struct LayoutOCRObservation: Codable, Equatable, Sendable {
  public let id: Int
  public let text: String
  public let bounds: NormalizedLayoutRect
  public let confidence: Double
  public init(id: Int, text: String, bounds: NormalizedLayoutRect, confidence: Double) {
    self.id = id; self.text = text; self.bounds = bounds; self.confidence = confidence
  }
}

public struct ImageLayoutAnalysisProvenance: Codable, Equatable, Sendable {
  public var requestID, sourceSHA256: String
  public var sourceWidth, sourceHeight: Int
  public var sourceName, modelIdentity, recipe: String
  public var warnings: [String]
  public init(requestID: String, sourceSHA256: String, sourceWidth: Int, sourceHeight: Int,
    sourceName: String, modelIdentity: String, recipe: String, warnings: [String]) {
    self.requestID = requestID; self.sourceSHA256 = sourceSHA256; self.sourceWidth = sourceWidth; self.sourceHeight = sourceHeight
    self.sourceName = sourceName; self.modelIdentity = modelIdentity; self.recipe = recipe; self.warnings = warnings
  }
}

public struct ImageLayoutAnalysis: Sendable {
  public let draft: ImageLayoutDraft
  public let warnings: [String]
  public init(draft: ImageLayoutDraft, warnings: [String]) { self.draft = draft; self.warnings = warnings }
}
