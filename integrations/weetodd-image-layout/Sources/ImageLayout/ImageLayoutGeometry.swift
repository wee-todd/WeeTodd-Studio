import Foundation

public struct PixelBounds: Equatable, Sendable {
  public let left, top, right, bottom: Int
  public let width, height: Int
  public var centeredYXYX: [Int] { [top - height / 2, left - width / 2, bottom - height / 2, right - width / 2] }
}

public enum ImageLayoutGeometry {
  public static func validateCanvas(width: Int, height: Int) throws {
    guard width > 0, height > 0, width <= 4096, height <= 4096,
      width.isMultiple(of: 16), height.isMultiple(of: 16), width * height <= 4_194_304 else {
      throw ImageLayoutError.invalid("Use multiples of 16, sides up to 4096 and at most 4,194,304 pixels.")
    }
  }
  public static func pixelBounds(_ rect: NormalizedLayoutRect, width: Int, height: Int) throws -> PixelBounds {
    try validateCanvas(width: width, height: height)
    guard [rect.x, rect.y, rect.width, rect.height].allSatisfy(\.isFinite),
      rect.x >= 0, rect.y >= 0, rect.width > 0, rect.height > 0,
      rect.x + rect.width <= 1 + 1e-14, rect.y + rect.height <= 1 + 1e-14 else {
      throw ImageLayoutError.invalid("Each region needs finite, positive bounds inside the canvas.")
    }
    // Decimal formatting rounds the original binary64 value, rather than a rounded x*1e9 intermediate.
    func edge(_ v: Double) -> Double { Double(String(format: "%.9f", locale: Locale(identifier: "en_US_POSIX"), v))! }
    let left = Int(floor(edge(rect.x * Double(width)))), top = Int(floor(edge(rect.y * Double(height))))
    let right = min(width, Int(ceil(edge((rect.x + rect.width) * Double(width)))))
    let bottom = min(height, Int(ceil(edge((rect.y + rect.height) * Double(height)))))
    guard left < right, top < bottom else { throw ImageLayoutError.invalid("A region is too small at export precision.") }
    return PixelBounds(left: left, top: top, right: right, bottom: bottom, width: width, height: height)
  }
}

public struct LayoutPoint: Equatable, Sendable { public let x, y: Double }

public struct LayoutViewport: Sendable {
  public let scale, originX, originY: Double
  public let canvasWidth, canvasHeight: Double
  public init(canvasWidth: Int, canvasHeight: Int, viewWidth: Double, viewHeight: Double,
    zoom: Double = 1, panX: Double = 0, panY: Double = 0) throws {
    guard canvasWidth > 0, canvasHeight > 0,
      [viewWidth, viewHeight, zoom, panX, panY].allSatisfy(\.isFinite),
      viewWidth > 0, viewHeight > 0, zoom > 0 else { throw ImageLayoutError.invalid("Invalid layout viewport.") }
    self.canvasWidth = Double(canvasWidth); self.canvasHeight = Double(canvasHeight)
    scale = min(viewWidth / Double(canvasWidth), viewHeight / Double(canvasHeight)) * zoom
    originX = (viewWidth - Double(canvasWidth) * scale) / 2 + panX
    originY = (viewHeight - Double(canvasHeight) * scale) / 2 + panY
  }
  public func viewPoint(x: Double, y: Double) -> LayoutPoint {
    .init(x: originX + x * canvasWidth * scale, y: originY + y * canvasHeight * scale)
  }
  public func normalizedPoint(x: Double, y: Double) -> LayoutPoint {
    .init(x: (x - originX) / (canvasWidth * scale), y: (y - originY) / (canvasHeight * scale))
  }
}
