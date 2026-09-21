import AppKit
import CryptoKit
import ImageIO
import StudioCore
import Vision

public enum CharacterPanelDetectorError: Error, Equatable { case unreadableImage, unsupportedRuntime }

enum VisionMaskPixels {
  static func bytes(from buffer: CVPixelBuffer, expectedWidth: Int, expectedHeight: Int) throws -> [UInt8] {
    guard CVPixelBufferGetWidth(buffer) == expectedWidth, CVPixelBufferGetHeight(buffer) == expectedHeight else {
      throw CharacterPanelDetectorError.unreadableImage
    }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw CharacterPanelDetectorError.unreadableImage }
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    var output = Array(repeating: UInt8(0), count: expectedWidth * expectedHeight)
    switch CVPixelBufferGetPixelFormatType(buffer) {
    case kCVPixelFormatType_OneComponent32Float:
      for y in 0..<expectedHeight {
        let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float32.self)
        for x in 0..<expectedWidth {
          let sample = row[x].isFinite ? row[x] : 0
          output[y * expectedWidth + x] = UInt8((max(0, min(1, sample)) * 255).rounded())
        }
      }
    case kCVPixelFormatType_OneComponent8:
      for y in 0..<expectedHeight {
        let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
        for x in 0..<expectedWidth { output[y * expectedWidth + x] = row[x] }
      }
    default:
      throw CharacterPanelDetectorError.unreadableImage
    }
    return output
  }
}

public struct CharacterPanelDetector: Sendable {
  public init() {}

  public func detect(source: URL) async throws -> CharacterPanelDetection {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      let data = try Data(contentsOf: source)
      let sourceHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
      guard #available(macOS 14.0, *) else { throw CharacterPanelDetectorError.unsupportedRuntime }
      guard let imageSource = CGImageSourceCreateWithData(data as CFData, nil),
            let sourceDimensions = Self.orientationNormalizedDimensions(imageSource),
            let image = Self.normalizedProxy(imageSource: imageSource, maximumDimension: 1600) else {
        throw CharacterPanelDetectorError.unreadableImage
      }
      let orientation = (CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any])?[kCGImagePropertyOrientation] as? Int ?? 1
      try Task.checkCancellation()
      let foregroundRequest = VNGenerateForegroundInstanceMaskRequest()
      let humanRequest = VNDetectHumanRectanglesRequest()
      humanRequest.upperBodyOnly = false
      let rectangleRequest = VNDetectRectanglesRequest()
      rectangleRequest.maximumObservations = 8
      rectangleRequest.minimumAspectRatio = 0.12
      let handler = VNImageRequestHandler(cgImage: image, orientation: .up)
      try await withTaskCancellationHandler(operation: {
        try handler.perform([foregroundRequest, humanRequest, rectangleRequest])
      }, onCancel: {
        foregroundRequest.cancel(); humanRequest.cancel(); rectangleRequest.cancel()
      })

      var foreground = [PanelPixelRect]()
      let edgeColumns = try Self.horizontalEdgeColumns(image)
      if let observation = foregroundRequest.results?.first {
        guard observation.allInstances.count <= 16 else {
          return CharacterPanelDetection(sourceSHA256: sourceHash, sourceOrientation: orientation,
            detectorVersion: CharacterPanelLayout.detectorVersion, candidates: [], status: .needsReview,
            diagnostics: ["Too many foreground instances for reliable four-panel detection; review crop boundaries."])
        }
        // Keep Vision's instance separation: gap filling within a single silhouette
        // must never bridge the narrow real gutter between neighboring instances.
        for instance in observation.allInstances {
          try Task.checkCancellation()
          foreground += try autoreleasepool {
            let buffer = try observation.generateScaledMaskForImage(forInstances: IndexSet(integer: instance), from: handler)
            return Self.horizontalComponents(in: buffer)
          }
        }
      }
      // Saliency may select only the large close-up. Full-body detections supply
      // smaller figures without imposing equal panel widths or inferred positions.
      let humans = (humanRequest.results ?? []).filter { $0.confidence >= 0.5 }.map {
        Self.pixelRect($0.boundingBox, imageWidth: image.width, imageHeight: image.height)
      }
      foreground = Self.mergedForegroundComponents(foreground + humans)
      let rectangles = (rectangleRequest.results ?? []).map {
        Self.pixelRect($0.boundingBox, imageWidth: image.width, imageHeight: image.height)
      }
      try Task.checkCancellation()
      let proxyDetection = CharacterPanelLayout.detect(imageWidth: image.width, imageHeight: image.height,
        foregroundComponents: foreground, edgeColumns: edgeColumns, rectangleHints: rectangles,
        detectionRevision: 1, sourceSHA256: sourceHash, sourceOrientation: orientation)
      return Self.mapToSource(proxyDetection, proxyWidth: image.width, proxyHeight: image.height,
        sourceWidth: sourceDimensions.width, sourceHeight: sourceDimensions.height)
    }
    return try await withTaskCancellationHandler(operation: {
      let result = try await worker.value
      try Task.checkCancellation()
      return result
    }, onCancel: { worker.cancel() })
  }

  public static func pixelRect(normalizedX: Double, normalizedY: Double,
    normalizedWidth: Double, normalizedHeight: Double, imageWidth: Int, imageHeight: Int) -> PanelPixelRect {
    let x = Int((normalizedX * Double(imageWidth)).rounded())
    let y = Int(((1 - normalizedY - normalizedHeight) * Double(imageHeight)).rounded())
    let maxX = Int(((normalizedX + normalizedWidth) * Double(imageWidth)).rounded())
    let maxY = Int(((1 - normalizedY) * Double(imageHeight)).rounded())
    return PanelPixelRect(x: x, y: y, width: maxX - x, height: maxY - y)
  }

  static func sourceRect(_ rect: PanelPixelRect, proxyWidth: Int, proxyHeight: Int,
    sourceWidth: Int, sourceHeight: Int) -> PanelPixelRect {
    let x0 = Int((Double(rect.x) * Double(sourceWidth) / Double(proxyWidth)).rounded())
    let y0 = Int((Double(rect.y) * Double(sourceHeight) / Double(proxyHeight)).rounded())
    let x1 = Int((Double(rect.maxX) * Double(sourceWidth) / Double(proxyWidth)).rounded())
    let y1 = Int((Double(rect.maxY) * Double(sourceHeight) / Double(proxyHeight)).rounded())
    return PanelPixelRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
  }

  private static func mapToSource(_ detection: CharacterPanelDetection, proxyWidth: Int, proxyHeight: Int,
    sourceWidth: Int, sourceHeight: Int) -> CharacterPanelDetection {
    var mapped = detection
    mapped.candidates = detection.candidates.map { panel in
      var panel = panel
      panel.sourcePixelRect = sourceRect(panel.sourcePixelRect, proxyWidth: proxyWidth, proxyHeight: proxyHeight,
        sourceWidth: sourceWidth, sourceHeight: sourceHeight)
      panel.evidence.foregroundBounds = panel.evidence.foregroundBounds.map {
        sourceRect($0, proxyWidth: proxyWidth, proxyHeight: proxyHeight,
          sourceWidth: sourceWidth, sourceHeight: sourceHeight)
      }
      return panel
    }
    return mapped
  }

  private static func pixelRect(_ rect: CGRect, imageWidth: Int, imageHeight: Int) -> PanelPixelRect {
    pixelRect(normalizedX: rect.origin.x, normalizedY: rect.origin.y,
      normalizedWidth: rect.width, normalizedHeight: rect.height,
      imageWidth: imageWidth, imageHeight: imageHeight)
  }

  private static func normalizedProxy(imageSource: CGImageSource, maximumDimension: Int) -> CGImage? {
    CGImageSourceCreateThumbnailAtIndex(imageSource, 0, [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: maximumDimension,
    ] as CFDictionary)
  }

  private static func orientationNormalizedDimensions(_ source: CGImageSource) -> (width: Int, height: Int)? {
    guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int else { return nil }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    return (5...8).contains(orientation) ? (height, width) : (width, height)
  }

  static func mergedForegroundComponents(_ components: [PanelPixelRect]) -> [PanelPixelRect] {
    var merged = [PanelPixelRect]()
    for rect in components.sorted(by: { $0.x < $1.x }) {
      if let previous = merged.last, rect.x < previous.maxX {
        merged[merged.count - 1] = PanelPixelRect(x: previous.x, y: min(previous.y, rect.y),
          width: max(previous.maxX, rect.maxX) - previous.x,
          height: max(previous.maxY, rect.maxY) - min(previous.y, rect.y))
      } else { merged.append(rect) }
    }
    return merged
  }

  /// Measure actual vertical edge occupancy, rather than filling contour bounding
  /// boxes that can span several figures and incorrectly cover their empty gutters.
  static func horizontalEdgeColumns(_ image: CGImage) throws -> [Double] {
    let width = image.width, height = image.height
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
      throw CharacterPanelDetectorError.unreadableImage
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let data = context.data?.assumingMemoryBound(to: UInt8.self) else {
      throw CharacterPanelDetectorError.unreadableImage
    }
    var columns = Array(repeating: 0.0, count: width)
    for y in 0..<height { for x in 1..<width {
      let offset = (y * width + x) * 4
      let difference = max(abs(Int(data[offset]) - Int(data[offset - 4])),
        abs(Int(data[offset + 1]) - Int(data[offset - 3])),
        abs(Int(data[offset + 2]) - Int(data[offset - 2])))
      if difference > 20 { columns[x] += 1.0 / Double(height) }
    }}
    return columns
  }

  private static func horizontalComponents(in buffer: CVPixelBuffer) -> [PanelPixelRect] {
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    guard let bytes = try? VisionMaskPixels.bytes(from: buffer, expectedWidth: width, expectedHeight: height) else { return [] }
    var occupied = Array(repeating: false, count: width)
    var minY = Array(repeating: height, count: width), maxY = Array(repeating: -1, count: width)
    for y in 0..<height { for x in 0..<width where bytes[y * width + x] >= 32 {
      occupied[x] = true; minY[x] = min(minY[x], y); maxY[x] = max(maxY[x], y)
    }}
    let bridgeGap = max(2, width / 100)
    var runs: [(Int, Int)] = [], start: Int?, last = -bridgeGap - 1
    for x in 0..<width where occupied[x] {
      if start == nil || x - last > bridgeGap { if let start { runs.append((start, last + 1)) }; start = x }
      last = x
    }
    if let start { runs.append((start, last + 1)) }
    return runs.map { run in
      let lowY = (run.0..<run.1).map { minY[$0] }.min() ?? 0
      let highY = (run.0..<run.1).map { maxY[$0] }.max() ?? height - 1
      return PanelPixelRect(x: run.0, y: lowY, width: run.1 - run.0, height: highY - lowY + 1)
    }.filter { $0.width * $0.height > width * height / 500 }
  }
}
