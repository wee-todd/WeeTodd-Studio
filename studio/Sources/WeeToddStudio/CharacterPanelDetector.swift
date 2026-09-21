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
      let contourRequest = VNDetectContoursRequest()
      contourRequest.contrastAdjustment = 1.0
      let rectangleRequest = VNDetectRectanglesRequest()
      rectangleRequest.maximumObservations = 8
      rectangleRequest.minimumAspectRatio = 0.12
      let handler = VNImageRequestHandler(cgImage: image, orientation: .up)
      try await withTaskCancellationHandler(operation: {
        try handler.perform([foregroundRequest, contourRequest, rectangleRequest])
      }, onCancel: {
        foregroundRequest.cancel(); contourRequest.cancel(); rectangleRequest.cancel()
      })

      var foreground = [PanelPixelRect]()
      var edgeColumns = Array(repeating: 0.0, count: image.width)
      if let observation = foregroundRequest.results?.first {
        let buffer = try observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler)
        foreground = Self.horizontalComponents(in: buffer)
      }
      if let contours = contourRequest.results?.first {
        for contour in contours.topLevelContours {
          let points = contour.normalizedPoints
          guard let first = points.first else { continue }
          var minX = first.x, maxX = first.x, minY = first.y, maxY = first.y
          for point in points.dropFirst() {
            minX = min(minX, point.x); maxX = max(maxX, point.x)
            minY = min(minY, point.y); maxY = max(maxY, point.y)
          }
          let rect = Self.pixelRect(CGRect(x: CGFloat(minX), y: CGFloat(minY),
            width: CGFloat(maxX - minX), height: CGFloat(maxY - minY)),
            imageWidth: image.width, imageHeight: image.height)
          for x in max(0, rect.x)..<min(image.width, rect.maxX) { edgeColumns[x] += Double(max(1, rect.height)) / Double(image.height) }
        }
      }
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
