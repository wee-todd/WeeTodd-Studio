import Foundation
import Vision
import CoreGraphics
import ImageLayout

enum MOSSLayoutVision {
  static func clippedTextBounds(_ box: CGRect) -> NormalizedLayoutRect? {
    guard [box.minX,box.minY,box.width,box.height].allSatisfy(\.isFinite), box.width>0, box.height>0 else { return nil }
    let clipped = box.intersection(CGRect(x:0,y:0,width:1,height:1))
    guard !clipped.isNull, clipped.width>0, clipped.height>0 else { return nil }
    return try? ImageLayoutAnalysisGeometry.visionBounds(x:clipped.minX,y:clipped.minY,width:clipped.width,height:clipped.height)
  }
  static func faces(_ image: CGImage) throws -> [NormalizedLayoutRect] {
    let request = VNDetectFaceRectanglesRequest()
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return try (request.results ?? []).map {
      try ImageLayoutAnalysisGeometry.visionBounds(x: $0.boundingBox.minX, y: $0.boundingBox.minY,
        width: $0.boundingBox.width, height: $0.boundingBox.height)
    }
  }
  static func recognize(_ image: CGImage) throws -> [LayoutOCRObservation] {
    try Task.checkCancellation()
    let full = CGRect(x: 0, y: 0, width: image.width, height: image.height)
    let crops = [full, CGRect(x: 0, y: 0, width: Double(image.width), height: Double(image.height)*0.20).integral.intersection(full),
      CGRect(x: 0, y: 0, width: Double(image.width), height: Double(image.height)*0.55).integral.intersection(full),
      CGRect(x: 0, y: Double(image.height)*0.45, width: Double(image.width), height: Double(image.height)*0.55).integral.intersection(full)]
    var observations: [(String, NormalizedLayoutRect, Double)] = []
    for rect in crops {
      try Task.checkCancellation()
      guard let crop = image.cropping(to: rect) else { continue }
      let request = VNRecognizeTextRequest()
      request.recognitionLevel = .accurate; request.usesLanguageCorrection = false
      request.automaticallyDetectsLanguage = true; request.minimumTextHeight = 0.005
      try VNImageRequestHandler(cgImage: crop, options: [:]).perform([request])
      let local = (request.results ?? []).compactMap { observation -> (String, NormalizedLayoutRect, Double)? in
      guard let candidate = observation.topCandidates(1).first, !candidate.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
      // Vision can extend curved/ornate lettering outside a crop. Keep the visible
      // intersection instead of discarding the entire recognized heading.
      guard let bounds = clippedTextBounds(observation.boundingBox) else { return nil }
      return (candidate.string, .init(x: (rect.minX + bounds.x*rect.width)/Double(image.width),
        y: (rect.minY + bounds.y*rect.height)/Double(image.height), width: bounds.width*rect.width/Double(image.width),
        height: bounds.height*rect.height/Double(image.height)), Double(candidate.confidence))
      }
      observations += local
    }
    guard observations.count <= 1152 else { throw MOSSError("The source contains too many text detections.") }
    let merged = try ImageLayoutOCR.merging(observations.enumerated().map {
      .init(id: $0.offset, text: $0.element.0, bounds: $0.element.1, confidence: $0.element.2)
    })
    guard merged.count <= 384, merged.reduce(0, { $0 + $1.text.count }) <= 30_000 else {
      throw MOSSError("The source has too much detected text for one layout. Use a smaller section or edit a layout manually.")
    }
    return merged
  }
}
