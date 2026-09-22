import CoreGraphics
import Foundation
import ImageIO
import StudioCore
import UniformTypeIdentifiers
import Vision

public struct CharacterSourceDetailImage: Codable, Equatable, Sendable {
  public var path: String
  public var label: String
  public var sha256: String
  public init(path: String, label: String, sha256: String) {
    self.path = path; self.label = label; self.sha256 = sha256
  }
}

public struct CharacterDetailImagePreparationResult: Codable, Equatable, Sendable {
  public var sourceSHA256: String
  public var subjectOverviewImage: CharacterSourceDetailImage?
  public var detailImage: CharacterSourceDetailImage?
  public var clothingDetailImage: CharacterSourceDetailImage?
  public var diagnostics: [String]
  public var preprocessingVersion: String
  public init(sourceSHA256: String, detailImage: CharacterSourceDetailImage?, diagnostics: [String],
              preprocessingVersion: String, clothingDetailImage: CharacterSourceDetailImage? = nil,
              subjectOverviewImage: CharacterSourceDetailImage? = nil) {
    self.sourceSHA256 = sourceSHA256; self.detailImage = detailImage
    self.subjectOverviewImage = subjectOverviewImage
    self.clothingDetailImage = clothingDetailImage
    self.diagnostics = diagnostics; self.preprocessingVersion = preprocessingVersion
  }

  public var sourceDetailImages: [CharacterSourceDetailImage] { [detailImage, clothingDetailImage].compactMap { $0 } }
}

public enum CharacterDetailImagePreparationError: Error, Equatable {
  case unreadableImage, unsupportedRuntime, invalidCrop, staleSource
}

public struct CharacterDetailImagePreparation: Sendable {
  struct VisionEvidence: Sendable {
    var faces: [CGRect]
    var people: [CGRect]
    var subjectMask: [UInt8]?
    var maskWidth: Int
    var maskHeight: Int
  }
  typealias VisionDetector = @Sendable (CGImage) async throws -> VisionEvidence

  public static let preprocessingVersion = "vision-primary-subject-detail-v3"
  public static let detailLabel = "primary head and face detail"
  public static let clothingDetailLabel = "primary torso and lap detail"
  public static let subjectOverviewLabel = "isolated primary character overview"
  public static let boundedOverviewLabel = "bounded character overview"
  private let visionDetector: VisionDetector

  public init() {
    visionDetector = { preview in try await Self.detectPrimarySubject(in: preview) }
  }

  init(visionDetector: @escaping VisionDetector) { self.visionDetector = visionDetector }

  public func prepare(source: URL, outputDirectory: URL) async throws -> CharacterDetailImagePreparationResult {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      let sourceHash = try CharacterArtifactHash.file(source.path)
      guard #available(macOS 14.0, *),
            let imageSource = Self.imageSource(source),
            let preview = Self.thumbnail(imageSource, maximumDimension: 1600) else {
        throw CharacterDetailImagePreparationError.unreadableImage
      }
      let evidence: VisionEvidence
      var visionDiagnostics: [String] = []
      do {
        evidence = try await visionDetector(preview)
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        try Task.checkCancellation()
        evidence = .init(faces: [], people: [], subjectMask: nil,
          maskWidth: preview.width, maskHeight: preview.height)
        visionDiagnostics.append("Apple Vision subject isolation failed. Review the bounded overview or choose a clearer single-character image.")
      }
      let result = try Self.prepare(imageSource: imageSource, sourceHash: sourceHash,
        outputDirectory: outputDirectory, normalizedFaces: evidence.faces,
        normalizedPeople: evidence.people, subjectMask: evidence.subjectMask,
        subjectMaskWidth: evidence.maskWidth, subjectMaskHeight: evidence.maskHeight,
        initialDiagnostics: visionDiagnostics)
      guard try CharacterArtifactHash.file(source.path) == sourceHash else {
        throw CharacterDetailImagePreparationError.staleSource
      }
      return result
    }
    return try await withTaskCancellationHandler(operation: {
      let result = try await worker.value
      try Task.checkCancellation()
      return result
    }, onCancel: { worker.cancel() })
  }

  private static func detectPrimarySubject(in preview: CGImage) async throws -> VisionEvidence {
    let request = VNDetectFaceRectanglesRequest()
    let peopleRequest = VNDetectHumanRectanglesRequest()
    let foregroundRequest = VNGenerateForegroundInstanceMaskRequest()
    peopleRequest.upperBodyOnly = false
    let handler = VNImageRequestHandler(cgImage: preview, orientation: .up)
    try await withTaskCancellationHandler(operation: {
      try handler.perform([request, peopleRequest, foregroundRequest])
    }, onCancel: { request.cancel(); peopleRequest.cancel(); foregroundRequest.cancel() })
    try Task.checkCancellation()
    let faces = (request.results ?? []).map(\.boundingBox)
    var selectedMask: [UInt8]?
    if faces.count == 1, let face = faces.first,
       let observation = foregroundRequest.results?.first {
      var matches: [[UInt8]] = []
      for instance in observation.allInstances {
        try Task.checkCancellation()
        let buffer = try observation.generateScaledMaskForImage(
          forInstances: IndexSet(integer: instance), from: handler)
        let bytes = try VisionMaskPixels.bytes(from: buffer,
          expectedWidth: preview.width, expectedHeight: preview.height)
        if mask(bytes, width: preview.width, height: preview.height, containsFace: face) {
          matches.append(bytes)
        }
      }
      if matches.count == 1 { selectedMask = matches[0] }
    }
    return .init(faces: faces, people: (peopleRequest.results ?? []).map(\.boundingBox),
      subjectMask: selectedMask, maskWidth: preview.width, maskHeight: preview.height)
  }

  static func prepare(source: URL, outputDirectory: URL,
    normalizedFaces: [CGRect], normalizedPeople: [CGRect]? = nil,
    subjectMask: [UInt8]? = nil, subjectMaskWidth: Int = 0,
    subjectMaskHeight: Int = 0) async throws -> CharacterDetailImagePreparationResult {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      let sourceHash = try CharacterArtifactHash.file(source.path)
      guard let imageSource = imageSource(source) else {
        throw CharacterDetailImagePreparationError.unreadableImage
      }
      let result = try prepare(imageSource: imageSource, sourceHash: sourceHash,
        outputDirectory: outputDirectory, normalizedFaces: normalizedFaces, normalizedPeople: normalizedPeople,
        subjectMask: subjectMask, subjectMaskWidth: subjectMaskWidth, subjectMaskHeight: subjectMaskHeight,
        initialDiagnostics: [])
      guard try CharacterArtifactHash.file(source.path) == sourceHash else {
        throw CharacterDetailImagePreparationError.staleSource
      }
      return result
    }
    return try await withTaskCancellationHandler(operation: { try await worker.value },
      onCancel: { worker.cancel() })
  }

  static func headContextRect(normalizedVisionFace face: CGRect,
    imageWidth: Int, imageHeight: Int) -> CGRect {
    let faceX = face.minX * CGFloat(imageWidth)
    let faceY = (1 - face.maxY) * CGFloat(imageHeight)
    let faceWidth = face.width * CGFloat(imageWidth)
    let faceHeight = face.height * CGFloat(imageHeight)
    let left = max(0, floor(faceX - faceWidth * 0.65))
    let top = max(0, floor(faceY - faceHeight))
    let right = min(CGFloat(imageWidth), ceil(faceX + faceWidth * 1.65))
    let bottom = min(CGFloat(imageHeight), ceil(faceY + faceHeight * 2.10))
    return CGRect(x: left, y: top, width: max(0, right - left), height: max(0, bottom - top))
  }

  static func clothingContextRect(normalizedVisionFace face: CGRect,
    normalizedPeople: [CGRect], imageWidth: Int, imageHeight: Int) -> CGRect? {
    let bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
    guard face.width > 0, face.height > 0, bounds.contains(face) else { return nil }
    let candidates = normalizedPeople.enumerated().filter { _, person in
      person.contains(CGPoint(x: face.midX, y: face.midY))
    }
    guard candidates.count == 1, let (index, person) = candidates.first else { return nil }
    let body = person.intersection(bounds)
    let bottom = body.minY
    let top = min(face.minY, body.maxY)
    let region = CGRect(x: body.minX, y: bottom, width: body.width, height: max(0, top - bottom))
    guard region.width > 0, region.height >= face.height else { return nil }
    // A face alone does not prove which visible limbs belong to that person.
    // Skip intersecting people instead of supplying a misleading close crop.
    guard !normalizedPeople.enumerated().contains(where: { otherIndex, other in
      otherIndex != index && !region.intersection(other).isNull
        && region.intersection(other).width * region.intersection(other).height > 0
    }) else { return nil }
    return CGRect(x: region.minX * CGFloat(imageWidth),
      y: (1 - region.maxY) * CGFloat(imageHeight),
      width: region.width * CGFloat(imageWidth), height: region.height * CGFloat(imageHeight)).integral
  }

  private static func prepare(imageSource: CGImageSource, sourceHash: String, outputDirectory: URL,
    normalizedFaces: [CGRect], normalizedPeople: [CGRect]?, subjectMask: [UInt8]?,
    subjectMaskWidth: Int, subjectMaskHeight: Int, initialDiagnostics: [String]) throws -> CharacterDetailImagePreparationResult {
    try Task.checkCancellation()
    guard let boundedSource = thumbnail(imageSource, maximumDimension: 1600) else {
      throw CharacterDetailImagePreparationError.unreadableImage
    }
    let artifactDirectory = outputDirectory.appendingPathComponent(
      "\(sourceHash)-\(preprocessingVersion)", isDirectory: true)
    try FileManager.default.createDirectory(at: artifactDirectory, withIntermediateDirectories: true)
    let validMask = subjectMask.flatMap { mask -> [UInt8]? in
      guard subjectMaskWidth == boundedSource.width, subjectMaskHeight == boundedSource.height,
            mask.count == boundedSource.width * boundedSource.height,
            mask.contains(where: { $0 >= 32 }) else { return nil }
      return mask
    }
    var diagnostics = initialDiagnostics
    var overviewLabel = subjectOverviewLabel
    let isolatedSource: CGImage
    let overviewSource: CGImage
    if let validMask, let white = whiteMatted(boundedSource, mask: validMask),
       let bounds = maskBounds(validMask, width: boundedSource.width, height: boundedSource.height),
       let cropped = white.cropping(to: expanded(bounds, imageWidth: boundedSource.width,
         imageHeight: boundedSource.height)) {
      isolatedSource = white; overviewSource = cropped
    } else {
      isolatedSource = boundedSource; overviewSource = boundedSource
      overviewLabel = boundedOverviewLabel
      diagnostics.append("Apple Vision could not isolate one primary person. Review the source or choose an image with one clearly visible character; analysis will use a bounded overview without claiming background removal.")
    }
    guard let overview = resizedTo512(overviewSource) else {
      throw CharacterDetailImagePreparationError.invalidCrop
    }
    let overviewURL = artifactDirectory.appendingPathComponent("primary-character-overview.jpg")
    try writeJPEG(overview, to: overviewURL)
    var result = CharacterDetailImagePreparationResult(sourceSHA256: sourceHash,
      detailImage: nil, diagnostics: diagnostics, preprocessingVersion: preprocessingVersion,
      subjectOverviewImage: .init(path: overviewURL.path, label: overviewLabel,
        sha256: try CharacterArtifactHash.file(overviewURL.path)))

    guard normalizedFaces.count == 1, let face = normalizedFaces.first else {
      let message = normalizedFaces.isEmpty
        ? "No face was detected; character extraction will use the overview image only."
        : "Detected \(normalizedFaces.count) faces; no automatic detail crop was created to avoid identity confusion."
      result.diagnostics.append(message)
      try writeManifest(result, to: artifactDirectory)
      return result
    }
    guard face.width >= 0.015, face.height >= 0.015 else {
      result.diagnostics.append("The detected face was too small for a useful head detail; character extraction will use the overview image only.")
      try writeManifest(result, to: artifactDirectory)
      return result
    }
    guard validMask != nil else {
      try writeManifest(result, to: artifactDirectory)
      return result
    }
    let cropRect = headContextRect(normalizedVisionFace: face,
      imageWidth: boundedSource.width, imageHeight: boundedSource.height).integral
    guard cropRect.width >= 32, cropRect.height >= 32,
          cropRect.minX >= 0, cropRect.minY >= 0,
          cropRect.maxX <= CGFloat(boundedSource.width), cropRect.maxY <= CGFloat(boundedSource.height),
          let crop = isolatedSource.cropping(to: cropRect), let resized = resizedTo512(crop) else {
      throw CharacterDetailImagePreparationError.invalidCrop
    }
    try Task.checkCancellation()
    let detailURL = artifactDirectory.appendingPathComponent("primary-head-face.jpg")
    try writeJPEG(resized, to: detailURL)
    let detailHash = try CharacterArtifactHash.file(detailURL.path)
    result.detailImage = .init(path: detailURL.path, label: detailLabel, sha256: detailHash)
    if let normalizedPeople {
      if let rect = clothingContextRect(normalizedVisionFace: face, normalizedPeople: normalizedPeople,
          imageWidth: boundedSource.width, imageHeight: boundedSource.height),
         rect.width >= 32, rect.height >= 32,
         let crop = isolatedSource.cropping(to: rect), let clothing = resizedTo512(crop) {
        let clothingURL = artifactDirectory.appendingPathComponent("primary-torso-lap.jpg")
        try writeJPEG(clothing, to: clothingURL)
        result.clothingDetailImage = .init(path: clothingURL.path, label: clothingDetailLabel,
          sha256: try CharacterArtifactHash.file(clothingURL.path))
      } else {
        result.diagnostics.append("The primary wearer's torso/lap is unavailable or ambiguous; wardrobe analysis will use the overview only.")
      }
    }
    try writeManifest(result, to: artifactDirectory)
    try Task.checkCancellation()
    return result
  }

  private static func mask(_ bytes: [UInt8], width: Int, height: Int,
    containsFace face: CGRect) -> Bool {
    guard bytes.count == width * height else { return false }
    let centerX = min(width - 1, max(0, Int(face.midX * CGFloat(width))))
    let centerY = min(height - 1, max(0, Int((1 - face.midY) * CGFloat(height))))
    let radius = max(1, min(width, height) / 100)
    var hits = 0, samples = 0
    for y in max(0, centerY - radius)...min(height - 1, centerY + radius) {
      for x in max(0, centerX - radius)...min(width - 1, centerX + radius) {
        samples += 1; if bytes[y * width + x] >= 64 { hits += 1 }
      }
    }
    return samples > 0 && hits * 2 >= samples
  }

  private static func maskBounds(_ mask: [UInt8], width: Int, height: Int) -> CGRect? {
    var minX = width, minY = height, maxX = -1, maxY = -1
    for y in 0..<height { for x in 0..<width where mask[y * width + x] >= 32 {
      minX = min(minX, x); minY = min(minY, y); maxX = max(maxX, x); maxY = max(maxY, y)
    }}
    guard maxX >= minX, maxY >= minY else { return nil }
    return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
  }

  private static func expanded(_ rect: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
    let margin = max(4, Int(ceil(max(rect.width, rect.height) * 0.04)))
    let x = max(0, Int(rect.minX) - margin), y = max(0, Int(rect.minY) - margin)
    let right = min(imageWidth, Int(rect.maxX) + margin)
    let bottom = min(imageHeight, Int(rect.maxY) + margin)
    return CGRect(x: x, y: y, width: right - x, height: bottom - y)
  }

  private static func whiteMatted(_ image: CGImage, mask: [UInt8]) -> CGImage? {
    guard mask.count == image.width * image.height else { return nil }
    var rgba = Array(repeating: UInt8(0), count: image.width * image.height * 4)
    guard let context = CGContext(data: &rgba, width: image.width, height: image.height,
      bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    for pixel in 0..<mask.count {
      let alpha = Int(mask[pixel])
      for channel in 0..<3 {
        let source = Int(rgba[pixel * 4 + channel])
        rgba[pixel * 4 + channel] = UInt8((source * alpha + 255 * (255 - alpha) + 127) / 255)
      }
      rgba[pixel * 4 + 3] = 255
    }
    guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
    return CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
      bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent)
  }

  private static func writeManifest(_ result: CharacterDetailImagePreparationResult,
    to artifactDirectory: URL) throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(result).write(to: artifactDirectory.appendingPathComponent("manifest.json"), options: .atomic)
  }

  private static func thumbnail(_ source: CGImageSource, maximumDimension: Int) -> CGImage? {
    CGImageSourceCreateThumbnailAtIndex(source, 0, [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: maximumDimension,
      kCGImageSourceShouldCacheImmediately: false,
    ] as CFDictionary)
  }

  private static func imageSource(_ url: URL) -> CGImageSource? {
    CGImageSourceCreateWithURL(url as CFURL,
      [kCGImageSourceShouldCache: false] as CFDictionary)
  }

  private static func resizedTo512(_ image: CGImage) -> CGImage? {
    let scale = min(1, 512 / CGFloat(max(image.width, image.height)))
    let width = max(1, Int((CGFloat(image.width) * scale).rounded()))
    let height = max(1, Int((CGFloat(image.height) * scale).rounded()))
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
  }

  private static func writeJPEG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(
      url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
      throw CharacterDetailImagePreparationError.unreadableImage
    }
    CGImageDestinationAddImage(destination, image,
      [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
      throw CharacterDetailImagePreparationError.unreadableImage
    }
  }
}
