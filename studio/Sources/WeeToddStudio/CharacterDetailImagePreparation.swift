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
  public var detailImage: CharacterSourceDetailImage?
  public var clothingDetailImage: CharacterSourceDetailImage?
  public var diagnostics: [String]
  public var preprocessingVersion: String
  public init(sourceSHA256: String, detailImage: CharacterSourceDetailImage?, diagnostics: [String],
              preprocessingVersion: String, clothingDetailImage: CharacterSourceDetailImage? = nil) {
    self.sourceSHA256 = sourceSHA256; self.detailImage = detailImage
    self.clothingDetailImage = clothingDetailImage
    self.diagnostics = diagnostics; self.preprocessingVersion = preprocessingVersion
  }

  public var sourceDetailImages: [CharacterSourceDetailImage] { [detailImage, clothingDetailImage].compactMap { $0 } }
}

public enum CharacterDetailImagePreparationError: Error, Equatable {
  case unreadableImage, unsupportedRuntime, invalidCrop, staleSource
}

public struct CharacterDetailImagePreparation: Sendable {
  public static let preprocessingVersion = "vision-primary-head-wardrobe-detail-v2"
  public static let detailLabel = "primary head and face detail"
  public static let clothingDetailLabel = "primary torso and lap detail"
  public init() {}

  public func prepare(source: URL, outputDirectory: URL) async throws -> CharacterDetailImagePreparationResult {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      let sourceHash = try CharacterArtifactHash.file(source.path)
      guard #available(macOS 14.0, *),
            let imageSource = Self.imageSource(source),
            let preview = Self.thumbnail(imageSource, maximumDimension: 1600) else {
        throw CharacterDetailImagePreparationError.unreadableImage
      }
      let request = VNDetectFaceRectanglesRequest()
      let peopleRequest = VNDetectHumanRectanglesRequest()
      peopleRequest.upperBodyOnly = false
      let handler = VNImageRequestHandler(cgImage: preview, orientation: .up)
      try await withTaskCancellationHandler(operation: { try handler.perform([request, peopleRequest]) },
        onCancel: { request.cancel(); peopleRequest.cancel() })
      try Task.checkCancellation()
      let faces = (request.results ?? []).map(\.boundingBox)
      let result = try Self.prepare(imageSource: imageSource, sourceHash: sourceHash,
        outputDirectory: outputDirectory, normalizedFaces: faces,
        normalizedPeople: (peopleRequest.results ?? []).map(\.boundingBox))
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

  static func prepare(source: URL, outputDirectory: URL,
    normalizedFaces: [CGRect], normalizedPeople: [CGRect]? = nil) async throws -> CharacterDetailImagePreparationResult {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      let sourceHash = try CharacterArtifactHash.file(source.path)
      guard let imageSource = imageSource(source) else {
        throw CharacterDetailImagePreparationError.unreadableImage
      }
      let result = try prepare(imageSource: imageSource, sourceHash: sourceHash,
        outputDirectory: outputDirectory, normalizedFaces: normalizedFaces, normalizedPeople: normalizedPeople)
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
    normalizedFaces: [CGRect], normalizedPeople: [CGRect]?) throws -> CharacterDetailImagePreparationResult {
    try Task.checkCancellation()
    guard normalizedFaces.count == 1, let face = normalizedFaces.first else {
      let message = normalizedFaces.isEmpty
        ? "No face was detected; character extraction will use the overview image only."
        : "Detected \(normalizedFaces.count) faces; no automatic detail crop was created to avoid identity confusion."
      return .init(sourceSHA256: sourceHash, detailImage: nil, diagnostics: [message],
        preprocessingVersion: preprocessingVersion)
    }
    guard face.width >= 0.015, face.height >= 0.015,
          let boundedSource = thumbnail(imageSource, maximumDimension: 2048) else {
      return .init(sourceSHA256: sourceHash, detailImage: nil,
        diagnostics: ["The detected face was too small for a useful head detail; character extraction will use the overview image only."],
        preprocessingVersion: preprocessingVersion)
    }
    let cropRect = headContextRect(normalizedVisionFace: face,
      imageWidth: boundedSource.width, imageHeight: boundedSource.height).integral
    guard cropRect.width >= 32, cropRect.height >= 32,
          cropRect.minX >= 0, cropRect.minY >= 0,
          cropRect.maxX <= CGFloat(boundedSource.width), cropRect.maxY <= CGFloat(boundedSource.height),
          let crop = boundedSource.cropping(to: cropRect), let resized = resizedTo512(crop) else {
      throw CharacterDetailImagePreparationError.invalidCrop
    }
    try Task.checkCancellation()
    let artifactDirectory = outputDirectory.appendingPathComponent(
      "\(sourceHash)-\(preprocessingVersion)", isDirectory: true)
    try FileManager.default.createDirectory(at: artifactDirectory, withIntermediateDirectories: true)
    let detailURL = artifactDirectory.appendingPathComponent("primary-head-face.jpg")
    try writeJPEG(resized, to: detailURL)
    let detailHash = try CharacterArtifactHash.file(detailURL.path)
    var result = CharacterDetailImagePreparationResult(sourceSHA256: sourceHash,
      detailImage: .init(path: detailURL.path, label: detailLabel, sha256: detailHash), diagnostics: [],
      preprocessingVersion: preprocessingVersion)
    if let normalizedPeople {
      if let rect = clothingContextRect(normalizedVisionFace: face, normalizedPeople: normalizedPeople,
          imageWidth: boundedSource.width, imageHeight: boundedSource.height),
         rect.width >= 32, rect.height >= 32,
         let crop = boundedSource.cropping(to: rect), let clothing = resizedTo512(crop) {
        let clothingURL = artifactDirectory.appendingPathComponent("primary-torso-lap.jpg")
        try writeJPEG(clothing, to: clothingURL)
        result.clothingDetailImage = .init(path: clothingURL.path, label: clothingDetailLabel,
          sha256: try CharacterArtifactHash.file(clothingURL.path))
      } else {
        result.diagnostics.append("The primary wearer's torso/lap is unavailable or ambiguous; wardrobe analysis will use the overview only.")
      }
    }
    let manifest = artifactDirectory.appendingPathComponent("manifest.json")
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(result).write(to: manifest, options: .atomic)
    try Task.checkCancellation()
    return result
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
