import AppKit
import CryptoKit
import ImageIO
import StudioCore
import UniformTypeIdentifiers
import Vision

public enum CharacterHeadPreparationError: Error, Equatable {
  case unreadableImage, inputTooLarge, invalidDimensions, selectionRequired, invalidSelection
  case unusableMask, unsupportedRuntime, staleSource
}

public struct CharacterHeadPixels: Equatable, Sendable {
  public var rgba: [UInt8]
  public var whiteRGB: [UInt8]
  public init(rgba: [UInt8], whiteRGB: [UInt8]) { self.rgba = rgba; self.whiteRGB = whiteRGB }
}

public struct CharacterHeadPreparation: Sendable {
  public static let preprocessingVersion = "vision-head-cutout-v2"
  static let maximumInputBytes = 64 * 1_024 * 1_024
  static let maximumSourceDimension = 32_768
  static let maximumSourcePixels = 268_435_456
  static let maximumAnalysisDimension = 2_048
  public init() {}

  public func prepare(source: URL, selection: CharacterHeadSelection?) async throws -> CharacterHeadReference {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-head-\(UUID().uuidString)", isDirectory: true)
    return try await prepare(source: source, selection: selection, outputDirectory: directory)
  }

  public func prepare(source: URL, selection: CharacterHeadSelection?, outputDirectory: URL) async throws -> CharacterHeadReference {
    let worker = Task.detached(priority: .userInitiated) {
      try Task.checkCancellation()
      try Self.validateFileSize(source)
      let initialData = try Data(contentsOf: source)
      guard initialData.count <= Self.maximumInputBytes else {
        throw CharacterHeadPreparationError.inputTooLarge
      }
      let initialHash = Self.hash(initialData)
      guard #available(macOS 14.0, *) else { throw CharacterHeadPreparationError.unsupportedRuntime }
      guard let imageSource = CGImageSourceCreateWithData(initialData as CFData, nil) else {
        throw CharacterHeadPreparationError.unreadableImage
      }
      let sourceDimensions = try Self.sourceDimensions(imageSource)
      guard let image = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, [
              kCGImageSourceCreateThumbnailFromImageAlways: true,
              kCGImageSourceCreateThumbnailWithTransform: true,
              kCGImageSourceThumbnailMaxPixelSize: Self.maximumAnalysisDimension,
            ] as CFDictionary) else { throw CharacterHeadPreparationError.unreadableImage }
      try Task.checkCancellation()
      let chosen: CharacterHeadSelection
      if let selection {
        chosen = selection
      } else {
        let request = VNDetectFaceRectanglesRequest()
        let faceHandler = VNImageRequestHandler(cgImage: image, orientation: .up)
        try await withTaskCancellationHandler(operation: { try faceHandler.perform([request]) },
          onCancel: { request.cancel() })
        let faces = request.results ?? []
        guard faces.count == 1, let face = faces.first else { throw CharacterHeadPreparationError.selectionRequired }
        var rect = CharacterPanelDetector.pixelRect(normalizedX: face.boundingBox.origin.x,
          normalizedY: face.boundingBox.origin.y, normalizedWidth: face.boundingBox.width,
          normalizedHeight: face.boundingBox.height, imageWidth: image.width, imageHeight: image.height)
        let marginX = rect.width / 2, top = rect.height * 3 / 4, bottom = rect.height / 2
        let x = max(0, rect.x - marginX), y = max(0, rect.y - top)
        rect = PanelPixelRect(x: x, y: y, width: min(image.width, rect.maxX + marginX) - x,
          height: min(image.height, rect.maxY + bottom) - y)
        let sourceRect = try Self.mapRect(rect, from: (image.width, image.height),
          to: sourceDimensions)
        chosen = CharacterHeadSelection(crop: sourceRect)
      }
      guard chosen.crop.isValid(inWidth: sourceDimensions.width,
        height: sourceDimensions.height) else {
        throw CharacterHeadPreparationError.invalidSelection
      }
      let proxyCrop = try Self.mapRect(chosen.crop, from: sourceDimensions,
        to: (image.width, image.height))
      guard let cropImage = image.cropping(to: CGRect(x: proxyCrop.x, y: proxyCrop.y,
              width: proxyCrop.width, height: proxyCrop.height)) else {
        throw CharacterHeadPreparationError.invalidSelection
      }

      let request = VNGenerateForegroundInstanceMaskRequest()
      let handler = VNImageRequestHandler(cgImage: cropImage, orientation: .up)
      try await withTaskCancellationHandler(operation: { try handler.perform([request]) },
        onCancel: { request.cancel() })
      guard let observation = request.results?.first else { throw CharacterHeadPreparationError.unusableMask }
      let instances: IndexSet
      if let selected = chosen.foregroundInstanceIndex {
        guard observation.allInstances.contains(selected) else { throw CharacterHeadPreparationError.invalidSelection }
        instances = IndexSet(integer: selected)
      } else if observation.allInstances.count == 1, let only = observation.allInstances.first {
        instances = IndexSet(integer: only)
      } else {
        throw CharacterHeadPreparationError.selectionRequired
      }
      let maskBuffer = try observation.generateScaledMaskForImage(forInstances: instances, from: handler)
      let rgba = try Self.rgbaBytes(cropImage)
      let mask = try VisionMaskPixels.bytes(from: maskBuffer,
        expectedWidth: cropImage.width, expectedHeight: cropImage.height)
      let pixels = try Self.applyMask(rgba: rgba, mask: mask, width: cropImage.width, height: cropImage.height)
      try Task.checkCancellation()

      try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
      let originalURL = outputDirectory.appendingPathComponent("original" + (source.pathExtension.isEmpty ? ".png" : ".\(source.pathExtension)"))
      let cropURL = outputDirectory.appendingPathComponent("head-crop.png")
      let maskURL = outputDirectory.appendingPathComponent("head-mask.png")
      let rgbaURL = outputDirectory.appendingPathComponent("head-cutout.png")
      let whiteURL = outputDirectory.appendingPathComponent("head-white.png")
      try initialData.write(to: originalURL, options: .atomic)
      try Self.writePNG(cropImage, to: cropURL)
      try Self.writeGray(mask, width: cropImage.width, height: cropImage.height, to: maskURL)
      try Self.writeRGBA(pixels.rgba, width: cropImage.width, height: cropImage.height, to: rgbaURL)
      try Self.writeRGB(pixels.whiteRGB, width: cropImage.width, height: cropImage.height, to: whiteURL)
      try Task.checkCancellation()
      try Self.validateFileSize(source)
      guard Self.hash(try Data(contentsOf: source)) == initialHash else {
        throw CharacterHeadPreparationError.staleSource
      }
      let artifactHashes = ["headCrop": Self.hash(try Data(contentsOf: cropURL)),
        "mask": Self.hash(try Data(contentsOf: maskURL)),
        "rgbaCutout": Self.hash(try Data(contentsOf: rgbaURL)),
        "whiteMatte": Self.hash(try Data(contentsOf: whiteURL))]
      return CharacterHeadReference(originalAssetID: UUID(), headCropAssetID: UUID(), maskAssetID: UUID(),
        rgbaCutoutAssetID: UUID(), whiteMatteAssetID: UUID(), sourcePath: originalURL.path,
        headCropPath: cropURL.path, maskPath: maskURL.path, rgbaCutoutPath: rgbaURL.path,
        whiteMattePath: whiteURL.path, crop: chosen.crop, sourceSHA256: initialHash,
        preprocessingVersion: Self.preprocessingVersion, artifactSHA256: artifactHashes)
    }
    return try await withTaskCancellationHandler(operation: {
      let result = try await worker.value
      try Task.checkCancellation()
      return result
    }, onCancel: { worker.cancel() })
  }

  public static func resolveSelection(candidates: [PanelPixelRect], selection: CharacterHeadSelection?) throws -> CharacterHeadSelection {
    if let selection { return selection }
    guard candidates.count == 1, let crop = candidates.first else { throw CharacterHeadPreparationError.selectionRequired }
    return CharacterHeadSelection(crop: crop)
  }

  static func orientedDimensions(rawWidth: Int, rawHeight: Int, orientation: Int)
    throws -> (width: Int, height: Int) {
    guard rawWidth > 0, rawHeight > 0,
          rawWidth <= maximumSourceDimension, rawHeight <= maximumSourceDimension,
          rawWidth <= maximumSourcePixels / rawHeight else {
      throw CharacterHeadPreparationError.invalidDimensions
    }
    return [5, 6, 7, 8].contains(orientation)
      ? (rawHeight, rawWidth) : (rawWidth, rawHeight)
  }

  static func mapRect(_ rect: PanelPixelRect, from source: (width: Int, height: Int),
                      to target: (width: Int, height: Int)) throws -> PanelPixelRect {
    guard rect.isValid(inWidth: source.width, height: source.height),
          source.width > 0, source.height > 0, target.width > 0, target.height > 0 else {
      throw CharacterHeadPreparationError.invalidSelection
    }
    let x = rect.x * target.width / source.width
    let y = rect.y * target.height / source.height
    let maxX = (rect.maxX * target.width + source.width - 1) / source.width
    let maxY = (rect.maxY * target.height + source.height - 1) / source.height
    return PanelPixelRect(x: max(0, x), y: max(0, y),
      width: min(target.width, maxX) - max(0, x),
      height: min(target.height, maxY) - max(0, y))
  }

  public static func applyMask(rgba: [UInt8], mask: [UInt8], width: Int, height: Int) throws -> CharacterHeadPixels {
    guard width > 0, height > 0, rgba.count == width * height * 4, mask.count == width * height,
          mask.contains(where: { $0 > 0 }) else { throw CharacterHeadPreparationError.unusableMask }
    var output = rgba, white = [UInt8](); white.reserveCapacity(width * height * 3)
    for index in 0..<(width * height) {
      let sourceAlpha = Int(rgba[index * 4 + 3]), maskAlpha = Int(mask[index])
      let alpha = UInt8(sourceAlpha * maskAlpha / 255)
      output[index * 4 + 3] = alpha
      for channel in 0..<3 {
        let value = Int(rgba[index * 4 + channel])
        let alphaValue = Int(alpha)
        let blended = (value * alphaValue + 255 * (255 - alphaValue) + 127) / 255
        white.append(UInt8(blended))
      }
    }
    return CharacterHeadPixels(rgba: output, whiteRGB: white)
  }

  static func unpremultiply(_ rgba: [UInt8]) -> [UInt8] {
    guard rgba.count.isMultiple(of: 4) else { return rgba }
    var straight = rgba
    for index in stride(from: 0, to: rgba.count, by: 4) {
      let alpha = Int(rgba[index + 3])
      guard alpha > 0 else {
        straight[index] = 0; straight[index + 1] = 0; straight[index + 2] = 0
        continue
      }
      for channel in 0..<3 {
        straight[index + channel] = UInt8(min(255, (Int(rgba[index + channel]) * 255 + alpha / 2) / alpha))
      }
    }
    return straight
  }

  private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
  private static func validateFileSize(_ source: URL) throws {
    let values = try source.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, let size = values.fileSize else {
      throw CharacterHeadPreparationError.unreadableImage
    }
    guard size <= maximumInputBytes else { throw CharacterHeadPreparationError.inputTooLarge }
  }
  private static func sourceDimensions(_ source: CGImageSource) throws -> (width: Int, height: Int) {
    guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int else {
      throw CharacterHeadPreparationError.invalidDimensions
    }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    return try orientedDimensions(rawWidth: width, rawHeight: height, orientation: orientation)
  }
  private static func rgbaBytes(_ image: CGImage) throws -> [UInt8] {
    var bytes = Array(repeating: UInt8(0), count: image.width * image.height * 4)
    guard let context = CGContext(data: &bytes, width: image.width, height: image.height, bitsPerComponent: 8,
      bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw CharacterHeadPreparationError.unreadableImage }
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return unpremultiply(bytes)
  }
  private static func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw CharacterHeadPreparationError.unreadableImage }
    CGImageDestinationAddImage(destination, image, nil); guard CGImageDestinationFinalize(destination) else { throw CharacterHeadPreparationError.unreadableImage }
  }
  private static func writeRGBA(_ bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    guard let provider = CGDataProvider(data: Data(bytes) as CFData), let image = CGImage(width: width, height: height,
      bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw CharacterHeadPreparationError.unreadableImage }
    try writePNG(image, to: url)
  }
  private static func writeRGB(_ bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    guard let provider = CGDataProvider(data: Data(bytes) as CFData), let image = CGImage(width: width, height: height,
      bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw CharacterHeadPreparationError.unreadableImage }
    try writePNG(image, to: url)
  }
  private static func writeGray(_ bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    guard let provider = CGDataProvider(data: Data(bytes) as CFData), let image = CGImage(width: width, height: height,
      bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw CharacterHeadPreparationError.unreadableImage }
    try writePNG(image, to: url)
  }
}

extension CharacterHeadPreparationError: LocalizedError {
  public var errorDescription: String? {
    switch self {
    case .selectionRequired: return "Select a crop around one complete head under Reference face, then remove its background again."
    case .invalidSelection: return "The head crop must stay inside the source image and select a valid foreground subject."
    case .unusableMask: return "No usable head foreground was found. Adjust the crop or choose a clearer reference face."
    case .unreadableImage: return "The reference image could not be decoded. Choose a supported image file."
    case .unsupportedRuntime: return "Reference background removal requires macOS 14 or newer."
    case .staleSource: return "The source face changed during preparation. Prepare the current image again."
    case .invalidDimensions: return "The reference image has invalid or excessive pixel dimensions. Choose a smaller image."
    case .inputTooLarge: return "Choose a reference face image of at most 64 MiB with supported dimensions."
    }
  }
}
