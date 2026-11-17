import CoreFoundation
import CryptoKit
import Foundation
import ImageIO
import ImageLayout

public struct PreparedMOSSLayoutRequest: Sendable {
  public let bytes: Data
  public let requestID: UUID
  public let sourceSHA256: String
  public let mossTokenizerDirectory: String?
  public let modelFamily: ImageLayoutVisionModel
}

public enum MOSSLayoutAnalysisPreparation {
  public static func prepare(source: URL, modelDirectory: String, mossTokenizerDirectory: String? = nil, requestID: UUID = UUID()) throws -> PreparedMOSSLayoutRequest {
    guard source.isFileURL, let family = visionModel(URL(fileURLWithPath: modelDirectory)),
      compatibleModel(URL(fileURLWithPath: modelDirectory)) else {
      throw StudioError.invalid("Choose a local image and a compatible MLX Qwen3.5-4B or Gemma4-12B vision model folder.")
    }
    if family == .gemma4_12b && mossTokenizerDirectory == nil {
      throw StudioError.invalid("Link the MOSS shared components before Gemma analysis so the layout is counted with MOSS’s Qwen tokenizer.")
    }
    if let folder = mossTokenizerDirectory {
      guard folder.hasPrefix("/"), !folder.utf8.contains(0), ["tokenizer.json", "tokenizer_config.json"].allSatisfy({
        FileManager.default.fileExists(atPath: URL(fileURLWithPath: folder).appendingPathComponent($0).path)
      }) else { throw StudioError.invalid("Link a complete MOSS Qwen tokenizer folder in the shared components.") }
    }
    let sha = try digest(source)
    var object: [String: Any] = ["schema": "weetodd-moss-layout-request-v1", "requestID": requestID.uuidString,
      "sourcePath": source.path, "sourceSHA256": sha, "modelDirectory": modelDirectory,
      "targetPixels": 2_000_000, "maxTokens": 8192, "seed": 42]
    if let mossTokenizerDirectory { object["mossTokenizerDirectory"] = mossTokenizerDirectory }
    return .init(bytes: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
      requestID: requestID, sourceSHA256: sha, mossTokenizerDirectory: mossTokenizerDirectory, modelFamily: family)
  }
  public static func digest(_ file: URL) throws -> String {
    let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, let count = values.fileSize, count > 0, count <= 256 * 1024 * 1024 else {
      throw StudioError.invalid("Choose a regular image file up to 256 MiB.")
    }
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }; var hash = SHA256()
    while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty { try Task.checkCancellation(); hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  public static func visionModel(_ directory: URL) -> ImageLayoutVisionModel? {
    let file = directory.appendingPathComponent("config.json")
    guard let properties = try? file.resourceValues(forKeys: [.fileSizeKey]), (properties.fileSize ?? Int.max) <= 1024 * 1024,
      let bytes = try? Data(contentsOf: file), let root = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] else { return nil }
    return try? ImageLayoutVisionModel.identify(root)
  }
  public static func compatibleModel(_ directory: URL) -> Bool {
    guard let family = visionModel(directory),
      let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil),
      files.contains(where: { $0.pathExtension == "safetensors" }),
      family.requiredAssets
        .allSatisfy({ FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }) else { return false }
    return true
  }
  public static func discoverModel(libraryDirectory: String?, sharedDirectory: String?) -> String? {
    for hint in [libraryDirectory, sharedDirectory].compactMap({ $0 }) where !hint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      if let model = discoverModel(libraryHint: hint) { return model }
    }
    return nil
  }
  public static func discoverModel(libraryHint: String) -> String? {
    guard libraryHint.hasPrefix("/") else { return nil }
    var root = URL(fileURLWithPath: libraryHint).standardizedFileURL
    let relatives = ["WeeTodd/assistant-qualification/Qwen3.5-4B-MLX-4bit",
      "WeeTodd/Assistant/Qwen3.5-4B-MLX-4bit", "assistant-qualification/Qwen3.5-4B-MLX-4bit",
      "Qwen3.5-4B-MLX-4bit", "ImgGen/mlx-vlm/gemma-4-12B-it-8bit", "gemma-4-12B-it-8bit"]
    var fallback: String?
    for _ in 0..<8 {
      if compatibleModel(root) { return root.path }
      for relative in relatives {
        let candidate = root.appendingPathComponent(relative)
        if compatibleModel(candidate) {
          if visionModel(candidate) == .gemma4_12b { return candidate.path }
          if fallback == nil { fallback = candidate.path }
        }
      }
      let parent = root.deletingLastPathComponent(); if parent == root { break }; root = parent
    }
    return fallback
  }
}

public struct MOSSLayoutAnalysisProposal: Sendable {
  public let layout: ImageLayoutDraft
  public let warnings: [String]
  public let sourceSHA256, modelIdentity: String
  public let requestID: UUID
  public init(result: [String: Any], output: URL, requestID: UUID, sourceSHA256: String, modelDirectory: String,
    mossTokenizerDirectory: String? = nil, expectedModelFamily: ImageLayoutVisionModel? = nil) throws {
    func integer(_ value: Any?, range: ClosedRange<Int>) -> Int? {
      guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
        number.doubleValue >= Double(range.lowerBound), number.doubleValue <= Double(range.upperBound) else { return nil }
      return number.intValue
    }
    func digest(_ text: String?) -> Bool {
      text?.count == 64 && text!.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }
    guard result["schema"] as? String == "weetodd-moss-layout-result-v1",
      result["requestID"] as? String == requestID.uuidString, result["sourceSHA256"] as? String == sourceSHA256,
      digest(sourceSHA256), let modelID = result["modelIdentity"] as? String, digest(modelID),
      let actualModel = result["modelDirectory"] as? String,
      URL(fileURLWithPath: actualModel).resolvingSymlinksInPath() == URL(fileURLWithPath: modelDirectory).resolvingSymlinksInPath(),
      let recipe = result["recipe"] as? String,
      ["moss-image-layout-vision-ocr-v1", "moss-image-layout-dense-v2", "moss-image-layout-dense-v3", "moss-image-layout-dense-v4"].contains(recipe),
      result["releasedStages"] as? [String] == ["analysis"],
      let width = integer(result["sourceWidth"], range: 1...131_072),
      let height = integer(result["sourceHeight"], range: 1...131_072),
      let snapshotWidth = integer(result["snapshotWidth"], range: 1...2048),
      let snapshotHeight = integer(result["snapshotHeight"], range: 1...2048),
      integer(result["promptTokens"], range: 1...4096) != nil,
      let normalized = result["normalizedRequest"] as? [String: Any],
      normalized["requestID"] as? String == requestID.uuidString,
      normalized["sourceSHA256"] as? String == sourceSHA256,
      normalized["modelDirectory"] as? String == modelDirectory,
      integer(normalized["targetPixels"], range: 2_000_000...2_000_000) != nil,
      let object = result["layout"] as? [String: Any],
      let warnings = result["warnings"] as? [String], warnings.count <= 128, warnings.allSatisfy({ $0.count <= 4096 }) else {
      throw StudioError.invalid("Image analysis does not match the captured source, model or cleanup contract.")
    }
    guard normalized["mossTokenizerDirectory"] as? String == mossTokenizerDirectory else {
      throw StudioError.invalid("Image analysis used a different MOSS counting tokenizer.")
    }
    if let expectedModelFamily {
      let actual = result["modelFamily"] as? String
      guard actual == expectedModelFamily.rawValue || (actual == nil && expectedModelFamily == .qwen35_4b) else {
        throw StudioError.invalid("Image analysis used a different vision model family.")
      }
    }
    if let folder = mossTokenizerDirectory {
      guard let actual = result["mossTokenizerDirectory"] as? String,
        URL(fileURLWithPath: actual).resolvingSymlinksInPath() == URL(fileURLWithPath: folder).resolvingSymlinksInPath(),
        digest(result["mossTokenizerIdentity"] as? String) else { throw StudioError.invalid("The MOSS counting tokenizer identity is missing.") }
    }
    var layout = try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONSerialization.data(withJSONObject: object))
    let expectedSize = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: width, sourceHeight: height)
    let guide = output.appendingPathComponent("source.png").standardizedFileURL
    guard layout.width == expectedSize.width, layout.height == expectedSize.height,
      layout.regions.count <= 128, layout.guideImagePath == guide.path,
      layout.analysis?.requestID == requestID.uuidString, layout.analysis?.sourceSHA256 == sourceSHA256,
      layout.analysis?.sourceWidth == width, layout.analysis?.sourceHeight == height,
      layout.analysis?.modelIdentity == modelID, layout.analysis?.recipe == recipe,
      layout.analysis?.warnings == warnings,
      result["effectivePrompt"] as? String == (try MOSSLayoutCompiler.compile(layout).effectivePrompt),
      let snapshotSHA = result["snapshotSHA256"] as? String, digest(snapshotSHA),
      try MOSSLayoutAnalysisPreparation.digest(guide) == snapshotSHA,
      let source = CGImageSourceCreateWithURL(guide as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
      CGImageSourceGetType(source) as String? == "public.png",
      let info = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      info[kCGImagePropertyPixelWidth] as? Int == snapshotWidth,
      info[kCGImagePropertyPixelHeight] as? Int == snapshotHeight else {
      throw StudioError.invalid("The proposed layout, source guide or compiled prompt is incomplete or changed.")
    }
    // The captured pixels are analysis evidence and an editor overlay, never generation input.
    layout.sourceReconstruction = nil
    self.layout = layout; self.warnings = warnings; self.requestID = requestID
    self.sourceSHA256 = sourceSHA256; self.modelIdentity = modelID
  }
  public func applying(to current: DrawThingsImageDraft, captured: DrawThingsImageDraft, sourceEnabled: Bool = false, noiseLevel: Double = 0.08) throws -> DrawThingsImageDraft {
    guard current == captured, current.executionProvider == .nativeMOSS, !current.hasManagedCharacterPrompt else {
      throw StudioError.invalid("The image draft changed while analyzing. Analyze again for the current draft.")
    }
    var result = current
    result.prompt = layout.summary.caption; result.width = layout.width; result.height = layout.height; result.mossLayout = layout
    var settings = result.mossImage ?? .init(); settings.regionalLayout = true; result.mossImage = settings
    result.mossLayout?.sourceReconstruction?.enabled = sourceEnabled
    result.mossLayout?.sourceReconstruction?.noiseLevel = noiseLevel
    try result.mossLayout?.sourceReconstruction?.validate()
    return result
  }
}
