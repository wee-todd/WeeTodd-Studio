import Foundation
import ImageLayout

/// Folder preferences only. Recipes, adapters and model weights remain separate.
public struct MOSSModelLinks: Codable, Equatable, Sendable {
  public var format = "weetodd-moss-model-links-v1"
  public var libraryDirectory = ""
  public var sharedDirectory = ""
  public var transformerDirectories: [String: String] = [:]
  public var analyzerDirectory = ""
  public init() {}

  public mutating func remember(_ query: MOSSModelQuery) {
    if Self.absolute(query.libraryDirectory) { libraryDirectory = query.libraryDirectory }
    if Self.absolute(query.sharedDirectory) { sharedDirectory = query.sharedDirectory }
    if ["sft", "base"].contains(query.variant), query.quantizationBits == nil || [4, 8].contains(query.quantizationBits!),
      Self.absolute(query.transformerDirectory) { transformerDirectories[query.profileKey] = query.transformerDirectory }
  }
  public func applying(to query: MOSSModelQuery) -> MOSSModelQuery {
    var value = query
    if value.libraryDirectory.isEmpty { value.libraryDirectory = libraryDirectory }
    if value.sharedDirectory.isEmpty { value.sharedDirectory = sharedDirectory }
    if value.transformerDirectory.isEmpty { value.transformerDirectory = transformerDirectories[value.profileKey] ?? "" }
    return value
  }
  public static func read(from url: URL) throws -> Self {
    let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
    let data = try file.read(upToCount: 65537) ?? Data()
    guard data.count <= 65536 else { throw StudioError.invalid("MOSS model links exceed 64 KiB.") }
    let value = try JSONDecoder().decode(Self.self, from: data)
    let paths = [value.libraryDirectory, value.sharedDirectory, value.analyzerDirectory] + Array(value.transformerDirectories.values)
    let keys = Set(["sft", "base"].flatMap { variant in ["bf16", "q4", "q8"].map { variant + ":" + $0 } })
    guard value.format == "weetodd-moss-model-links-v1", paths.allSatisfy({ $0.isEmpty || absolute($0) }),
      Set(value.transformerDirectories.keys).isSubset(of: keys) else { throw StudioError.invalid("Invalid MOSS model folder preferences.") }
    return value
  }
  public func write(to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(self).write(to: url, options: .atomic)
  }
  private static func absolute(_ value: String) -> Bool { value.hasPrefix("/") && !value.utf8.contains(0) }
}

/// Captures only selection fields, so background discovery cannot overwrite a later selection.
public struct MOSSModelQuery: Hashable, Sendable {
  public var libraryDirectory, sharedDirectory, transformerDirectory, variant: String
  public var quantizationBits: Int?
  public init(_ settings: MOSSImageSettings) {
    libraryDirectory = settings.libraryDirectory; sharedDirectory = settings.sharedDirectory
    transformerDirectory = settings.transformerDirectory; variant = settings.variant; quantizationBits = settings.quantizationBits
  }
  public var profileKey: String { variant + ":" + (quantizationBits.map { "q\($0)" } ?? "bf16") }
  public func apply(to settings: inout MOSSImageSettings) {
    settings.libraryDirectory = libraryDirectory; settings.sharedDirectory = sharedDirectory
    settings.transformerDirectory = transformerDirectory
  }
}

public enum MOSSModelDiscovery {
  public static func mountedLibraries() -> [URL] {
    (FileManager.default.mountedVolumeURLs(includingResourceValuesForKeys: nil) ?? [])
      .map { $0.appendingPathComponent("ai-models", isDirectory: true) }.sorted { $0.path < $1.path }
  }
  /// Bounded known paths only; no recursive scan, weight reads, inference or downloads.
  public static func discover(query: MOSSModelQuery, remembered: MOSSModelLinks, libraries: [URL]) -> MOSSModelLinks {
    var links = remembered; links.remember(query)
    var resolved = links.applying(to: query)
    var roots: [URL] = [], seen = Set<String>()
    for hint in [resolved.libraryDirectory, resolved.sharedDirectory, resolved.transformerDirectory] where hint.hasPrefix("/") {
      var directory = URL(fileURLWithPath: hint).standardizedFileURL
      for _ in 0..<8 {
        if seen.insert(directory.path).inserted { roots.append(directory) }
        let parent = directory.deletingLastPathComponent(); if parent == directory { break }; directory = parent
      }
    }
    for root in libraries where root.isFileURL {
      if seen.insert(root.standardizedFileURL.path).inserted { roots.append(root.standardizedFileURL) }
    }
    let manager = FileManager.default
    func exists(_ directory: URL, _ name: String) -> Bool { manager.fileExists(atPath: directory.appendingPathComponent(name).path) }
    for root in roots {
      for family in [root, root.appendingPathComponent("WeeTodd/MOSS-Image"), root.appendingPathComponent("MOSS-Image")] {
        for shared in [family.appendingPathComponent("Shared"), family.appendingPathComponent("moss-image-shared"), family] where resolved.sharedDirectory.isEmpty {
          if ["text_encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "vae/config.json"].allSatisfy({ exists(shared, $0) }) {
            resolved.sharedDirectory = shared.path
            if resolved.libraryDirectory.isEmpty { resolved.libraryDirectory = family.path }
          }
        }
        // Affine conversions must be explicitly selected or remembered for their precision.
        let officialKey = resolved.variant + ":bf16"
        if ["sft", "base"].contains(resolved.variant), links.transformerDirectories[officialKey] == nil {
          for transformer in [family.appendingPathComponent(resolved.variant == "sft" ? "SFT/transformer" : "Base/transformer"),
            family.appendingPathComponent("moss-image-" + resolved.variant + "/transformer")] {
            if exists(transformer, "config.json"), !exists(transformer, "moss_quantization.json"),
              let files = try? manager.contentsOfDirectory(at: transformer, includingPropertiesForKeys: nil),
              files.contains(where: { $0.pathExtension == "safetensors" }) {
              links.transformerDirectories[officialKey] = transformer.path
              if resolved.quantizationBits == nil && resolved.transformerDirectory.isEmpty { resolved.transformerDirectory = transformer.path }
              if resolved.libraryDirectory.isEmpty { resolved.libraryDirectory = family.path }; break
            }
          }
        }
      }
    }
    links.remember(resolved)
    if links.analyzerDirectory.isEmpty {
      var fallback: String?
      for root in roots {
        if let model = MOSSLayoutAnalysisPreparation.discoverModel(libraryHint: root.path) {
          if MOSSLayoutAnalysisPreparation.visionModel(URL(fileURLWithPath: model)) == .gemma4_12b {
            links.analyzerDirectory = model; break
          }
          if fallback == nil { fallback = model }
        }
      }
      if links.analyzerDirectory.isEmpty { links.analyzerDirectory = fallback ?? "" }
    }
    return links
  }
}
