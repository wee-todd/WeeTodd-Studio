import AVFoundation
import CryptoKit
import Darwin
import Foundation

public enum MovieUpscaleMethod: String, Codable, CaseIterable, Identifiable {
  case ltx25, relayVSR
  public var id: String { rawValue }
  public var label: String { self == .relayVSR ? "RelayVSR 2×" : "LTX 2×" }
}

extension Clip {
  public var usesRelayVSR: Bool {
    engine == .ltx25 && inferredTask == "video_upscale"
      && generationSelection?.movieUpscaleMethod == .relayVSR
  }
}

public enum NativeRelayVSRPreparation {
  private static let checkpointNames = [
    "relayvsr_x2_lightx2v_int8_convrot.safetensors",
    "relayvsr_flash_vae_x2_fp32.safetensors",
  ]

  public static func discoverModels() -> String? {
    let manager = FileManager.default
    let candidates = (manager.mountedVolumeURLs(includingResourceValuesForKeys: nil) ?? [])
      .map { $0.appendingPathComponent("ai-models/WeeTodd/RelayVSR/x2", isDirectory: true) }
    return candidates.first { folder in
      checkpointNames.allSatisfy { manager.isReadableFile(atPath: folder.appendingPathComponent($0).path) }
    }?.path
  }

  private struct FileSnapshot: Codable, Equatable {
    let path: String
    let identity: [Int64]

    init(_ path: String) throws {
      guard path.hasPrefix("/"), !path.utf8.contains(0) else {
        throw StudioError.invalid("RelayVSR requires an absolute local file path.")
      }
      let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
      var status = stat()
      guard url.path.withCString({ Darwin.lstat($0, &status) }) == 0,
        status.st_mode & S_IFMT == S_IFREG, status.st_size > 0,
        FileManager.default.isReadableFile(atPath: url.path) else {
        throw StudioError.invalid("RelayVSR could not read \(url.lastPathComponent). Check the source and model folder.")
      }
      self.path = url.path
      identity = [Int64(status.st_dev), Int64(bitPattern: UInt64(status.st_ino)), status.st_size,
        Int64(status.st_mtimespec.tv_sec), Int64(status.st_mtimespec.tv_nsec),
        Int64(status.st_ctimespec.tv_sec), Int64(status.st_ctimespec.tv_nsec)]
    }

    func verify() throws {
      guard try Self(path) == self else {
        throw StudioError.invalid("A RelayVSR input changed. Prepare the clip again.")
      }
    }
  }

  private struct Recipe: Codable {
    let format: String
    let source: FileSnapshot
    let worker: FileSnapshot
    let models: String
    let checkpoints: [FileSnapshot]
    let start: Double
    let duration: Double?
    let seed: Int

    func verify() throws {
      guard format == "weetodd-relayvsr-v1", seed >= 0, start.isFinite, start >= 0,
        duration.map({ $0.isFinite && $0 > 0 }) ?? true,
        checkpoints.count == checkpointNames.count else {
        throw StudioError.invalid("Invalid prepared RelayVSR job. Prepare the clip again.")
      }
      try source.verify()
      try worker.verify()
      for (file, name) in zip(checkpoints, checkpointNames) {
        let path = URL(fileURLWithPath: models).appendingPathComponent(name).path
        guard try FileSnapshot(path) == file else {
          throw StudioError.invalid("The RelayVSR checkpoint changed. Prepare the clip again.")
        }
      }
    }
  }

  private static func recipe(_ request: [String: Any]) throws -> Recipe {
    guard let raw = request["project"], let id = request["clipID"] as? String else {
      throw StudioError.invalid("RelayVSR requires a clip.")
    }
    let decoder = JSONDecoder()
    let project = try decoder.decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: raw))
    let globals = try decoder.decode([MediaAsset].self,
      from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    guard let clip = project.clips.first(where: { $0.id.uuidString == id }), clip.usesRelayVSR,
      !project.isContinuousSceneMember(clip), clip.seed >= 0,
      clip.attachments.allSatisfy({ [.reference, .first, .last, .audioDriver].contains($0.role) }) else {
      throw StudioError.invalid("RelayVSR requires an independent source-movie clip and a nonnegative seed.")
    }
    let references = clip.attachments.filter { $0.role == .reference && $0.isEnabled }
    guard references.count == 1, let attachment = references.first,
      let asset = (project.assets + globals).first(where: { $0.id == attachment.assetID }),
      asset.kind == .video else {
      throw StudioError.invalid("RelayVSR needs one enabled Reference movie.")
    }
    let runtime = request["runtime"] as? [String: Any] ?? [:]
    return try recipe(source: asset.path, start: attachment.sourceStartSeconds ?? 0,
      duration: attachment.sourceDurationSeconds, seed: clip.seed,
      worker: runtime["relayVSRWorkerPath"] as? String,
      models: runtime["relayVSRModelDirectory"] as? String)
  }

  private static func recipe(source: String, start: Double = 0, duration: Double? = nil,
    seed: Int = 42, worker: String?, models: String?) throws -> Recipe {
    guard let worker,
      FileManager.default.isExecutableFile(atPath: worker) else {
      throw StudioError.invalid("The native RelayVSR worker is missing. Use a complete Studio build.")
    }
    guard let models, models.hasPrefix("/") else {
      throw StudioError.invalid("Choose the folder containing both RelayVSR 2× checkpoints.")
    }
    let value = try Recipe(format: "weetodd-relayvsr-v1", source: FileSnapshot(source),
      worker: FileSnapshot(worker), models: models, checkpoints: checkpointNames.map {
        try FileSnapshot(URL(fileURLWithPath: models).appendingPathComponent($0).path)
      }, start: start, duration: duration, seed: seed)
    try value.verify()
    return value
  }

  private static func encode(_ recipe: Recipe) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(recipe)
  }

  public static func describe(request: [String: Any]) throws -> [String: Any] {
    let value = try recipe(request)
    let fingerprint = try SHA256.hash(data: encode(value)).map { String(format: "%02x", $0) }.joined()
    return ["fingerprint": fingerprint, "sourcePaths": [value.source.path, value.worker.path]
      + value.checkpoints.map(\.path), "readinessErrors": []]
  }

  public static func prepare(request: [String: Any], destination: URL) async throws -> [String: Any] {
    try await prepare(recipe(request), destination: destination)
  }

  public static func prepareMovie(source: String, worker: String?, models: String?,
    destination: URL) async throws -> String {
    let result = try await prepare(recipe(source: source, worker: worker, models: models),
      destination: destination)
    guard let path = result["recipePath"] as? String else {
      throw StudioError.invalid("RelayVSR did not prepare a recipe.")
    }
    return path
  }

  private static func prepare(_ value: Recipe, destination: URL) async throws -> [String: Any] {
    let movie = AVURLAsset(url: URL(fileURLWithPath: value.source.path),
      options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard try await !movie.loadTracks(withMediaType: .video).isEmpty else {
      throw StudioError.invalid("The RelayVSR reference has no video track.")
    }
    let duration = try await movie.load(.duration).seconds
    guard duration.isFinite, value.start < duration,
      value.duration.map({ value.start + $0 <= duration + 0.001 }) ?? true else {
      throw StudioError.invalid("The RelayVSR source interval falls outside the movie.")
    }
    try value.verify()
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let file = destination.appendingPathComponent("relayvsr.json")
    try encode(value).write(to: file, options: .atomic)
    return ["recipePath": file.path, "report": ["task": "video_upscale",
      "method": "RelayVSR 2×", "nativeRuntime": "swift-mlx", "audio": "source",
      "frameRate": "source", "source": value.source.path, "sourceStart": value.start]]
  }

  public static func arguments(recipePath: String, worker: String, output: URL) throws -> [String] {
    let recipe = try JSONDecoder().decode(Recipe.self, from: Data(contentsOf: URL(fileURLWithPath: recipePath)))
    try recipe.verify()
    guard try FileSnapshot(worker) == recipe.worker else {
      throw StudioError.invalid("The native RelayVSR worker changed. Prepare the clip again.")
    }
    var result = ["--studio", recipe.source.path, output.path, "--models", recipe.models,
      "--seed", String(recipe.seed)]
    if recipe.start > 0 || recipe.duration != nil {
      result += ["--source-start", String(recipe.start)]
      if let duration = recipe.duration { result += ["--source-duration", String(duration)] }
    }
    return result
  }
}
