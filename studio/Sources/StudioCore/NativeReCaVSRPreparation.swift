import AVFoundation
import Foundation
import InferenceContracts

public enum NativeReCaVSRPreparation {
  public static func discoverModels() -> String? {
    (FileManager.default.mountedVolumeURLs(includingResourceValuesForKeys: nil) ?? []).map {
      $0.appendingPathComponent("ai-models/WeeTodd/ReCaVSR")
    }.first { (try? ReCaReleasedAssets.resolve(directory: $0, precision: .bf16)) != nil }?.path
  }
  private static func make(source: String, interval: ReCaSourceInterval?, worker: String?, models: String?,
    settings: ReCaVSRSettings, destination: URL) throws -> NativeReCaVSRRecipe {
    guard let worker, FileManager.default.isExecutableFile(atPath: worker), let models, models.hasPrefix("/") else {
      throw StudioError.invalid("Choose the ReCaVSR model folder and use a complete Studio build containing its native worker.")
    }
    let executable = URL(fileURLWithPath: worker)
    return try NativeReCaVSRRecipe(source: ReCaFileIdentity.capture(URL(fileURLWithPath: source)), interval: interval,
      worker: ReCaFileIdentity.capture(executable), metal: ReCaFileIdentity.capture(NativeMLXMetalLibrary.resolve(executable: executable)),
      assets: ReCaReleasedAssets.resolve(directory: URL(fileURLWithPath: models), precision: settings.precision), settings: settings,
      outputPath: destination.appendingPathComponent("video.mov").path)
  }
  private static func make(request: [String: Any], destination: URL) throws -> NativeReCaVSRRecipe {
    guard let raw = request["project"], let id = request["clipID"] as? String else { throw StudioError.invalid("Select a ReCaVSR clip.") }
    let project = try JSONDecoder().decode(StudioProject.self, from: JSONSerialization.data(withJSONObject: raw))
    let globals = try JSONDecoder().decode([MediaAsset].self, from: JSONSerialization.data(withJSONObject: request["globalAssets"] ?? []))
    guard let clip = project.clips.first(where: { $0.id.uuidString == id }), clip.usesReCaVSR,
      !project.isContinuousSceneMember(clip), clip.seed >= 0 else { throw StudioError.invalid("ReCaVSR needs an independent source movie and nonnegative seed.") }
    let references = clip.attachments.filter { $0.role == .reference && $0.isEnabled }
    guard references.count == 1, let reference = references.first,
      let asset = (project.assets + globals).first(where: { $0.id == reference.assetID }), asset.kind == .video else {
      throw StudioError.invalid("ReCaVSR needs one enabled Reference movie.")
    }
    var settings = try clip.generationSelection?.reCaVSRSettings ?? ReCaVSRSettings()
    settings.seed = UInt64(clip.seed)
    let interval = try ReCaSourceInterval(startSeconds: reference.sourceStartSeconds ?? 0,
      durationSeconds: reference.sourceDurationSeconds ?? clip.duration)
    let runtime = request["runtime"] as? [String: Any] ?? [:]
    return try make(source: asset.path, interval: interval, worker: runtime["reCaVSRWorkerPath"] as? String,
      models: runtime["reCaVSRModelDirectory"] as? String, settings: settings, destination: destination)
  }
  public static func describe(request: [String: Any]) throws -> [String: Any] {
    let recipe = try make(request: request, destination: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
    return ["fingerprint": try recipe.fingerprint(), "sourcePaths": [recipe.source.path, recipe.worker.path, recipe.metal.path] + recipe.assets.files.map { $0.file.path }, "readinessErrors": []]
  }
  public static func prepare(request: [String: Any], destination: URL) async throws -> [String: Any] {
    try await save(make(request: request, destination: destination), destination: destination)
  }
  public static func prepareMovie(source: String, worker: String?, models: String?, settings: ReCaVSRSettings,
    destination: URL) async throws -> String {
    let recipe = try make(source: source, interval: nil, worker: worker, models: models, settings: settings, destination: destination)
    let result = try await save(recipe, destination: destination)
    return result["recipePath"] as! String
  }
  private static func save(_ recipe: NativeReCaVSRRecipe, destination: URL) async throws -> [String: Any] {
    try recipe.verifyInputs(); try ReCaReleasedAssets.validate(recipe.assets, precision: recipe.settings.precision)
    let asset = AVURLAsset(url: URL(fileURLWithPath: recipe.source.path), options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard try await !asset.loadTracks(withMediaType: .video).isEmpty else { throw StudioError.invalid("ReCaVSR reference has no video track.") }
    let duration = try await asset.load(.duration).seconds
    if let interval = recipe.interval {
      guard interval.startSeconds < duration, interval.startSeconds + interval.durationSeconds <= duration + 0.001 else { throw StudioError.invalid("ReCaVSR interval is outside the source movie.") }
    }
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let url = destination.appendingPathComponent("recavsr.json")
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    try encoder.encode(recipe).write(to: url, options: .atomic)
    return ["recipePath": url.path, "report": ["task": "video_upscale", "method": "ReCaVSR", "nativeRuntime": "swift-mlx",
      "frameRate": "source", "audio": "source", "fingerprint": try recipe.fingerprint()]]
  }
  public static func arguments(recipePath: String, worker: String, output: URL) throws -> [String] {
    let recipe = try NativeReCaVSRRecipe.prepare(Data(contentsOf: URL(fileURLWithPath: recipePath)))
    guard URL(fileURLWithPath: worker).resolvingSymlinksInPath().path == recipe.worker.path else { throw StudioError.invalid("The ReCaVSR worker changed. Prepare again.") }
    // Full content verification runs in the native worker, never on the main actor.
    return ["--studio", "--recipe", recipePath, "--output", output.path]
  }
}
