import Foundation

public struct NativePixelCQRuntime: Codable, Equatable, Sendable {
  public var workerPath: String
  public var ffmpegPath: String
  public var ffprobePath: String
  public var components: [String: String]
  public var cqAdapterPath: String?
  public init(workerPath: String, ffmpegPath: String, ffprobePath: String,
    components: [String: String], cqAdapterPath: String?) {
    self.workerPath = workerPath; self.ffmpegPath = ffmpegPath; self.ffprobePath = ffprobePath
    self.components = components; self.cqAdapterPath = cqAdapterPath
  }
  public static func resolve(settings: NativePixelCQSettings, profilesDirectory: String,
    workerPath: String, ffmpegPath: String, ffprobePath: String, cqCandidates: [String] = [],
    pixelCandidates: [String] = []) throws -> Self {
    try settings.validate()
    let catalog = try NativeLTXPreparation.catalog(directory: profilesDirectory)
    let candidates = catalog.filter {
      (settings.profileID == "auto" || $0["id"] as? String == settings.profileID)
        && ($0["generation"] as? [String: Any])?["pipelineMode"] as? String == "distilled"
    }
    let cq = settings.cqAdapterPath ?? (cqCandidates.count == 1 ? cqCandidates[0] : nil)
    guard settings.cqStrength == 0 || cq != nil else {
      throw StudioError.invalid("Choose the installed CQ Enhancer V2 LoRA for native upscaling. Add it to the LoRA library or use Choose CQ V2.")
    }
    for candidate in candidates {
      guard let path = candidate["id"] as? String,
        let raw = try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
        var profile = raw["components"] as? [String: Any],
        (profile["loras"] as? [Any] ?? []).isEmpty, (profile["stage_two_loras"] as? [Any] ?? []).isEmpty else { continue }
      let config = raw["config"] as? [String: Any] ?? [:]
      if let pixel = config["dfr_detailing_lora_path"] { profile["dfr_detailing_lora_path"] = pixel }
      let override = settings.pixelAdapterPath ?? (pixelCandidates.count == 1 ? pixelCandidates[0] : nil)
      if let components = try? NativeLTXMoviePreparation.components(from: profile, mode: .pixelSpatial, pixelSpatialAdapterPath: override) {
        return .init(workerPath: workerPath, ffmpegPath: ffmpegPath, ffprobePath: ffprobePath,
          components: components, cqAdapterPath: cq)
      }
    }
    throw StudioError.invalid("Link native LTX 2.5 distilled components and the Pixel Spatial x2 adapter in Model Setup, or choose a compatible native profile.")
  }
  public func validate(settings: NativePixelCQSettings) throws {
    try settings.validate()
    for executable in [workerPath, ffmpegPath, ffprobePath] {
      guard NativePixelCQSettings.validPath(executable), FileManager.default.isExecutableFile(atPath: executable) else {
        throw StudioError.invalid("Native Pixel Spatial + CQ needs the bundled LTX worker and executable FFmpeg/FFprobe. Configure Runtime Settings.")
      }
    }
    guard settings.cqStrength == 0 || cqAdapterPath.map(NativePixelCQSettings.validPath) == true else {
      throw StudioError.invalid("Choose a local CQ Enhancer V2 adapter.")
    }
  }
  var mediaRuntime: FlashVSRRuntime {
    var r = FlashVSRRuntime(); r.ffmpegPath = ffmpegPath; r.ffprobePath = ffprobePath; return r
  }
  func adapters(_ settings: NativePixelCQSettings) -> [NativeLTXMoviePreparation.RefinementLoRA] {
    guard settings.cqStrength > 0, let path = cqAdapterPath else { return [] }
    return [.init(path: path, strength: settings.cqStrength)]
  }
}

public struct NativePixelCQResources: Codable, Sendable {
  public let runtime: NativePixelCQRuntime
  public let workerFiles: [FlashVSRFile]
  public let componentIdentitySHA256: String
}

public struct NativePixelCQReceipt: Codable, Sendable {
  public let requestFingerprint: String
  public let source: FlashVSRFile
  public let video: FlashVSRFile
  public let workerMetadata: FlashVSRFile
  public let workerRecipe: FlashVSRFile
  public let media: FlashVSRMediaInfo
  public let elapsedSeconds: Double
  public let sourceAudioSHA256: String?
  public let outputAudioSHA256: String?
  /// Verified decoder RGB, before delivery encoding. Absent in legacy receipts.
  public let restored: FlashVSRFile?
  public init(requestFingerprint: String, source: FlashVSRFile, video: FlashVSRFile,
    workerMetadata: FlashVSRFile, workerRecipe: FlashVSRFile, media: FlashVSRMediaInfo,
    elapsedSeconds: Double, sourceAudioSHA256: String?, outputAudioSHA256: String?, restored: FlashVSRFile? = nil) {
    self.requestFingerprint = requestFingerprint; self.source = source; self.video = video
    self.workerMetadata = workerMetadata; self.workerRecipe = workerRecipe; self.media = media
    self.elapsedSeconds = elapsedSeconds; self.sourceAudioSHA256 = sourceAudioSHA256
    self.outputAudioSHA256 = outputAudioSHA256; self.restored = restored
  }
}
