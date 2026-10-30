import Foundation

public enum FlashVSRService {
  public static func preflightResources(settings: FlashVSRSettings, runtime: FlashVSRRuntime) async throws -> FlashVSRResources {
    if settings.backend == .swiftMLX {
      return try await NativeFlashVSRBackend().preflightResources(settings: settings, runtime: runtime)
    }
    return try await VpipeFlashVSRBackend().preflightResources(settings: settings, runtime: runtime)
  }
  public static func resume(plan: FlashVSRResolvedPlan, directory: URL) throws -> FlashVSRReceipt {
    let url = directory.appendingPathComponent("receipt.json")
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let size = attributes[.size] as? NSNumber, size.intValue > 0, size.intValue <= 1024 * 1024 else {
      throw StudioError.invalid("No bounded completed upscale receipt is available. Restart the interrupted upscale in a new directory.")
    }
    let receipt = try JSONDecoder().decode(FlashVSRReceipt.self, from: Data(contentsOf: url))
    guard receipt.format == "weetodd-video-upscale-result-v1", receipt.requestFingerprint == (try plan.fingerprint),
      receipt.requestFingerprint == (try receipt.plan.fingerprint) else {
      throw StudioError.invalid("The source, settings or engine changed. Start a new upscale.")
    }
    let root = directory.resolvingSymlinksInPath().path + "/"
    for file in [receipt.video, receipt.restored] + (receipt.audio.map { [$0] } ?? []) + (receipt.sourcePreview.map { [$0] } ?? []) {
      guard URL(fileURLWithPath: file.path).resolvingSymlinksInPath().path.hasPrefix(root) else {
        throw StudioError.invalid("The upscale receipt references media outside its output directory.")
      }
      try file.verify()
    }
    return receipt
  }

  public static func preflight(request: VideoUpscaleRequest) async throws -> FlashVSRResolvedPlan {
    try request.settings.validateExecution()
    return try await backend(request.settings.backend).preflight(request: request)
  }
  public static func run(plan: FlashVSRResolvedPlan, directory: URL,
    emit: @escaping @Sendable (FlashVSREvent) -> Void = { _ in }) async throws -> FlashVSRReceipt {
    try plan.request.settings.validateExecution()
    return try await backend(plan.request.settings.backend).run(plan: plan, directory: directory, emit: emit)
  }
  public static func backend(_ selected: FlashVSRSettings.Backend) -> any FlashVSRBackend {
    selected == .swiftMLX ? NativeFlashVSRBackend() : VpipeFlashVSRBackend()
  }
  public static func runIntermediate(plan: FlashVSRResolvedPlan, directory: URL,
    emit: @escaping @Sendable (FlashVSREvent) -> Void = { _ in }) async throws -> FlashVSRReceipt {
    guard let selected = backend(plan.request.settings.backend) as? any FlashVSRIntermediateBackend else {
      throw StudioError.invalid("This processing sequence requires the native Swift/MLX FlashVSR+ backend for lossless transport.")
    }
    return try await selected.runIntermediate(plan: plan, directory: directory, emit: emit)
  }
}
