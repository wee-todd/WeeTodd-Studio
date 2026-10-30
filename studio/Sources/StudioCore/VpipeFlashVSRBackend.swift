import Darwin
import Foundation

public struct VpipeFlashVSRBackend: FlashVSRBackend {
  public init() {}
  static func newWorkspace() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-flashvsr-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    return root
  }
  public func preflight(request: VideoUpscaleRequest) async throws -> FlashVSRResolvedPlan {
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation: {
      try await Task.detached {
        try request.settings.validateExecution()
        guard request.format == "weetodd-video-upscale-v1", request.firstFrame >= 0,
          request.frameCount == nil || request.frameCount! > 0 else { throw StudioError.invalid("Invalid FlashVSR+ request or selected interval.") }
        let runtime = request.runtime
        guard request.sourcePath.hasPrefix("/"), FileManager.default.isReadableFile(atPath: request.sourcePath) else {
          throw StudioError.invalid("Connect a readable local source movie.")
        }
        let workspace = try Self.newWorkspace(); defer { try? FileManager.default.removeItem(at: workspace) }
        let resources = try Self.inspectResources(settings: request.settings, runtime: runtime, workspace: workspace, cancellation: cancellation)
        let source = try FlashVSRFile.capture(URL(fileURLWithPath: request.sourcePath))
        if let expected = request.sourceSHA256, expected != source.sha256 { throw StudioError.invalid("The selected source movie changed.") }
        let media = try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: source.path), runtime: runtime, directory: workspace, cancellation: cancellation)
        let count = request.frameCount ?? (media.frames - request.firstFrame)
        guard count > 0, request.firstFrame <= media.frames - count else { throw StudioError.invalid("The selected frame interval extends beyond the source movie.") }
        try cancellation.check()
        let temporaryBytes = try FlashVSRResourcePlan.temporaryBytes(media: media, settings: request.settings, frames: count)
        return FlashVSRResolvedPlan(request: request, source: source, media: media, components: resources.components,
          engineFiles: resources.engineFiles, frames: count, temporaryBytes: temporaryBytes)
      }.value
    }, onCancel: { cancellation.cancel() })
  }
  public func preflightResources(settings: FlashVSRSettings, runtime: FlashVSRRuntime) async throws -> FlashVSRResources {
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation: {
      try await Task.detached {
        let workspace = try Self.newWorkspace(); defer { try? FileManager.default.removeItem(at: workspace) }
        return try Self.inspectResources(settings: settings, runtime: runtime, workspace: workspace, cancellation: cancellation)
      }.value
    }, onCancel: { cancellation.cancel() })
  }
  private static func inspectResources(settings: FlashVSRSettings, runtime: FlashVSRRuntime,
    workspace: URL, cancellation: NativeHeadlessCancellation) throws -> FlashVSRResources {
    try settings.validateExecution()
    guard settings.backend == .vpipe else { throw StudioError.invalid("The vpipe adapter cannot execute a native Swift/MLX request.") }
    for (name, value) in [("vpipe", runtime.vpipePath), ("FFmpeg", runtime.ffmpegPath), ("FFprobe", runtime.ffprobePath), ("WeeToddCLI", runtime.supervisorPath)] {
      guard value.hasPrefix("/"), FileManager.default.isExecutableFile(atPath: value) else {
        throw StudioError.invalid("Connect the optional " + name + " executable before running FlashVSR+.")
      }
    }
    let version = ProcessInfo.processInfo.operatingSystemVersion
    guard version.majorVersion >= 26 else { throw StudioError.invalid("The optional vpipe engine requires macOS 26 or later. Other Studio features remain available.") }
    guard runtime.modelRoot.hasPrefix("/") else { throw StudioError.invalid("Connect the local FlashVSR+ model folder.") }
    let engine = try FlashVSRFile.capture(URL(fileURLWithPath: runtime.vpipePath))
    let versionResult = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: engine.path), arguments: ["--version"], directory: workspace, log: workspace.appendingPathComponent("version.log"), cancellation: cancellation)
    guard versionResult.exitCode == 0, String(decoding: versionResult.output, as: UTF8.self).contains("c9fef0a*0") else {
      throw StudioError.invalid("Connect the qualified official vpipe 0.1.104 build. A different engine build needs qualification.")
    }
    var engineFiles = [engine]
    let helper = URL(fileURLWithPath: engine.path).deletingLastPathComponent()
    let frameworks = helper.deletingLastPathComponent().appendingPathComponent("Frameworks")
    for directory in [helper, frameworks] {
      if let urls = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
        for url in urls.sorted(by: { $0.path < $1.path }) where url.pathExtension == "dylib" { engineFiles.append(try FlashVSRFile.capture(url)) }
      }
    }
    for name in [runtime.supervisorPath, runtime.ffmpegPath, runtime.ffprobePath] {
      engineFiles.append(try FlashVSRFile.capture(URL(fileURLWithPath: name)))
    }
    let components = try FlashVSRModelManifest.resolve(root: URL(fileURLWithPath: runtime.modelRoot), settings: settings)
    try FlashVSRModelManifest.verifyPublished(components, settings: settings)
    try cancellation.check()
    return FlashVSRResources(components: components, engineFiles: engineFiles)
  }
  public func run(plan: FlashVSRResolvedPlan, directory: URL, emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> FlashVSRReceipt {
    guard plan.request.settings.backend == .vpipe else { throw StudioError.invalid("The vpipe adapter cannot execute a native Swift/MLX plan.") }
    let cancellation = NativeHeadlessCancellation()
    return try await withTaskCancellationHandler(operation: {
      try await Task.detached { try FlashVSRMediaPipeline.execute(plan: plan, directory: directory, cancellation: cancellation, emit: emit, restore: Self.restore) }.value
    }, onCancel: { cancellation.cancel() })
  }
  private static func restore(plan: FlashVSRResolvedPlan, selected: URL, restored: URL,
    directory: URL, cancellation: NativeHeadlessCancellation,
    emit: @escaping @Sendable (FlashVSREvent) -> Void) throws {
    let runtime = plan.request.runtime, settings = plan.request.settings
    let graph = try VpipeFlashVSRPipeline.make(settings: settings, modelRoot: plan.components.root, input: selected.path,
      output: restored.path, fps: plan.media.fps.value, frames: plan.frames)
    let graphURL = directory.appendingPathComponent("pipeline.json"); try graph.write(to: graphURL, options: .withoutOverwriting)
    let lease = runtime.leasePath.isEmpty ? ExternalInferenceLease.defaultURL.path : runtime.leasePath
    let command = FlashVSREngineCommand(executable: plan.engineFiles[0], arguments: ["--config", "{\"memory_cap_mb\":12288,\"wired_pool_mb\":4096,\"swap_allowance_mb\":0}", "--launch", graphURL.path],
      directory: directory.path, log: directory.appendingPathComponent("vpipe.log").path, lease: lease, ownerPID: getpid())
    var checkedCommand = command
    checkedCommand.modelCheck = .init(components: plan.components, settings: settings)
    let commandURL = directory.appendingPathComponent("engine-command.json")
    try JSONEncoder().encode(checkedCommand).write(to: commandURL, options: .withoutOverwriting)
    try cancellation.check()
    emit(.init(stage: "restore", message: "Turbo · " + settings.decoder.label))
    let engineResult = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.supervisorPath),
      arguments: ["--flashvsr-engine-host", commandURL.path], directory: directory,
      log: directory.appendingPathComponent("engine.log"), cancellation: cancellation, maximumOutput: 65536, captureOutput: false, onLine: { line in
        if line.contains("[ERROR]") || line.contains("Impacting Interactivity") || line.contains("GPU command buffer error") || line.contains("decoding with the VAE") {
          throw StudioError.invalid("FlashVSR+ engine failed or attempted decoder substitution. See vpipe.log: " + String(line.suffix(400)))
        }
        if line.contains("[PROGRESS]") || line.contains("tiny decoder") || line.contains("Wan VAE") { emit(.init(stage: "restore", message: String(line.prefix(500)))) }
      })
    guard engineResult.exitCode == 0 else { throw StudioError.invalid("The optional FlashVSR+ engine failed. See engine.log and vpipe.log.") }
  }
}
