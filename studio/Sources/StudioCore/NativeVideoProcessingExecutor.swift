import Darwin
import Foundation

public struct NativeVideoProcessingResult: Sendable {
  public let source: FlashVSRFile
  public let recipe: VideoProcessingRecipe
  public let receipt: VideoProcessingReceipt
  public let receiptPath: String
  public let reusedSteps: Int
  public init(source: FlashVSRFile, recipe: VideoProcessingRecipe, receipt: FlashVSRReceipt, receiptPath: String, reusedSteps: Int) {
    self.source = source; self.recipe = recipe; self.receipt = .flashVSR(receipt); self.receiptPath = receiptPath; self.reusedSteps = reusedSteps
  }
  public init(source: FlashVSRFile, recipe: VideoProcessingRecipe, nativeReceipt: NativePixelCQReceipt, receiptPath: String, reusedSteps: Int) {
    self.source = source; self.recipe = recipe; self.receipt = .pixelCQ(nativeReceipt); self.receiptPath = receiptPath; self.reusedSteps = reusedSteps
  }
  public init(source: FlashVSRFile, recipe: VideoProcessingRecipe, chainReceipt: VideoProcessingChainReceipt, receiptPath: String, reusedSteps: Int) {
    self.source = source; self.recipe = recipe; self.receipt = .chain(chainReceipt); self.receiptPath = receiptPath; self.reusedSteps = reusedSteps
  }
  public init(source:FlashVSRFile,recipe:VideoProcessingRecipe,sequenceReceipt:VideoProcessingSequenceReceipt,receiptPath:String,reusedSteps:Int) {
    self.source=source;self.recipe=recipe;self.receipt = .sequence(sequenceReceipt);self.receiptPath=receiptPath;self.reusedSteps=reusedSteps
  }
  public var fingerprint: String { get throws {
    NativeHeadlessJob.hash(Data((try recipe.fingerprint + "|" + receipt.requestFingerprint).utf8))
  } }
  public func lineage(sourceVersionID: UUID?) throws -> VideoProcessingLineage {
    .init(source: source, sourceVersionID: sourceVersionID, receiptPath: receiptPath,
      fingerprint: try fingerprint, recipe: recipe)
  }
}

public enum NativeVideoProcessingExecutor {
  private struct Checkpoint: Codable {
    let fingerprint: String
    let attempt: UUID
    let receiptSHA256: String
  }
  public static func run(plan: NativeVideoProcessingPlan, source: FlashVSRFile, directory: URL,
    backend: (any FlashVSRBackend)? = nil, nativeWorker: NativePixelCQService.Worker? = nil,
    sequenceWorker:VideoProcessingSequenceExecutor.Worker? = nil,
    cancellation: NativeHeadlessCancellation = NativeHeadlessCancellation(),
    emit: @escaping @Sendable (FlashVSREvent) -> Void = { _ in }) async throws -> NativeVideoProcessingResult {
    try cancellation.check(); try plan.recipe.validate()
    let task = Task { try await execute(plan: plan, source: source, directory: directory,
      backend: backend, nativeWorker: nativeWorker, sequenceWorker:sequenceWorker,cancellation: cancellation, emit: emit) }
    return try await withTaskCancellationHandler(operation: {
      do { try cancellation.onCancellation { task.cancel() } }
      catch { task.cancel(); _ = try? await task.value; throw error }
      defer { try? cancellation.onCancellation(nil) }
      return try await task.value
    }, onCancel: { cancellation.cancel(); task.cancel() })
  }
  private static func verify(_ file: FlashVSRFile) async throws {
    try await Task.detached { try file.verify() }.value
  }
  private static func execute(plan: NativeVideoProcessingPlan, source: FlashVSRFile, directory: URL,
    backend: (any FlashVSRBackend)?, nativeWorker: NativePixelCQService.Worker?, sequenceWorker:VideoProcessingSequenceExecutor.Worker?,cancellation: NativeHeadlessCancellation,
    emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> NativeVideoProcessingResult {
    try await verify(source); try cancellation.check(); try Task.checkCancellation()
    guard plan.recipe.isSequence || ((!plan.recipe.hasNative || plan.native != nil) &&
      plan.recipe.steps.filter({ !$0.isNative }).allSatisfy({ plan.resources[$0.id] != nil })) else {
      throw StudioError.invalid("Every processing stage must be admitted before restoration starts.")
    }
    guard directory.isFileURL else { throw StudioError.invalid("Choose a local processing cache directory.") }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let root = directory.resolvingSymlinksInPath().standardizedFileURL
    let fd = Darwin.open(root.appendingPathComponent(".processing.lock").path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0o600)
    guard fd >= 0 else { throw StudioError.invalid("Cannot open the processing cache lease.") }
    defer { flock(fd, LOCK_UN); close(fd) }
    var attributes = stat()
    guard fstat(fd, &attributes) == 0, attributes.st_mode & S_IFMT == S_IFREG,
      flock(fd, LOCK_EX | LOCK_NB) == 0 else {
      throw StudioError.invalid("Another process owns this processing cache. Retry after it finishes.")
    }
    if plan.recipe.isSequence {
      return try await VideoProcessingSequenceExecutor.run(plan:plan,source:source,root:root,backend:backend,nativeWorker:nativeWorker,worker:sequenceWorker,cancellation:cancellation,emit:emit)
    }
    var nativeResult: NativeVideoProcessingResult?
    if let prefix = plan.recipe.nativePrefix {
      guard let resources = plan.native else { throw StudioError.invalid("Native upscale resources were not admitted. Retry after native model setup.") }
      let result = try await NativePixelCQService.execute(resources: resources, recipe: prefix, source: source,
        directory: root, cancellation: cancellation, worker: nativeWorker, emit: emit)
      if plan.recipe.isNative { return result }
      nativeResult = result
    }
    var input = nativeResult?.receipt.video ?? source, final: FlashVSRReceipt?, receiptPath = "", reused = nativeResult?.reusedSteps ?? 0
    for step in plan.recipe.steps where !step.isNative {
      let selectedBackend = backend ?? FlashVSRService.backend(step.settings.backend)
      try cancellation.check(); try Task.checkCancellation(); try await verify(input)
      var request = VideoUpscaleRequest(sourcePath: input.path, settings: step.settings, runtime: plan.runtime)
      request.sourceSHA256 = input.sha256
      let resolved = try await selectedBackend.preflight(request: request)
      guard resolved.source == input, plan.resources[step.id] == FlashVSRResources(components: resolved.components, engineFiles: resolved.engineFiles) else {
        throw StudioError.invalid("FlashVSR+ inputs or components changed after recipe admission. Retry upscaling from the original take.")
      }
      let key = try resolved.fingerprint, folder = root.appendingPathComponent(key)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      guard folder.resolvingSymlinksInPath().path.hasPrefix(root.path + "/") else {
        throw StudioError.invalid("The processing cache escaped its output directory.")
      }
      let checkpointURL = folder.appendingPathComponent("completed.json")
      var accepted: FlashVSRReceipt?, attempt: URL?
      if FileManager.default.fileExists(atPath: checkpointURL.path) {
        do {
          let size = try FileManager.default.attributesOfItem(atPath: checkpointURL.path)[.size] as? NSNumber
          guard let size, size.intValue > 0, size.intValue <= 1024 * 1024 else { throw StudioError.invalid("Invalid processing checkpoint.") }
          let checkpoint = try JSONDecoder().decode(Checkpoint.self, from: Data(contentsOf: checkpointURL))
          let cached = folder.appendingPathComponent(checkpoint.attempt.uuidString)
          guard checkpoint.fingerprint == key, cached.resolvingSymlinksInPath().path.hasPrefix(folder.path + "/"),
            checkpoint.receiptSHA256 == (try NativeHeadlessJob.fileHash(cached.appendingPathComponent("receipt.json"))) else {
            throw StudioError.invalid("The completed processing receipt changed.")
          }
          accepted = try await Task.detached { try FlashVSRService.resume(plan: resolved, directory: cached) }.value
          attempt = cached; reused += 1
          emit(.init(stage: "reuse", message: "Reusing verified Turbo " + step.settings.decoder.label))
        } catch {
          emit(.init(stage: "retry", message: "Cached upscale is incomplete or changed; preserving it and starting a new attempt."))
        }
      }
      if accepted == nil {
        let id = UUID(), destination = folder.appendingPathComponent(id.uuidString)
        let result = try await selectedBackend.run(plan: resolved, directory: destination, emit: emit)
        try cancellation.check(); try Task.checkCancellation(); try await verify(source)
        let verified = try await Task.detached { try FlashVSRService.resume(plan: resolved, directory: destination) }.value
        guard result.video == verified.video, result.requestFingerprint == verified.requestFingerprint,
          !resolved.media.hasAudio || verified.audio != nil else {
          throw StudioError.invalid("The processing worker returned different media or omitted source audio.")
        }
        let checkpoint = Checkpoint(fingerprint: key, attempt: id,
          receiptSHA256: try NativeHeadlessJob.fileHash(destination.appendingPathComponent("receipt.json")))
        try JSONEncoder().encode(checkpoint).write(to: checkpointURL, options: .atomic)
        accepted = verified; attempt = destination
      }
      final = accepted; input = accepted!.video; receiptPath = attempt!.appendingPathComponent("receipt.json").path
    }
    try await verify(source); try cancellation.check(); try Task.checkCancellation()
    if let nativeResult {
      return try await finalizeChain(plan: plan, source: source, nativeResult: nativeResult, flash: final!,
        flashReceiptPath: receiptPath, root: root, reusedSteps: reused, cancellation: cancellation)
    }
    return NativeVideoProcessingResult(source: source, recipe: plan.recipe, receipt: final!, receiptPath: receiptPath, reusedSteps: reused)
  }

  private static func audioHash(_ file: FlashVSRFile, pcm: Bool, runtime: FlashVSRRuntime,
    directory: URL, cancellation: NativeHeadlessCancellation) throws -> String {
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffmpegPath),
      arguments: ["-v", "error", "-nostdin", "-i", file.path, "-map", "0:a:0", "-vn", "-c:a", pcm ? "pcm_f32le" : "copy", "-f", "hash", "-hash", "sha256", "-"],
      directory: directory, log: directory.appendingPathComponent(UUID().uuidString + "-audio.log"), cancellation: cancellation)
    let hash = String(decoding: result.output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    guard result.exitCode == 0, hash.hasPrefix("SHA256="), hash.count == 71 else {
      throw StudioError.invalid("Cannot verify the processing chain's unchanged original audio.")
    }
    return hash
  }

  private static func finalizeChain(plan: NativeVideoProcessingPlan, source: FlashVSRFile,
    nativeResult: NativeVideoProcessingResult, flash: FlashVSRReceipt, flashReceiptPath: String,
    root: URL, reusedSteps: Int, cancellation: NativeHeadlessCancellation) async throws -> NativeVideoProcessingResult {
    let native = nativeResult.receipt.native!
    guard flash.plan.source == native.video, flash.plan.frames == native.media.frames,
      abs(flash.plan.media.fps.value - native.media.fps.value) < 0.0000001 else {
      throw StudioError.invalid("The FlashVSR stage must use the verified native intermediate without changing the source clock.")
    }
    let nativeReceiptFile = try FlashVSRFile.capture(URL(fileURLWithPath: nativeResult.receiptPath))
    let flashReceiptFile = try FlashVSRFile.capture(URL(fileURLWithPath: flashReceiptPath))
    let key = NativeHeadlessJob.hash(Data((try plan.recipe.fingerprint + "|" + source.sha256 + "|"
      + nativeReceiptFile.sha256 + "|" + flashReceiptFile.sha256).utf8))
    let folder = root.appendingPathComponent("chain-" + key)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    guard folder.resolvingSymlinksInPath().path.hasPrefix(root.path + "/") else {
      throw StudioError.invalid("The processing chain cache escaped its output directory.")
    }
    let checkpointURL = folder.appendingPathComponent("completed.json")
    if FileManager.default.fileExists(atPath: checkpointURL.path) {
      do {
        guard let size = try FileManager.default.attributesOfItem(atPath: checkpointURL.path)[.size] as? NSNumber,
          size.intValue > 0, size.intValue <= 1024 * 1024 else { throw StudioError.invalid("Invalid chain checkpoint.") }
        let saved = try JSONDecoder().decode(Checkpoint.self, from: Data(contentsOf: checkpointURL))
        let attempt = folder.appendingPathComponent(saved.attempt.uuidString), receiptURL = attempt.appendingPathComponent("receipt.json")
        let receiptSize = try FileManager.default.attributesOfItem(atPath: receiptURL.path)[.size] as? NSNumber
        guard saved.fingerprint == key, attempt.resolvingSymlinksInPath().path.hasPrefix(folder.path + "/"),
          saved.receiptSHA256 == (try NativeHeadlessJob.fileHash(receiptURL)),
          let receiptSize, receiptSize.intValue > 0, receiptSize.intValue <= 1024 * 1024 else {
          throw StudioError.invalid("The chain receipt changed.")
        }
        let receipt = try JSONDecoder().decode(VideoProcessingChainReceipt.self, from: Data(contentsOf: receiptURL))
        guard receipt.format == "weetodd-video-processing-chain-result-v1", receipt.requestFingerprint == key,
          receipt.recipe == plan.recipe, receipt.source == source,
          receipt.nativeReceiptFile == nativeReceiptFile, receipt.flashReceiptFile == flashReceiptFile,
          receipt.native.requestFingerprint == native.requestFingerprint, receipt.native.video == native.video,
          receipt.flashVSR.requestFingerprint == flash.requestFingerprint, receipt.flashVSR.video == flash.video,
          receipt.sourceAudioSHA256 == receipt.outputAudioSHA256, receipt.sourcePCMHash == receipt.outputPCMHash,
          !native.media.hasAudio || receipt.sourceAudioSHA256 != nil && receipt.sourcePCMHash != nil,
          URL(fileURLWithPath: receipt.video.path).resolvingSymlinksInPath().path.hasPrefix(attempt.resolvingSymlinksInPath().path + "/") else {
          throw StudioError.invalid("The completed chain no longer matches its verified stages.")
        }
        try await verify(receipt.video); try cancellation.check(); try Task.checkCancellation()
        return .init(source: source, recipe: plan.recipe, chainReceipt: receipt, receiptPath: receiptURL.path, reusedSteps: reusedSteps)
      } catch {
        try cancellation.check(); try Task.checkCancellation()
      }
    }
    let attemptID = UUID(), attempt = folder.appendingPathComponent(attemptID.uuidString)
    try FileManager.default.createDirectory(at: attempt, withIntermediateDirectories: false)
    let videoURL = attempt.appendingPathComponent("processed.mov")
    let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: plan.runtime.ffmpegPath),
      arguments: ["-v", "error", "-nostdin", "-n", "-i", flash.video.path, "-i", source.path,
        "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-movflags", "+faststart", videoURL.path],
      directory: attempt, log: attempt.appendingPathComponent("preserve-original-audio.log"), cancellation: cancellation)
    guard result.exitCode == 0 else { throw StudioError.invalid("The processing chain could not preserve original audio. Both restoration stages are retained for retry.") }
    let video = try FlashVSRFile.capture(videoURL)
    let checked = try FlashVSRMediaInfo.inspect(videoURL, runtime: plan.runtime, directory: attempt, cancellation: cancellation)
    guard checked.frames == native.media.frames, abs(checked.fps.value - native.media.fps.value) < 0.0000001,
      checked.width == flash.plan.request.settings.width, checked.height == flash.plan.request.settings.height,
      checked.hasAudio == native.media.hasAudio else {
      throw StudioError.invalid("The processing chain changed source FPS, frame count, audio presence or target dimensions.")
    }
    let sourceAudio = try native.media.hasAudio ? audioHash(source, pcm: false, runtime: plan.runtime, directory: attempt, cancellation: cancellation) : nil
    let outputAudio = try native.media.hasAudio ? audioHash(video, pcm: false, runtime: plan.runtime, directory: attempt, cancellation: cancellation) : nil
    let sourcePCM = try native.media.hasAudio ? audioHash(source, pcm: true, runtime: plan.runtime, directory: attempt, cancellation: cancellation) : nil
    let outputPCM = try native.media.hasAudio ? audioHash(video, pcm: true, runtime: plan.runtime, directory: attempt, cancellation: cancellation) : nil
    guard sourceAudio == outputAudio, sourcePCM == outputPCM else {
      throw StudioError.invalid("The processing chain's delivered audio differs from the original take.")
    }
    for file in [source, native.video, nativeReceiptFile, flashReceiptFile, flash.video, flash.restored] + (flash.audio.map { [$0] } ?? []) { try await verify(file) }
    try cancellation.check(); try Task.checkCancellation()
    let receipt = VideoProcessingChainReceipt(format: "weetodd-video-processing-chain-result-v1", requestFingerprint: key,
      recipe: plan.recipe, source: source, native: native, flashVSR: flash, nativeReceiptFile: nativeReceiptFile,
      flashReceiptFile: flashReceiptFile, video: video, sourceAudioSHA256: sourceAudio, outputAudioSHA256: outputAudio,
      sourcePCMHash: sourcePCM, outputPCMHash: outputPCM)
    let receiptURL = attempt.appendingPathComponent("receipt.json"), encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    try encoder.encode(receipt).write(to: receiptURL, options: .withoutOverwriting)
    try encoder.encode(Checkpoint(fingerprint: key, attempt: attemptID, receiptSHA256: try NativeHeadlessJob.fileHash(receiptURL))).write(to: checkpointURL, options: .atomic)
    return .init(source: source, recipe: plan.recipe, chainReceipt: receipt, receiptPath: receiptURL.path, reusedSteps: reusedSteps)
  }
}
