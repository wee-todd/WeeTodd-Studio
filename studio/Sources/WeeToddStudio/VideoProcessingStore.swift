import AppKit
import Foundation
import UniformTypeIdentifiers
import StudioCore

struct VideoProcessingDestination {
  let clips: [Clip]
  let session: UUID
  let projectID: UUID
}

@MainActor extension StudioStore {
  func videoProcessingDestination(for clip: Clip) -> VideoProcessingDestination {
    .init(clips: videoProcessingClips(for: clip), session: documentSessionID, projectID: project.id)
  }
  private func matchesVideoProcessingDestination(_ target: VideoProcessingDestination) -> Bool {
    documentSessionID == target.session && project.id == target.projectID && target.clips.allSatisfy { frozen in
      project.clips.first(where: { $0.id == frozen.id }) == frozen
    }
  }
  func videoProcessingClips(for clip: Clip) -> [Clip] {
    if let members = try? project.continuousSceneMembers(for: clip), !members.isEmpty { return members }
    return [clip]
  }
  func setVideoProcessingEnabled(_ enabled: Bool) {
    guard let clip = selectedClip else { return }
    let ids = Set(videoProcessingClips(for: clip).map(\.id))
    change { project in
      for i in project.clips.indices where ids.contains(project.clips[i].id) {
        project.clips[i].videoProcessing = enabled ? VideoProcessingRecipe(settings: Self.recipeFlashDefaults) : nil
      }
    }
  }
  static var recipeFlashDefaults: FlashVSRSettings {
    var settings = FlashVSRSettings(); settings.width = 1920; settings.height = 1080; settings.decoder = .tiny
    return settings
  }
  func setVideoProcessingEngine(_ kind: String) {
    guard let clip = selectedClip, ["native-pixel-cq", "flashvsr-turbo", "native-pixel-cq-flashvsr", "h3-regenerate", "h3-regenerate-flashvsr"].contains(kind) else { return }
    var flash = FlashVSRSettings(); flash.width = project.settings.width; flash.height = project.settings.height
    var recipe: VideoProcessingRecipe
    switch kind {
    case "h3-regenerate": recipe = .init(h3Regenerate: clip.videoProcessing?.steps.first(where: \.isH3Regenerate)?.h3Regenerate ?? h3RegenerateSettings)
    case "h3-regenerate-flashvsr":
      var first = VideoProcessingStep(h3Regenerate: clip.videoProcessing?.steps.first(where: \.isH3Regenerate)?.h3Regenerate ?? h3RegenerateSettings)
      first.id = "regenerate"
      var second = VideoProcessingStep(settings: flash); second.id = "flash"; second.input = first.id
      do { recipe = try .init(steps: [first, second]) } catch { self.error = error.localizedDescription; return }
    case "native-pixel-cq": recipe = .init(pixelCQ: .init())
    case "native-pixel-cq-flashvsr": recipe = .init(pixelCQ: clip.videoProcessing?.steps.first(where: \.isNative)?.pixelCQ ?? .init(), flashVSR: Self.recipeFlashDefaults)
    default: recipe = .init(settings: flash)
    }
    if recipe.isChain, let prefix = clip.videoProcessing?.nativePrefix {
      recipe.steps[0] = prefix.steps[0]; recipe.steps[1].input = prefix.steps[0].id
    }
    if recipe.isChain { recipe.format = "weetodd-video-processing-v2" }
    let ids = Set(videoProcessingClips(for: clip).map(\.id))
    change { project in for i in project.clips.indices where ids.contains(project.clips[i].id) {
      project.clips[i].videoProcessing = recipe
    } }
  }
  func editPixelCQProcessing(_ edit: (inout NativePixelCQSettings) -> Void) {
    guard let clip = selectedClip, var recipe = clip.videoProcessing, let index = recipe.steps.firstIndex(where: \.isNative),
      var settings = recipe.steps[index].pixelCQ else { return }
    edit(&settings); recipe.steps[index].pixelCQ = settings
    let ids = Set(videoProcessingClips(for: clip).map(\.id))
    change { project in for i in project.clips.indices where ids.contains(project.clips[i].id) {
      project.clips[i].videoProcessing = recipe
    } }
  }
  func choosePixelCQAdapter(cq: Bool) {
    let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
    panel.message = cq ? "Choose the installed LTX 2.5 CQ Enhancer V2 LoRA." : "Choose the official LTX 2.5 Pixel Spatial x2 IC LoRA."
    guard panel.runModal() == .OK, let file = panel.url else { return }
    editPixelCQProcessing { if cq { $0.cqAdapterPath = file.path } else { $0.pixelAdapterPath = file.path } }
  }
  func nativePixelCQRuntime(for recipe: VideoProcessingRecipe) async throws -> NativePixelCQRuntime {
    guard let options = recipe.steps.first(where: \.isNative)?.pixelCQ, let worker = runtime.ltx25WorkerPath else {
      throw StudioError.invalid("Native Pixel Spatial + CQ needs the bundled LTX worker. Use a complete Studio build.")
    }
    let settings = runtime, candidates = localLoRALibraryAssets.map(\.path)
    return try await Task.detached(priority: .userInitiated) {
      func executable(_ configured: String, _ name: String) -> String {
        if !configured.isEmpty { return configured }
        return ["/opt/homebrew/bin/", "/usr/local/bin/"].map { $0 + name }.first { FileManager.default.isExecutableFile(atPath: $0) } ?? ""
      }
      let cq = Array(Set(candidates.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased() == "ltx2.5-cq-enhancer-lora-v2.safetensors" })).sorted()
      let pixel = Array(Set(candidates.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased() == "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" })).sorted()
      return try NativePixelCQRuntime.resolve(settings: options, profilesDirectory: settings.profilesDirectory,
        workerPath: worker, ffmpegPath: executable(settings.ffmpegPath, "ffmpeg"), ffprobePath: executable(settings.ffprobePath, "ffprobe"),
        cqCandidates: cq, pixelCandidates: pixel)
    }.value
  }
  func editVideoProcessing(_ edit: (inout FlashVSRSettings) -> Void) {
    guard let clip = selectedClip, var recipe = clip.videoProcessing, let index = recipe.steps.firstIndex(where: \.isFlashVSR) else { return }
    edit(&recipe.steps[index].settings)
    if recipe.steps[index].settings.variant == .base { recipe.format = "weetodd-video-processing-v2" }
    let ids = Set(videoProcessingClips(for: clip).map(\.id))
    change { project in for i in project.clips.indices where ids.contains(project.clips[i].id) {
      project.clips[i].videoProcessing = recipe
    } }
  }
  func videoProcessingOriginal(for clip: Clip) -> (path: String, frozen: FlashVSRFile?) {
    if let lineage = clip.versions.last(where: { $0.path == clip.sourcePath })?.processingLineage {
      return (lineage.source.path, lineage.source)
    }
    return (clip.sourcePath, nil)
  }
  func preflightClipVideoProcessing(_ clip: Clip, admission: NativeHeadlessExecutor.ProcessingPreflight? = nil) async throws -> NativeVideoProcessingPlan? {
    let target = videoProcessingDestination(for: clip), clips = videoProcessingClips(for: clip)
    guard let recipe = clips.first?.videoProcessing else {
      guard clips.allSatisfy({ $0.videoProcessing == nil }) else {
        throw StudioError.invalid("Use the same upscale recipe for every shot in a continuous scene.")
      }
      return nil
    }
    guard clips.allSatisfy({ $0.videoProcessing == recipe }) else {
      throw StudioError.invalid("Use the same upscale recipe for every shot in a continuous scene.")
    }
    let plan: NativeVideoProcessingPlan
    if !recipe.hasNative && !recipe.hasH3Regenerate, let admission {
      plan = try await admission(recipe, videoUpscaleRuntime)
    } else {
      loadVideoUpscalePreferences()
      let pixel = recipe.hasNative ? try await nativePixelCQRuntime(for: recipe) : nil
      plan = try await NativeVideoProcessingPlan.preflight(recipe: recipe,
        runtime: recipe.hasFlashVSR ? videoUpscaleRuntime : nil,
        nativeRuntime: pixel, h3Runtime: recipe.hasH3Regenerate ? h3RegenerateRuntime : nil)
    }
    guard matchesVideoProcessingDestination(target) else {
      throw StudioError.invalid("The destination project, source or recipe changed during upscale admission. Retry the current take.")
    }
    return plan
  }
  func retryClipVideoProcessing() {
    guard !operationBusy, let clip = selectedClip, clip.videoProcessing != nil, !clip.sourcePath.isEmpty else { return }
    let target = videoProcessingDestination(for: clip)
    videoUpscaleBusy = true
    videoUpscaleTask = Task { [weak self] in
      guard let self else { return }
      defer { self.videoUpscaleTask = nil; self.videoUpscaleBusy = false }
      do {
        let plan = try await self.preflightClipVideoProcessing(clip)
        try await self.processAcceptedClipVideo(target, plan: plan)
      } catch is CancellationError { self.videoUpscaleStatus = "Upscale cancelled. Original take retained; retry without generating again." }
      catch { self.error = error.localizedDescription }
    }
  }
  /// Runs only after the generation worker has returned and its original take has been accepted.
  func processAcceptedClipVideo(_ target: VideoProcessingDestination, plan: NativeVideoProcessingPlan?) async throws {
    guard let plan else { return }
    guard matchesVideoProcessingDestination(target), let clip = target.clips.first else {
      throw StudioError.invalid("The destination project or take changed. Retry upscaling the current take.")
    }
    let clips = target.clips, session = target.session, projectID = target.projectID
    guard clips.allSatisfy({ $0.videoProcessing == plan.recipe && !$0.sourcePath.isEmpty }) else {
      throw StudioError.invalid("The processing recipe changed. Original take retained; use Upscale accepted take.")
    }
    let original = videoProcessingOriginal(for: clip)
    guard clips.allSatisfy({ videoProcessingOriginal(for: $0).path == original.path }) else {
      throw StudioError.invalid("The continuous scene no longer shares one original movie. Restore its take before upscaling.")
    }
    let cancellation = NativeHeadlessCancellation(); videoProcessingCancellation = cancellation
    videoUpscaleBusy = true; videoUpscaleStatus = "Original accepted · checking upscale source…"
    defer { videoUpscaleBusy = false; videoProcessingCancellation = nil }
    let source: FlashVSRFile
    if let frozen = original.frozen { source = frozen }
    else { source = try await Task.detached { try FlashVSRFile.capture(URL(fileURLWithPath: original.path)) }.value }
    let directory = dataDirectory.appendingPathComponent("Processing/" + projectID.uuidString)
    do {
      let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: directory,
        cancellation: cancellation, emit: { [weak self] event in
          Task { @MainActor in self?.videoUpscaleStatus = event.message }
        })
      try Task.checkCancellation()
      if acceptClipVideoProcessing(result, clips: clips, session: session, projectID: projectID) {
        videoUpscaleResult = result.recipe.steps.count == 1 ? result.receipt.flashVSR : nil
        videoUpscaleStatus = result.receipt.label + " · " + (result.reusedSteps > 0 ? "reused \(result.reusedSteps) stage(s) · recorded render " : "") + String(Int(result.receipt.elapsedSeconds.rounded())) + " s"
        notice = "Upscale accepted. Original take and prior upscales remain in Versions."
      } else { notice = "Upscale saved at " + result.receipt.video.path + ". The destination clip changed." }
    } catch {
      notice = "Original generation retained. Retry with Upscale accepted take; generation is not repeated."
      throw error
    }
  }
  @discardableResult func acceptClipVideoProcessing(_ result: NativeVideoProcessingResult, clips: [Clip],
    session: UUID, projectID: UUID) -> Bool {
    guard documentSessionID == session, project.id == projectID,
      clips.allSatisfy({ frozen in project.clips.first(where: { $0.id == frozen.id }) == frozen }) else { return false }
    do {
      var updated = project
      for clip in clips { try VideoProcessingAcceptance.apply(result, to: clip.id, project: &updated) }
      change { $0 = updated }; refreshPreview(); return true
    } catch { self.error = error.localizedDescription; return false }
  }
  func cancelVideoProcessing() { videoProcessingCancellation?.cancel(); videoUpscaleTask?.cancel() }
}
