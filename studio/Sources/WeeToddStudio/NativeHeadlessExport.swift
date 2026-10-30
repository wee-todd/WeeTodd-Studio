import Foundation
import StudioCore

@MainActor extension StudioStore {
  func pixelRuntimeForExport(_ plan: NativeVideoProcessingPlan, previous: NativePixelCQRuntime?) throws -> NativePixelCQRuntime {
    let steps = plan.recipe.steps.filter(\.isNative)
    let candidates: [NativePixelCQRuntime]
    if plan.recipe.isSequence {
      candidates = steps.compactMap { plan.pixelResources?[$0.id]?.runtime }
    } else { candidates = plan.native.map { [$0.runtime] } ?? [] }
    guard !steps.isEmpty, candidates.count == steps.count, let first = candidates.first else {
      throw StudioError.invalid("Every native Pixel stage must have admitted components before job export.")
    }
    let shared = previous ?? first
    guard candidates.allSatisfy({ $0.workerPath == shared.workerPath && $0.ffmpegPath == shared.ffmpegPath && $0.ffprobePath == shared.ffprobePath }),
      plan.recipe.isSequence || previous == nil || previous == first else {
      throw StudioError.invalid("Use the same native LTX worker and media tools throughout the exported job. Legacy recipes also require one shared component selection.")
    }
    if plan.recipe.isSequence {
      for (step, actual) in zip(steps, candidates) {
        guard let settings = step.pixelCQ, settings.profileID == "auto" else { continue }
        var expected = try NativeVideoProcessingPlan.pixelRuntime(shared, settings: settings), selected = actual
        if settings.cqStrength == 0 { expected.cqAdapterPath = nil; selected.cqAdapterPath = nil }
        guard expected == selected else {
          throw StudioError.invalid("Automatic Pixel stages resolved different models or adapters. Choose explicit native profiles/adapters when mixing model selections in one exported job.")
        }
      }
    }
    return shared
  }
  var nativeHeadlessEligible: Bool {
    !project.clips.isEmpty && project.clips.allSatisfy { clip in
      clip.engine == .movie || clip.engine == .h3 && runtime.usesNativeH3
        || clip.engine == .ltx25 && runtime.usesNativeLTX25
    }
  }
  /// Export reuses the native preparation/compiler; it never recomposes through Python.
  func exportNativeHeadlessJob(body: [String: Any], to url: URL, clipOnly: Bool) async throws {
    let snapshot = project, settings = runtime, session = documentSessionID
    let execution = try productionExecutionFingerprint()
    var frozenProject = snapshot
    if clipOnly {
      guard let selected = selectedClip else { throw StudioError.invalid("Select a clip first.") }
      // The native finishing admission runs before any preparation. Added timeline media
      // remains explicit rather than silently disappearing from a clip-only export.
      guard snapshot.audio.isEmpty, snapshot.titles.isEmpty,
        try snapshot.continuousSceneMembers(for: selected).isEmpty else {
        throw StudioError.invalid("Export the complete scene or use the Python job exporter for timeline titles/audio.")
      }
      var clip = selected; clip.transition = "cut"
      frozenProject.clips = [clip]; frozenProject.name = clip.name
    }
    var generated = Set(body["generateIDs"] as? [String] ?? [])
    // A pending draft targets its original source, including imported movie clips.
    // Applied takes no longer match that source and remain ordinary accepted media.
    let rippleIDs = Set(frozenProject.clips.filter { $0.rippleDraft?.sourceMatches($0) == true }.map(\.id))
    guard rippleIDs.isEmpty || settings.nativeRippleEnabled == true else {
      throw StudioError.invalid("Enable native Ripple before exporting its pending edit.")
    }
    generated.formUnion(rippleIDs.map(\.uuidString))
    var sceneOwners: [UUID: UUID] = [:]
    for clip in frozenProject.clips where generated.contains(clip.id.uuidString) && !rippleIDs.contains(clip.id) {
      let members = try snapshot.continuousSceneMembers(for: clip)
      for member in members {
        guard !rippleIDs.contains(member.id) else { throw StudioError.invalid("Apply the pending Ripple edit before exporting its continuous scene.") }
        generated.insert(member.id.uuidString); sceneOwners[member.id] = members[0].id
      }
    }
    var workerPaths: [String: String] = [:]
    if let worker = settings.h3WorkerPath { workerPaths["h3"] = worker }
    if let worker = settings.ltx25WorkerPath { workerPaths["ltx25"] = worker }
    try NativeHeadlessJob.validateFinishing(frozenProject)
    var processingRuntime: FlashVSRRuntime?
    var nativeProcessingRuntime: NativePixelCQRuntime?
    var h3ProcessingRuntime: H3RegenerateRuntime?
    if frozenProject.clips.contains(where: { $0.videoProcessing != nil }) {
      for clip in frozenProject.clips {
        guard let recipe = clip.videoProcessing else { continue }
        let plan = try await preflightClipVideoProcessing(clip)
        if recipe.hasNative {
          guard let plan else { throw StudioError.invalid("Native processing was not admitted before job export.") }
          nativeProcessingRuntime = try pixelRuntimeForExport(plan, previous: nativeProcessingRuntime)
        }
        if recipe.hasFlashVSR { loadVideoUpscalePreferences(); processingRuntime = videoUpscaleRuntime }
        if recipe.hasH3Regenerate { loadVideoUpscalePreferences(); h3ProcessingRuntime = h3RegenerateRuntime }
      }
    }
    let inputs = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".inputs-" + UUID().uuidString)
    var published = false
    defer { if !published { try? FileManager.default.removeItem(at: inputs) } }
    var recipes: [String: NativeHeadlessJob.Recipe] = [:]
    for clip in frozenProject.clips where generated.contains(clip.id.uuidString)
      && (sceneOwners[clip.id] == nil || sceneOwners[clip.id] == clip.id) {
      guard project == snapshot, documentSessionID == session,
        try productionExecutionFingerprint() == execution else {
        throw StudioError.invalid("The movie changed during job export. Export the current edit again.")
      }
      let target = inputs.appendingPathComponent(clip.id.uuidString)
      let ripple = rippleIDs.contains(clip.id), h3 = !ripple && clip.engine == .h3
      var request = body; request["clipID"] = clip.id.uuidString
      let take = inputs.appendingPathComponent("ripple-take-" + clip.id.uuidString)
      if ripple {
        request = ["draft": try clip.rippleDraft!.object(), "takeOutput": take.path]
      }
      let result = try await bridge.invoke(ripple ? "ripple-native-prepare" : h3 ? "h3-native-prepare" : "ltx-native-prepare",
        runtime: settings, payload: request, output: target)
      guard let recipePath = result["recipePath"] as? String else { throw StudioError.invalid("Native job preparation returned no recipe.") }
      _ = try await bridge.invoke(h3 ? "h3-native-preflight" : "ltx-native-preflight",
        runtime: settings, payload: ["recipePath": recipePath], output: ripple ? take : target.appendingPathComponent("preflight"))
      let report: Any = ripple ? ["rippleDraft": try clip.rippleDraft!.object()] : result["report"] ?? [:]
      recipes[clip.id.uuidString] = NativeHeadlessJob.Recipe(engine: h3 ? "h3" : "ltx25",
        bytes: try Data(contentsOf: URL(fileURLWithPath: recipePath)),
        signature: ripple ? clip.rippleDraft!.inputFingerprint : signature(for: clip),
        report: try JSONSerialization.data(withJSONObject: report))
    }
    guard project == snapshot, documentSessionID == session,
      try productionExecutionFingerprint() == execution else {
      throw StudioError.invalid("The movie changed during job export. Export the current edit again.")
    }
    let job = try await Task.detached(priority: .userInitiated) {
      try NativeHeadlessJob(project: frozenProject, recipes: recipes, workers: workerPaths, ffmpeg: settings.ffmpegPath, processingRuntime: processingRuntime, nativeProcessingRuntime: nativeProcessingRuntime, h3ProcessingRuntime: h3ProcessingRuntime)
    }.value
    guard project == snapshot, documentSessionID == session,
      try productionExecutionFingerprint() == execution else {
      throw StudioError.invalid("The movie changed during job export. Export the current edit again.")
    }
    try job.write(to: url); published = true
  }
}
