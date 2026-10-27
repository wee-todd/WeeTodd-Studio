import Foundation
import MLX

/// An in-memory, ordered visual-reference request. The worker is responsible
/// for bounded decode and source-file identity before constructing RGB bytes.
public struct H3Ref2VAStillRequest: Sendable {
  public let omni: H3OmniRecipe.Selection?
  public let transformerWeightCacheGB:Int
  public let prompt: String
  public let references: [H3Ref2VAReference]
  public let geometry: H3Geometry
  public let seed: UInt64
  public let requestedSteps: Int
  public let samplingMethod: H3SamplingMethod
  public let referenceNoise: H3ReferenceNoiseControls?
  public let transformer: URL
  public let qwenPages: URL
  public let qwenVision: URL
  public let tokenizer: URL
  public let videoDecodeMemoryMode: H3VideoDecodeMemoryMode?
  public let videoDecodePrecision: H3VideoDecodePrecision
  public let videoVAE: URL
  public let audioVAE: URL
  public let turboLoRA: URL?
  public let turboLoRAStrength: Float
  public let additionalLoRAs: [H3LoRAAdapter]
  public let loRAAdapters: [H3LoRAAdapter]
  public let transformerBackend: H3TransformerBackend
  public let attentionPolicy: H3AttentionPolicy

  public init(prompt: String, references: [H3StillReference], width: Int,
    height: Int, durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, qwenVision: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    loRAAdapters: [H3LoRAAdapter]? = nil,
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil,
    videoDecodePrecision: H3VideoDecodePrecision = .float32,
    samplingMethod: H3SamplingMethod = .euler,
    referenceNoise: H3ReferenceNoiseControls? = nil,
    canvasAdmission: H3CanvasAdmission = .ordinary,
    transformerBackend: H3TransformerBackend = .mlx,
    attentionPolicy: H3AttentionPolicy = .dense, transformerWeightCacheGB:Int = 0,
    stillImageMode: H3StillImageMode? = nil, omni: H3OmniRecipe.Selection? = nil) throws {
    try self.init(prompt: prompt, mediaReferences: references.map { .image($0) },
      width: width, height: height, durationSeconds: durationSeconds,
      seed: seed, requestedSteps: requestedSteps, transformer: transformer,
      qwenPages: qwenPages, qwenVision: qwenVision, tokenizer: tokenizer,
      videoVAE: videoVAE, audioVAE: audioVAE, turboLoRA: turboLoRA,
      turboLoRAStrength: turboLoRAStrength,
      additionalLoRAs: additionalLoRAs, loRAAdapters: loRAAdapters,
      videoDecodeMemoryMode: videoDecodeMemoryMode, videoDecodePrecision: videoDecodePrecision, samplingMethod: samplingMethod,
      referenceNoise: referenceNoise, canvasAdmission: canvasAdmission,
      transformerBackend: transformerBackend, attentionPolicy: attentionPolicy, transformerWeightCacheGB:transformerWeightCacheGB,
      stillImageMode: stillImageMode, omni: omni)
  }

  public init(prompt: String, mediaReferences: [H3Ref2VAReference], width: Int,
    height: Int, durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, qwenVision: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    loRAAdapters: [H3LoRAAdapter]? = nil,
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil,
    videoDecodePrecision: H3VideoDecodePrecision = .float32,
    samplingMethod: H3SamplingMethod = .euler,
    referenceNoise: H3ReferenceNoiseControls? = nil,
    canvasAdmission: H3CanvasAdmission = .ordinary,
    transformerBackend: H3TransformerBackend = .mlx,
    attentionPolicy: H3AttentionPolicy = .dense, transformerWeightCacheGB:Int = 0,
    stillImageMode: H3StillImageMode? = nil, omni: H3OmniRecipe.Selection? = nil) throws {
    if omni != nil {
      guard stillImageMode == nil, (5...15).contains(durationSeconds), requestedSteps == 9,
        samplingMethod == .euler, referenceNoise == nil, canvasAdmission == .ordinary,
        transformerBackend == .mlx, attentionPolicy == .dense, transformerWeightCacheGB == 0,
        turboLoRA == nil, additionalLoRAs.isEmpty, loRAAdapters?.isEmpty ?? true,
        mediaReferences.allSatisfy({ switch $0 { case .image, .video, .audio: true; default: false } }) else {
        throw H3CheckpointError.invalid("Omni rejects competing sampling, placement, adapter and image controls.")
      }
    }
    self.omni = omni
    if attentionPolicy == .solExperimental {
      try H3SolTaskPolicy.validateTask(task: "ref2va", contextFrames: 0,
        isRefinement: false, ordinaryCanvas: canvasAdmission == .ordinary,
        mlxBackend: transformerBackend == .mlx, hasFast: false, hasVDN: false,
        hasFun: false, hasMotion: false)
    }
    try videoDecodePrecision.validate(memoryMode: videoDecodeMemoryMode)
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= 65_536, (2...101).contains(requestedSteps),
      [transformer, qwenPages, qwenVision, tokenizer, videoVAE, audioVAE]
        .allSatisfy({ $0.isFileURL && $0.path.hasPrefix("/") }),
      turboLoRA == nil || (turboLoRA!.isFileURL &&
        turboLoRA!.path.hasPrefix("/") && turboLoRAStrength.isFinite &&
        (-10...10).contains(turboLoRAStrength)),
      (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count <= 8,
      Set(([turboLoRA].compactMap { $0 } + additionalLoRAs.map(\.url))
        .map { $0.standardizedFileURL.path }).count ==
        (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count else {
      throw H3CheckpointError.invalid("Invalid H3 visual-reference request.")
    }
    try H3VideoReferencePreparation.validate(mediaReferences)
    let geometry: H3Geometry
    if let stillImageMode {
      guard canvasAdmission == .ordinary, transformerBackend == .mlx, attentionPolicy == .dense,
        mediaReferences.allSatisfy({ if case .image = $0 { return true }; return false }) else {
        throw H3CheckpointError.invalid("H3 still editing accepts only ordinary MLX image references.")
      }
      geometry = try H3Geometry(width: width, height: height, stillImageMode: stillImageMode)
    } else {
      geometry = try H3Geometry(width: width, height: height,
        durationSeconds: durationSeconds, canvasAdmission: canvasAdmission)
    }
    let conditionRows = mediaReferences.reduce(0) { rows, reference in
      switch reference {
      case .image(let image), .timedImage(let image, _):
        rows + image.width * image.height / 1024
      case .video(let video), .timedVideo(let video, _):
        rows + ((video.persistentFrameCount - 5) / 17 * 5 + 2) *
          video.width * video.height / 1024
      case .audio, .timedAudio: rows
      }
    }
    let conditionAudioRows = mediaReferences.reduce(0) { rows, reference in
      switch reference {
      case .audio(let audio), .timedAudio(let audio, _):
        return rows + 2 * ((audio.frames + 799) / 800)
      default: break
      }
      if let audio = reference.videoAudio {
        return rows + 2 * ((audio.frames + 799) / 800)
      }
      return rows
    }
    guard width * height <= geometry.maximumPixels,
      try geometry.packedRows(textRows: 1,
        conditionVideoRows: conditionRows,
        conditionAudioRows: conditionAudioRows) <= geometry.maximumPackedRows else {
      throw H3CheckpointError.invalid("H3 reference canvas exceeds packed-row admission.")
    }
    try H3Noise.validate(videoLatentFrames: geometry.videoLatentFrames,
      latentHeight: height / 16, latentWidth: width / 16,
      audioLatentFrames: geometry.audioLatentFrames, stillImageMode: stillImageMode)
    if geometry.canvasAdmission == .spatialRefinement {
      try geometry.canvasAdmission.validate(width: width, height: height)
      _ = try H3VideoVAEDecoder.preflightSpatial(geometry: geometry)
    }
    self.prompt = prompt
    self.references = mediaReferences
    self.geometry = geometry
    self.seed = seed
    self.requestedSteps = requestedSteps
    self.samplingMethod = samplingMethod
    self.referenceNoise = referenceNoise
    self.transformer = transformer
    self.qwenPages = qwenPages
    self.qwenVision = qwenVision
    self.tokenizer = tokenizer
    self.videoDecodeMemoryMode = videoDecodeMemoryMode
    self.videoDecodePrecision = videoDecodePrecision
    self.videoVAE = videoVAE
    self.audioVAE = audioVAE
    self.turboLoRA = turboLoRA
    self.turboLoRAStrength = turboLoRAStrength
    self.additionalLoRAs = additionalLoRAs
    let primary = try turboLoRA.map {
      [try H3LoRAAdapter(url: $0, strength: turboLoRAStrength)]
    } ?? []
    let effective = loRAAdapters ?? (primary + additionalLoRAs)
    guard effective.count <= 8,
      Set(effective.map { $0.url.standardizedFileURL.path }).count == effective.count else {
      throw H3CheckpointError.invalid("H3 requires at most eight distinct LoRA descriptors.")
    }
    for adapter in effective { try adapter.validate(requestedSteps: requestedSteps, samplingMethod: samplingMethod) }
    if attentionPolicy == .solExperimental {
      try H3SolTaskPolicy.validateSettings(steps: requestedSteps, samplingMethod: samplingMethod,
        adapters: effective, packedRows: try geometry.packedRows(textRows: 1,
          conditionVideoRows: conditionRows, conditionAudioRows: conditionAudioRows))
    }
    self.loRAAdapters = effective
    self.transformerBackend = transformerBackend
    guard H3TransformerCachePlan.budgets.contains(transformerWeightCacheGB),
      transformerWeightCacheGB == 0 || (transformerBackend == .mlx && canvasAdmission == .ordinary && effective.allSatisfy({ $0.startAfterEvaluations == 0 })) else {
      throw H3CheckpointError.invalid("H3 weight cache requires ordinary MLX generation without deferred adapters or refinement.")
    }
    self.transformerWeightCacheGB=transformerWeightCacheGB
    self.attentionPolicy = attentionPolicy
  }
}

public enum H3Ref2VAStillRunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let textRows: Int
    public let packedRows: Int
    public let evaluations: Int
    public var omniSource: H3OmniCheckpoint? { omni?.source }
    let layout: H3ReferenceLayout
    let referenceLayout: H3ReferenceLayout
    let contextVideoRows: Int
    let contextAudioRows: Int
    let videoSchedule: H3Schedule?
    let audioSchedule: H3Schedule?
    let rowSchedule: H3ReferenceRowSchedule
    let nativeWorkerURL: URL?
    let experimentalSol: H3SolTaskPolicy?
    let omni: H3OmniRunner.Context?
  }

  public typealias Result = H3AVOutputDecoder.Result

  public static func preflight(_ request: H3Ref2VAStillRequest,
    contextFrames: Int = 0, refinement: H3JointRefinement? = nil) throws -> Admission {
    guard request.geometry.stillImageMode == nil || (contextFrames == 0 && refinement == nil) else {
      throw H3CheckpointError.invalid("H3 stills cannot combine video continuation or refinement.")
    }
    guard request.geometry.canvasAdmission == .ordinary || refinement != nil else {
      throw H3CheckpointError.invalid("Expanded H3 canvas requires explicit initialized spatial refinement.")
    }
    guard request.omni == nil || (contextFrames == 0 && refinement == nil) else {
      throw H3CheckpointError.invalid("Omni does not support continuation or refinement.")
    }
    try request.omni?.checkUnchanged()
    try Task.checkCancellation()
    if let diagnostic = H3SolTaskContext.policy {
      guard request.attentionPolicy == .solExperimental, diagnostic.tau == 0.5 else {
        throw H3CheckpointError.invalid("Diagnostic Sol policy conflicts with the fixed saved Ref attention selection.")
      }
    }
    if request.transformerWeightCacheGB > 0 {
      _ = try H3TransformerCachePlan.inspect(checkpointURL:request.transformer,blockCount:50,budgetGB:request.transformerWeightCacheGB,adapters:request.loRAAdapters)
    }
    let experimentalSol = request.attentionPolicy == .solExperimental ? try H3SolTaskPolicy(tau: 0.5) : nil
    if experimentalSol != nil {
      try H3SolTaskPolicy.validateTask(task: "ref2va", contextFrames: contextFrames,
        isRefinement: refinement != nil, ordinaryCanvas: request.geometry.canvasAdmission == .ordinary,
        mlxBackend: request.transformerBackend == .mlx, hasFast: false, hasVDN: false,
        hasFun: false, hasMotion: false)
      try H3SolTaskPolicy.validateSettings(steps: request.requestedSteps,
        samplingMethod: request.samplingMethod, adapters: request.loRAAdapters, packedRows: 1)
      let adapter = request.loRAAdapters[0] // count admitted above
      try H3SolTaskPolicy.inspectQualifiedFiles(checkpoint: request.transformer, adapter: adapter.url)
      let checkpoint = try H3CheckpointLayout(url: request.transformer)
      guard checkpoint.fastVariant == nil, checkpoint.curveRank == nil else {
        throw H3CheckpointError.invalid("Experimental Sol rejects Fast and curve-rank checkpoints.")
      }
    }
    let nativeWorkerURL: URL?
    if request.transformerBackend == .nncExperimental {
      try H3NativeBlockAdmission.validateSettings(task: "ref2va", steps: request.requestedSteps,
        samplingMethod: request.samplingMethod, adapters: request.loRAAdapters,
        contextFrames: contextFrames, isRefinement: refinement != nil, packedRows: 1)
      let worker = try H3NativeBlockWorker.resolve(
        executableURL: URL(fileURLWithPath: CommandLine.arguments[0]),
        environment: ProcessInfo.processInfo.environment)
      try H3NativeBlockWorker.preflight(workerURL: worker)
      try H3NativeBlockAdmission.inspect(checkpoint: request.transformer,
        adapter: request.loRAAdapters[0].url)
      nativeWorkerURL = worker
    } else { nativeWorkerURL = nil }
    let prepared = try H3VideoReferencePreparation.prepare(prompt: request.prompt,
      geometry: request.geometry, references: request.references,
      tokenizerURL: request.tokenizer)
    guard contextFrames == 0 || refinement == nil else {
      throw H3CheckpointError.invalid("H3 refinement cannot combine latent-tail continuation.")
    }
    // Omni owns its PDD grid; never construct ordinary schedules for that profile.
    let grids = try refinement?.schedules(requestedSteps: request.requestedSteps)
    let video = try request.omni == nil ? (grids?.video ?? H3Schedule(requestedSteps: request.requestedSteps, shift: 12)) : nil
    let audio = try request.omni == nil ? (grids?.audio ?? H3Schedule(requestedSteps: request.requestedSteps, shift: 3)) : nil
    let layout = contextFrames == 0 ? prepared.layout
      : try H3ReferenceLayout(referenceLayout: prepared.layout,
        geometry: request.geometry, contextFrames: contextFrames)
    let contextVideoRows = layout.conditionVideoIndices.count - prepared.layout.conditionVideoIndices.count
    let contextAudioRows = layout.conditionAudioIndices.count - prepared.layout.conditionAudioIndices.count
    guard layout.tags.count <= request.geometry.maximumPackedRows else {
      throw H3CheckpointError.invalid("H3 reference packed rows exceed engine admission.")
    }
    if experimentalSol != nil {
      try H3SolTaskPolicy.validateSettings(steps: request.requestedSteps,
        samplingMethod: request.samplingMethod, adapters: request.loRAAdapters, packedRows: layout.tags.count)
    }
    if nativeWorkerURL != nil {
      try H3NativeBlockAdmission.validateSettings(task: "ref2va", steps: request.requestedSteps,
        samplingMethod: request.samplingMethod, adapters: request.loRAAdapters,
        contextFrames: contextFrames, isRefinement: refinement != nil, packedRows: layout.tags.count)
    }
    _ = try H3QwenCheckpointLayout.inspect(root: request.qwenPages)
    if !prepared.qwenRequest.visualRanges.isEmpty {
      let vision = try H3QwenCheckpointLayout.inspect(root: request.qwenVision)
      guard vision.visionFile != nil else {
        throw H3CheckpointError.invalid("The H3 visual conditioner needs a vision tower.")
      }
    }
    let omni: H3OmniRunner.Context?
    if let selection = request.omni {
      let plan = try H3PDDPlan(contract: selection.contract)
      let attention = try H3OmniRunner.attention(references:request.references, geometry:request.geometry,
        textRows:prepared.qwenRequest.tags.count, layout:layout)
      omni = H3OmniRunner.Context(source:try H3OmniCheckpoint.admit(url:request.transformer), plan:plan, attention:attention)
    } else { omni = nil }
    let rows: H3ReferenceRowSchedule
    if let omni { rows = try H3ReferenceRowSchedule(layout:layout, omni:omni.plan) }
    else if let video, let audio {
      rows = try H3ReferenceRowSchedule(layout: layout,
        video: video, audio: audio,
        visualConditionStrength: request.referenceNoise?.visual ?? 0.999,
        audioConditionStrength: request.referenceNoise?.audio ?? 1,
        cleanVideoPrefixRows: contextVideoRows, cleanAudioPrefixRows: contextAudioRows)
    } else { throw H3CheckpointError.invalid("Missing ordinary reference schedules.") }
    guard try H3CheckpointSource.layout(request.transformer, omni:omni?.source).fastVariant == nil else {
      throw H3CheckpointError.invalid("FastH3 Preview v1 does not support reference-conditioned generation.")
    }
    try H3VideoVAELayout(url: request.videoVAE).validateVideoUse()
    if request.geometry.stillImageMode == nil { _ = try H3AudioVAELayout(url: request.audioVAE) }
    if !prepared.layout.conditionAudioIndices.isEmpty {
      try H3AudioVAEEncoder.inspect(checkpointURL: request.audioVAE)
    }
    for adapter in request.loRAAdapters {
      guard let video else { throw H3CheckpointError.invalid("Omni cannot use LoRA adapters.") }
      try adapter.validate(requestedSteps: video.timesteps.count + 1, samplingMethod: request.samplingMethod)
      _ = try H3LoRAFile(url: adapter.url,
        strength: adapter.strength, requestedSteps: video.timesteps.count + 1, samplingMethod: request.samplingMethod, qkvLayout: adapter.qkvLayout, profile: adapter.profile, startAfterEvaluations: adapter.startAfterEvaluations, requiresStandardProfile: refinement.map { $0.strength < 1 || $0.startVideoSigma != nil } ?? false)
    }
    return Admission(geometry: request.geometry,
      textRows: prepared.qwenRequest.tags.count,
      packedRows: layout.tags.count,
      evaluations: rows.indicesByStep.count, layout: layout, referenceLayout: prepared.layout,
      contextVideoRows: contextVideoRows, contextAudioRows: contextAudioRows,
      videoSchedule: video, audioSchedule: audio, rowSchedule: rows, nativeWorkerURL: nativeWorkerURL,
      experimentalSol: experimentalSol, omni: omni)
  }

  public static func run(_ request: H3Ref2VAStillRequest,
    contextFrames: Int = 0, context: H3Continuation.Rows? = nil,
    initialRows: H3JointLatentArtifact.Rows? = nil, refinement: H3JointRefinement? = nil,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    onLatents: (([Float], [Float]) throws -> Void)? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let sampled = try sample(request, contextFrames: contextFrames, context: context, initialRows: initialRows, refinement: refinement, progress: progress)
    try onLatents?(sampled.rows.video, sampled.rows.audio)
    return try H3AVOutputDecoder.decode(videoRows: sampled.rows.video, audioRows: sampled.rows.audio,
      geometry: request.geometry, videoVAE: request.videoVAE, audioVAE: request.audioVAE,
      videoDecodeMemoryMode: request.videoDecodeMemoryMode, videoDecodePrecision: request.videoDecodePrecision,
      backendReport: sampled.backendReport, onFrame: onFrame, onAudio: onAudio, progress: progress)
  }

  public static func sample(_ request: H3Ref2VAStillRequest,
    contextFrames: Int = 0, context: H3Continuation.Rows? = nil,
    initialRows: H3JointLatentArtifact.Rows? = nil, refinement: H3JointRefinement? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> H3SampledLatents {
    guard (initialRows == nil) == (refinement == nil), context == nil || refinement == nil else {
      throw H3CheckpointError.invalid("H3 refinement requires full initial rows and cannot combine tail context.")
    }
    try initialRows?.validate(geometry: request.geometry)
    let admission = try preflight(request, contextFrames: contextFrames, refinement: refinement)
    guard (context == nil) == (contextFrames == 0),
      context.map({ $0.video.count == admission.contextVideoRows * 96
        && $0.audio.count == admission.contextAudioRows * 32
        && $0.video.allSatisfy(\.isFinite) && $0.audio.allSatisfy(\.isFinite) }) ?? true else {
      throw H3CheckpointError.invalid("Ref2VA history must contain the complete finite synchronized tail.")
    }
    let geometry = admission.geometry
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    let textRows = try autoreleasepool { () throws -> [Float] in
      let prepared = try H3VideoReferencePreparation.prepare(prompt: request.prompt,
        geometry: geometry, references: request.references,
        tokenizerURL: request.tokenizer)
      let encoded = try H3QwenTextEncoder.encodeReferences(
        prompt: request.prompt, pixels: prepared.qwenPixels,
        references: prepared.qwenReferences,
        checkpointRoot: request.qwenPages,
        visionCheckpointURL: request.qwenVision,
        tokenizerURL: request.tokenizer) { completed, total in
          progress("text", completed, total)
        }
      guard encoded.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 reference conditioner changed admitted text rows.")
      }
      return encoded.hidden.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("text_weights_released", 1, 1)
    try Task.checkCancellation()

    let conditionVideoRows = try autoreleasepool { () throws -> [Float] in
      guard !admission.referenceLayout.conditionVideoIndices.isEmpty else { return [] }
      let rows = try H3VideoReferencePreparation.encodeVideoRows(
        references: request.references, layout: admission.referenceLayout,
        videoVAEURL: request.videoVAE)
      return rows.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("reference_video_weights_released", 1, 1)
    try Task.checkCancellation()

    let conditionAudioRows: [Float] = try autoreleasepool {
      guard admission.referenceLayout.conditionAudioIndices.count > 0 else { return [] }
      let rows = try H3VideoReferencePreparation.encodeAudioRows(
        references: request.references, layout: admission.referenceLayout,
        audioVAEURL: request.audioVAE)
      return rows.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    if !conditionAudioRows.isEmpty {
      progress("reference_audio_weights_released", 1, 1)
    }
    try Task.checkCancellation()

    var backendReport: H3BackendReport?
    let rawRows = try autoreleasepool { () throws -> ([Float], [Float]) in
      return try H3ReferenceDiTState.withStage({ try H3ReferenceDiTState(checkpointURL: request.transformer,
        layout: admission.layout,
        textEmbeddings: MLXArray(textRows,
          [1, admission.textRows, 5120]).asType(.bfloat16),
        timestepTable: admission.rowSchedule.table,
        blockCount: 50, omni: admission.omni, transformerWeightCacheGB:request.transformerWeightCacheGB,
        allowMPP: H3MPPProjection.isTaskEligible(.ref2va, contextFrames: contextFrames,
          isRefinement: refinement != nil),
        nativeWorkerURL: admission.nativeWorkerURL, experimentalSol: admission.experimentalSol,
        turboLoRAURL: request.turboLoRA,
        turboLoRAStrength: request.turboLoRAStrength,
        additionalLoRAs: request.additionalLoRAs, loRAAdapters: request.loRAAdapters, progress: { completed, total in
          progress("transformer_prepare", completed, total)
        }) }) { state in
        let initial = try H3Noise.makeReference(seed: request.seed,
          conditionVideo: conditionVideoRows, conditionAudio: conditionAudioRows,
          videoLatentFrames: geometry.videoLatentFrames,
          latentHeight: geometry.height / 16, latentWidth: geometry.width / 16,
          audioLatentFrames: geometry.audioLatentFrames,
          referenceNoise: request.referenceNoise, stillImageMode: geometry.stillImageMode)
        let refinedVideo = try initialRows.map {
          guard let schedule = admission.videoSchedule else { throw H3CheckpointError.invalid("Omni rejects initialized video refinement.") }
          return try H3JointRefinement.targetRows(initial.video, source: $0.video,
            prefix: admission.referenceLayout.conditionVideoIndices.count, schedule: schedule, canvasAdmission: geometry.canvasAdmission)
        } ?? initial.video
        let refinedAudio = try initialRows.map {
          guard let schedule = admission.audioSchedule else { throw H3CheckpointError.invalid("Omni rejects initialized audio refinement.") }
          return try H3JointRefinement.targetRows(initial.audio, source: $0.audio,
            prefix: admission.referenceLayout.conditionAudioIndices.count, schedule: schedule, canvasAdmission: geometry.canvasAdmission)
        } ?? initial.audio
        let initialVideo = context.map { concatenated([
          MLXArray($0.video, [1, admission.contextVideoRows, 96]), refinedVideo], axis: 1) } ?? refinedVideo
        let initialAudio = context.map { concatenated([
          MLXArray($0.audio, [1, admission.contextAudioRows, 32]), refinedAudio], axis: 1) } ?? refinedAudio
        let sampled: H3ReferenceSampler.Result
        if let omni = admission.omni {
          sampled = try H3OmniRunner.sample(layout:admission.layout, plan:omni.plan,
            rowSchedule:admission.rowSchedule, videoNoise:initialVideo, audioNoise:initialAudio,
            progress:{ progress("sampling",$0,$1) }) { step,video,audio,indices in
              try state.predict(videoLatents:video,audioLatents:audio,timestepIndices:indices,
                progress:{ progress("sampling_block_\(step + 1)",$0,$1) })
            }
        } else {
        guard let videoSchedule = admission.videoSchedule, let audioSchedule = admission.audioSchedule else { throw H3CheckpointError.invalid("Missing ordinary sampler schedules.") }
        sampled = try H3ReferenceSampler.run(predictor: state,
          videoSchedule: videoSchedule,
          audioSchedule: audioSchedule,
          rowSchedule: admission.rowSchedule,
          videoLatents: initialVideo, audioLatents: initialAudio,
          samplingMethod: request.samplingMethod,
          progress: { completed, total in progress("sampling", completed, total) },
          blockProgress: { step, completed, total in
            progress("sampling_block_\(step)", completed, total)
          })
        }
        let targetVideo = sampled.video[0..<1,
          admission.layout.conditionVideoIndices.count..<admission.layout.videoIndices.count, 0..<96]
        let targetAudio = sampled.audio[0..<1,
          admission.layout.conditionAudioIndices.count..<admission.layout.audioIndices.count,
          0..<32]
        let videoRows = targetVideo.asType(.float32).asArray(Float.self)
        let audioRows = refinement?.preserveAudio == true ? initialRows!.audio : targetAudio.asType(.float32).asArray(Float.self)
        backendReport = state.unloadAndReport()
        return (videoRows, audioRows)
      }
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    try Task.checkCancellation()
    return H3SampledLatents(video: rawRows.0, audio: rawRows.1, backendReport: backendReport)

  }
}
