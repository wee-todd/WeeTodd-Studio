import Foundation
import MLX

/// Narrow text-to-audiovisual H3 vertical slice. It owns no movie files: a
/// worker must stage every decoded frame and waveform before atomic publish.
/// Other H3 tasks have separate conditioning contracts and cannot enter here.
public struct H3T2VARequest: Sendable {
  public let prompt: String
  public let geometry: H3Geometry
  public let durationSeconds: Double
  public let seed: UInt64
  public let requestedSteps: Int
  public let samplingMethod: H3SamplingMethod
  public let transformer: URL
  public let qwenPages: URL
  public let tokenizer: URL
  public let videoDecodeMemoryMode: H3VideoDecodeMemoryMode?
  public let videoVAE: URL
  public let audioVAE: URL
  public let turboLoRA: URL?
  public let turboLoRAStrength: Float
  public let additionalLoRAs: [H3LoRAAdapter]
  public let loRAAdapters: [H3LoRAAdapter]
  public let funControl: H3FunControlGuide?
  public let vdn:H3VDNSelection?

  public init(prompt: String, width: Int, height: Int,
    durationSeconds: Double, seed: UInt64, requestedSteps: Int,
    transformer: URL, qwenPages: URL, tokenizer: URL,
    videoVAE: URL, audioVAE: URL, turboLoRA: URL? = nil,
    turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    loRAAdapters: [H3LoRAAdapter]? = nil,
    funControl: H3FunControlGuide? = nil,
    vdn:H3VDNSelection? = nil,
    videoDecodeMemoryMode: H3VideoDecodeMemoryMode? = nil,
    samplingMethod: H3SamplingMethod = .euler,
    canvasAdmission: H3CanvasAdmission = .ordinary) throws {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= 65_536, (2...101).contains(requestedSteps),
      [transformer, qwenPages, tokenizer, videoVAE, audioVAE]
        .allSatisfy({ $0.isFileURL && $0.path.hasPrefix("/") }),
      turboLoRA == nil || (turboLoRA!.isFileURL &&
        turboLoRA!.path.hasPrefix("/") && turboLoRAStrength.isFinite &&
        (-10...10).contains(turboLoRAStrength)),
      (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count <= 8,
      Set(([turboLoRA].compactMap { $0 } + additionalLoRAs.map(\.url))
        .map { $0.standardizedFileURL.path }).count ==
        (turboLoRA == nil ? 0 : 1) + additionalLoRAs.count else {
      throw H3CheckpointError.invalid("Invalid H3 text-to-audiovisual request.")
    }
    let geometry = try H3Geometry(width: width, height: height,
      durationSeconds: durationSeconds, canvasAdmission: canvasAdmission)
    guard width * height <= geometry.canvasAdmission.maximumPixels else {
      throw H3CheckpointError.invalid("H3 video area exceeds the current Swift canvas budget (1376 × 768 pixels).")
    }
    guard try geometry.packedRows(textRows: 1,
      conditionVideoRows: 0, conditionAudioRows: 0) <= geometry.maximumPackedRows else {
      throw H3CheckpointError.invalid("H3 packed rows exceed the current Swift engine limit.")
    }
    if let funControl {
      try funControl.validate(geometry: geometry)
    }
    self.funControl = funControl
    if geometry.canvasAdmission == .spatialRefinement {
      try geometry.canvasAdmission.validate(width: width, height: height)
      _ = try H3VideoVAEDecoder.preflightSpatial(geometry: geometry)
      guard funControl == nil else { throw H3CheckpointError.invalid("Spatial refinement cannot combine Fun control.") }
    }
    self.prompt = prompt
    self.geometry = geometry
    self.durationSeconds = durationSeconds
    self.seed = seed
    self.requestedSteps = requestedSteps
    self.samplingMethod = samplingMethod
    self.transformer = transformer
    self.qwenPages = qwenPages
    self.tokenizer = tokenizer
    self.videoDecodeMemoryMode = videoDecodeMemoryMode
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
    self.loRAAdapters = effective
    if let vdn {
      guard requestedSteps == vdn.schedulePoints,samplingMethod == .euler,
        effective.isEmpty,turboLoRA == nil,additionalLoRAs.isEmpty,funControl == nil,canvasAdmission == .ordinary else {
        throw H3CheckpointError.invalid("VDN requires its released Euler schedule, mandatory adapter stack and ordinary T2VA canvas, without other controls.")
      }
    }
    self.vdn=vdn
  }
}

public enum H3T2VARunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let textRows: Int
    public let packedRows: Int
    public let evaluations: Int
    let layout: H3PackedLayout
    let videoSchedule: H3Schedule
    let audioSchedule: H3Schedule
    let rowSchedule: H3RowSchedule
  }

  public struct Result {
    public let videoFrames: Int
    public let audioSamplesPerChannel: Int
    public let audioSampleRate: Int
  }

  /// Inspect every installed component and all task geometry before the first
  /// weighted stage. A missing VAE cannot fail after a long transformer run.
  public static func preflight(_ request: H3T2VARequest, refinement: H3JointRefinement? = nil) throws -> Admission {
    guard request.geometry.canvasAdmission == .ordinary || refinement != nil else {
      throw H3CheckpointError.invalid("Expanded H3 canvas requires explicit initialized spatial refinement.")
    }
    try Task.checkCancellation()
    let tokenizer = try H3QwenTokenizer(url: request.tokenizer)
    let qwen = try H3QwenRequest.text(request.prompt, tokenizer: tokenizer)
    let layout = try H3PackedLayout(geometry: request.geometry,
      textTags: qwen.tags, anchors: [])
    let grids = try refinement?.schedules(requestedSteps: request.requestedSteps)
    let video = try grids?.video ?? H3Schedule(requestedSteps: request.requestedSteps, shift: 12)
    let audio = try grids?.audio ?? H3Schedule(requestedSteps: request.requestedSteps, shift: 3)
    let rows = try H3RowSchedule(layout: layout, video: video, audio: audio)
    guard rows.table.count <= 128, layout.tags.count <= request.geometry.maximumPackedRows else {
      throw H3CheckpointError.invalid("H3 timestep table or packed rows exceed engine admission.")
    }
    _ = try H3QwenCheckpointLayout.inspect(root: request.qwenPages)
    let transformer = try H3CheckpointLayout(url: request.transformer)
    if let vdn=request.vdn {
      guard refinement == nil,
        transformer.curveRank == nil ? vdn.adalnInputGrid == nil :
          (vdn.variant == .fiftyStep || vdn.adalnInputGrid != nil) else {
        throw H3CheckpointError.invalid("VDN cannot combine refinement; pruned eight-step H3 requires the explicit original-width AdaLN input grid, while an unpruned base uses its own timestep encoder.")
      }
      _ = try H3VDNLayout(packed:layout)
      try vdn.preflight()
    }
    if let control = request.funControl {
      try control.validate(geometry: request.geometry)
      _ = try H3FunControlLayout(url: control.checkpoint, base: transformer)
    }
    _ = try H3VideoVAELayout(url: request.videoVAE)
    _ = try H3AudioVAELayout(url: request.audioVAE)
    for adapter in request.loRAAdapters {
      try adapter.validate(requestedSteps: video.timesteps.count + 1, samplingMethod: request.samplingMethod)
      _ = try H3LoRAFile(url: adapter.url,
        strength: adapter.strength, requestedSteps: video.timesteps.count + 1, samplingMethod: request.samplingMethod, qkvLayout: adapter.qkvLayout, profile: adapter.profile, startAfterEvaluations: adapter.startAfterEvaluations, requiresStandardProfile: refinement.map { $0.strength < 1 || $0.startVideoSigma != nil } ?? false)
    }
    return Admission(geometry: request.geometry,
      textRows: qwen.tags.count, packedRows: layout.tags.count,
      evaluations: video.timesteps.count, layout: layout,
      videoSchedule: video, audioSchedule: audio, rowSchedule: rows)
  }

  /// `onFrame` receives exactly one RGB8 frame at a time. `onAudio` receives
  /// channel-major float32 stereo at 32 kHz after the video stage is released.
  public static func run(_ request: H3T2VARequest,
    initialRows: H3JointLatentArtifact.Rows? = nil, refinement: H3JointRefinement? = nil,
    publicationAudio: H3AudioReference? = nil,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    onLatents: (([Float], [Float]) throws -> Void)? = nil,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    guard (initialRows == nil) == (refinement == nil) else {
      throw H3CheckpointError.invalid("H3 refinement requires both joint initial rows and explicit controls.")
    }
    if let publicationAudio {
      guard refinement?.preserveAudio == true, initialRows != nil,
        publicationAudio.frames > 0, publicationAudio.frames <= 480_000,
        publicationAudio.samples.count == publicationAudio.frames * 2,
        publicationAudio.samples.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("Original audio publication requires explicit initialized AV preservation.")
      }
    }
    try initialRows?.validate(geometry: request.geometry)
    let admission = try preflight(request, refinement: refinement)
    let geometry = admission.geometry
    // Guide VAE weights are released before Qwen or the transformer loads.
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    var controlCondition = try autoreleasepool {
      try request.funControl?.encode(videoVAE: request.videoVAE, geometry: geometry) {
        progress("control_video_encode", $0, $1)
      }
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    if controlCondition != nil { progress("control_video_weights_released", 1, 1) }
    let rawRows = try autoreleasepool { () throws -> ([Float], [Float]) in
      try Task.checkCancellation()
      let conditioned = try H3QwenTextEncoder.encode(prompt: request.prompt,
        checkpointRoot: request.qwenPages,
        tokenizerURL: request.tokenizer) { completed, total in
        progress("text", completed, total)
      }
      guard conditioned.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 conditioner changed its admitted text rows.")
      }
      progress("text_weights_released", 1, 1)
      let state = try H3DiTState(checkpointURL: request.transformer,
        layout: admission.layout,
        textEmbeddings: conditioned.hidden
          .reshaped([1, admission.textRows, 5120]),
        timestepTable: admission.rowSchedule.table,
        blockCount: 50,
        turboLoRAURL: request.turboLoRA,
        turboLoRAStrength: request.turboLoRAStrength,
        additionalLoRAs: request.additionalLoRAs, loRAAdapters: request.loRAAdapters,
        funControl: controlCondition,vdn:request.vdn) { completed, total in
        progress("transformer_prepare", completed, total)
      }
      defer { state.unload() }
      let noise = try H3Noise.make(seed: request.seed,
        videoLatentFrames: geometry.videoLatentFrames,
        latentHeight: geometry.height / 16,
        latentWidth: geometry.width / 16,
        audioLatentFrames: geometry.audioLatentFrames)
      let initializedVideo = try initialRows.map {
        try H3JointRefinement.targetRows(noise.video, source: $0.video, prefix: 0, schedule: admission.videoSchedule, canvasAdmission: geometry.canvasAdmission)
      } ?? noise.video
      let initializedAudio = try initialRows.map {
        try H3JointRefinement.targetRows(noise.audio, source: $0.audio, prefix: 0, schedule: admission.audioSchedule, canvasAdmission: geometry.canvasAdmission)
      } ?? noise.audio
      let sampled = try H3AVSampler.run(predictor: state,
        videoSchedule: admission.videoSchedule,
        audioSchedule: admission.audioSchedule,
        rowSchedule: admission.rowSchedule,
        videoLatents: initializedVideo, audioLatents: initializedAudio,
        samplingMethod: request.samplingMethod,
        progress: { completed, total in progress("sampling", completed, total) },
        blockProgress: { step, completed, total in
          progress("sampling_block_\(step)", completed, total)
        })
      let video = sampled.video.asType(.float32).asArray(Float.self)
      let audio = refinement?.preserveAudio == true ? initialRows!.audio : sampled.audio.asType(.float32).asArray(Float.self)
      return (video, audio)
    }
    controlCondition = nil
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    try Task.checkCancellation()
    try onLatents?(rawRows.0, rawRows.1)

    let result = try H3AVOutputDecoder.decode(videoRows: rawRows.0,
      audioRows: rawRows.1, geometry: geometry, videoVAE: request.videoVAE,
      audioVAE: request.audioVAE, videoDecodeMemoryMode: request.videoDecodeMemoryMode,
      publicationAudio: publicationAudio,
      onFrame: onFrame, onAudio: onAudio,
      progress: progress)
    return Result(videoFrames: result.videoFrames,
      audioSamplesPerChannel: result.audioSamplesPerChannel,
      audioSampleRate: result.audioSampleRate)
  }
}
