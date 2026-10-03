import Foundation
import MLX
import MLXRandom

/// FL2VA anchors are conditioned video rows at explicit generated-frame times,
/// never untimed Ref2VA appearance references.
public struct H3FL2VARequest: Sendable {
  public let base: H3T2VARequest
  public let vision: URL
  public let images: [H3StillReference]
  public let anchors: [H3PackedLayout.Anchor]

  public init(base: H3T2VARequest, vision: URL,
    images: [H3StillReference], anchors: [H3PackedLayout.Anchor]) throws {
    guard base.funControl == nil, vision.isFileURL, vision.path.hasPrefix("/"),
      (1...8).contains(images.count), images.count == anchors.count,
      anchors.enumerated().allSatisfy({ index, anchor in
        let frame: Int
        switch anchor {
        case .first: frame = 0
        case .last: frame = base.geometry.frames - 1
        case .frame(let value): frame = value
        }
        guard (0..<base.geometry.frames).contains(frame) else { return false }
        if index == 0 { return true }
        let previous: Int
        switch anchors[index - 1] {
        case .first: previous = 0
        case .last: previous = base.geometry.frames - 1
        case .frame(let value): previous = value
        }
        return previous < frame
      }),
      images.allSatisfy({ $0.width == base.geometry.width &&
        $0.height == base.geometry.height &&
        $0.rgb8.count == $0.width * $0.height * 3 }) else {
      throw H3CheckpointError.invalid("H3 FL2VA requires one to eight prepared canvas images at unique ascending frame positions, without a Fun control branch.")
    }
    self.base = base
    self.vision = vision
    self.images = images
    self.anchors = anchors
  }
}

public enum H3FL2VARunner {
  public struct Admission {
    public let geometry: H3Geometry
    public let packedRows: Int
    public let evaluations: Int
    let textRows: Int
    let layout: H3PackedLayout
    let videoSchedule: H3Schedule
    let audioSchedule: H3Schedule
    let rowSchedule: H3RowSchedule
  }

  public typealias Result = H3AVOutputDecoder.Result

  public static func preflight(_ request: H3FL2VARequest) throws -> Admission {
    try Task.checkCancellation()
    let base = request.base
    let tokenizer = try H3QwenTokenizer(url: base.tokenizer)
    let qwen = try H3FL2VAQwenFrames.prepare(images: request.images,
      prompt: base.prompt, tokenizer: tokenizer).request
    let layout = try H3PackedLayout(geometry: base.geometry,
      textTags: qwen.tags, anchors: request.anchors)
    let video = try H3Schedule(requestedSteps: base.requestedSteps, shift: 12)
    let audio = try H3Schedule(requestedSteps: base.requestedSteps, shift: 3)
    let rows = try H3RowSchedule(layout: layout, video: video, audio: audio)
    guard rows.table.count <= 128, layout.tags.count <= 40_000 else {
      throw H3CheckpointError.invalid("H3 FL2VA packed rows exceed engine admission.")
    }
    _ = try H3QwenCheckpointLayout.inspect(root: base.qwenPages)
    let vision = try H3QwenCheckpointLayout.inspect(root: request.vision)
    guard vision.visionFile != nil else {
      throw H3CheckpointError.invalid("H3 FL2VA needs a Qwen vision tower.")
    }
    let checkpoint = try H3CheckpointLayout(url: base.transformer)
    guard checkpoint.curveRank == 64 else {
      throw H3CheckpointError.invalid("Swift FL2VA requires a verified FL2VA transformer partition.")
    }
    try H3VideoVAEEncoder.preflight(checkpointURL: base.videoVAE)
    _ = try H3AudioVAELayout(url: base.audioVAE)
    for adapter in base.loRAAdapters {
      _ = try H3LoRAFile(url: adapter.url, strength: adapter.strength,
        requestedSteps: base.requestedSteps)
    }
    return Admission(geometry: base.geometry, packedRows: layout.tags.count,
      evaluations: video.timesteps.count, textRows: qwen.tags.count,
      layout: layout, videoSchedule: video, audioSchedule: audio,
      rowSchedule: rows)
  }

  public static func run(_ request: H3FL2VARequest,
    onFrame: (Int, Data) throws -> Void,
    onAudio: ([Float], Int) throws -> Void,
    progress: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let admission = try preflight(request)
    let base = request.base
    let geometry = admission.geometry
    let textRows = try autoreleasepool { () throws -> [Float] in
      let tokenizer = try H3QwenTokenizer(url: base.tokenizer)
      let visual = try H3FL2VAQwenFrames.prepare(images: request.images,
        prompt: base.prompt, tokenizer: tokenizer)
      let packed = try visual.images.map {
        try H3QwenImageProcessor.packRGB8(image: $0.rgb8,
          width: $0.width, height: $0.height)
      }
      let pixels = concatenated(packed.map(\.pixels), axis: 0).asType(.bfloat16)
      let encoded = try H3QwenTextEncoder.encodeKeyframes(prompt: base.prompt,
        pixels: pixels, grids: packed.map(\.grid),
        checkpointRoot: base.qwenPages, visionCheckpointURL: request.vision,
        tokenizerURL: base.tokenizer) { completed, total in
          progress("text", completed, total)
        }
      guard encoded.tags == Array(admission.layout.tags.prefix(admission.textRows)) else {
        throw H3CheckpointError.invalid("H3 FL2VA text rows changed after admission.")
      }
      return encoded.hidden.asType(.float32).asArray(Float.self)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("text_weights_released", 1, 1)
    try Task.checkCancellation()

    let conditionRows = try autoreleasepool { () throws -> [Float] in
      let metadata = try H3VideoVAELayout(url: base.videoVAE)
      MLXRandom.seed(42)
      var pieces: [MLXArray] = []
      for image in request.images {
        try Task.checkCancellation()
        let latent = try H3VideoVAEEncoder.encodeKeyframe(
          checkpointURL: base.videoVAE, rgb8: Array(image.rgb8),
          width: image.width, height: image.height)
        pieces.append(try H3LatentCodec.videoEncoderRows(latents: latent,
          mean: metadata.latentsMean,
          standardDeviation: metadata.latentsStandardDeviation))
      }
      let joined = concatenated(pieces, axis: 1)
      guard joined.shape == [1, admission.layout.conditionVideoRows, 96] else {
        throw H3CheckpointError.invalid("H3 FL2VA condition rows changed after admission.")
      }
      let rows = joined.asType(.float32).asArray(Float.self)
      guard rows.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("H3 FL2VA keyframe latents contain non-finite values.")
      }
      return rows
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("keyframe_video_weights_released", 1, 1)
    try Task.checkCancellation()

    let rawRows = try autoreleasepool { () throws -> ([Float], [Float]) in
      let state = try H3DiTState(checkpointURL: base.transformer,
        layout: admission.layout,
        textEmbeddings: MLXArray(textRows,
          [1, admission.textRows, 5120]).asType(.bfloat16),
        timestepTable: admission.rowSchedule.table,
        turboLoRAURL: base.turboLoRA,
        turboLoRAStrength: base.turboLoRAStrength,
        additionalLoRAs: base.additionalLoRAs) { completed, total in
          progress("transformer_prepare", completed, total)
        }
      defer { state.unload() }
      let noise = try H3Noise.makeWithCondition(seed: base.seed,
        conditionRows: admission.layout.conditionVideoRows,
        videoLatentFrames: geometry.videoLatentFrames,
        latentHeight: geometry.height / 16,
        latentWidth: geometry.width / 16,
        audioLatentFrames: geometry.audioLatentFrames)
      let clean = MLXArray(conditionRows,
        [1, admission.layout.conditionVideoRows, 96])
      let anchored = Float(0.999) * clean + Float(0.001) * noise.condition
      let video = concatenated([anchored, noise.video], axis: 1)
      let sampled = try H3AVSampler.run(predictor: state,
        videoSchedule: admission.videoSchedule,
        audioSchedule: admission.audioSchedule,
        rowSchedule: admission.rowSchedule,
        videoLatents: video, audioLatents: noise.audio,
        progress: { completed, total in progress("sampling", completed, total) },
        blockProgress: { step, completed, total in
          progress("sampling_block_\(step)", completed, total)
        })
      let targetVideo = sampled.video[0..<1,
        admission.layout.conditionVideoRows..<sampled.video.shape[1], 0..<96]
      let videoRows = targetVideo.asType(.float32).asArray(Float.self)
      let audioRows = sampled.audio.asType(.float32).asArray(Float.self)
      guard videoRows.allSatisfy(\.isFinite), audioRows.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("H3 FL2VA sampled AV latents contain non-finite values.")
      }
      return (videoRows, audioRows)
    }
    Stream.gpu.synchronize()
    Memory.clearCache()
    progress("transformer_weights_released", 1, 1)
    return try H3AVOutputDecoder.decode(videoRows: rawRows.0,
      audioRows: rawRows.1, geometry: geometry,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE,
      onFrame: onFrame, onAudio: onAudio, progress: progress)
  }
}
