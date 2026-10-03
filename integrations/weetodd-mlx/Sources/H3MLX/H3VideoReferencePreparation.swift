import Foundation
import MLX

/// Already-resampled, bounded 24 fps RGB frames. Media I/O owns file identity,
/// decoding, resizing and source-frame selection before constructing this value.
public struct H3VideoReference: Sendable {
  public let rgb8: Data
  public let frameCount: Int
  public let width: Int
  public let height: Int
  public let audio: H3AudioReference?

  public init(rgb8: Data, frameCount: Int, width: Int, height: Int,
    audio: H3AudioReference? = nil) {
    self.rgb8 = rgb8
    self.frameCount = frameCount
    self.width = width
    self.height = height
    self.audio = audio
  }
}

public struct H3AudioReference: Sendable {
  /// Planar stereo Float32 PCM at 32 kHz: all left samples, then all right.
  public let samples: [Float]
  public let frames: Int

  public init(samples: [Float], frames: Int) {
    self.samples = samples
    self.frames = frames
  }
}

public enum H3Ref2VAReference: Sendable {
  case image(H3StillReference)
  case video(H3VideoReference)
  case audio(H3AudioReference)
  case timedImage(H3StillReference, frame: Int)
  case timedAudio(H3AudioReference, frame: Int)
}

/// Stages ordered visual and sound references without co-resident Qwen,
/// VAE and transformer weights. Video Qwen frames are sampled at 2 fps;
/// its persistent VAE rows retain the admitted 24 fps source motion.
public enum H3VideoReferencePreparation {
  private static func validAudio(_ audio: H3AudioReference) -> Bool {
    (800...480_000).contains(audio.frames) &&
      audio.samples.count == 2 * audio.frames &&
      audio.samples.allSatisfy(\.isFinite)
  }

  public struct Prepared {
    public let qwenRequest: H3QwenRequest
    public let qwenGrids: [H3QwenRequest.Grid]
    public let qwenReferences: [H3QwenRequest.Reference]
    public let qwenPixels: MLXArray
    public let layout: H3ReferenceLayout
  }

  public static func validate(_ references: [H3Ref2VAReference]) throws {
    guard (1...12).contains(references.count) else {
      throw H3CheckpointError.invalid("H3 Ref2VA needs one to twelve ordered references.")
    }
    let images = references.filter {
      switch $0 { case .image, .timedImage: true; default: false }
    }.count
    let videos = references.filter { if case .video = $0 { true } else { false } }.count
    let audios = references.filter {
      switch $0 {
      case .audio, .timedAudio: true
      case .video(let video): video.audio != nil
      case .image, .timedImage: false
      }
    }.count
    guard (images + videos > 0 || references.allSatisfy({
      if case .timedAudio = $0 { return true }
      return false
    })), images <= 9, videos <= 3, audios <= 3 else {
      throw H3CheckpointError.invalid("H3 Ref2VA needs a visual reference or timed audio, with at most nine images, three videos and three audio sources.")
    }
    for reference in references {
      switch reference {
      case .image(let image), .timedImage(let image, _):
        try H3StillReferenceMedia.validateCanvas(width: image.width, height: image.height)
        guard image.rgb8.count == image.width * image.height * 3 else {
          throw H3CheckpointError.invalid("H3 reference image needs bounded RGB8 pixels on the 32-pixel grid.")
        }
      case .video(let video):
        guard (5...362).contains(video.frameCount),
          (video.frameCount - 5).isMultiple(of: 17),
          (64...256).contains(video.width), (64...256).contains(video.height),
          video.width.isMultiple(of: 32), video.height.isMultiple(of: 32),
          video.rgb8.count == video.frameCount * video.width * video.height * 3 else {
          throw H3CheckpointError.invalid("H3 reference video needs bounded 24 fps RGB8 frames on the 5 + 17*n grid.")
        }
        if let audio = video.audio {
          guard validAudio(audio) else {
            throw H3CheckpointError.invalid("H3 movie soundtrack needs at most 15 seconds of finite 32 kHz stereo PCM.")
          }
        }
      case .audio(let audio), .timedAudio(let audio, _):
        guard validAudio(audio) else {
          throw H3CheckpointError.invalid("H3 reference audio needs at most 15 seconds of finite 32 kHz stereo PCM.")
        }
      }
    }
  }

  public static func prepare(prompt: String, geometry: H3Geometry,
    references: [H3Ref2VAReference], tokenizerURL: URL) throws -> Prepared {
    try validate(references)
    var grids: [H3QwenRequest.Grid] = []
    var qwenReferences: [H3QwenRequest.Reference] = []
    var pixels: [MLXArray] = []
    var specs: [H3ReferenceSpec] = []
    for reference in references {
      try Task.checkCancellation()
      switch reference {
      case .image(let image), .timedImage(let image, _):
        let packed = try H3QwenImageProcessor.packRGB8(image: image.rgb8,
          width: image.width, height: image.height)
        pixels.append(packed.pixels)
        grids.append(packed.grid)
        qwenReferences.append(.image(grid: packed.grid))
        let frame: Int?
        if case .timedImage(_, let target) = reference { frame = target }
        else { frame = nil }
        specs.append(.image(latentHeight: image.height / 16,
          latentWidth: image.width / 16, targetFrame: frame))
      case .video(let video):
        let frameBytes = video.width * video.height * 3
        // Keep at most six paired visual blocks for one movie, reserving
        // Qwen rows for the explicit seam image and prompt. Full-rate VAE
        // latents still carry the source motion and soundtrack.
        let visualStride = max(12, (video.frameCount + 11) / 12)
        let indices = Array(stride(from: 0, to: video.frameCount, by: visualStride))
        var blocks: [H3QwenRequest.VideoBlock] = []
        for pairStart in stride(from: 0, to: indices.count, by: 2) {
          let first = indices[pairStart]
          let second = indices[min(pairStart + 1, indices.count - 1)]
          let pair = video.rgb8.subdata(in: first * frameBytes..<(first + 1) * frameBytes)
            + video.rgb8.subdata(in: second * frameBytes..<(second + 1) * frameBytes)
          let packed = try H3QwenVideoProcessor.packRGB8(frames: pair,
            frameCount: 2, width: video.width, height: video.height)
          pixels.append(packed.pixels)
          grids.append(packed.grid)
          let timestamp = (Double(first) + Double(second)) / 48
          blocks.append(.init(timestampSeconds: timestamp, grid: packed.grid))
        }
        qwenReferences.append(.video(blocks: blocks,
          hasAudio: video.audio != nil))
        let latentFrames = (video.frameCount - 5) / 17 * 5 + 2
        specs.append(.video(latentFrames: latentFrames,
          latentHeight: video.height / 16,
          latentWidth: video.width / 16,
          audioLatents: video.audio.map { ($0.frames + 799) / 800 } ?? 0,
          sourceLatentFrames: latentFrames))
      case .audio(let audio), .timedAudio(let audio, _):
        qwenReferences.append(.audio)
        let frame: Int?
        if case .timedAudio(_, let target) = reference { frame = target }
        else { frame = nil }
        specs.append(.audio(latents: (audio.frames + 799) / 800,
          targetFrame: frame))
      }
    }
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let request = try H3QwenRequest.references(prompt: prompt,
      references: qwenReferences, tokenizer: tokenizer)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: request.tags, references: specs)
    let joined = pixels.isEmpty ? MLXArray([Float](), [0, 1536]).asType(.bfloat16)
      : concatenated(pixels, axis: 0).asType(.bfloat16)
    eval(joined)
    return Prepared(qwenRequest: request, qwenGrids: grids,
      qwenReferences: qwenReferences, qwenPixels: joined, layout: layout)
  }

  public static func encodeVideoRows(references: [H3Ref2VAReference],
    layout: H3ReferenceLayout, videoVAEURL: URL) throws -> MLXArray {
    if layout.conditionVideoIndices.isEmpty {
      guard references.allSatisfy({
        switch $0 { case .audio, .timedAudio: true; default: false }
      }) else {
        throw H3CheckpointError.invalid("H3 visual references need video condition rows.")
      }
      return MLXArray([Float](), [1, 0, 96])
    }
    let metadata = try H3VideoVAELayout(url: videoVAEURL)
    var pieces: [MLXArray] = []
    for reference in references {
      try Task.checkCancellation()
      let latent: MLXArray
      switch reference {
      case .image(let image), .timedImage(let image, _):
        latent = try H3VideoVAEEncoder.encodeStill(
          checkpointURL: videoVAEURL, rgb8: Array(image.rgb8),
          width: image.width, height: image.height)
      case .video(let video):
        latent = try H3VideoVAEEncoder.encodeVideo(
          checkpointURL: videoVAEURL, rgb8: Array(video.rgb8),
          frameCount: video.frameCount, width: video.width, height: video.height)
      case .audio, .timedAudio:
        continue
      }
      let rows = try H3LatentCodec.videoEncoderRows(latents: latent,
        mean: metadata.latentsMean,
        standardDeviation: metadata.latentsStandardDeviation)
      pieces.append(rows)
      eval(rows)
      Memory.clearCache()
    }
    let joined = pieces.isEmpty ? MLXArray([Float](), [1, 0, 96])
      : concatenated(pieces, axis: 1)
    guard joined.shape == [1, layout.conditionVideoIndices.count, 96] else {
      throw H3CheckpointError.invalid("H3 reference video rows changed admitted geometry.")
    }
    eval(joined)
    return joined
  }

  public static func encodeAudioRows(references: [H3Ref2VAReference],
    layout: H3ReferenceLayout, audioVAEURL: URL) throws -> MLXArray {
    let metadata = try H3AudioVAELayout(url: audioVAEURL)
    var pieces: [MLXArray] = []
    for reference in references {
      try Task.checkCancellation()
      let audio: H3AudioReference
      switch reference {
      case .audio(let value), .timedAudio(let value, _): audio = value
      case .video(let video):
        guard let value = video.audio else { continue }
        audio = value
      case .image, .timedImage: continue
      }
      let waveform = MLXArray(audio.samples, [2, audio.frames, 1])
      let latent = try H3AudioVAEEncoder.encode(checkpointURL: audioVAEURL,
        waveform: waveform)
      pieces.append(try H3LatentCodec.audioEncoderRows(latents: latent,
        mean: metadata.latentsMean,
        standardDeviation: metadata.latentsStandardDeviation))
      Memory.clearCache()
    }
    guard !pieces.isEmpty else { return MLXArray([Float](), [1, 0, 32]) }
    let joined = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
    guard joined.shape == [1, layout.conditionAudioIndices.count, 32] else {
      throw H3CheckpointError.invalid("H3 reference audio rows changed admitted geometry.")
    }
    eval(joined)
    return joined
  }
}
