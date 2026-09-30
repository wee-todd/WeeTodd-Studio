import Foundation
import MLX

/// Already-resampled, bounded 24 fps RGB frames. Media I/O owns file identity,
/// decoding, resizing and source-frame selection before constructing this value.
public struct H3VideoReference: Sendable {
  public let rgb8: Data
  public let frameCount: Int
  public let width: Int
  public let height: Int

  public init(rgb8: Data, frameCount: Int, width: Int, height: Int) {
    self.rgb8 = rgb8
    self.frameCount = frameCount
    self.width = width
    self.height = height
  }
}

public enum H3Ref2VAReference: Sendable {
  case image(H3StillReference)
  case video(H3VideoReference)
}

/// Stages ordered still and silent-video references without co-resident Qwen,
/// video-VAE and transformer weights. Video Qwen frames are sampled at 2 fps;
/// its persistent VAE rows retain the admitted 24 fps source motion.
public enum H3VideoReferencePreparation {
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
    let images = references.filter { if case .image = $0 { true } else { false } }.count
    let videos = references.count - images
    guard images <= 9, videos <= 3 else {
      throw H3CheckpointError.invalid("H3 Ref2VA allows at most nine images and three videos.")
    }
    for reference in references {
      switch reference {
      case .image(let image):
        guard (64...256).contains(image.width), (64...256).contains(image.height),
          image.width.isMultiple(of: 32), image.height.isMultiple(of: 32),
          image.rgb8.count == image.width * image.height * 3 else {
          throw H3CheckpointError.invalid("H3 reference image needs bounded RGB8 pixels on the 32-pixel grid.")
        }
      case .video(let video):
        guard (5...175).contains(video.frameCount),
          (video.frameCount - 5).isMultiple(of: 17),
          (64...256).contains(video.width), (64...256).contains(video.height),
          video.width.isMultiple(of: 32), video.height.isMultiple(of: 32),
          video.rgb8.count == video.frameCount * video.width * video.height * 3 else {
          throw H3CheckpointError.invalid("H3 reference video needs bounded 24 fps RGB8 frames on the 5 + 17*n grid.")
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
      case .image(let image):
        let packed = try H3QwenImageProcessor.packRGB8(image: image.rgb8,
          width: image.width, height: image.height)
        pixels.append(packed.pixels)
        grids.append(packed.grid)
        qwenReferences.append(.image(grid: packed.grid))
        specs.append(.image(latentHeight: image.height / 16,
          latentWidth: image.width / 16))
      case .video(let video):
        let frameBytes = video.width * video.height * 3
        let indices = Array(stride(from: 0, to: video.frameCount, by: 12))
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
          let timestamp = Double(pairStart + min(pairStart + 1, indices.count - 1)) / 4
          blocks.append(.init(timestampSeconds: timestamp, grid: packed.grid))
        }
        qwenReferences.append(.video(blocks: blocks, hasAudio: false))
        let latentFrames = (video.frameCount - 5) / 17 * 5 + 2
        specs.append(.video(latentFrames: latentFrames,
          latentHeight: video.height / 16,
          latentWidth: video.width / 16, audioLatents: 0,
          sourceLatentFrames: latentFrames))
      }
    }
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let request = try H3QwenRequest.references(prompt: prompt,
      references: qwenReferences, tokenizer: tokenizer)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: request.tags, references: specs)
    let joined = concatenated(pixels, axis: 0).asType(.bfloat16)
    eval(joined)
    return Prepared(qwenRequest: request, qwenGrids: grids,
      qwenReferences: qwenReferences, qwenPixels: joined, layout: layout)
  }

  public static func encodeVideoRows(references: [H3Ref2VAReference],
    layout: H3ReferenceLayout, videoVAEURL: URL) throws -> MLXArray {
    let metadata = try H3VideoVAELayout(url: videoVAEURL)
    var pieces: [MLXArray] = []
    for reference in references {
      try Task.checkCancellation()
      let latent: MLXArray
      switch reference {
      case .image(let image):
        latent = try H3VideoVAEEncoder.encodeStill(
          checkpointURL: videoVAEURL, rgb8: Array(image.rgb8),
          width: image.width, height: image.height)
      case .video(let video):
        latent = try H3VideoVAEEncoder.encodeVideo(
          checkpointURL: videoVAEURL, rgb8: Array(video.rgb8),
          frameCount: video.frameCount, width: video.width, height: video.height)
      }
      let rows = try H3LatentCodec.videoEncoderRows(latents: latent,
        mean: metadata.latentsMean,
        standardDeviation: metadata.latentsStandardDeviation)
      pieces.append(rows)
      eval(rows)
      Memory.clearCache()
    }
    let joined = concatenated(pieces, axis: 1)
    guard joined.shape == [1, layout.conditionVideoIndices.count, 96] else {
      throw H3CheckpointError.invalid("H3 reference video rows changed admitted geometry.")
    }
    eval(joined)
    return joined
  }
}
