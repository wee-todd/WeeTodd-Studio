import Foundation
import MLX

/// Immutable, already-decoded RGB bytes. Media I/O owns source identity,
/// resizing and mutation checks before constructing one of these values.
public struct H3StillReference: Sendable {
  public let rgb8: Data
  public let width: Int
  public let height: Int
  public let pixelBudgetPercent: Int?

  public init(rgb8: Data, width: Int, height: Int, pixelBudgetPercent: Int? = nil) {
    self.rgb8 = rgb8
    self.width = width
    self.height = height
    self.pixelBudgetPercent = pixelBudgetPercent
  }
}

/// Preserves the same still order in Qwen presentation and packed H3 rows.
/// Each prepared still retains at most 64 visual pads on an aspect-preserving
/// 32-pixel grid; the media adapter owns the bounded source decode.
public enum H3StillReferencePreparation {
  public struct Prepared {
    public let qwenRequest: H3QwenRequest
    public let qwenGrids: [H3QwenRequest.Grid]
    public let qwenPixels: MLXArray
    public let layout: H3ReferenceLayout
    fileprivate let references: [H3StillReference]

    /// Stage the installed video encoder after Qwen has released its weights.
    /// Each still contributes exactly one deterministic posterior-mean frame.
    public func encodeVideoRows(videoVAEURL: URL) throws -> MLXArray {
      try H3StillReferencePreparation.encodeVideoRows(references: references,
        layout: layout, videoVAEURL: videoVAEURL)
    }
  }

  /// Allows a runner to release the Qwen presentation (including its patch
  /// tensor) before loading video-VAE weights for the same ordered stills.
  public static func encodeVideoRows(references: [H3StillReference],
    layout: H3ReferenceLayout, videoVAEURL: URL) throws -> MLXArray {
      let metadata = try H3VideoVAELayout(url: videoVAEURL)
      var pieces: [MLXArray] = []
      pieces.reserveCapacity(references.count)
      for reference in references {
        try Task.checkCancellation()
        let moments = try H3VideoVAEEncoder.encodeStill(
          checkpointURL: videoVAEURL, rgb8: Array(reference.rgb8),
          width: reference.width, height: reference.height,
          maximumReferencePixels: reference.pixelBudgetPercent == nil ? nil : 4 * H3Geometry.maximumCanvasPixels)
        let rows = try H3LatentCodec.videoEncoderRows(latents: moments,
          mean: metadata.latentsMean,
          standardDeviation: metadata.latentsStandardDeviation)
        pieces.append(rows)
        eval(rows)
        Memory.clearCache()
      }
      let result = concatenated(pieces, axis: 1)
      guard result.shape == [1, layout.conditionVideoIndices.count, 96] else {
        throw H3CheckpointError.invalid("Still-reference VAE rows changed admitted geometry.")
      }
      eval(result)
      return result
  }

  public static func prepare(prompt: String, geometry: H3Geometry,
    references: [H3StillReference], tokenizerURL: URL) throws -> Prepared {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      (1...9).contains(references.count) else {
      throw H3CheckpointError.invalid("Still Ref2VA needs a prompt and one to nine images.")
    }
    var qwenGrids: [H3QwenRequest.Grid] = []
    var pixels: [MLXArray] = []
    var specs: [H3ReferenceSpec] = []
    qwenGrids.reserveCapacity(references.count)
    pixels.reserveCapacity(references.count)
    specs.reserveCapacity(references.count)
    for reference in references {
      try Task.checkCancellation()
      try H3StillReferenceMedia.validateCanvas(width: reference.width,
        height: reference.height, pixelBudgetPercent: reference.pixelBudgetPercent)
      guard reference.rgb8.count == reference.width * reference.height * 3 else {
        throw H3CheckpointError.invalid("Still reference must be bounded, prepared RGB8 on a 32-pixel grid.")
      }
      let packed = try H3QwenImageProcessor.packRGB8(image: reference.rgb8,
        width: reference.width, height: reference.height)
      qwenGrids.append(packed.grid)
      pixels.append(packed.pixels)
      specs.append(.image(latentHeight: reference.height / 16,
        latentWidth: reference.width / 16))
    }
    let tokenizer = try H3QwenTokenizer(url: tokenizerURL)
    let qwenReferences = qwenGrids.map { H3QwenRequest.Reference.image(grid: $0) }
    let request = try H3QwenRequest.references(prompt: prompt,
      references: qwenReferences, tokenizer: tokenizer)
    let layout = try H3ReferenceLayout(geometry: geometry,
      textTags: request.tags, references: specs)
    let joined = concatenated(pixels, axis: 0).asType(.bfloat16)
    eval(joined)
    return Prepared(qwenRequest: request, qwenGrids: qwenGrids,
      qwenPixels: joined, layout: layout, references: references)
  }
}
