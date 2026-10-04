import Foundation
import MLX

/// One H3 denoiser input in the released text, reference, audio, video order.
/// Row selection for the two velocity heads retains every reference video row.
public struct H3PackedSequence {
  public let embeddings: MLXArray
  public let positions: MLXArray
  public let timestepIndices: MLXArray
  public let modulationIndices: MLXArray
  public let videoIndices: MLXArray
  public let audioIndices: MLXArray

  public init(layout: H3PackedLayout, text: MLXArray,
    video: MLXArray, audio: MLXArray,
    timestepIndices: [Int32]) throws {
    let total = layout.tags.count
    let textRows = layout.audioStart - layout.conditionVideoRows
    let audioRows = layout.videoStart - layout.audioStart
    let targetVideoRows = total - layout.videoStart
    let videoRows = layout.conditionVideoRows + targetVideoRows
    guard (1...layout.maximumPackedRows).contains(total),
      textRows > 0, audioRows > 0, targetVideoRows > 0,
      text.shape == [1, textRows, 5376],
      video.shape == [1, videoRows, 5376],
      audio.shape == [1, audioRows, 5376],
      text.dtype == .bfloat16, video.dtype == .bfloat16,
      audio.dtype == .bfloat16,
      layout.positions.count == total,
      timestepIndices.count == total,
      timestepIndices.allSatisfy({ (0..<128).contains(Int($0)) }),
      layout.positions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
      throw H3CheckpointError.invalid("Invalid H3 packed audiovisual sequence.")
    }
    let condition = video[0..<1, 0..<layout.conditionVideoRows, 0..<5376]
    let target = video[0..<1, layout.conditionVideoRows..<videoRows, 0..<5376]
    embeddings = concatenated([text, condition, audio, target], axis: 1)
    positions = MLXArray(layout.positions.flatMap { [$0.x, $0.y, $0.z] }, [total, 3])
    self.timestepIndices = MLXArray(timestepIndices)
    modulationIndices = MLXArray(zip(timestepIndices, layout.tags).map { timestep, tag in
      timestep * 3 + max(0, tag)
    })
    videoIndices = MLXArray(
      (textRows..<layout.audioStart).map(Int32.init)
        + (layout.videoStart..<total).map(Int32.init))
    audioIndices = MLXArray((layout.audioStart..<layout.videoStart).map(Int32.init))
    eval(embeddings, positions, self.timestepIndices,
      modulationIndices, videoIndices, audioIndices)
  }

  /// Ref2VA stores reference blocks in presentation order, while the two
  /// projected latent tensors remain grouped by modality. One gather maps
  /// those tensors into the packed transformer sequence without host copies.
  public init(layout: H3ReferenceLayout, text: MLXArray,
    video: MLXArray, audio: MLXArray,
    timestepIndices: [Int32]) throws {
    let total = layout.tags.count
    let textRows = total - layout.videoIndices.count - layout.audioIndices.count
    guard (1...layout.maximumPackedRows).contains(total), textRows > 0,
      text.shape == [1, textRows, 5376],
      video.shape == [1, layout.videoIndices.count, 5376],
      audio.shape == [1, layout.audioIndices.count, 5376],
      text.dtype == .bfloat16, video.dtype == .bfloat16,
      audio.dtype == .bfloat16,
      layout.positions.count == total,
      timestepIndices.count == total,
      timestepIndices.allSatisfy({ (0..<128).contains(Int($0)) }),
      layout.positions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
      throw H3CheckpointError.invalid("Invalid Ref2VA packed audiovisual sequence.")
    }
    var permutation = [Int32](repeating: -1, count: total)
    for row in 0..<textRows { permutation[row] = Int32(row) }
    for (offset, row) in layout.videoIndices.enumerated() {
      permutation[row] = Int32(textRows + offset)
    }
    for (offset, row) in layout.audioIndices.enumerated() {
      permutation[row] = Int32(textRows + layout.videoIndices.count + offset)
    }
    guard permutation.allSatisfy({ $0 >= 0 }) else {
      throw H3CheckpointError.invalid("Ref2VA modality indices do not cover all packed rows.")
    }
    let grouped = concatenated([text, video, audio], axis: 1)
    embeddings = take(grouped, MLXArray(permutation), axis: 1)
    positions = MLXArray(layout.positions.flatMap { [$0.x, $0.y, $0.z] }, [total, 3])
    self.timestepIndices = MLXArray(timestepIndices)
    modulationIndices = MLXArray(zip(timestepIndices, layout.tags).map { timestep, tag in
      timestep * 3 + max(0, tag)
    })
    videoIndices = MLXArray(layout.videoIndices.map(Int32.init))
    audioIndices = MLXArray(layout.audioIndices.map(Int32.init))
    eval(embeddings, positions, self.timestepIndices,
      modulationIndices, videoIndices, audioIndices)
  }
}
