import Foundation
import MLX

/// A velocity prediction advances video and audio together. The predictor
/// returns every video row, including fixed visual references.
public protocol H3VelocityPredictor {
  var layout: H3PackedLayout { get }
  func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void) throws
    -> H3FinalLayer.Output
}

extension H3DiTState: H3VelocityPredictor {}

public enum H3AVSampler {
  public struct Result {
    public let video: MLXArray
    public let audio: MLXArray
  }

  public static func run<P: H3VelocityPredictor>(predictor: P,
    videoSchedule: H3Schedule, audioSchedule: H3Schedule,
    rowSchedule: H3RowSchedule,
    videoLatents: MLXArray, audioLatents: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in },
    blockProgress: (Int, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let layout = predictor.layout
    let conditionRows = layout.conditionVideoRows
    let targetVideoRows = layout.tags.count - layout.videoStart
    let videoRows = conditionRows + targetVideoRows
    let audioRows = layout.videoStart - layout.audioStart
    let steps = videoSchedule.timesteps.count
    guard steps > 0, audioSchedule.timesteps.count == steps,
      rowSchedule.indicesByStep.count == steps,
      (1...128).contains(rowSchedule.table.count),
      videoLatents.shape == [1, videoRows, 96],
      audioLatents.shape == [1, audioRows, 32],
      videoLatents.dtype == .float32,
      audioLatents.dtype == .float32,
      rowSchedule.indicesByStep.allSatisfy({ $0.count == layout.tags.count
        && $0.allSatisfy({ (0..<rowSchedule.table.count).contains(Int($0)) }) }) else {
      throw H3CheckpointError.invalid("Invalid synchronized H3 sampling inputs.")
    }
    var video = videoLatents
    var audio = audioLatents
    for index in 0..<steps {
      try Task.checkCancellation()
      let velocity = try predictor.predict(videoLatents: video,
        audioLatents: audio,
        timestepIndices: rowSchedule.indicesByStep[index],
        progress: { completed, total in
          blockProgress(index + 1, completed, total)
        })
      guard velocity.video.shape == video.shape,
        velocity.audio.shape == audio.shape,
        velocity.video.dtype.isFloatingPoint,
        velocity.audio.dtype.isFloatingPoint else {
        throw H3CheckpointError.invalid("H3 predictor returned mismatched AV velocity rows.")
      }
      let advancedVideo = try videoSchedule.advance(
        sample: video[0..<1, conditionRows..<videoRows, 0..<96],
        velocity: velocity.video[0..<1, conditionRows..<videoRows, 0..<96]
          .asType(.float32), index: index)
      video = conditionRows > 0
        ? concatenated([video[0..<1, 0..<conditionRows, 0..<96],
            advancedVideo], axis: 1)
        : advancedVideo
      audio = try audioSchedule.advance(sample: audio,
        velocity: velocity.audio.asType(.float32), index: index)
      eval(video, audio)
      progress(index + 1, steps)
      try Task.checkCancellation()
    }
    return Result(video: video, audio: audio)
  }
}
