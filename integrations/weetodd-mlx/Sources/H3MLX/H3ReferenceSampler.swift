import Foundation
import MLX

/// A Ref2VA predictor returns velocities for all condition and target rows.
/// The sampler must advance only target rows, preserving encoded references.
public protocol H3ReferenceVelocityPredictor {
  var layout: H3ReferenceLayout { get }
  func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void) throws
    -> H3FinalLayer.Output
}

public enum H3ReferenceSampler {
  public struct Result {
    public let video: MLXArray
    public let audio: MLXArray
  }

  public static func run<P: H3ReferenceVelocityPredictor>(predictor: P,
    videoSchedule: H3Schedule, audioSchedule: H3Schedule,
    rowSchedule: H3ReferenceRowSchedule,
    videoLatents: MLXArray, audioLatents: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in },
    blockProgress: (Int, Int, Int) -> Void = { _, _, _ in }) throws -> Result {
    let layout = predictor.layout
    let conditionVideo = layout.conditionVideoIndices.count
    let conditionAudio = layout.conditionAudioIndices.count
    let videoRows = layout.videoIndices.count
    let audioRows = layout.audioIndices.count
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
      throw H3CheckpointError.invalid("Invalid Ref2VA synchronized sampling inputs.")
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
        throw H3CheckpointError.invalid("Ref2VA predictor returned mismatched AV velocity rows.")
      }
      let advancedVideo = try videoSchedule.advance(
        sample: video[0..<1, conditionVideo..<videoRows, 0..<96],
        velocity: velocity.video[0..<1, conditionVideo..<videoRows, 0..<96]
          .asType(.float32), index: index)
      let advancedAudio = try audioSchedule.advance(
        sample: audio[0..<1, conditionAudio..<audioRows, 0..<32],
        velocity: velocity.audio[0..<1, conditionAudio..<audioRows, 0..<32]
          .asType(.float32), index: index)
      video = conditionVideo > 0
        ? concatenated([video[0..<1, 0..<conditionVideo, 0..<96], advancedVideo], axis: 1)
        : advancedVideo
      audio = conditionAudio > 0
        ? concatenated([audio[0..<1, 0..<conditionAudio, 0..<32], advancedAudio], axis: 1)
        : advancedAudio
      eval(video, audio)
      progress(index + 1, steps)
      try Task.checkCancellation()
    }
    return Result(video: video, audio: audio)
  }
}
