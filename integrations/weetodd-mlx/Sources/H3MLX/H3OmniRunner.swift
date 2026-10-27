import Foundation
import MLX

/// Omni orchestration plugs into the shared reference preparation, weighted DiT
/// and output decoder. This strategy only owns target PDD state transitions.
enum H3OmniRunner {
  struct Context {
    let source: H3OmniCheckpoint
    let plan: H3PDDPlan
    let attention: H3OmniVSAPlan
  }

  static func attention(references: [H3Ref2VAReference], geometry: H3Geometry,
    textRows: Int, layout: H3ReferenceLayout) throws -> H3OmniVSAPlan {
    var segments: [H3OmniSegment] = [.dense(role:.text, rows:textRows)]
    func audio(_ value: H3AudioReference) { segments.append(.dense(role:.referenceAudio, rows:2 * ((value.frames + 799) / 800))) }
    for reference in references {
      switch reference {
      case .image(let image): segments.append(.dense(role:.image, rows:image.width * image.height / 1024))
      case .audio(let value): audio(value)
      case .video(let video):
        if let value = video.audio { audio(value) }
        segments.append(.referenceVideo(grid:[(video.persistentFrameCount - 5) / 17 * 5 + 2, video.height / 32, video.width / 32]))
      default: throw H3CheckpointError.invalid("Omni rejects timed reference placement.")
      }
    }
    segments.append(.dense(role:.targetAudio, rows:layout.targetAudioIndices.count))
    let plan = try H3OmniVSAPlan(prefixSegments:segments,
      targetVideoGrid:[geometry.videoLatentFrames, geometry.height / 32, geometry.width / 32])
    guard plan.rows == layout.tags.count else { throw H3CheckpointError.invalid("Omni media roles differ from packed spans.") }
    return plan
  }

  static func sample(layout: H3ReferenceLayout, plan: H3PDDPlan,
    rowSchedule: H3ReferenceRowSchedule, videoNoise: MLXArray, audioNoise: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in },
    predict: (Int, MLXArray, MLXArray, [Int32]) throws -> H3FinalLayer.Output) throws -> H3ReferenceSampler.Result {
    guard videoNoise.shape == [1,layout.videoIndices.count,96], audioNoise.shape == [1,layout.audioIndices.count,32],
      videoNoise.dtype == .float32, audioNoise.dtype == .float32,
      rowSchedule.indicesByStep.count == 8 else { throw H3CheckpointError.invalid("Invalid Omni synchronized sampling state.") }
    func replace(_ original: MLXArray, _ target: MLXArray, _ prefix: Int, _ width: Int) -> MLXArray {
      prefix == 0 ? target : concatenated([original[0..<1,0..<prefix,0..<width],target],axis:1)
    }
    let cv = layout.conditionVideoIndices.count, ca = layout.conditionAudioIndices.count
    let vr = layout.videoIndices.count, ar = layout.audioIndices.count
    var video = replace(videoNoise, videoNoise[0..<1,cv..<vr,0..<96] * Float(plan.coarseSigmas(for:.video)[0]),cv,96)
    var audio = replace(audioNoise, audioNoise[0..<1,ca..<ar,0..<32] * Float(plan.coarseSigmas(for:.audio)[0]),ca,32)
    for step in 0..<plan.evaluationCount {
      try Task.checkCancellation()
      let mean = try predict(step,video,audio,rowSchedule.indicesByStep[step])
      guard mean.video.shape == video.shape, mean.audio.shape == audio.shape else { throw H3CheckpointError.invalid("Omni PDD head shape changed.") }
      video = replace(video,try plan.advance(sample:video[0..<1,cv..<vr,0..<96],mean:mean.video[0..<1,cv..<vr,0..<96],step:step,modality:.video),cv,96)
      audio = replace(audio,try plan.advance(sample:audio[0..<1,ca..<ar,0..<32],mean:mean.audio[0..<1,ca..<ar,0..<32],step:step,modality:.audio),ca,32)
      eval(video,audio)
      progress(step+1,plan.evaluationCount)
      try Task.checkCancellation()
    }
    return H3ReferenceSampler.Result(video:video,audio:audio)
  }
}
