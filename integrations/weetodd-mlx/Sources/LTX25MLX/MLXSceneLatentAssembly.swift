import Foundation
import MLX
import LTX25Engine

/// Joins sampled scene windows while they are still audiovisual latents.
/// Later windows' first causal video token is dropped before overlap blending;
/// audio overlap is trimmed on the same delivered timeline for a single decode.
enum MLXSceneLatentAssembly {
  struct Output {
    let video: MLXArray
    let audio: MLXArray
  }

  static func assemble(video: [MLXArray], audio: [MLXArray],
    plan: LTX25ScenePlan, latentHeight: Int, latentWidth: Int) throws -> Output {
    guard video.count == plan.windowFrames.count,
      audio.count == plan.windowFrames.count,
      (1...128).contains(latentHeight), (1...128).contains(latentWidth) else {
      throw LTXError.invalid("LTX scene windows or spatial layout do not match the plan.")
    }
    let pixels = latentHeight * latentWidth
    for index in video.indices {
      let frames = (plan.windowFrames[index] - 1) / 8 + 1
      guard video[index].dtype == .float32,
        video[index].shape == [frames * pixels, 128],
        audio[index].dtype == .float32,
        audio[index].shape == [plan.windowAudioTokens[index], 128] else {
        throw LTXError.invalid("LTX scene window \(index + 1) latent shape differs from the plan.")
      }
    }
    let blendFrames = plan.videoOverlapLatentFrames - 1
    guard blendFrames >= 1 else {
      throw LTXError.invalid("LTX scene needs at least two video overlap latents.")
    }
    var current = video[0].reshaped([
      (plan.windowFrames[0] - 1) / 8 + 1, latentHeight, latentWidth, 128])
    var audioParts = [audio[0]]
    for index in 1..<video.count {
      let nextFrames = (plan.windowFrames[index] - 1) / 8 + 1
      let next = video[index].reshaped([nextFrames, latentHeight, latentWidth, 128])
      let trim = plan.joinAudioTokens[index - 1]
      guard nextFrames > plan.videoOverlapLatentFrames,
        audio[index].shape[0] > trim else {
        throw LTXError.invalid("LTX scene window is shorter than its audiovisual overlap.")
      }
      let following = next[1..<nextFrames]
      let alphas = (0..<blendFrames).map {
        Float($0) / Float(max(1, blendFrames - 1))
      }
      let alpha = MLXArray(alphas, [blendFrames, 1, 1, 1])
      let previousCount = current.shape[0]
      let blended = current[(previousCount - blendFrames)..<previousCount] * (1 - alpha)
        + following[0..<blendFrames] * alpha
      current = concatenated([
        current[0..<(previousCount - blendFrames)], blended,
        following[blendFrames..<following.shape[0]],
      ], axis: 0)
      audioParts.append(audio[index][trim..<audio[index].shape[0]])
      try Task.checkCancellation()
    }
    let assembledAudio = concatenated(audioParts, axis: 0)
    let expectedFrames = (plan.totalFrames - 1) / 8 + 1
    guard current.shape[0] == expectedFrames,
      assembledAudio.shape == [plan.expectedAudioTokens, 128] else {
      throw LTXError.invalid("LTX scene assembled audiovisual length differs from its plan.")
    }
    let assembledVideo = current.reshaped([expectedFrames * pixels, 128])
    eval(assembledVideo, assembledAudio)
    return Output(video: assembledVideo, audio: assembledAudio)
  }
}
