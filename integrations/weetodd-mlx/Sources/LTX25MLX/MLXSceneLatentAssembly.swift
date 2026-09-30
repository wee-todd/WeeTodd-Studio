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

  static func assembleVideoGroup(video: [MLXArray], plan: LTX25ScenePlan,
    windows: Range<Int>, latentHeight: Int, latentWidth: Int) throws -> MLXArray {
    guard video.count == plan.windowFrames.count,
      windows.lowerBound >= 0, windows.upperBound <= video.count,
      !windows.isEmpty, (1...128).contains(latentHeight),
      (1...128).contains(latentWidth) else {
      throw LTXError.invalid("LTX scene video group does not match its frame plan.")
    }
    let pixels = latentHeight * latentWidth
    for index in windows {
      let latentFrames = (plan.windowFrames[index] - 1) / 8 + 1
      guard video[index].dtype == .float32,
        video[index].shape == [latentFrames * pixels, 128] else {
        throw LTXError.invalid("LTX scene video group has an invalid sampled window.")
      }
    }
    let first = windows.lowerBound
    var current = video[first].reshaped([
      (plan.windowFrames[first] - 1) / 8 + 1, latentHeight, latentWidth, 128])
    let blendFrames = plan.videoOverlapLatentFrames - 1
    guard blendFrames >= 1 else {
      throw LTXError.invalid("LTX scene video group has no causal overlap.")
    }
    for index in windows.dropFirst() {
      let nextFrames = (plan.windowFrames[index] - 1) / 8 + 1
      guard nextFrames > plan.videoOverlapLatentFrames else {
        throw LTXError.invalid("LTX scene video group window is shorter than its overlap.")
      }
      let next = video[index].reshaped([nextFrames, latentHeight, latentWidth, 128])
      let following = next[1..<nextFrames]
      let alpha = MLXArray((0..<blendFrames).map {
        Float($0) / Float(max(1, blendFrames - 1))
      }, [blendFrames, 1, 1, 1])
      let count = current.shape[0]
      let blended = current[(count - blendFrames)..<count] * (1 - alpha)
        + following[0..<blendFrames] * alpha
      current = concatenated([current[0..<(count - blendFrames)], blended,
        following[blendFrames..<following.shape[0]]], axis: 0)
      try Task.checkCancellation()
    }
    let expectedFrames = plan.windowFrames[first] +
      windows.dropFirst().reduce(0) { $0 + plan.segmentFrames[$1] }
    let latentFrames = (expectedFrames - 1) / 8 + 1
    guard current.shape == [latentFrames, latentHeight, latentWidth, 128] else {
      throw LTXError.invalid("LTX scene video group did not cover its editorial frames.")
    }
    let result = current.reshaped([latentFrames * pixels, 128])
    eval(result)
    return result
  }

  static func assemble(video: [MLXArray], audio: [MLXArray],
    plan: LTX25ScenePlan, latentHeight: Int, latentWidth: Int,
    strictBoundaries: Set<Int> = []) throws -> Output {
    guard video.count == plan.windowFrames.count,
      audio.count == plan.windowFrames.count,
      strictBoundaries.allSatisfy({ (1..<video.count).contains($0) }),
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
      let alphas = (0..<blendFrames).map { frame in
        strictBoundaries.contains(index)
          ? (frame == blendFrames - 1 ? Float(1) : Float(0))
          : Float(frame) / Float(max(1, blendFrames - 1))
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
