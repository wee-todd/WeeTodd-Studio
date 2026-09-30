import Foundation
import MLX
import LTX25Engine
import LTX25Video
import LTX25Audio

/// Samples native scene windows with the same released two-stage renderer used
/// for ordinary clips. Only compact audiovisual tails cross a window boundary.
public enum MLXSceneSampler {
  public struct Result {
    public let video: MLXArray
    public let audio: MLXArray
    public let geometry: AVGeometry
    public let windowSeconds: [Double]
  }

  private static func detachedTail(_ source: MLXArray,
    range: Range<Int>) -> MLXArray {
    let slice = source[range]
    return MLXArray(slice.asArray(Float.self), [range.count, 128])
  }

  static func interiorVideoTail(_ source: MLXArray,
    latentFrames: Int, pixels: Int, overlapFrames: Int) throws -> MLXArray {
    guard source.dtype == .float32,
      source.shape == [latentFrames * pixels, 128],
      pixels > 0, overlapFrames >= 2, latentFrames > overlapFrames else {
      throw LTXError.invalid("Swift LTX scene video history cannot fit its interior overlap.")
    }
    return detachedTail(source, range:
      ((latentFrames - overlapFrames) * pixels)..<((latentFrames - 1) * pixels))
  }

  static func audioTail(_ source: MLXArray, count: Int) throws -> MLXArray {
    guard source.dtype == .float32, source.shape.count == 2,
      source.shape[1] == 128, count > 0, count < source.shape[0] else {
      throw LTXError.invalid("Swift LTX scene audio history exceeds its sampled window.")
    }
    return detachedTail(source, range:
      (source.shape[0] - count)..<source.shape[0])
  }

  public static func preflight(_ compiled: MLXStudioSceneRecipe.Compiled,
    maximumActivationBytes: Int, videoActivationBytes: Int) throws -> AVGeometry {
    let plan = compiled.plan, requests = compiled.requests
    guard requests.count == plan.windowFrames.count, let first = requests.first else {
      throw LTXError.invalid("Swift LTX scene requests do not match their frame plan.")
    }
    let geometry = try AVGeometry(width: first.width, height: first.height,
      frames: plan.totalFrames, fps: plan.fps)
    try MLXSceneMediaPublisher.admit(geometry: geometry,
      videoActivationBytes: videoActivationBytes)
    let text = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: first.gemmaRoot),
      connectorURL: URL(fileURLWithPath: first.connectorCheckpoint))
    _ = try VideoDecoder(checkpoint: URL(fileURLWithPath: first.videoCheckpoint))
    try AudioDecoder.validateCheckpoint(URL(fileURLWithPath: first.audioCheckpoint))
    for index in requests.indices {
      let request = requests[index], recipe = try request.recipe()
      guard request.frames == plan.windowFrames[index],
        request.width == geometry.width, request.height == geometry.height,
        request.fps == geometry.fps,
        request.gemmaRoot == first.gemmaRoot,
        request.transformerRoot == first.transformerRoot,
        request.connectorCheckpoint == first.connectorCheckpoint,
        request.videoCheckpoint == first.videoCheckpoint,
        request.audioCheckpoint == first.audioCheckpoint,
        request.spatialUpscalerCheckpoint == first.spatialUpscalerCheckpoint,
        request.task == "t2v", request.referenceImages.isEmpty,
        request.audioReference == nil, request.noisePolicy == .releasedMLX else {
        throw LTXError.invalid("Swift LTX scene windows need identical components and admitted geometry.")
      }
      _ = try MLXTextEncodingPlan(promptTokens: text.tokenize(request.prompt).count)
      _ = try MLXDistilledSamplingRunner(recipe: recipe,
        transformerRoot: URL(fileURLWithPath: request.transformerRoot),
        upscalerCheckpoint: URL(fileURLWithPath: request.spatialUpscalerCheckpoint),
        statisticsCheckpoint: URL(fileURLWithPath: request.videoCheckpoint),
        extensionContextFrames: index == 0 ? nil : plan.overlapFrames,
        extensionVideoGuideLatentFrames: index == 0 ? nil : plan.videoOverlapLatentFrames - 1,
        extensionAudioGuideTokens: index == 0 ? nil : plan.joinAudioTokens[index - 1],
        stageOneLoras: request.stageOneLoras,
        stageTwoLoras: request.stageTwoLoras,
        noisePolicy: request.noisePolicy,
        maximumActivationBytes: maximumActivationBytes)
    }
    return geometry
  }

  public static func sample(_ compiled: MLXStudioSceneRecipe.Compiled,
    maximumActivationBytes: Int,
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> Result {
    let plan = compiled.plan, requests = compiled.requests
    guard requests.count == plan.windowFrames.count, let first = requests.first else {
      throw LTXError.invalid("Swift LTX scene requests do not match their frame plan.")
    }
    let geometry = try AVGeometry(width: first.width, height: first.height,
      frames: plan.totalFrames, fps: plan.fps)
    let contexts = try autoreleasepool { () -> [MLXTextEncoder.Output] in
      let encoder = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: first.gemmaRoot),
        connectorURL: URL(fileURLWithPath: first.connectorCheckpoint))
      var encoded: [MLXTextEncoder.Output] = []
      for index in requests.indices {
        try Task.checkCancellation()
        let result = try encoder.encode(prompt: requests[index].prompt) { event in
          try progress("scene_text_\(index + 1):" + event.stage,
            event.completed, event.total)
        }
        encoded.append(result)
      }
      return encoded
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    try progress("scene_text_weights_released", 1, 1)
    var videoWindows: [MLXArray] = [], audioWindows: [MLXArray] = []
    var prior: MLXDistilledSamplingRunner.ExtensionGuides?
    var seconds: [Double] = []
    for index in requests.indices {
      try Task.checkCancellation()
      let request = requests[index], recipe = try request.recipe()
      guard request.task == "t2v", request.referenceImages.isEmpty,
        request.audioReference == nil, request.noisePolicy == .releasedMLX,
        request.width == geometry.width, request.height == geometry.height,
        request.fps == geometry.fps, request.frames == plan.windowFrames[index] else {
        throw LTXError.invalid("Swift LTX scene window changed its validated conditioning or geometry.")
      }
      let sampler = try MLXDistilledSamplingRunner(recipe: recipe,
        transformerRoot: URL(fileURLWithPath: request.transformerRoot),
        upscalerCheckpoint: URL(fileURLWithPath: request.spatialUpscalerCheckpoint),
        statisticsCheckpoint: URL(fileURLWithPath: request.videoCheckpoint),
        extensionContextFrames: index == 0 ? nil : plan.overlapFrames,
        extensionVideoGuideLatentFrames: index == 0 ? nil : plan.videoOverlapLatentFrames - 1,
        extensionAudioGuideTokens: index == 0 ? nil : plan.joinAudioTokens[index - 1],
        stageOneLoras: request.stageOneLoras,
        stageTwoLoras: request.stageTwoLoras,
        noisePolicy: request.noisePolicy,
        maximumActivationBytes: maximumActivationBytes)
      let started = Date()
      var lowTail: MLXArray?
      let sampled = try sampler.evaluateWithStageOneCapture(
        videoContext: contexts[index].video,
        audioContext: contexts[index].audio,
        extensionGuides: prior,
        stageOneVideoObserver: { low in
          guard index < requests.count - 1 else { return }
          lowTail = try interiorVideoTail(low,
            latentFrames: recipe.low.latentFrames,
            pixels: recipe.low.latentHeight * recipe.low.latentWidth,
            overlapFrames: plan.videoOverlapLatentFrames)
        },
        progress: { stage, completed, total in
          try progress("scene_window_\(index + 1):" + stage, completed, total)
        })
      let video = sampled["video"]!, audio = sampled["audio"]!
      seconds.append(Date().timeIntervalSince(started))
      videoWindows.append(video); audioWindows.append(audio)
      if index < requests.count - 1 {
        let highTail = try interiorVideoTail(video,
          latentFrames: recipe.high.latentFrames,
          pixels: recipe.high.latentHeight * recipe.high.latentWidth,
          overlapFrames: plan.videoOverlapLatentFrames)
        let audioCount = plan.joinAudioTokens[index]
        let audioTail = try Self.audioTail(audio, count: audioCount)
        guard let lowTail else {
          throw LTXError.invalid("Swift LTX scene lost its first-stage continuation tail.")
        }
        prior = .init(stageOneVideo: lowTail, stageTwoVideo: highTail,
          audio: audioTail)
      } else { prior = nil }
      Stream.gpu.synchronize(); Memory.clearCache()
      try progress("scene_window_released", index + 1, requests.count)
    }
    let joined = try MLXSceneLatentAssembly.assemble(video: videoWindows,
      audio: audioWindows, plan: plan,
      latentHeight: geometry.latentHeight, latentWidth: geometry.latentWidth)
    return Result(video: joined.video, audio: joined.audio,
      geometry: geometry, windowSeconds: seconds)
  }
}
