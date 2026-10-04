import Foundation
import MLX
import LTX25Engine
import LTX25Video
import LTX25Audio

/// Samples native scene windows with the same released two-stage renderer used
/// for ordinary clips. Version-one boundary images retain their old placement;
/// version-two images use exact local frames and appended rows. Only compact
/// audiovisual tails cross window boundaries.
public enum MLXSceneSampler {
  public struct Result {
    public let video: MLXArray
    public let videoWindows: [MLXArray]
    public let audio: MLXArray
    public let geometry: AVGeometry
    public let windowSeconds: [Double]
    public let strictBoundaries: Set<Int>
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

  /// Exact scene-specific row admission. Ordinary base transports do not carry
  /// the appended image rows; the worker combines this with its stage ceiling.
  public static func estimatedTransformerActivationBytes(_ compiled:MLXStudioSceneRecipe.Compiled) throws -> Int {
    var maximum=0
    for index in compiled.requests.indices {
      try Task.checkCancellation()
      let request=compiled.requests[index],recipe=try request.recipe()
      let hasImages = !(try compiled.references(in:index)).isEmpty
      for geometry in [recipe.low,recipe.high] {
        let history=try index == 0 ? nil : MLXExtensionGuideLayout(geometry:geometry,
          contextFrames:compiled.plan.overlapFrames,videoGuideLatentFrames:compiled.plan.videoOverlapLatentFrames-1,
          audioGuideTokens:compiled.plan.joinAudioTokens[index-1])
        let scene=try compiled.imageRouting.map { try MLXSceneKeyframeLayout(geometry:geometry,
          anchors:$0.windows[index].map(\.layoutAnchor),extensionGuide:history) }
        let configuration=try AVBlockConfiguration(videoTokens:scene?.videoTokens ?? history?.videoTokens ?? geometry.videoTokens,
          audioTokens:scene?.audioTokens ?? history?.audioTokens ?? geometry.audioFrames,textTokens:1024)
        maximum=max(maximum,try MLXAVBlock.estimatedActivationBytes(configuration:configuration,
          perTokenVideo:history != nil || hasImages,perTokenAudio:history != nil))
      }
    }
    return maximum+(compiled.imageRouting?.retainedConditioningBytes ?? 0)
  }

  public static func preflight(_ compiled: MLXStudioSceneRecipe.Compiled,
    maximumActivationBytes: Int,
    decodePlan: MLXSceneDecodeWindowPlan) throws -> AVGeometry {
    let plan = compiled.plan, requests = compiled.requests
    guard requests.count == plan.windowFrames.count, let first = requests.first else {
      throw LTXError.invalid("Swift LTX scene requests do not match their frame plan.")
    }
    let geometry = try AVGeometry(width: first.width, height: first.height,
      frames: plan.totalFrames, fps: plan.fps)
    let windowImages=try requests.indices.map { try compiled.references(in:$0) }
    guard try estimatedTransformerActivationBytes(compiled)<=maximumActivationBytes else {
      throw LTXError.invalid("Scene image and history rows exceed the admitted transformer activation budget.")
    }
    let selected=try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:first.videoCheckpoint),settings:first.diffusionVAE)
    if selected.isDiffusion {
      guard decodePlan.overlapFrames == 0,decodePlan.latentRanges == [0..<geometry.latentFrames],compiled.decodeMode.maximumWindowFrames == nil else {
        throw LTXError.invalid("DiffVAE cannot reuse convolutional scene temporal overlap windows.")
      }
      _ = try MLXDiffusionScenePublication.admit(geometry:geometry,plan:plan,strictBoundaries:compiled.strictBoundaries,
        checkpoint:URL(fileURLWithPath:first.videoCheckpoint),settings:first.diffusionVAE,maximumWorkspaceBytes:decodePlan.admittedActivationBytes)
    } else {
    try MLXSceneMediaPublisher.admit(geometry: geometry,
      decodePlan: decodePlan)
    if !compiled.strictBoundaries.isEmpty {
      for route in try MLXSceneMediaPublisher.strictFrameRoutes(plan: plan,
        strictBoundaries: compiled.strictBoundaries) {
        let group = try AVGeometry(width: geometry.width, height: geometry.height,
          frames: route.frames, fps: geometry.fps)
        _ = try MLXSceneDecodeWindowPlan(geometry: group,
          maximumActivationBytes: decodePlan.admittedActivationBytes,
          maximumWindowFrames: compiled.decodeMode.maximumWindowFrames)
      }
    }
    }
    let text = try MLXTextEncoder(gemmaRoot: URL(fileURLWithPath: first.gemmaRoot),
      connectorURL: URL(fileURLWithPath: first.connectorCheckpoint))
    _ = try MLXVideoDecoderSelection(checkpoint:URL(fileURLWithPath:first.videoCheckpoint),settings:first.diffusionVAE)
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
        request.diffusionVAE == first.diffusionVAE,
        request.audioCheckpoint == first.audioCheckpoint,
        request.spatialUpscalerCheckpoint == first.spatialUpscalerCheckpoint,
        request.task == (request.audioReference != nil ? "a2v" :
          (request.referenceImages.isEmpty ? "t2v" : "i2v")),
        request.referenceImages.count <= 1,
        request.noisePolicy == .releasedMLX else {
        throw LTXError.invalid("Swift LTX scene windows need identical components and admitted geometry.")
      }
      _ = try MLXTextEncodingPlan(promptTokens: text.tokenize(request.prompt).count)
      if !windowImages[index].isEmpty {
        for image in windowImages[index] { try MLXReferenceImage.inspect(URL(fileURLWithPath:image.path)) }
        for g in [recipe.low, recipe.high] {
          _ = try MLXImageEncodePlan(width: g.width, height: g.height)
        }
        _ = try MLXImageEncoder(checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
      }
      if let source = request.audioReference {
        _ = try MLXSourceAudioInterval(source: URL(fileURLWithPath: source.path),
          sourceStartSeconds: source.sourceStartSeconds,
          sourceDurationSeconds: source.sourceDurationSeconds,
          durationSeconds: Double(request.frames) / request.fps)
        _ = try MLXAudioEncoder(checkpoint: URL(fileURLWithPath: request.audioCheckpoint))
      }
      _ = try MLXDistilledSamplingRunner(recipe: recipe,
        transformerRoot: URL(fileURLWithPath: request.transformerRoot),
        upscalerCheckpoint: URL(fileURLWithPath: request.spatialUpscalerCheckpoint),
        statisticsCheckpoint: URL(fileURLWithPath: request.videoCheckpoint),
        firstStrength: request.referenceImages.first?.strength,
        firstFrame: index == 0 ? 0 : plan.overlapFrames - 1,
        sceneAnchors:compiled.imageRouting?.windows[index].map(\.layoutAnchor),
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
    ffmpeg: URL,
    preparedAudio: [MLXSourceAudioInterval.Prepared?] = [],
    maximumActivationBytes: Int,
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> Result {
    let plan = compiled.plan, requests = compiled.requests
    guard requests.count == plan.windowFrames.count, let first = requests.first else {
      throw LTXError.invalid("Swift LTX scene requests do not match their frame plan.")
    }
    guard preparedAudio.isEmpty || preparedAudio.count == requests.count,
      requests.indices.allSatisfy({ index in
        (requests[index].audioReference != nil) ==
          (!preparedAudio.isEmpty && preparedAudio[index] != nil)
      }) else {
      throw LTXError.invalid("Swift LTX scene audio intervals must be prepared for every driven window.")
    }
    let geometry = try AVGeometry(width: first.width, height: first.height,
      frames: plan.totalFrames, fps: plan.fps)
    let imageDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("weetodd-scene-image-" + UUID().uuidString)
    let windowImages=try requests.indices.map { try compiled.references(in:$0) }
    if windowImages.contains(where: { !$0.isEmpty }) {
      try FileManager.default.createDirectory(at: imageDirectory,
        withIntermediateDirectories: false)
    }
    defer { try? FileManager.default.removeItem(at: imageDirectory) }
    var preparedImages: [[[MLXReferenceImage.Prepared]]] = []
    for (window, request) in requests.enumerated() {
      let recipe = try request.recipe()
      let stages = try windowImages[window].isEmpty && compiled.imageRouting == nil ? [] :
        MLXReferenceImage.prepareStages(windowImages[window],
          sizes: [(recipe.low.width, recipe.low.height),
            (recipe.high.width, recipe.high.height)],
          ffmpeg: ffmpeg, directory: imageDirectory) { stage, role in
            try progress("scene_reference_prepare:\(window + 1):\(stage + 1):" + role,
              stage + 1, 2)
          }
      preparedImages.append(stages)
    }
    let audioDrivers: [MLXArray?] = try autoreleasepool {
      guard !preparedAudio.isEmpty else { return Array(repeating: nil, count: requests.count) }
      let encoder = try MLXAudioEncoder(checkpoint: URL(fileURLWithPath: first.audioCheckpoint))
      var drivers: [MLXArray?] = []
      for (index, source) in preparedAudio.enumerated() {
        guard let source else { drivers.append(nil); continue }
        try Task.checkCancellation()
        let mel = try MLXAudioMel.encode(wav: source.conditioning)
        let raw = try encoder.encode(mel: mel) { completed, total in
          try progress("scene_audio_encode:\(index + 1)", completed, total)
        }
        let count = try requests[index].recipe().high.audioFrames
        let fitted = raw.shape[0] < count
          ? concatenated([raw, MLXArray.zeros([count - raw.shape[0],128])],axis:0)
          : raw[0..<count]
        eval(fitted)
        drivers.append(fitted)
      }
      return drivers
    }
    Stream.gpu.synchronize(); Memory.clearCache()
    if audioDrivers.contains(where: { $0 != nil }) {
      try progress("scene_audio_encoder_weights_released", 1, 1)
    }
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
      guard request.task == (request.audioReference != nil ? "a2v" :
          (request.referenceImages.isEmpty ? "t2v" : "i2v")),
        request.noisePolicy == .releasedMLX,
        request.width == geometry.width, request.height == geometry.height,
        request.fps == geometry.fps, request.frames == plan.windowFrames[index] else {
        throw LTXError.invalid("Swift LTX scene window changed its validated conditioning or geometry.")
      }
      let sceneReferences:[[MLXArray]] = try autoreleasepool {
        guard compiled.imageRouting != nil else { return [] }
        guard !windowImages[index].isEmpty else { return [[],[]] }
        let encoder=try MLXImageEncoder(checkpoint:URL(fileURLWithPath:request.videoCheckpoint))
        return try preparedImages[index].enumerated().map { stage,images in
          try images.enumerated().map { ordinal,image in
            try Task.checkCancellation()
            return try encoder.encode(MLXArray(try image.pixels(),[1,image.height,image.width,3])) { completed in
              try progress("scene_reference_encode:\(index+1):\(stage+1):\(ordinal+1)",completed,42)
            }
          }
        }
      }
      let references: [(first: MLXArray, last: MLXArray?)] = try autoreleasepool {
        guard compiled.imageRouting == nil,!preparedImages[index].isEmpty else { return [] }
        let encoder = try MLXImageEncoder(
          checkpoint: URL(fileURLWithPath: request.videoCheckpoint))
        var stages: [(first: MLXArray, last: MLXArray?)] = []
        for (stage, images) in preparedImages[index].enumerated() {
          let image = images[0]
          let encoded = try encoder.encode(MLXArray(try image.pixels(),
            [1, image.height, image.width, 3])) { completed in
              try progress("scene_reference_encode:\(index + 1):\(stage + 1)", completed, 42)
            }
          stages.append((encoded, nil))
        }
        return stages
      }
      Stream.gpu.synchronize(); Memory.clearCache()
      if !references.isEmpty || sceneReferences.contains(where: { !$0.isEmpty }) {
        try progress("scene_reference_weights_released", index + 1, requests.count)
      }
      let sampler = try MLXDistilledSamplingRunner(recipe: recipe,
        transformerRoot: URL(fileURLWithPath: request.transformerRoot),
        upscalerCheckpoint: URL(fileURLWithPath: request.spatialUpscalerCheckpoint),
        statisticsCheckpoint: URL(fileURLWithPath: request.videoCheckpoint),
        firstStrength: request.referenceImages.first?.strength,
        firstFrame: index == 0 ? 0 : plan.overlapFrames - 1,
        sceneAnchors:compiled.imageRouting?.windows[index].map(\.layoutAnchor),
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
        sceneReferences:sceneReferences,references: references,
        frozenAudio: audioDrivers[index],
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
      latentHeight: geometry.latentHeight, latentWidth: geometry.latentWidth,
      strictBoundaries: compiled.strictBoundaries)
    return Result(video: joined.video, videoWindows: videoWindows,
      audio: joined.audio,
      geometry: geometry, windowSeconds: seconds,
      strictBoundaries: compiled.strictBoundaries)
  }
}
