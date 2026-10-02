import Foundation
import MLX
import LTX25Engine
import AdapterRuntime

/// Spatial DFR uses the shared streamed denoiser for both stages. The low-pass
/// generated keyframes are carried into stage two; stage-one video is also
/// appended as a clean half-resolution reference. Audio comes from stage one.
public final class MLXDFRSamplingRunner {
  private let recipe: DistilledTwoStageRecipe
  private let low: MLXDFRLayout
  private let high: MLXDFRLayout
  private let weights: [MLXDenoiserWeights]
  private let upscaler: MLXLatentUpscaler
  private let temporalUpscaler: MLXTemporalUpscaler?
  private let temporalRounds: Int
  private let requestedFrames: Int
  private let transformerRoot: URL
  private let firstStrength: Float?
  private let lastStrength: Float?
  private let maximumActivationBytes: Int
  private let requiresEndpoints:Bool
  private let gate = NSLock()
  public private(set) var stageSeconds: [String: Double] = [:]

  public init(recipe: DistilledTwoStageRecipe, transformerRoot: URL,
    upscalerCheckpoint: URL, statisticsCheckpoint: URL,
    slotFrames: [Int], detailingAdapter: LoRAAdapter,
    firstStrength:Float?=nil,lastStrength:Float?=nil,
    temporalUpscalerCheckpoint: URL?=nil,temporalRounds:Int=0,requestedFrames:Int?=nil,
    maximumActivationBytes: Int = 2 * 1024 * 1024 * 1024) throws {
    self.recipe = recipe
    self.maximumActivationBytes = maximumActivationBytes
    self.transformerRoot = transformerRoot
    self.temporalRounds = temporalRounds
    self.requestedFrames = requestedFrames ?? recipe.high.frames
    self.firstStrength = firstStrength
    self.lastStrength = lastStrength
    guard (0...2).contains(temporalRounds),
      (temporalRounds == 0) == (temporalUpscalerCheckpoint == nil),
      self.requestedFrames <= recipe.high.frames,
      self.requestedFrames > 1,
      (self.requestedFrames - 1) % 8 == 0 else {
      throw LTXError.invalid("DFR temporal rounds, checkpoint and requested canvas differ.")
    }
    _ = try MLXDFRTemporalPlan.outputFrames(inputFrames:recipe.high.frames,rounds:temporalRounds)
    guard recipe.high.fps * Double(1 << temporalRounds) <= 120 else {
      throw LTXError.invalid("DFR temporal output FPS exceeds 120.")
    }
    temporalUpscaler = try temporalUpscalerCheckpoint.map {
      try MLXTemporalUpscaler(checkpoint:$0,statisticsCheckpoint:statisticsCheckpoint)
    }
    if temporalRounds > 0 {
      var count=recipe.high.latentFrames
      for _ in 0..<temporalRounds {
        _ = try MLXTemporalUpscaler.admit(shape:[count,recipe.high.latentHeight,
          recipe.high.latentWidth,128],maximumActivationBytes:maximumActivationBytes)
        count=2*count-1
      }
    }
    self.requiresEndpoints=firstStrength != nil
    low = try MLXDFRLayout(geometry: recipe.low, slotFrames: slotFrames,
      firstStrength:firstStrength,lastStrength:lastStrength,
      lastFrame:lastStrength == nil ? nil : self.requestedFrames-1)
    high = try MLXDFRLayout(geometry: recipe.high, slotFrames: slotFrames, reference: recipe.low,
      firstStrength:firstStrength,lastStrength:lastStrength,
      lastFrame:lastStrength == nil ? nil : self.requestedFrames-1)
    for layout in [low, high] {
      let block = try MLXAVBlock(configuration: Self.configuration(layout),
        maximumActivationBytes: maximumActivationBytes)
      if layout.referenceTokens > 0 || firstStrength != nil { try block.admitPerTokenVideo() }
    }
    _ = try MLXLatentUpscaler.admit(shape: [recipe.low.latentFrames,
      recipe.low.latentHeight, recipe.low.latentWidth, 128],
      maximumActivationBytes: maximumActivationBytes)
    _ = try MLXLatentUpscaler.admit(shape: [slotFrames.count,
      recipe.low.latentHeight, recipe.low.latentWidth, 128],
      maximumActivationBytes: maximumActivationBytes)
    weights = try [(low, [LoRAAdapter]()), (high, [detailingAdapter])].map { layout, adapters in
      let source = try MLXDenoiserWeights(root: transformerRoot,
        configuration: Self.configuration(layout), adapters: adapters,
        maximumActivationBytes: maximumActivationBytes, requireKeyframeMarker: true,
        pixelSpatialDFRAdapterPath:adapters.first?.path)
      guard source.sourceCheckpoint == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
        throw LTXError.invalid("DFR requires the released distilled transformer checkpoint.")
      }
      return source
    }
    upscaler = try MLXLatentUpscaler(checkpoint: upscalerCheckpoint,
      statisticsCheckpoint: statisticsCheckpoint)
  }

  private func runTemporal(video original: MLXArray, slots originalSlots: MLXArray,
    audio: MLXArray, videoContext: MLXArray, audioContext: MLXArray,
    highReference: (first: MLXArray, last: MLXArray?)?,
    progress: @escaping (String, Int, Int) throws -> Void) throws -> MLXArray {
    guard let temporalUpscaler else { throw LTXError.invalid("DFR temporal upscaler is missing.") }
    let frameTokens = recipe.high.latentHeight * recipe.high.latentWidth
    var carry: [Int: MLXArray] = [:]
    for (index, frame) in high.slotFrames.enumerated() {
      carry[frame] = originalSlots[index*frameTokens..<(index+1)*frameTokens]
        .reshaped([frameTokens,128])
    }
    struct ExplicitAnchor {
      let latent: MLXArray
      let strength: Float
      let replace: Bool
    }
    var explicit: [Int: ExplicitAnchor] = [:]
    if let highReference, let firstStrength {
      explicit[0] = ExplicitAnchor(latent:highReference.first,strength:firstStrength,replace:true)
      if let last = highReference.last, let lastStrength {
        explicit[requestedFrames-1] = ExplicitAnchor(latent:last,strength:lastStrength,replace:false)
      }
    }
    var video = original, frames = recipe.high.frames, playbackFPS = recipe.high.fps
    let schedule = try SamplingSchedule(sigmas:[0.975,0.909375,0.725,0.421875,0],eta:0.5)
    // Stage one generated audio for the padded canvas. Publication trims it
    // later; temporal conditioning must retain that original wall clock.
    let sourceSeconds = Double(recipe.high.frames) / recipe.high.fps
    for round in 1...temporalRounds {
      let roundStart = Date()
      let upscaled = try temporalUpscaler.upscale(video.reshaped([frames/8+1,
        recipe.high.latentHeight,recipe.high.latentWidth,128]),
        maximumActivationBytes:maximumActivationBytes) {
          try progress("temporal_upscale_\(round):"+$0,0,1)
        }
      frames = 2*(frames-1)+1
      playbackFPS *= 2
      video = upscaled.reshaped([(frames/8+1)*frameTokens,128])
      eval(video)
      Memory.clearCache()
      try progress("temporal_upscaler_weights_released",round,temporalRounds)
      carry = Dictionary(uniqueKeysWithValues:carry.map { ($0.key*2,$0.value) })
      explicit = Dictionary(uniqueKeysWithValues:explicit.map { ($0.key*2,$0.value) })
      let seams = carry.keys.sorted()
      let tiles = try MLXDFRTemporalPlan.tiles(seams:seams,frames:frames,
        maximumTiles:1 << round)
      let conditionedFPS = try MLXDFRTemporalPlan.conditioningFPS(playbackFPS)
      var owned: [MLXArray] = [], newSlots: [Int: MLXArray] = [:]
      var planeAt = carry
      var previousTile: (baseCell: Int, latent: MLXArray)?
      for (tileIndex,tile) in tiles.enumerated() {
        try Task.checkCancellation()
        let geometry = try AVGeometry(width:recipe.high.width,height:recipe.high.height,
          frames:tile.frames,fps:conditionedFPS)
        let sourceCells = video[tile.latentStart*frameTokens..<tile.latentEnd*frameTokens]
        let base: MLXArray
        let pinned: MLXArray?
        if tile.dropLatentPrefix > 0 {
          guard let plane = planeAt[tile.pixelStart], let previousTile else {
            throw LTXError.invalid("Temporal DFR has no generated plane or previous tile for its seam.")
          }
          base = concatenated([plane,sourceCells],axis:0)
            .reshaped([geometry.videoTokens,128])
          let start = (tile.latentStart - previousTile.baseCell)*frameTokens
          let count = (tile.dropLatentPrefix-1)*frameTokens
          guard start >= 0, start+count <= previousTile.latent.shape[0] else {
            throw LTXError.invalid("Temporal DFR pinned seam extends beyond its prior tile.")
          }
          pinned = concatenated([plane,previousTile.latent[start..<start+count]],axis:0)
            .reshaped([tile.dropLatentPrefix*frameTokens,128])
        } else {
          base = sourceCells.reshaped([geometry.videoTokens,128])
          pinned = nil
        }
        var anchors: [MLXDFRTemporalLayout.Anchor] = []
        for frame in Set((tileIndex == 0 ? [0] : []) + tile.anchorFrames).sorted() {
          if let image = explicit[frame] {
            anchors.append(.init(frame:frame-tile.pixelStart,latent:image.latent,
              strength:image.strength,replace:image.replace && frame == tile.pixelStart))
          } else if frame > 0, let keyframe = carry[frame] {
            anchors.append(.init(frame:frame-tile.pixelStart,latent:keyframe,
              strength:0.95,replace:false))
          }
        }
        for (frame,image) in explicit where frame > tile.pixelStart && frame < tile.pixelEnd &&
          !anchors.contains(where: { $0.frame == frame-tile.pixelStart }) {
          anchors.append(.init(frame:frame-tile.pixelStart,latent:image.latent,
            strength:image.strength,replace:false))
        }
        anchors.sort { $0.frame < $1.frame }
        let localSlots=tile.slotFrames.filter { explicit[$0] == nil }.map { $0-tile.pixelStart }
        let slotInitials=localSlots.map { local -> MLXArray in
          let latent=min(max(Int((Double(local)/8).rounded(.toNearestOrEven)),0),geometry.latentFrames-1)
          return base[latent*frameTokens..<(latent+1)*frameTokens]
        }
        let seeded=concatenated(slotInitials,axis:0).reshaped([localSlots.count*frameTokens,128])
        let layout=try MLXDFRTemporalLayout(geometry:geometry,slots:localSlots,anchors:anchors)
        let prepared=try layout.prepare(generated:base,initialSlots:seeded,
          pinnedPrefix:pinned)
        let frozen=try MLXDFRFrozenAudio.tile(audio,pixelStart:tile.pixelStart,
          frames:tile.frames,playbackFPS:playbackFPS,sourceSeconds:sourceSeconds)
        let config=try AVBlockConfiguration(videoTokens:layout.videoTokens,
          audioTokens:frozen.latent.shape[0],textTokens:1024)
        let block=try MLXAVBlock(configuration:config,maximumActivationBytes:maximumActivationBytes)
        if !anchors.isEmpty || pinned != nil { try block.admitPerTokenVideo() }
        let source=try MLXDenoiserWeights(root:transformerRoot,configuration:config,
          adapters:[],maximumActivationBytes:maximumActivationBytes,
          requireKeyframeMarker:true)
        let runner=try MLXSamplingRunner(configuration:config,
          maximumActivationBytes:maximumActivationBytes,keyframeMarkerRows:layout.slotTokens)
        let seed=recipe.seed &+ UInt64(round*1000+tileIndex)
        let mask=MLXArray(prepared.condition.mask,[layout.videoTokens,1])
        let noise=MLXRandom.normal([layout.videoTokens,128],key:MLXRandom.key(seed))
        let state=noise*(mask*Float(schedule.sigmas[0]))
          + prepared.latent*(1-mask*Float(schedule.sigmas[0]))
        var key=MLXRandom.key(seed &+ 10000)
        let sampled=try runner.evaluate([
          "video_latent":state,"audio_latent":frozen.latent,
          "video_text":videoContext,"audio_text":audioContext,
          "video_positions":MLXArray(layout.positions,[layout.videoTokens,3]),
          "audio_positions":frozen.positions],schedule:schedule,
          videoConditioning:prepared.condition,frozenAudio:true,
          fixedWeights:source.readFixed,blockWeights:source.readBlock,
          fixedAdapters:source.fixedAdapters,blockAdapters:source.blockAdapters,
          noise:{ _,_,shape in
            let (next,draw)=MLXRandom.split(key:key);key=next
            return MLXRandom.normal([1]+shape,key:draw).reshaped(shape)
          },stageProgress:{ _,event in
            try progress("temporal_\(round)_tile_\(tileIndex+1):"+event.stage,
              event.completedBlocks,48)
          },progress:{ event in
            try progress("temporal_sampling",event.completedSteps,4)
          })
        let result=sampled["video"]!
        let primary=result[0..<geometry.videoTokens].reshaped([geometry.videoTokens,128])
        previousTile=(tile.dropLatentPrefix == 0 ? tile.latentStart : tile.latentStart-1,
          primary)
        let drop=tile.dropLatentPrefix*frameTokens
        owned.append(primary[drop..<geometry.videoTokens])
        let generated=result[layout.videoTokens-layout.slotTokens..<layout.videoTokens]
        for (index,frame) in tile.slotFrames.filter({ explicit[$0] == nil }).enumerated() {
          if newSlots[frame] == nil {
            let plane=generated[index*frameTokens..<(index+1)*frameTokens]
              .reshaped([frameTokens,128])
            newSlots[frame]=plane
            planeAt[frame]=plane
          }
        }
        try progress("temporal_tile_complete",tileIndex+1,tiles.count)
      }
      video=concatenated(owned,axis:0).reshaped([(frames/8+1)*frameTokens,128])
      for (frame,latent) in newSlots where carry[frame] == nil { carry[frame]=latent }
      eval(video)
      Memory.clearCache()
      stageSeconds["temporal_round_\(round)"]=Date().timeIntervalSince(roundStart)
    }
    let published=try MLXDFRTemporalPlan.outputFrames(inputFrames:requestedFrames,rounds:temporalRounds)
    let tokens=(published/8+1)*frameTokens
    let output=video[0..<tokens].reshaped([tokens,128])
    eval(output)
    return output
  }

  private static func configuration(_ layout: MLXDFRLayout) throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoTokens: layout.videoTokens,
      audioTokens: layout.geometry.audioFrames, textTokens: 1024)
  }

  public func evaluate(videoContext: MLXArray, audioContext: MLXArray,
    references:[(first:MLXArray,last:MLXArray?)]=[],
    progress: @escaping (String, Int, Int) throws -> Void = { _, _, _ in }) throws -> [String: MLXArray] {
    guard gate.try() else { throw LTXError.invalid("DFR sampler is already active.") }
    defer { Stream.gpu.synchronize(); Memory.clearCache(); gate.unlock() }
    for (value, width) in [(videoContext, 4096), (audioContext, 2048)] {
      guard value.dtype == .float32, value.shape == [1024, width],
        MLX.isFinite(value).all().item(Bool.self) else {
        throw LTXError.invalid("DFR text context differs from the trained layout.")
      }
    }
    guard references.count == (requiresEndpoints ? 2 : 0) else {
      throw LTXError.invalid("DFR endpoint references must be encoded at both resolutions.")
    }
    if !references.isEmpty {
      for (index,pair) in references.enumerated() {
        let layout=index == 0 ? low : high
        let count=layout.geometry.latentHeight*layout.geometry.latentWidth
        guard pair.first.dtype == .float32,pair.first.shape == [count,128],
          pair.last.map({ $0.dtype == .float32 && $0.shape == [count,128] }) ?? true else {
          throw LTXError.invalid("DFR endpoint latent differs from stage geometry.")
        }
      }
    }
    stageSeconds = [:]
    let stageOne = try autoreleasepool { () throws -> (MLXArray, MLXArray, MLXArray) in
      let start = Date()
      let g = recipe.low
      let video = MLXNoisePolicy.seeded(recipe.seed, tokens: g.videoTokens).asType(.float32)
      let audio = MLXNoisePolicy.seeded(recipe.seed &+ 1, tokens: g.audioFrames).asType(.float32)
      let pair=references.first
      let prepared = try low.prepare(generated: video,first:pair?.first,last:pair?.last)
      let sampled = try sample(stage: 1, layout: low, video: prepared.latent,
        audio: audio, condition: prepared.condition,
        videoContext: videoContext, audioContext: audioContext,
        schedule: recipe.first, bfloat16: ["video", "audio"], progress: progress)
      let primary = sampled["video"]![0..<g.videoTokens].reshaped([g.videoTokens, 128])
      let slots = sampled["video"]![low.endpointTokens..<low.videoTokens]
        .reshaped([low.slotTokens, 128])
      let resultAudio = sampled["audio"]!.reshaped([g.audioFrames, 128])
      eval(primary, slots, resultAudio)
      stageSeconds["stage1"] = Date().timeIntervalSince(start)
      try progress("stage1_weights_released", 1, 2)
      return (primary, slots, resultAudio)
    }
    Memory.clearCache()
    let upscaled = try autoreleasepool { () throws -> (MLXArray, MLXArray) in
      let start = Date()
      let main = try upscaler.upscale(stageOne.0.reshaped([recipe.low.latentFrames,
        recipe.low.latentHeight, recipe.low.latentWidth, 128]),
        maximumActivationBytes: maximumActivationBytes,
        progress: { try progress("upscale:" + $0, 0, 1) })
      let slots = try upscaler.upscale(stageOne.1.reshaped([low.slotFrames.count,
        recipe.low.latentHeight, recipe.low.latentWidth, 128]),
        maximumActivationBytes: maximumActivationBytes,
        progress: { try progress("slot_upscale:" + $0, 0, 1) })
      let primary = main.reshaped([recipe.high.videoTokens, 128])
      let seededSlots = slots.reshaped([high.slotTokens, 128])
      eval(primary, seededSlots)
      stageSeconds["latent_upscale_mlx"] = Date().timeIntervalSince(start)
      try progress("upscaler_weights_released", 1, 1)
      return (primary, seededSlots)
    }
    Memory.clearCache()
    let sigma = Float(recipe.second.sigmas[0])
    let videoNoise = MLXNoisePolicy.seeded(recipe.seed &+ 2,
      tokens: recipe.high.videoTokens).asType(.float32)
    let audioNoise = MLXNoisePolicy.seeded(recipe.seed &+ 2,
      tokens: recipe.high.audioFrames).asType(.float32)
    let video = videoNoise * sigma + upscaled.0 * (1 - sigma)
    let audio = audioNoise * sigma + stageOne.2 * (1 - sigma)
    let pair=references.count == 2 ? references[1] : nil
    let prepared = try high.prepare(generated: video, first:pair?.first,last:pair?.last,
      reference: stageOne.0,
      slots: upscaled.1)
    let started = Date()
    let sampled = try sample(stage: 2, layout: high, video: prepared.latent,
      audio: audio, condition: prepared.condition,
      videoContext: videoContext, audioContext: audioContext,
      schedule: recipe.second, bfloat16: ["audio"], progress: progress)
    let primary = sampled["video"]![0..<recipe.high.videoTokens]
      .reshaped([recipe.high.videoTokens, 128])
    eval(primary)
    stageSeconds["stage2"] = Date().timeIntervalSince(started)
    try progress("stage2_weights_released", 2, 2)
    guard temporalRounds > 0 else { return ["video": primary, "audio": stageOne.2] }
    let finalSlots=sampled["video"]![high.videoTokens-high.slotTokens..<high.videoTokens]
      .reshaped([high.slotTokens,128])
    eval(finalSlots)
    let temporal=try runTemporal(video:primary,slots:finalSlots,audio:stageOne.2,
      videoContext:videoContext,audioContext:audioContext,
      highReference:references.count == 2 ? references[1] : nil,progress:progress)
    return ["video":temporal,"audio":stageOne.2]
  }

  private func sample(stage: Int, layout: MLXDFRLayout, video: MLXArray,
    audio: MLXArray, condition: MLXVideoDenoiseCondition,
    videoContext: MLXArray, audioContext: MLXArray,
    schedule: SamplingSchedule, bfloat16: Set<String>,
    progress: @escaping (String, Int, Int) throws -> Void) throws -> [String: MLXArray] {
    let g = layout.geometry
    let source = weights[stage - 1]
    let runner = try MLXSamplingRunner(configuration: Self.configuration(layout),
      maximumActivationBytes: maximumActivationBytes,
      keyframeMarkerRows: layout.slotTokens)
    let slotNoise=MLXNoisePolicy.seeded(recipe.seed &+ UInt64(stage+2),
      tokens:layout.slotTokens).asType(.float32)
    let initializedVideo=try layout.noiseSlots(video,noise:slotNoise,
      sigma:Float(schedule.sigmas[0]))
    let inputs: [String: MLXArray] = [
      "video_latent": initializedVideo, "audio_latent": audio,
      "video_text": videoContext, "audio_text": audioContext,
      "video_positions": MLXArray(layout.positions, [layout.videoTokens, 3]),
      "audio_positions": MLXArray(g.audioPositions, [g.audioFrames, 1])]
    var key = MLXRandom.key(recipe.seed &+ 10000)
    let sampled = try runner.evaluate(inputs, schedule: schedule,
      videoConditioning: condition, bfloat16State: bfloat16,
      fixedWeights: source.readFixed, blockWeights: source.readBlock,
      fixedAdapters: source.fixedAdapters, blockAdapters: source.blockAdapters,
      noise: { _, _, shape in
        let (next, draw) = MLXRandom.split(key: key)
        key = next
        return MLXRandom.normal([1] + shape, key: draw).reshaped(shape)
      }, stageProgress: { _, event in
        try progress("stage\(stage):" + event.stage, event.completedBlocks, 48)
      }, progress: { event in
        try progress("sampling", (stage == 1 ? 0 : 8) + event.completedSteps, 11)
      })
    return sampled
  }
}
