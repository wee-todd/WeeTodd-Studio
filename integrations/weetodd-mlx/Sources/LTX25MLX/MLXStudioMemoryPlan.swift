import Foundation
import LTX25Engine

/// Per-job activation admission, not eager allocation or a process-memory cap.
/// Stages unload sequentially, so their workspaces are not added together.
public struct MLXStudioMemoryPlan:Sendable {
  public let transformerActivationBytes:Int
  public let videoActivationBytes:Int
  public let activationCeilingBytes:Int

  public init(request:MLXDistilledRequest,extensionContextFrames:Int?=nil,
    physicalMemory:UInt64,recommendedWorkingSet:UInt64) throws {
    let gib:UInt64=1024*1024*1024
    // Leave half of RAM outside this worker's admission, and reserve another
    // 4 GiB inside it for stage weights, bounded caches and media buffers.
    // Respect Metal's working set and the engine's existing 32 GiB stage limit.
    let stageEnvelope=min(physicalMemory/2,recommendedWorkingSet)
    let ceiling=min(stageEnvelope > 4*gib ? stageEnvelope-4*gib : 0,
      UInt64(min(MLXMediaPipeline.maximumVideoActivationMiB,MLXMediaPipeline.maximumTransformerActivationMiB))*1024*1024)
    let recipe=try request.recipe()
    var transformer=0
    let geometries=request.ingredientsSheet == nil && request.msr == nil && request.singleStageSampling == nil ? [recipe.low,recipe.high] : [recipe.high]
    for (index,geometry) in geometries.enumerated() {
      let ordinary=try request.ordinaryKeyframeLayout(geometry:geometry,stage:index)
      let singleStage=try request.singleStageControlLayout()
      let layout=try (request.usesOrdinaryKeyframes ? nil : request.referenceImages.first).map { try MLXReferenceLayout(geometry:geometry,
        firstStrength:$0.strength,lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil) }
      let guide=try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:geometry,contextFrames:$0) }
      let union=try index == 0 ? request.unionControlGuide.map {
        try MLXUnionControlLayout(geometry:geometry,strength:$0.referenceStrength)
      } : nil
      let ic=try index == 0 ? request.icControl.map { try MLXICControlLayout(geometry:geometry,control:$0) } : nil
      let ingredients=try request.ingredientsSheet.map {
        try MLXReferenceVideoLayout(geometry:geometry,strength:$0.referenceStrength)
      }
      let msr=try MLXMSRReferencePlan.resolve(request,target:geometry)
      let msrAudio=try MLXMSRAudioLayout.resolve(request,target:geometry)
      let dfr=try request.dfr.map { _ in try MLXDFRLayout(geometry:geometry,
        slotFrames:MLXDFRCanvas(frames:recipe.high.frames).slotFrames,
        reference:index == 1 ? recipe.low : nil,
        firstStrength:request.referenceImages.first?.strength,
        lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil,
        lastFrame:request.referenceImages.count == 2 ? request.frames-1 : nil) }
      guard layout?.lastStrength == nil || guide == nil else {
        throw LTXError.invalid("A last-frame image cannot share an LTX history guide.")
      }
      transformer=max(transformer,try MLXAVBlock.estimatedActivationBytes(configuration:
        AVBlockConfiguration(videoTokens:singleStage?.videoTokens ?? ordinary?.videoTokens ?? dfr?.videoTokens ?? guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ic?.videoTokens ?? ingredients?.videoTokens ?? msr?.layout.videoTokens ?? geometry.videoTokens,
          audioTokens:msrAudio?.audioTokens ?? guide?.audioTokens ?? geometry.audioFrames,textTokens:1024),
        perTokenVideo:singleStage?.requiresPerTokenVideo == true || ordinary?.anchors.isEmpty == false || layout != nil || guide != nil || union != nil || ic != nil || ingredients != nil || msr != nil ||
          (dfr?.referenceTokens ?? 0)>0 || (dfr != nil && !request.referenceImages.isEmpty),perTokenAudio:guide != nil || msrAudio != nil) + (index == 0 && request.guidedSampling != nil ?
          MLXGuidedSampling.reserveBytes(videoTokens:ingredients?.videoTokens ?? ordinary?.videoTokens ?? layout?.videoTokens ?? geometry.videoTokens,
            audioTokens:geometry.audioFrames) : 0))
    }
    if request.singleStageSampling?.method == .cfgpp {
      let layout=try request.singleStageControlLayout()!
      transformer += try layout.cfgppReserveBytes(audioTokens:recipe.high.audioFrames)
    }
    if request.ingredientsSampling == .ancestralCFGPP {
      transformer += MLXSingleStageRipple.cfgppReserveBytes(geometry:recipe.high)
    }
    if let dfr=request.dfr,dfr.temporalRounds>0 {
      let configurations=try MLXDFRTemporalPlan.admissionConfigurations(
        geometry:recipe.high,requestedFrames:request.frames,
        slots:MLXDFRCanvas(frames:recipe.high.frames).slotFrames,
        rounds:dfr.temporalRounds,endpointCount:request.referenceImages.count)
      for config in configurations {
        transformer=max(transformer,try MLXAVBlock.estimatedActivationBytes(
          configuration:config,perTokenVideo:true))
      }
      var latentFrames=recipe.high.latentFrames
      for _ in 0..<dfr.temporalRounds {
        transformer=max(transformer,try MLXTemporalUpscaler.estimatedActivationBytes(shape:
          [latentFrames,recipe.high.latentHeight,recipe.high.latentWidth,128]))
        latentFrames=2*latentFrames-1
      }
    }
    // This plan validates geometry and calculates bytes without allocating a VAE.
    let output=try request.dfr.map { dfr in
      try dfr.temporalRounds == 0 ? recipe.high : AVGeometry(width:recipe.high.width,height:recipe.high.height,
        frames:MLXDFRTemporalPlan.outputFrames(inputFrames:request.frames,rounds:dfr.temporalRounds),
        fps:recipe.high.fps*Double(1 << dfr.temporalRounds))
    } ?? recipe.high
    let decoder=try MLXNativeVideoDecoder.admit(checkpoint:URL(fileURLWithPath:request.videoCheckpoint),settings:request.diffusionVAE,shape:output.videoShape,configuration:
      MLXMediaPipeline.videoConfiguration(for:output,activationBytes:Int.max),backend:.mlx).bytes
    let unionGuideEncoder=try request.unionControlGuide.map { _ in
      try MLXVideoEncodeTilePlan(frames:recipe.low.frames,width:recipe.low.width/2,
        height:recipe.low.height/2,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    } ?? 0
    let ingredientsGuideEncoder=try request.ingredientsSheet.map { _ in
      try MLXVideoEncodeTilePlan(frames:request.guidedSampling?.singleStage == true ? recipe.high.frames : 1,width:recipe.high.width,
        height:recipe.high.height,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    } ?? 0
    let msrGuideEncoder=try MLXMSRReferencePlan.resolve(request,target:recipe.high)?.plans.map { plan in
      try MLXVideoEncodeTilePlan(frames:plan.geometry.frames,width:plan.geometry.width,
        height:plan.geometry.height,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    }.max() ?? 0
    let icGuideEncoder=try request.icControl.map { control in
      let geometry=try control.guideGeometry(target:recipe.low)
      return try MLXVideoEncodeTilePlan(frames:geometry.frames,width:geometry.width,
        height:geometry.height,maximumOwnedBufferBytes:Int.max).tiles.map(\.ownedBufferBytes).max() ?? 0
    } ?? 0
    let video=max(decoder,unionGuideEncoder,ingredientsGuideEncoder,msrGuideEncoder,icGuideEncoder)
    for (stage,needed) in [("transformer",transformer),("video decoder",video)] {
      guard UInt64(needed) <= ceiling else {
        throw LTXError.invalid("LTX \(request.width)×\(request.height), \(request.frames) frames: \(stage) needs an estimated \(needed) activation bytes; this Mac's stage allowance is \(ceiling) bytes after memory reserves (engine maximum 32 GiB).")
      }
    }
    transformerActivationBytes=transformer;videoActivationBytes=video;activationCeilingBytes=Int(ceiling)
  }

  public init(ripple request: MLXRippleRequest, physicalMemory: UInt64,
    recommendedWorkingSet: UInt64) throws {
    let gib: UInt64 = 1024 * 1024 * 1024
    let stageEnvelope = min(physicalMemory / 2, recommendedWorkingSet)
    let ceiling = min(stageEnvelope > 4 * gib ? stageEnvelope - 4 * gib : 0,
      UInt64(min(MLXMediaPipeline.maximumVideoActivationMiB,
        MLXMediaPipeline.maximumTransformerActivationMiB)) * 1024 * 1024)
    guard ceiling > 0, ceiling <= UInt64(Int.max) else {
      throw LTXError.invalid("Ripple has no admitted activation budget after system reserves.")
    }
    let layout = try MLXReferenceVideoLayout(geometry: request.geometry,
      strength: request.referenceStrength,
      anchors: request.anchors.map { RippleImageAnchor(frame: $0.frame, strength: $0.strength) })
    let configuration = try AVBlockConfiguration(videoTokens: layout.videoTokens,
      audioTokens: request.geometry.audioFrames, textTokens: 1024)
    let transformer = try MLXAVBlock.estimatedActivationBytes(configuration: configuration,
      perTokenVideo: true)
    let video = try MLXNativeVideoDecoder.admit(checkpoint:URL(fileURLWithPath:request.videoCheckpoint),settings:request.diffusionVAE,shape:request.geometry.videoShape,
      configuration:MLXMediaPipeline.videoConfiguration(for:request.geometry,activationBytes:Int.max),backend:.mlx).bytes
    for (stage, needed) in [("transformer", transformer), ("video decoder", video)] {
      guard UInt64(needed) <= ceiling else {
        throw LTXError.invalid("Ripple \(request.width)×\(request.height), \(request.frames) frames: \(stage) needs \(needed) activation bytes; this Mac admits \(ceiling) after reserves.")
      }
    }
    _ = try request.plan(maximumActivationBytes: transformer)
    _ = try MLXVideoEncodeTilePlan(frames: request.frames, width: request.width,
      height: request.height)
    for _ in request.anchors {
      _ = try MLXImageEncodePlan(width: request.width, height: request.height,
        maximumOwnedBufferBytes: Int(ceiling))
    }
    transformerActivationBytes = transformer
    videoActivationBytes = video
    activationCeilingBytes = Int(ceiling)
  }
}
