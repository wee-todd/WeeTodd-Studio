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
    let geometries=request.ingredientsSheet == nil && request.msr == nil ? [recipe.low,recipe.high] : [recipe.high]
    for (index,geometry) in geometries.enumerated() {
      let layout=try request.referenceImages.first.map { try MLXReferenceLayout(geometry:geometry,
        firstStrength:$0.strength,lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil) }
      let guide=try extensionContextFrames.map { try MLXExtensionGuideLayout(geometry:geometry,contextFrames:$0) }
      let union=try index == 0 ? request.unionControlGuide.map {
        try MLXUnionControlLayout(geometry:geometry,strength:$0.referenceStrength)
      } : nil
      let ingredients=try request.ingredientsSheet.map {
        try MLXReferenceVideoLayout(geometry:geometry,strength:$0.referenceStrength)
      }
      let msr=try MLXMSRReferencePlan.resolve(request,target:geometry)
      let dfr=try request.dfr.map { _ in try MLXDFRLayout(geometry:geometry,
        slotFrames:MLXDFRCanvas(frames:recipe.high.frames).slotFrames,
        reference:index == 1 ? recipe.low : nil,
        firstStrength:request.referenceImages.first?.strength,
        lastStrength:request.referenceImages.count == 2 ? request.referenceImages[1].strength : nil) }
      guard layout?.lastStrength == nil || guide == nil else {
        throw LTXError.invalid("A last-frame image cannot share an LTX history guide.")
      }
      transformer=max(transformer,try MLXAVBlock.estimatedActivationBytes(configuration:
        AVBlockConfiguration(videoTokens:dfr?.videoTokens ?? guide?.videoTokens ?? layout?.videoTokens ?? union?.videoTokens ?? ingredients?.videoTokens ?? msr?.layout.videoTokens ?? geometry.videoTokens,
          audioTokens:guide?.audioTokens ?? geometry.audioFrames,textTokens:1024),
        perTokenVideo:layout != nil || guide != nil || union != nil || ingredients != nil || msr != nil ||
          (dfr?.referenceTokens ?? 0)>0 || (dfr != nil && !request.referenceImages.isEmpty),perTokenAudio:guide != nil))
    }
    // This plan validates geometry and calculates bytes without allocating a VAE.
    let decoder=try MLXVideoDecodePlan(shape:recipe.high.videoShape,configuration:
      MLXMediaPipeline.videoConfiguration(for:recipe.high,activationBytes:Int.max)).admittedActivationBytes
    let unionGuideEncoder=try request.unionControlGuide.map { _ in
      try MLXVideoEncodeTilePlan(frames:recipe.low.frames,width:recipe.low.width/2,
        height:recipe.low.height/2,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    } ?? 0
    let ingredientsGuideEncoder=try request.ingredientsSheet.map { _ in
      try MLXVideoEncodeTilePlan(frames:recipe.high.frames,width:recipe.high.width,
        height:recipe.high.height,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    } ?? 0
    let msrGuideEncoder=try MLXMSRReferencePlan.resolve(request,target:recipe.high)?.plans.map { plan in
      try MLXVideoEncodeTilePlan(frames:plan.geometry.frames,width:plan.geometry.width,
        height:plan.geometry.height,maximumOwnedBufferBytes:Int.max)
        .tiles.map(\.ownedBufferBytes).max() ?? 0
    }.max() ?? 0
    let video=max(decoder,unionGuideEncoder,ingredientsGuideEncoder,msrGuideEncoder)
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
    let video = try MLXVideoDecodePlan(shape: request.geometry.videoShape,
      configuration: MLXMediaPipeline.videoConfiguration(for: request.geometry,
        activationBytes: Int.max)).admittedActivationBytes
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
