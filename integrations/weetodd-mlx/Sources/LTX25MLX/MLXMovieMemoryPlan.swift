import Foundation
import LTX25Engine
import LTX25Video

/// Pure per-stage admission; not eager allocation or a process-memory cap.
/// Mirrors Studio's host reserve policy and the actual movie chunk geometry.
public struct MLXMovieMemoryPlan: Sendable {
  public let transformerActivationBytes:Int,videoActivationBytes:Int,textOwnedBufferBytes:Int
  public let activationCeilingBytes:Int
  public let chunks:[MLXMovieUpscalePlan.Chunk]
  public init(request:MLXMovieUpscaleRequest,cutFrames:[Int]=[],
    physicalMemory:UInt64,recommendedWorkingSet:UInt64) throws {
    let gib:UInt64=1024*1024*1024
    let envelope=min(physicalMemory/2,recommendedWorkingSet)
    let ceiling=min(envelope>4*gib ? envelope-4*gib : 0,
      UInt64(min(MLXMediaPipeline.maximumVideoActivationMiB,
        MLXMediaPipeline.maximumTransformerActivationMiB))*1024*1024)
    guard ceiling>0,ceiling<=UInt64(Int.max) else {
      throw LTXError.invalid("Movie stages have no activation allowance after system reserves.")
    }
    chunks=try request.chunkPlans(cutFrames:cutFrames)
    let largest=chunks.max { $0.paddedFrames<$1.paddedFrames }!
    let local=try MLXMovieUpscalePlan(mode:request.plan.mode,width:request.plan.size.width,
      height:request.plan.size.height,frames:largest.frames,fps:request.plan.fps,
      sizePolicy:.strict,refinementStrength:request.plan.refinementStrength)
    let g=try AVGeometry(width:local.size.outputWidth,height:local.size.outputHeight,
      frames:largest.paddedFrames,fps:local.fps)
    let sourceShape=[g.latentFrames,local.size.height/32,local.size.width/32,128]
    var upscaleConfiguration=LatentUpscaleConfiguration()
    upscaleConfiguration.maximumActivationBytes=Int.max
    let upscale=try LatentUpscalePlan(shape:sourceShape,configuration:upscaleConfiguration)
    // This is the SAME full im2col reserve enforced by MLXLatentUpscaler.admit.
    let upscaleWorkspace=upscale.outputShape.dropLast().reduce(1,*)*1024*27*4
    var transformer=upscale.activationBytes+upscaleWorkspace
    let usesFirst=request.anchors != .none || request.referenceImages.contains { $0.role=="first" }
    let usesLast=request.anchors == .firstLast || request.referenceImages.contains { $0.role=="last" }
    if local.mode != .latentOnly {
      let layout=try MLXMovieUpscaleLayout(plan:local,
        firstStrength:usesFirst ? request.anchorStrength : nil,
        lastStrength:usesLast ? request.anchorStrength : nil)
      let configuration=try AVBlockConfiguration(videoTokens:layout.videoTokens,
        audioTokens:g.audioFrames,textTokens:1024)
      let needsPerToken=layout.mask.contains { $0 != 1 }
      transformer=max(transformer,try MLXAVBlock.estimatedActivationBytes(
        configuration:configuration,perTokenVideo:needsPerToken))
      // Rotary grids are included within the block estimate; verify that reserve.
      _ = try MLXDenoiser.admitRotary(configuration:configuration,maximumActivationBytes:transformer)
    }
    _ = try MLXLatentUpscaler.admit(shape:sourceShape,maximumActivationBytes:transformer)
    let encoder=try MLXVideoEncodeTilePlan(frames:largest.paddedFrames,
      width:local.size.width,height:local.size.height,maximumOwnedBufferBytes:Int.max)
      .tiles.map(\.ownedBufferBytes).max()!
    let decoder=try MLXNativeVideoDecoder.admit(checkpoint:URL(fileURLWithPath:request.components["video_checkpoint"]!),settings:request.diffusionVAE,shape:g.videoShape,
      configuration:MLXMediaPipeline.videoConfiguration(for:g,activationBytes:Int.max),backend:.mlx).bytes
    let image=try local.mode != .latentOnly && (usesFirst || usesLast) ?
      MLXImageEncodePlan(width:g.width,height:g.height,maximumOwnedBufferBytes:Int.max).ownedBufferBytes : 0
    let video=max(encoder,decoder,image)
    for (stage,needed) in [("transformer/upscaler",transformer),("video encoder/decoder",video)] {
      guard UInt64(needed)<=ceiling else {
        throw LTXError.invalid("Movie \(g.width)×\(g.height), \(largest.paddedFrames) padded frames: \(stage) needs \(needed) activation bytes; this Mac admits \(ceiling) after reserves (engine maximum 32 GiB).")
      }
    }
    transformerActivationBytes=transformer;videoActivationBytes=video
    // Text preflight tokenizes the actual prompt and validates its existing plan.
    // This is a bounded allowance, not a claim of exact text memory consumption.
    textOwnedBufferBytes=Int(min(3*gib,ceiling));activationCeilingBytes=Int(ceiling)
  }
}
