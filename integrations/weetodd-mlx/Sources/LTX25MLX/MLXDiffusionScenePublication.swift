import Foundation
import MLX
import LTX25Engine
import LTX25Video

/// DiffVAE's native context/query/width tiling is internal to a complete decode.
/// Convolutional temporal overlap joins are never applied to diffusion noise.
public enum MLXDiffusionScenePublication {
  public static func admit(geometry:AVGeometry,plan:LTX25ScenePlan,
    strictBoundaries:Set<Int>,checkpoint:URL,settings:MLXDiffusionVideoSettings?,
    maximumWorkspaceBytes:Int) throws -> Int {
    let groups = strictBoundaries.isEmpty ? [geometry.frames] :
      try MLXSceneMediaPublisher.strictFrameRoutes(plan:plan,strictBoundaries:strictBoundaries).map(\.frames)
    var maximum=0
    for frames in groups {
      try Task.checkCancellation()
      let group=try AVGeometry(width:geometry.width,height:geometry.height,frames:frames,fps:geometry.fps)
      let admitted=try MLXNativeVideoDecoder.admit(checkpoint:checkpoint,settings:settings,shape:group.videoShape,
        configuration:MLXMediaPipeline.videoConfiguration(for:group,activationBytes:maximumWorkspaceBytes),backend:.mlx)
      guard admitted.shape == [frames,geometry.height,geometry.width,3] else {
        throw LTXError.invalid("DiffVAE scene decoder geometry differs from its complete causal interval.")
      }
      maximum=max(maximum,admitted.bytes)
    }
    return maximum
  }
  public static func decode(_ sampled:MLXSceneSampler.Result,plan:LTX25ScenePlan,
    checkpoint:URL,settings:MLXDiffusionVideoSettings?,maximumWorkspaceBytes:Int,
    progress:(Int,Int)throws->Void,receive:(Int,Data)throws->Void) throws {
    _ = try admit(geometry:sampled.geometry,plan:plan,strictBoundaries:sampled.strictBoundaries,
      checkpoint:checkpoint,settings:settings,maximumWorkspaceBytes:maximumWorkspaceBytes)
    let decoder=try MLXNativeVideoDecoder(checkpoint:checkpoint,settings:settings,maximumWorkspaceBytes:maximumWorkspaceBytes)
    let g=sampled.geometry
    if sampled.strictBoundaries.isEmpty {
      let unpacked=try g.unpackVideo(sampled.video.asArray(Float.self))
      try decoder.decodeRGB8(latent:MLXArray(unpacked,g.videoShape),
        configuration:MLXMediaPipeline.videoConfiguration(for:g,activationBytes:maximumWorkspaceBytes),progress:progress) { index,bytes in
        if index<MLXSceneMediaPublisher.deliveredFrames(plan:plan) { try receive(index,bytes) }
      }
    } else {
      for route in try MLXSceneMediaPublisher.strictFrameRoutes(plan:plan,strictBoundaries:sampled.strictBoundaries) {
        try Task.checkCancellation()
        let group=try AVGeometry(width:g.width,height:g.height,frames:route.frames,fps:g.fps)
        let packed=try MLXSceneLatentAssembly.assembleVideoGroup(video:sampled.videoWindows,plan:plan,
          windows:route.windows,latentHeight:g.latentHeight,latentWidth:g.latentWidth)
        let unpacked=try group.unpackVideo(packed.asArray(Float.self))
        try decoder.decodeRGB8(latent:MLXArray(unpacked,group.videoShape),
          configuration:MLXMediaPipeline.videoConfiguration(for:group,activationBytes:maximumWorkspaceBytes),progress:progress) { local,bytes in
          if route.local.contains(local) { try receive(route.outputStart+local-route.local.lowerBound,bytes) }
        }
        Stream.gpu.synchronize();Memory.clearCache()
      }
    }
  }
}
