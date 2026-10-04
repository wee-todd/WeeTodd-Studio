import Foundation
import MLX
import LTX25Engine
import LTX25Video

/// Header-based dispatch for the shared native media stage. Convolutional
/// defaults remain unchanged; DiffVAE cannot silently fall back after failure.
public final class MLXNativeVideoDecoder {
  private let convolutional:MLXVideoDecoder?
  private let diffusion:MLXDiffusionVideoDecoder?
  public let isDiffusion:Bool
  public var cacheLimitBytes:Int { convolutional?.cacheLimitBytes ?? 128*1024*1024 }
  public var stageSeconds:[String:Double] { convolutional?.stageSeconds ?? diffusion!.stageSeconds }
  public init(checkpoint:URL,settings:MLXDiffusionVideoSettings?=nil,maximumWorkspaceBytes:Int) throws {
    let selection=try MLXVideoDecoderSelection(checkpoint:checkpoint,settings:settings)
    isDiffusion=selection.isDiffusion
    if isDiffusion {
      convolutional=nil;diffusion=try MLXDiffusionVideoDecoder(checkpoint:checkpoint,
        options:(try settings ?? MLXDiffusionVideoSettings()).options(maximumWorkspaceBytes:maximumWorkspaceBytes))
    } else { convolutional=try MLXVideoDecoder(checkpoint:checkpoint);diffusion=nil }
  }
  public static func admit(checkpoint:URL,settings:MLXDiffusionVideoSettings?=nil,shape:[Int],
    configuration:VideoDecodeConfiguration,backend:MLXVideoBackend) throws -> (shape:[Int],bytes:Int) {
    let selection=try MLXVideoDecoderSelection(checkpoint:checkpoint,settings:settings)
    if case .diffusion=selection {
      guard backend == .mlx else { throw LTXError.invalid("Diffusion VAE requires the native MLX video backend; MPS has no DiffVAE fallback.") }
      let options=try (try settings ?? MLXDiffusionVideoSettings()).options(maximumWorkspaceBytes:configuration.maximumActivationBytes)
      let plan=try MLXDiffusionVideoPlan(shape:shape,options:options,elementBytes:2)
      return ([plan.outputShape[2],plan.outputShape[3],plan.outputShape[4],3],plan.conservativeWorkspaceBytes)
    }
    if backend == .mlx {
      let plan=try MLXVideoDecodePlan(shape:shape,configuration:configuration)
      return (plan.outputShape,plan.admittedActivationBytes)
    }
    let plan=try VideoDecodePlan(shape:shape,configuration:configuration)
    return (plan.outputShape,plan.admittedActivationBytes)
  }
  /// Shared receipt fields describe the stage actually executed; Conv returns
  /// no additions so its existing receipt format/defaults remain unchanged.
  public static func publicationMetadata(isDiffusion:Bool,settings:MLXDiffusionVideoSettings?) -> [String:Any] {
    guard isDiffusion else { return [:] }
    return ["video_decoder_architecture":"ltx25-one-step-diffusion-vae",
      "video_input_latent_dtype":"float32","video_stage_latent_dtype":"bfloat16",
      "video_precision":"bfloat16","diffvae_noise_seed":0,
      "diffvae_noise_layout":"B3FHW-before-c-w-h-patching",
      "diffvae_optimization":settings?.optimization.rawValue ?? "combined",
      "diffvae_query_chunk_size":settings?.queryChunkSize ?? 512,
      "diffvae_context_width_chunks":settings?.contextWidthChunks ?? 4,
      "diffvae_stage4_tile_width":settings?.stage4TileWidth ?? 0,
      "video_tiling_policy":"diffusion-internal-query-context-width-tiling"]
  }
  public func decodeRGB8(latent:MLXArray,configuration:VideoDecodeConfiguration,
    progress:(Int,Int) throws -> Void={ _,_ in },receive:(Int,Data) throws -> Void) throws {
    if let convolutional { try convolutional.decodeRGB8(latent:latent,configuration:configuration,progress:progress,receive:receive) }
    else {
      // The native media stage already declares BF16 video precision. Keep the
      // cast explicit; the standalone DiffVAE decode API preserves caller dtype.
      try diffusion!.decodeRGB8(latent:latent.asType(.bfloat16),progress:{ _,a,b in try progress(a,b) },receive:receive)
    }
  }
  public func decode(latent:MLXArray,configuration:VideoDecodeConfiguration,
    progress:(Int,Int) throws -> Void={ _,_ in },receive:(VideoFrameChunk) throws -> Void) throws {
    guard let convolutional else { throw LTXError.invalid("DiffVAE frame publication must use its RGB24 path to preserve output-dtype quantization.") }
    try convolutional.decode(latent:latent,configuration:configuration,progress:progress,receive:receive)
  }
}
