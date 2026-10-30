import Foundation
import MLX

/// Resolved execution detail shared by the sampler and its result trace.
/// This changes batching only; every query continues to see every packed key.
public struct H3RegenerateAttentionPlan: Sendable, Equatable {
  public let policyVersion:Int
  public let rows:Int,heads:Int,headWidth:Int,queryChunk:Int,headsPerGroup:Int,outputProjectionChunk:Int
  public let forceFused:Bool
  public let useUnfencedKernel:Bool
  public let route:String,architecture:String
  public static let mlxSwiftRevision="901941965d82e4a216d4d117231d847d194c563d"
  public static let mlxRevision="1f8e74e3f12f31365464a6867c6579f0e9b29d85"

  static func resolve(rows:Int,heads:Int,headWidth:Int,isMetalGPU:Bool,isBF16:Bool,
    architecture:String,supportsUnfenced:Bool=true) throws -> Self {
    let group=try H3RegenerateAttention.preferredHeadGroupSize(rows:rows,heads:heads,headWidth:headWidth)
    // The wider route is qualified only for the pinned non-NAX g15d kernel.
    // Larger grids, other devices/precisions/dimensions retain phase-one admission.
    let qualified=isMetalGPU && isBF16 && architecture == "applegpu_g15d"
      && heads == 56 && headWidth == 128 && group == 56 && rows > 8
    let unfenced=qualified && supportsUnfenced
    return Self(policyVersion:3,rows:rows,heads:heads,headWidth:headWidth,
      queryChunk:qualified ? 8192:256,headsPerGroup:group,outputProjectionChunk:256,forceFused:qualified,useUnfencedKernel:unfenced,
      route:unfenced ? "metal-steel-unfenced-bf16-d128":qualified ? "metal-steel-bf16-d128":"mlx-sdpa-conservative",architecture:architecture)
  }

  static func runtime(rows:Int,heads:Int,headWidth:Int,dtype:DType) throws -> Self {
    let gpu=Device.defaultDevice().deviceType == .gpu
    return try resolve(rows:rows,heads:heads,headWidth:headWidth,isMetalGPU:gpu,
      isBF16:dtype == .bfloat16,architecture:gpu ? GPU.deviceInfo().architecture:"cpu",
      supportsUnfenced:unfencedRuntimeIsQualified(ProcessInfo.processInfo.operatingSystemVersion))
  }

  /// JIT compiler qualification is deliberately separate from shape admission.
  /// Other OS releases retain the previously qualified MLX fused route.
  static func unfencedRuntimeIsQualified(_ version:OperatingSystemVersion) -> Bool {
    version.majorVersion == 27 && version.minorVersion == 0 && version.patchVersion == 1
  }

  public var attentionCallsPerBlock:Int {
    ((rows+queryChunk-1)/queryChunk)*((heads+headsPerGroup-1)/headsPerGroup)
  }
  public var outputProjectionCallsPerBlock:Int { (rows+outputProjectionChunk-1)/outputProjectionChunk }

  public var metadata:[String:Any] {
    ["policyVersion":policyVersion,"packedRows":rows,"heads":heads,"headWidth":headWidth,
      "queryChunk":queryChunk,"headsPerGroup":headsPerGroup,"forceFused":forceFused,
      "attentionQueryChunk":queryChunk,"attentionCallsPerBlock":attentionCallsPerBlock,
      "outputProjectionChunk":outputProjectionChunk,
      "outputProjectionCallsPerBlock":outputProjectionCallsPerBlock,
      "simdExecutionFencesRemoved":useUnfencedKernel,
      "route":route,"architecture":architecture,"allKeysVisible":true,
      "operatingSystem":ProcessInfo.processInfo.operatingSystemVersionString,
      "smallQueryTail":"conservative SDPA for at most 8 rows",
      "mlxSwiftRevision":Self.mlxSwiftRevision,"mlxRevision":Self.mlxRevision]
  }
}
