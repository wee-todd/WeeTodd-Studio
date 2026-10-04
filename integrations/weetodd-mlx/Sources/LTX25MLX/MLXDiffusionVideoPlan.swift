import Foundation
import CoreFoundation
import TensorIO
import LTX25Engine

/// Public Python decoder choices. Width tiling and fused Metal paths require their
/// own numerical/media qualification; selecting them never changes the default.
public enum MLXDiffusionVideoOptimization:String,Codable,Sendable {
  case combined
  case deferredStage4="deferred_stage4"
  case stage4WidthTiles="stage4_width_tiles"
  case metalNA3D="metal_na3d_experimental"
  case metalNA3DQueryTiled="metal_na3d_query_tiled_experimental"
}
public struct MLXDiffusionVideoOptions:Sendable {
  public let optimization:MLXDiffusionVideoOptimization
  public let queryChunkSize:Int
  public let tokenChunkSize:Int
  public let contextWidthChunks:Int
  public let stage4TileWidth:Int
  public let maximumWorkspaceBytes:Int
  public let maximumWeightBytes:Int
  public init(optimization:MLXDiffusionVideoOptimization = .combined,queryChunkSize:Int=512,
    tokenChunkSize:Int=4096,contextWidthChunks:Int=4,stage4TileWidth:Int=0,
    maximumWorkspaceBytes:Int=32*1024*1024*1024,maximumWeightBytes:Int=256*1024*1024) throws {
    guard (1...65_536).contains(queryChunkSize),(1...65_536).contains(tokenChunkSize),
      (1...4096).contains(contextWidthChunks),(0...4096).contains(stage4TileWidth),
      maximumWorkspaceBytes>0,(1...512*1024*1024).contains(maximumWeightBytes),
      (optimization == .stage4WidthTiles ? stage4TileWidth>0 : stage4TileWidth == 0) else {
      throw LTXError.invalid("Invalid Diffusion VAE chunk, workspace or optimization controls.")
    }
    self.optimization=optimization;self.queryChunkSize=queryChunkSize;self.tokenChunkSize=tokenChunkSize
    self.contextWidthChunks=contextWidthChunks;self.stage4TileWidth=stage4TileWidth
    self.maximumWorkspaceBytes=maximumWorkspaceBytes;self.maximumWeightBytes=maximumWeightBytes
  }
  var isMetal:Bool { optimization == .metalNA3D || optimization == .metalNA3DQueryTiled }
}

/// Header-only admission, independent of MLX/device/parameter materialization.
/// Products are checked before allocating query indices, pixel noise or weights.
public struct MLXDiffusionVideoPlan:Sendable {
  public let outputShape:[Int]
  public let stageShapes:[[Int]]
  public let maximumGatherBytes:Int
  public let conservativeWorkspaceBytes:Int
  public let widthHalo=24
  public static let channels=[2048,1024,512,512,256]
  public static let depths=[4,6,4,2,8]
  public static let kernels=[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]
  public static let strides=[[1,2,2],[2,1,1],[2,2,2],[2,2,2]]
  public static let reductions=[2,2,1,2]
  static func product(_ values:[Int],limit:Int=Int.max) throws -> Int {
    var result=1
    for value in values {
      let next=result.multipliedReportingOverflow(by:value)
      guard value>0,!next.overflow,next.partialValue<=limit else { throw LTXError.invalid("Diffusion VAE shape/workspace overflows its admitted bound.") }
      result=next.partialValue
    }
    return result
  }
  public init(shape:[Int],options:MLXDiffusionVideoOptions,elementBytes:Int=2) throws {
    guard shape.count == 5,shape[0] == 1,shape[1] == 128,shape[2]>=3,
      shape[3]>=7,shape[4]>=7,shape[2...4].allSatisfy({ $0<=4096 }),[2,4].contains(elementBytes) else {
      throw LTXError.invalid("Diffusion VAE requires batch-one B128FHW with initial F/H/W covering3×7×7.")
    }
    let target=try Self.product([shape[2],8])-7
    outputShape=[1,3,target,try Self.product([shape[3],32]),try Self.product([shape[4],32])]
    _=try Self.product(outputShape,limit:Int(Int32.max))
    var stages:[[Int]]=[],current=[1,shape[2]+2,shape[3],shape[4],2048]
    var peakActivation=0,gather=0
    for index in 0..<5 {
      stages.append(current)
      let rows=try Self.product(Array(current[0..<4]),limit:Int(Int32.max))
      let active=try Self.product(current+[elementBytes])
      peakActivation=max(peakActivation,active)
      let kernel=Self.kernels[index]
      guard zip(Array(current[1...3]),kernel).allSatisfy({ $0 >= $1 }) else {
        throw LTXError.invalid("Diffusion VAE stage geometry does not cover its shifted neighborhood.")
      }
      if !options.isMetal {
        let chunk=min(options.queryChunkSize,rows)
        let gathered=try Self.product([chunk,try Self.product(kernel),current[4],elementBytes],limit:Int(Int32.max)*elementBytes)
        let scores=try Self.product([chunk,current[4]/64,try Self.product(kernel),4])
        let indices=try Self.product([chunk,try Self.product(kernel),4],limit:64*1024*1024)
        gather=max(gather,try Self.product([2,gathered])+scores*2+indices)
      }
      if index<4 {
        let s=Self.strides[index]
        current=[1,current[1]*s[0]-(s[0] == 2 ? 1 : 0),current[2]*s[1],current[3]*s[2],Self.channels[index+1]]
      }
    }
    // Stage5 uses only the published interval (at least11 frames for its kernel).
    // Four large activations account for QKV+residual/context. Gather and MLP
    // intermediates are independent bounded chunks, not whole-video 4x hidden.
    let outputBytes=try Self.product(outputShape+[elementBytes])
    let mlpChunk=try Self.product([options.tokenChunkSize,2048*4,elementBytes,3])
    let estimate=try Self.product([peakActivation,8])+gather+mlpChunk+outputBytes*3+options.maximumWeightBytes
    guard estimate<=options.maximumWorkspaceBytes else { throw LTXError.invalid("Diffusion VAE decode exceeds its conservative stage workspace allowance.") }
    stageShapes=stages;maximumGatherBytes=gather;conservativeWorkspaceBytes=estimate
  }
}

public struct MLXDiffusionVideoCheckpoint {
  public let url:URL
  public let decoderTensorBytes:UInt64
  let file:SafeTensorFile
  public init(checkpoint:URL) throws {
    try Task.checkCancellation()
    let file=try SafeTensorFile(url:checkpoint,maximumHeaderBytes:1024*1024)
    guard file.metadata["model_version"] == "2.5.0",let raw=file.metadata["config"],
      let config=try JSONSerialization.jsonObject(with:Data(raw.utf8)) as? [String:Any],
      let vae=config["vae"] as? [String:Any],let decoder=vae["decoder"] as? [String:Any],
      decoder["_class_name"] as? String == "NADiffusionDecoder",vae["model_output_type"] as? String == "x0" else {
      throw LTXError.invalid("The checkpoint is not the released LTX2.5 one-step Diffusion VAE.")
    }
    func integral(_ value:Any?) -> Int? {
      guard let number=value as? NSNumber,CFGetTypeID(number) != CFBooleanGetTypeID(),number.doubleValue.isFinite,
        number.doubleValue.rounded(.towardZero) == number.doubleValue,number.doubleValue>=0,number.doubleValue<Double(Int.max) else { return nil }
      return number.intValue
    }
    func ints(_ value:Any?) -> [Int]? { (value as? [Any]).flatMap { values in let result=values.compactMap(integral);return result.count == values.count ? result:nil } }
    guard integral(decoder["in_channels"]) == 128,integral(decoder["out_channels"]) == 3,
      integral(decoder["patch_size"]) == 4,integral(decoder["head_dim"]) == 64,
      ints(decoder["stage_channels"]) == MLXDiffusionVideoPlan.channels,
      ints(decoder["stage_depths"]) == MLXDiffusionVideoPlan.depths,
      (decoder["stage_kernels"] as? [Any])?.compactMap(ints) == MLXDiffusionVideoPlan.kernels,
      ints(decoder["stage5_kernel"]) == [11,11,11],
      integral(decoder["default_num_inference_steps"]) == 1,
      integral(decoder["timestep_scale_multiplier"]) == 1000,
      decoder["stage5_channels"] == nil || decoder["stage5_channels"] is NSNull || integral(decoder["stage5_channels"]) == 256,
      decoder["resampler_kind"] as? String == "linear",decoder["spatial_padding_mode"] as? String == "zeros",
      let upsample=decoder["upsamples"] as? [[Any]],upsample.count == 4,
      zip(upsample.indices,upsample).allSatisfy({ index,row in row.count == 2 && ints(row[0]) == MLXDiffusionVideoPlan.strides[index] && integral(row[1]) == MLXDiffusionVideoPlan.reductions[index] }) else {
      throw LTXError.invalid("Unsupported Diffusion VAE architecture or step/output configuration.")
    }
    var expected=Self.shapes
    // Historical static gates are compatible only at their exact projection
    // output width. Folding happens FP32→stored dtype before the linear op.
    for key in file.tensors.keys where key.hasPrefix("decoder.") && ["gate_ctx","gate_msa","gate_mlp"].contains(String(key.split(separator:".").last!)) {
      let stem=String(key.dropLast(key.split(separator:".").last!.count+1))
      guard stem.hasPrefix("decoder.diff_blocks."),let block=Int(stem.split(separator:".").last!), (0..<8).contains(block) else { throw LTXError.invalid("Unexpected legacy Diffusion VAE gate.") }
      expected[key]=[256]
    }
    let keys=Set(file.tensors.keys.filter { $0.hasPrefix("decoder.") || $0.hasPrefix("per_channel_statistics.") })
    guard keys == Set(expected.keys) else { throw LTXError.invalid("Incomplete or unexpected Diffusion VAE decoder subtree.") }
    var total:UInt64=0
    for (name,shape) in expected {
      try Task.checkCancellation()
      guard let tensor=file.tensors[name],tensor.shape == shape.map(UInt64.init),tensor.dtype == "BF16" else {
        throw LTXError.invalid("Diffusion VAE tensor shape/dtype mismatch: \(name)")
      }
      total+=tensor.byteCount
    }
    try file.checkUnchanged(at:checkpoint)
    self.url=checkpoint;self.file=file;self.decoderTensorBytes=total
  }
  static var shapes:[String:[Int]] {
    var result:[String:[Int]]=[:]
    func linear(_ name:String,_ output:Int,_ input:Int,_ bias:Bool=true) {
      result[name+".weight"]=[output,input];if bias { result[name+".bias"]=[output] }
    }
    func block(_ name:String,_ dim:Int) {
      result[name+".norm1.weight"]=[dim];result[name+".norm2.weight"]=[dim]
      result[name+".attn.q_norm.weight"]=[64];result[name+".attn.k_norm.weight"]=[64]
      linear(name+".attn.qkv",3*dim,dim);linear(name+".attn.proj",dim,dim)
      linear(name+".mlp.w_gate",4*dim,dim,false);linear(name+".mlp.w_up",4*dim,dim,false);linear(name+".mlp.w_down",dim,4*dim,false)
    }
    linear("decoder.conv_in",2048,128);linear("decoder.conv_in_x_t",256,48);linear("decoder.conv_out",48,256)
    linear("decoder.t_embedder.mlp.0",384,256);linear("decoder.t_embedder.mlp.2",384,384);linear("decoder.shared_adaln.proj",1792,384)
    result["decoder.norm_out.weight"]=[256];result["decoder.type_emb"]=[128]
    for i in 0..<4 {
      for b in 0..<MLXDiffusionVideoPlan.depths[i] { block("decoder.det_stages.\(i).\(b)",MLXDiffusionVideoPlan.channels[i]) }
      let dim=MLXDiffusionVideoPlan.channels[i],out=dim/MLXDiffusionVideoPlan.reductions[i]*MLXDiffusionVideoPlan.strides[i].reduce(1,*)
      linear("decoder.upsamples.\(i).proj",out,dim)
    }
    for b in 0..<8 {
      let name="decoder.diff_blocks.\(b)";block(name,256);linear(name+".context_proj",256,256);result[name+".scale_shift_table"]=[7,256]
    }
    result["per_channel_statistics.mean-of-means"]=[128];result["per_channel_statistics.std-of-means"]=[128]
    return result
  }
}
