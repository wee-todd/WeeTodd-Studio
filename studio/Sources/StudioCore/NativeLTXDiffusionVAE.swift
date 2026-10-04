import Foundation
import CoreFoundation
import CryptoKit
import Darwin

public enum LTX25DiffusionVAEOptimization:String,Codable,CaseIterable,Sendable {
  case combined
  case deferredStage4="deferred_stage4"
  case stage4WidthTiles="stage4_width_tiles"
  case metalNA3D="metal_na3d_experimental"
  case metalNA3DQueryTiled="metal_na3d_query_tiled_experimental"
}
public struct LTX25DiffusionVAESettings:Codable,Equatable,Sendable {
  public var experimentalEnabled:Bool
  public var optimization:LTX25DiffusionVAEOptimization
  public var queryChunkSize:Int,contextWidthChunks:Int,stage4TileWidth:Int
  public init(experimentalEnabled:Bool=false,optimization:LTX25DiffusionVAEOptimization = .combined,
    queryChunkSize:Int=512,contextWidthChunks:Int=4,stage4TileWidth:Int=0) {
    self.experimentalEnabled=experimentalEnabled;self.optimization=optimization
    self.queryChunkSize=queryChunkSize;self.contextWidthChunks=contextWidthChunks;self.stage4TileWidth=stage4TileWidth
  }
}
/// Header-only admission. Tensor payloads are never read or copied by Studio.
public enum NativeLTXDiffusionVAE {
  public static let prefix="decoder."
  static func integer(_ value:Any?) -> Int? {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
      n.doubleValue>=0,n.doubleValue<Double(Int.max),n.doubleValue.rounded(.towardZero)==n.doubleValue else { return nil }
    return n.intValue
  }
  static func dimensions(_ value:Any?) -> [Int]? {
    guard let a=value as? [Any] else { return nil };let values=a.compactMap(integer)
    return values.count == a.count ? values:nil
  }
  public static func matches(metadata:[String:String],tensors:[String:Any]) -> Bool {
    guard metadata["model_version"] == "2.5.0",let raw=metadata["config"],raw.utf8.count<=1024*1024,
      let config=(try? JSONSerialization.jsonObject(with:Data(raw.utf8))) as? [String:Any],
      let vae=config["vae"] as? [String:Any],let d=vae["decoder"] as? [String:Any],
      d["_class_name"] as? String == "NADiffusionDecoder",vae["model_output_type"] as? String == "x0",
      integer(d["in_channels"]) == 128,integer(d["out_channels"]) == 3,integer(d["patch_size"]) == 4,
      integer(d["head_dim"]) == 64,integer(d["default_num_inference_steps"]) == 1,
      integer(d["timestep_scale_multiplier"]) == 1000,dimensions(d["stage_channels"]) == [2048,1024,512,512,256],
      dimensions(d["stage_depths"]) == [4,6,4,2,8],dimensions(d["stage5_kernel"]) == [11,11,11],
      (d["stage_kernels"] as? [Any])?.compactMap(dimensions) == [[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]],
      d["resampler_kind"] as? String == "linear",d["spatial_padding_mode"] as? String == "zeros",
      let up=d["upsamples"] as? [[Any]],up.count == 4,
      zip(up.indices,up).allSatisfy({ i,a in a.count == 2 && dimensions(a[0]) == [[1,2,2],[2,1,1],[2,2,2],[2,2,2]][i] && integer(a[1]) == [2,2,1,2][i] }),
      d["stage5_channels"] == nil || d["stage5_channels"] is NSNull || integer(d["stage5_channels"]) == 256 else { return false }
    var expected=shapes
    for name in tensors.keys where name.hasPrefix(prefix) && ["gate_ctx","gate_msa","gate_mlp"].contains(String(name.split(separator:".").last ?? "")) {
      let parts=name.split(separator:".")
      guard parts.count == 4,parts[1] == "diff_blocks",let block=Int(parts[2]),(0..<8).contains(block) else { return false }
      expected[name]=[256]
    }
    guard Set(tensors.keys.filter { $0.hasPrefix(prefix) || $0.hasPrefix("per_channel_statistics.") }) == Set(expected.keys) else { return false }
    return expected.allSatisfy { name,shape in
      guard let tensor=tensors[name] as? [String:Any] else { return false }
      return tensor["dtype"] as? String == "BF16" && dimensions(tensor["shape"]) == shape
    }
  }
  public static func validate(at input:URL) throws -> String {
    guard input.isFileURL,input.path.utf8.count<=4096,!input.path.utf8.contains(0) else { throw StudioError.invalid("Choose a local Diffusion VAE checkpoint.") }
    try Task.checkCancellation()
    let url=input.resolvingSymlinksInPath().standardizedFileURL
    let fd=Darwin.open(url.path,O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_NOFOLLOW)
    guard fd>=0 else { throw StudioError.invalid("Relink the Diffusion VAE checkpoint.") }
    let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? file.close() }
    var before=stat(),after=stat(),current=stat()
    guard fstat(fd,&before) == 0,before.st_mode&S_IFMT == S_IFREG,before.st_size>8,
      let prefix=try file.read(upToCount:8),prefix.count == 8 else { throw StudioError.invalid("Diffusion VAE must be a regular safetensors file.") }
    let length=prefix.enumerated().reduce(UInt64(0)) { $0|UInt64($1.element)<<(8*$1.offset) }
    guard length>0,length<=1024*1024,length+8<=UInt64(before.st_size),
      let raw=try file.read(upToCount:Int(length)),raw.count==Int(length),
      let header=try JSONSerialization.jsonObject(with:raw) as? [String:Any],let metadata=header["__metadata__"] as? [String:String],
      matches(metadata:metadata,tensors:header.filter { $0.key != "__metadata__" }) else { throw StudioError.invalid("Choose the released compatible one-step LTX2.5 Diffusion VAE.") }
    var spans:[(Int,Int)]=[]
    for (name,value) in header where name != "__metadata__" {
      try Task.checkCancellation()
      guard let tensor=value as? [String:Any],tensor["dtype"] as? String == "BF16",let shape=dimensions(tensor["shape"]),!shape.isEmpty,
        let offsets=dimensions(tensor["data_offsets"]),offsets.count==2,offsets[1]>=offsets[0] else { throw StudioError.invalid("DiffVAE tensor descriptor is malformed.") }
      var bytes=2
      for dimension in shape { let product=bytes.multipliedReportingOverflow(by:dimension);guard dimension>0,!product.overflow else { throw StudioError.invalid("DiffVAE tensor shape overflows.") };bytes=product.partialValue }
      guard offsets[1]-offsets[0]==bytes else { throw StudioError.invalid("DiffVAE tensor payload bounds differ from its shape.") };spans.append((offsets[0],offsets[1]))
    }
    var end=0
    for span in spans.sorted(by:{ $0.0<$1.0 }) { guard span.0==end else { throw StudioError.invalid("DiffVAE payload is overlapping or incomplete.") };end=span.1 }
    guard UInt64(end)+length+8==UInt64(before.st_size),fstat(fd,&after)==0,lstat(url.path,&current)==0,
      after.st_size==before.st_size,after.st_mtimespec.tv_sec==before.st_mtimespec.tv_sec,after.st_mtimespec.tv_nsec==before.st_mtimespec.tv_nsec,
      after.st_ctimespec.tv_sec==before.st_ctimespec.tv_sec,after.st_ctimespec.tv_nsec==before.st_ctimespec.tv_nsec,
      current.st_dev==before.st_dev,current.st_ino==before.st_ino else { throw StudioError.invalid("DiffVAE checkpoint changed during header admission.") }
    return SHA256.hash(data:raw).map { String(format:"%02x",$0) }.joined()
  }
  public static func wire(_ settings:LTX25DiffusionVAESettings,checkpoint:URL) throws -> [String:Any] {
    guard settings.experimentalEnabled,(1...65_536).contains(settings.queryChunkSize),(1...4096).contains(settings.contextWidthChunks),
      (0...4096).contains(settings.stage4TileWidth),(settings.optimization == .stage4WidthTiles ? settings.stage4TileWidth>0 : settings.stage4TileWidth==0) else {
      throw StudioError.invalid("Enable experimental Diffusion VAE explicitly and select valid query/context/width controls.")
    }
    _ = try validate(at:checkpoint)
    return ["optimization":settings.optimization.rawValue,"query_chunk_size":settings.queryChunkSize,
      "context_width_chunks":settings.contextWidthChunks,"stage4_tile_width":settings.stage4TileWidth]
  }
  public static func apply(_ settings:LTX25DiffusionVAESettings?,checkpoint:URL,config:[String:Any]) throws -> [String:Any] {
    guard let settings else { return config }
    let wire=try wire(settings,checkpoint:checkpoint)
    var result=config
    result["diffvae_optimization"]=wire["optimization"];result["diffvae_query_chunk_size"]=wire["query_chunk_size"]
    result["diffvae_context_width_chunks"]=wire["context_width_chunks"];result["diffvae_stage4_tile_width"]=wire["stage4_tile_width"]
    return result
  }
  public static func profileControls(_ config:[String:Any]) -> [String:Any] {
    let defaults:[String:Any] = ["diffvae_optimization":"combined","diffvae_query_chunk_size":512,
      "diffvae_context_width_chunks":4,"diffvae_stage4_tile_width":0]
    return config.filter { key,value in
      guard let expected=defaults[key] else { return false }
      return !NSDictionary(dictionary:["value":value]).isEqual(to:["value":expected])
    }
  }
  public static func resolvedWire(_ settings:LTX25DiffusionVAESettings?,profile:[String:Any],checkpoint:URL) throws -> [String:Any]? {
    if let settings { return try wire(settings,checkpoint:checkpoint) }
    guard profile.isEmpty else {
      throw StudioError.invalid("Ripple cannot ignore custom profile Diffusion VAE controls; enable the matching advanced decoder controls explicitly.")
    }
    return nil
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
      for b in 0..<[4,6,4,2,8][i] { block("decoder.det_stages.\(i).\(b)",[2048,1024,512,512,256][i]) }
      let dim=[2048,1024,512,512,256][i],out=dim/[2,2,1,2][i]*[[1,2,2],[2,1,1],[2,2,2],[2,2,2]][i].reduce(1,*)
      linear("decoder.upsamples.\(i).proj",out,dim)
    }
    for b in 0..<8 {
      let name="decoder.diff_blocks.\(b)";block(name,256);linear(name+".context_proj",256,256);result[name+".scale_shift_table"]=[7,256]
    }
    result["per_channel_statistics.mean-of-means"]=[128];result["per_channel_statistics.std-of-means"]=[128]
    return result
  }
}
