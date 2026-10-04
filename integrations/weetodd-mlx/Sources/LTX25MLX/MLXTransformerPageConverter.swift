import Foundation
import Darwin
import CoreFoundation
import CryptoKit
import MLX
import TensorIO
import LTX25Engine

/// Native, tensor-at-a-time conversion of released LTX 2.5 transformer weights
/// into the existing affine group-64 Q8 page format. No LoRAs are fused. The
/// caller serializes MLX use and keeps the raw checkpoint immutable throughout.
public enum MLXTransformerPageConverter {
  /// Resolve the closest existing ancestor with POSIX realpath, then append
  /// normalized missing components. Foundation can leave a symlink ancestor
  /// unresolved when the final output does not exist yet.
  public static func canonicalLocalURL(_ url:URL) throws -> URL {
    guard url.isFileURL,url.path.hasPrefix("/"),!url.path.utf8.contains(0),url.path.utf8.count<=4096 else {
      throw LTXError.invalid("Transformer paths must be bounded absolute local paths.")
    }
    var ancestor=url.standardizedFileURL
    var missing:[String]=[]
    while true {
      if let resolved=realpath(ancestor.path,nil) {
        let path=String(cString:resolved);free(resolved)
        // Keep the OS spelling. Foundation standardization/appending can turn
        // /private/var or /private/tmp back into their symlink aliases.
        let suffix=missing.reversed().joined(separator:"/")
        return URL(fileURLWithPath:suffix.isEmpty ? path : (path == "/" ? "/" : path+"/")+suffix)
      }
      let failure=errno
      var info=stat()
      guard failure==ENOENT,lstat(ancestor.path,&info) != 0,errno==ENOENT,
        ancestor.path != "/",!ancestor.lastPathComponent.isEmpty else {
        throw LTXError.invalid("Transformer path has an inaccessible or unresolved ancestor.")
      }
      missing.append(ancestor.lastPathComponent)
      ancestor=ancestor.deletingLastPathComponent()
    }
  }
  public struct SourceIdentity: Codable, Equatable, Sendable {
    public let device:UInt64
    public let inode:UInt64
    public let bytes:UInt64
    public let modifiedSeconds:Int
    public let modifiedNanos:Int
    public let changedSeconds:Int
    public let changedNanos:Int
    public let headerSHA256:String
    fileprivate init(url:URL,headerSHA256:String) throws {
      var s=stat()
      guard fstatat(AT_FDCWD,url.path,&s,0)==0,s.st_mode&S_IFMT==S_IFREG,s.st_size>0 else {
        throw LTXError.invalid("Raw transformer source must be a regular file.")
      }
      device=UInt64(UInt32(bitPattern:s.st_dev));inode=UInt64(s.st_ino);bytes=UInt64(s.st_size)
      modifiedSeconds=s.st_mtimespec.tv_sec;modifiedNanos=s.st_mtimespec.tv_nsec
      changedSeconds=s.st_ctimespec.tv_sec;changedNanos=s.st_ctimespec.tv_nsec
      self.headerSHA256=headerSHA256
    }
  }
  public struct Progress:Sendable {
    public let stage:String
    public let completedTensors:Int
    public let totalTensors:Int
    public let completedPages:Int
    public let totalPages:Int
    public let tensorName:String?
  }
  public final class Plan {
    public let source:URL
    public let sourceIdentity:SourceIdentity
    public let sourceTensorBytes:UInt64
    public let outputTensorBytes:UInt64
    public let maximumWorkingBytes:Int
    public let largestQuantizationInputBytes:UInt64
    public var blockCount:Int { layers.count }
    fileprivate let file:SafeTensorFile
    fileprivate let fixed:[String]
    fileprivate let layers:[[String]]
    fileprivate let metadata:[String:Any]
    fileprivate init(source:URL,file:SafeTensorFile,fixed:[String],layers:[[String]],
      metadata:[String:Any],maximumTensorBytes:Int,maximumWorkingBytes:Int) throws {
      self.source=source;self.file=file;self.fixed=fixed;self.layers=layers;self.metadata=metadata
      self.maximumWorkingBytes=maximumWorkingBytes
      let header=try MLXTransformerPageConverter.headerData(source)
      sourceIdentity=try SourceIdentity(url:source,headerSHA256:MLXTransformerPageConverter.sha(header))
      try file.checkUnchanged(at:source)
      sourceTensorBytes=file.tensors.values.reduce(0) { $0+$1.byteCount }
      var output:UInt64=0,largest:UInt64=0
      for (index,keys) in ([fixed]+layers).enumerated() {
        let page=try MLXTransformerPageConverter.outputDescriptors(file:file,keys:keys,quantize:index>0)
        output += page.values.reduce(0) { $0+$1.byteCount }
      }
      for name in layers.flatMap({ $0 }) where MLXTransformerPageConverter.quantizable(name,file.tensors[name]!) {
        let d=file.tensors[name]!
        let factors=try MLXTransformerPageConverter.outputDescriptors(file:file,keys:[name],quantize:true)
        let packed=factors.values.reduce(0) { $0+$1.byteCount }
        // Known storage reserve, not an OS/allocator peak guarantee. Covers one
        // input, Float32 quantizer scratch, output factors and the write window.
        let reserve=d.byteCount.multipliedReportingOverflow(by:4)
        guard !reserve.overflow,d.byteCount<=UInt64(maximumTensorBytes),
          reserve.partialValue+packed+4*1024*1024<=UInt64(maximumWorkingBytes) else {
          throw LTXError.invalid("Transformer tensor exceeds the bounded quantization working allowance: \(name)")
        }
        largest=max(largest,d.byteCount)
      }
      largestQuantizationInputBytes=largest;outputTensorBytes=output
      try MLXTransformerPageConverter.validateOutputPlan(self)
    }
    public func checkUnchanged() throws {
      try file.checkUnchanged(at:source)
      guard try SourceIdentity(url:source,headerSHA256:sourceIdentity.headerSHA256)==sourceIdentity else {
        throw LTXError.invalid("Raw transformer source identity changed after preflight.")
      }
    }
  }
  /// CPU/header-only admission. No MLX arrays or checkpoint payloads are read.
  public static func preflight(source:URL,maximumTensorBytes:Int=512*1024*1024,
    maximumWorkingBytes:Int=2*1024*1024*1024) throws -> Plan {
    guard source.isFileURL,source.pathExtension=="safetensors",(1...512*1024*1024).contains(maximumTensorBytes),
      (1...8*1024*1024*1024).contains(maximumWorkingBytes) else {
      throw LTXError.invalid("Transformer conversion requires a local safetensors file and bounded allowances.")
    }
    let source=try canonicalLocalURL(source)
    let file=try SafeTensorFile(url:source,maximumHeaderBytes:16*1024*1024)
    let metadata=try decodedMetadata(file.metadata)
    guard metadata["model_version"] as? String == "2.5.0",metadata["weetodd_baked_loras"] == nil,
      let config=metadata["config"] as? [String:Any],let architecture=config["transformer"] as? [String:Any] else {
      throw LTXError.invalid("Expected an unmerged released LTX 2.5 transformer checkpoint.")
    }
    try validateArchitecture(architecture)
    let c=try AVBlockConfiguration(videoTokens:1,audioTokens:1,textTokens:1)
    let blockShapes=expectedBlockShapes(c)
    var fixedShapes=DenoiserLayout.weightShapes(c)
    fixedShapes["keyframes_abs_pos_embedding"]=[1,c.videoDimension]
    return try plan(source:source,file:file,metadata:metadata,blockCount:48,blockShapes:blockShapes,
      fixedShapes:fixedShapes,maximumTensorBytes:maximumTensorBytes,maximumWorkingBytes:maximumWorkingBytes)
  }
  private static func decodedMetadata(_ values:[String:String]) throws -> [String:Any] {
    values.mapValues { text in
      (try? JSONSerialization.jsonObject(with:Data(text.utf8),options:[.fragmentsAllowed])) ?? text
    }
  }
  private static func validateArchitecture(_ a:[String:Any]) throws {
    let expected:[String:Any]=[
      "num_layers":48,"num_attention_heads":32,"audio_num_attention_heads":32,
      "attention_head_dim":128,"audio_attention_head_dim":64,"cross_attention_dim":4096,"audio_cross_attention_dim":2048,
      "in_channels":128,"out_channels":128,"audio_out_channels":128,
      "timestep_scale_multiplier":1000,"av_ca_timestep_scale_multiplier":1000,
      "positional_embedding_theta":10000,"frequencies_precision":"float64",
      "positional_embedding_max_pos":[20,2048,2048],"audio_positional_embedding_max_pos":[20],
      "ff_bias":false,"apply_gated_attention":true,"cross_attention_adaln":true,"use_audio_video_cross_attention":true,
      "rope_type":"split","qk_norm":"rms_norm","norm_eps":1e-6,"activation_fn":"gelu-approximate",
      "attention_bias":true,"double_self_attention":false,"only_cross_attention":false,"share_ff":false,
      "av_cross_ada_norm":true,"norm_elementwise_affine":false]
    for (key,value) in expected {
      guard let actual=a[key],jsonEqual(actual,value) else {
        throw LTXError.invalid("Raw transformer architecture differs at \(key).")
      }
    }
    for (key,value) in ["audio_ff_bias":true,"attention_type":"default","dropout":0] as [String:Any] {
      if let actual=a[key],!jsonEqual(actual,value) { throw LTXError.invalid("Unsupported transformer control: \(key)") }
    }
  }
  private static func jsonEqual(_ a:Any,_ b:Any) -> Bool {
    // Do not let NSNumber bridge true into an architecture integer of one.
    if let x=a as? NSNumber,let y=b as? NSNumber,
      (CFGetTypeID(x)==CFBooleanGetTypeID()) != (CFGetTypeID(y)==CFBooleanGetTypeID()) { return false }
    return NSDictionary(dictionary:["value":a]).isEqual(to:["value":b])
  }
  static func expectedBlockShapes(_ c:AVBlockConfiguration) -> [String:[Int]] {
    var result:[String:[Int]]=[:]
    func linear(_ name:String,_ input:Int,_ output:Int,_ bias:Bool=true) {
      result[name+".weight"]=[output,input]
      if bias { result[name+".bias"]=[output] }
    }
    let vd=c.videoDimension,ad=c.audioDimension
    for (name,q,k,inner) in [("attn1",vd,vd,vd),("audio_attn1",ad,ad,ad),("attn2",vd,vd,vd),
      ("audio_attn2",ad,ad,ad),("audio_to_video_attn",vd,ad,ad),("video_to_audio_attn",ad,vd,ad)] {
      linear(name+".to_q",q,inner);linear(name+".to_k",k,inner);linear(name+".to_v",k,inner)
      linear(name+".to_out",inner,q);linear(name+".to_gate_logits",q,c.heads)
      result[name+".q_norm.weight"]=[inner];result[name+".k_norm.weight"]=[inner]
    }
    for (name,d,bias) in [("ff",vd,false),("audio_ff",ad,true)] {
      linear(name+".proj_in",d,4*d,bias);linear(name+".proj_out",4*d,d,bias)
    }
    for (name,rows,d) in [("scale_shift_table",9,vd),("audio_scale_shift_table",9,ad),
      ("prompt_scale_shift_table",2,vd),("audio_prompt_scale_shift_table",2,ad),
      ("scale_shift_table_a2v_ca_video",5,vd),("scale_shift_table_a2v_ca_audio",5,ad)] { result[name]=[rows,d] }
    return result
  }
  static func plan(source:URL,file:SafeTensorFile,metadata:[String:Any],blockCount:Int,
    blockShapes:[String:[Int]],fixedShapes:[String:[Int]],maximumTensorBytes:Int,maximumWorkingBytes:Int) throws -> Plan {
    guard (1...48).contains(blockCount),!blockShapes.isEmpty else { throw LTXError.invalid("Invalid transformer page layout.") }
    var normalized:[String:String]=[:],fixed:[String]=[],layers=[[String]](repeating:[],count:blockCount)
    for name in file.tensors.keys.sorted() {
      try Task.checkCancellation()
      let d=file.tensors[name]!
      guard ["BF16","F16","F32"].contains(d.dtype),!d.shape.contains(0),
        let key=LTXAdapterCompatibility.normalize(name),normalized.updateValue(name,forKey:key)==nil else {
        throw LTXError.invalid("Raw transformer tensor is unsupported or repeats a normalized target: \(name)")
      }
      if key.hasPrefix("transformer_blocks.") {
        let parts=key.split(separator:".",maxSplits:2,omittingEmptySubsequences:false)
        guard parts.count==3,let index=Int(parts[1]),String(index)==parts[1],layers.indices.contains(index),
          let shape=blockShapes[String(parts[2])],d.shape==shape.map(UInt64.init) else {
          throw LTXError.invalid("Raw transformer block tensor has an invalid index/shape: \(name)")
        }
        layers[index].append(name)
      } else { fixed.append(name) }
    }
    guard (1...1024).contains(fixed.count),layers.allSatisfy({ $0.count==blockShapes.count }) else {
      throw LTXError.invalid("Raw transformer fixed weights or complete contiguous block set is missing.")
    }
    for (key,shape) in fixedShapes {
      guard let name=normalized[key],file.tensors[name]!.shape==shape.map(UInt64.init) else {
        throw LTXError.invalid("Missing or incompatible raw fixed transformer weight: \(key)")
      }
    }
    return try Plan(source:source,file:file,fixed:fixed,layers:layers,metadata:metadata,
      maximumTensorBytes:maximumTensorBytes,maximumWorkingBytes:maximumWorkingBytes)
  }
  static func quantizable(_ name:String,_ d:TensorDescriptor) -> Bool {
    name.hasSuffix(".weight") && d.shape.count==2 && d.shape[1]%64==0
  }
  private static func outputDescriptors(file:SafeTensorFile,keys:[String],quantize:Bool) throws -> [String:SafeTensorStreamWriter.Tensor] {
    var result:[String:SafeTensorStreamWriter.Tensor]=[:]
    for name in keys {
      let d=file.tensors[name]!
      let descriptors:[String:SafeTensorStreamWriter.Tensor]
      if quantize && quantizable(name,d) {
        let stem=String(name.dropLast(7)),rows=d.shape[0],columns=d.shape[1]
        descriptors=[name:try .init(dtype:"U32",shape:[rows,columns/4]),
          stem+".scales":try .init(dtype:d.dtype,shape:[rows,columns/64]),
          stem+".biases":try .init(dtype:d.dtype,shape:[rows,columns/64])]
      } else { descriptors=[name:try .init(dtype:d.dtype,shape:d.shape)] }
      for (key,value) in descriptors {
        guard result.updateValue(value,forKey:key)==nil else { throw LTXError.invalid("Quantization would overwrite a source tensor.") }
      }
    }
    return result
  }
  private static func sha(_ data:Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
  private static func headerData(_ url:URL) throws -> Data {
    let handle=try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    guard let prefix=try handle.read(upToCount:8),prefix.count==8 else { throw LTXError.invalid("Missing source header length.") }
    let count=prefix.withUnsafeBytes { $0.loadUnaligned(as:UInt64.self).littleEndian }
    guard count<=16*1024*1024,let data=try handle.read(upToCount:Int(count)),data.count==Int(count) else {
      throw LTXError.invalid("Invalid source header identity.")
    }
    return prefix+data
  }
  private static func sourceSHA(_ plan:Plan) throws -> String {
    try plan.checkUnchanged()
    let handle=try FileHandle(forReadingFrom:plan.source);defer { try? handle.close() }
    var digest=SHA256(),bytes:UInt64=0
    while let chunk=try handle.read(upToCount:4*1024*1024),!chunk.isEmpty {
      try Task.checkCancellation();try plan.checkUnchanged();digest.update(data:chunk);bytes += UInt64(chunk.count)
    }
    guard bytes==plan.sourceIdentity.bytes else { throw LTXError.invalid("Source hash byte count changed.") }
    try plan.checkUnchanged()
    return digest.finalize().map { String(format:"%02x",$0) }.joined()
  }
  private static func manifestData(_ plan:Plan,fixed:[String:Any],layers:[[String:Any]],sourceHash:String) throws -> Data {
    let identity=try JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.sourceIdentity))
    let manifest:[String:Any]=["format":"weetodd-ltx25-transformer-paged-q8-v1","kind":"transformer",
      "num_layers":plan.blockCount,"group_size":64,"bits":8,"source":plan.source.lastPathComponent,
      "source_tensor_bytes":plan.sourceTensorBytes,"output_tensor_bytes":plan.outputTensorBytes,
      "metadata":plan.metadata,"fixed":fixed,"layers":layers,
      "conversion_provenance":["implementation":"swift-mlx-affine-q8-v1","source_sha256":sourceHash,
        "source_identity":identity,"maximum_working_bytes":plan.maximumWorkingBytes,"cache_limit_bytes":128*1024*1024]]
    let data=try JSONSerialization.data(withJSONObject:manifest,options:[.sortedKeys,.prettyPrinted])
    guard data.count<=1024*1024 else { throw LTXError.invalid("Converted manifest exceeds the native reader allowance.") }
    return data
  }
  private static func validateOutputPlan(_ plan:Plan) throws {
    var records:[[String:Any]]=[]
    for (index,keys) in ([plan.fixed]+plan.layers).enumerated() {
      let descriptors=try outputDescriptors(file:plan.file,keys:keys,quantize:index>0)
      try SafeTensorStreamWriter.validateHeader(tensors:descriptors,metadata:["format":"mlx"],
        maximumHeaderBytes:index==0 ? 1024*1024 : 512*1024)
      let name=index==0 ? "fixed.safetensors" : String(format:"layer-%03d.safetensors",index-1)
      records.append(["file":"pages/"+name,"tensor_count":descriptors.count,
        "tensor_bytes":descriptors.values.reduce(UInt64(0),{$0+$1.byteCount}),"sha256":String(repeating:"0",count:64)])
    }
    _ = try manifestData(plan,fixed:records[0],layers:Array(records.dropFirst()),sourceHash:String(repeating:"0",count:64))
  }
  /// Publication is one exclusive directory rename. Cancellation/failure never
  /// leaves a usable manifest or overwrites an existing destination.
  @discardableResult public static func convert(_ plan:Plan,destination:URL,
    progress:(Progress) throws -> Void = { _ in }) throws -> URL {
    guard destination.isFileURL,!destination.lastPathComponent.isEmpty,
      ![".","..","/"].contains(destination.lastPathComponent) else { throw LTXError.invalid("Paged destination must be local.") }
    try Task.checkCancellation();try plan.checkUnchanged()
    let parent=try canonicalLocalURL(destination.deletingLastPathComponent())
    let destination=parent.appendingPathComponent(destination.lastPathComponent)
    var existing=stat()
    guard lstat(destination.path,&existing) != 0 && errno==ENOENT else { throw LTXError.invalid("Paged destination already exists or is inaccessible.") }
    try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
    let temporary=parent.appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString)")
    try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:false)
    var published=false
    defer { if !published { try? FileManager.default.removeItem(at:temporary) } }
    let pages=temporary.appendingPathComponent("pages")
    try FileManager.default.createDirectory(at:pages,withIntermediateDirectories:false)
    let stream=Stream.defaultStream,priorLimit=Memory.cacheLimit
    Memory.cacheLimit=128*1024*1024
    defer { stream.synchronize();Memory.clearCache();Memory.cacheLimit=priorLimit }
    let total=plan.file.tensors.count,totalPages=plan.layers.count+1
    var completed=0,completedPages=0
    func report(_ stage:String,_ tensor:String?=nil) throws {
      try Task.checkCancellation();try plan.checkUnchanged()
      try progress(Progress(stage:stage,completedTensors:completed,totalTensors:total,
        completedPages:completedPages,totalPages:totalPages,tensorName:tensor))
      try Task.checkCancellation();try plan.checkUnchanged()
    }
    try report("source_hash")
    let sourceHash=try sourceSHA(plan)
    func writePage(_ name:String,_ keys:[String],quantize:Bool) throws -> [String:Any] {
      let descriptors=try outputDescriptors(file:plan.file,keys:keys,quantize:quantize)
      let url=pages.appendingPathComponent(name)
      let writer=try SafeTensorStreamWriter(url:url,tensors:descriptors,metadata:["format":"mlx"],
        maximumHeaderBytes:quantize ? 512*1024 : 1024*1024)
      for key in keys.sorted() {
        try report(quantize ? "quantize_tensor" : "copy_fixed_tensor",key)
        let d=plan.file.tensors[key]!
        try autoreleasepool {
          if quantize && quantizable(key,d) {
            let input=try MLXWeight.read(plan.file,key)
            guard MLX.isFinite(input).all().item(Bool.self) else { throw LTXError.invalid("Non-finite raw quantization input.") }
            let q=quantized(input,groupSize:64,bits:8,mode:.affine)
            guard let biases=q.biases else { throw LTXError.invalid("Affine Q8 quantization did not produce biases.") }
            eval(q.wq,q.scales,biases)
            let stem=String(key.dropLast(7)),values=[key:q.wq,stem+".scales":q.scales,stem+".biases":biases]
            for outputKey in values.keys.sorted() {
              let array=values[outputKey]!,expected=descriptors[outputKey]!
              let dtype:[String:DType]=["U32":.uint32,"BF16":.bfloat16,"F16":.float16,"F32":.float32]
              guard array.shape==expected.shape.map(Int.init),array.dtype==dtype[expected.dtype] else {
                throw LTXError.invalid("Q8 output differs from its declared shape/dtype.")
              }
              let data=array.asData(access:.noCopyIfContiguous).data
              try data.withUnsafeBytes { raw in
                for start in stride(from:0,to:raw.count,by:4*1024*1024) {
                  try writer.append(tensor:outputKey,bytes:UnsafeRawBufferPointer(rebasing:raw[start..<min(start+4*1024*1024,raw.count)]))
                }
              }
              withExtendedLifetime(array) {}
            }
          } else {
            for start in stride(from:UInt64(0),to:d.byteCount,by:4*1024*1024) {
              try plan.file.withTensorBytes(named:key,range:start..<min(start+4*1024*1024,d.byteCount),access:.buffered) {
                try writer.append(tensor:key,bytes:$0)
              }
            }
          }
        }
        stream.synchronize();Memory.clearCache()
        completed += 1;try report("tensor_complete",key)
      }
      let hash=try writer.finish()
      let verified=try SafeTensorFile(url:url)
      guard verified.tensors.count==descriptors.count,verified.tensors.values.reduce(0,{$0+$1.byteCount})==writer.payloadBytes else {
        throw LTXError.invalid("Converted page validation differs from planned output.")
      }
      completedPages += 1;try report("page_complete")
      return ["file":"pages/"+name,"tensor_count":descriptors.count,"tensor_bytes":writer.payloadBytes,"sha256":hash]
    }
    let fixed=try writePage("fixed.safetensors",plan.fixed,quantize:false)
    var layers:[[String:Any]]=[]
    for (index,keys) in plan.layers.enumerated() { layers.append(try writePage(String(format:"layer-%03d.safetensors",index),keys,quantize:true)) }
    let data=try manifestData(plan,fixed:fixed,layers:layers,sourceHash:sourceHash)
    try data.write(to:temporary.appendingPathComponent("paged_manifest.json"),options:.atomic)
    try report("publish")
    guard renameatx_np(AT_FDCWD,temporary.path,AT_FDCWD,destination.path,UInt32(RENAME_EXCL))==0 else {
      throw LTXError.invalid("Cannot exclusively publish converted transformer pages.")
    }
    published=true
    return destination
  }
}
