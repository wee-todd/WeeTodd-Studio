import CryptoKit
import Foundation
import TensorIO

/// Deterministic preparation of the pinned Ref2VA INT8/ConvRot release.
/// Quantized payloads remain byte-identical; only six fixed matrices are cast.
public enum H3Ref2VABaseConverter {
  public static let version="h3-ref2va-int8-convrot-legacy-bf16-v1"
  public static let sourceSHA256="9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
  public static let sourceBytes:UInt64=34_038_894_550
  public struct Result:Sendable { public let url:URL,sha256:String,payloadBytes:UInt64 }
  static let castShapes:[String:[UInt64]]=[
    "audio_patch_proj.weight":[5376,32],"video_patch_proj.weight":[5376,96],
    "final_layer.audio_out.weight":[32,5376],"final_layer.video_out.weight":[96,5376],
    "time_embedder.proj_in.weight":[5376,256],"time_embedder.proj_out.weight":[2688,5376]]
  static func bfloat16Bits(_ word:UInt32) -> UInt16 {
    if word & 0x7fffffff > 0x7f800000 { return UInt16(truncatingIfNeeded:word >> 16)|0x0040 }
    return UInt16(truncatingIfNeeded:(word &+ 0x7fff &+ ((word >> 16)&1)) >> 16)
  }
  static func preparedTensor(name:String,tensor:H3TensorInfo) throws -> (name:String,info:H3TensorInfo) {
    guard !name.hasPrefix("model."),!name.hasPrefix("diffusion_model."),
      ["F32","BF16","I8","U8"].contains(tensor.dtype) else {
      throw H3CheckpointError.invalid("Ref2VA preparation requires the pinned rootless source namespace and dtypes.")
    }
    var dtype=tensor.dtype
    if let shape=castShapes[name] {
      guard tensor.dtype == "F32",tensor.shape == shape else { throw H3CheckpointError.invalid("Unexpected Ref2VA fixed matrix: \(name)") }
      dtype="BF16"
    }
    return ("model.diffusion_model."+name,H3TensorInfo(dtype:dtype,shape:tensor.shape))
  }
  private static func hash(_ url:URL) throws -> String {
    let handle=try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    var value=SHA256()
    while try autoreleasepool(invoking:{ () throws -> Bool in
      try Task.checkCancellation()
      guard let data=try handle.read(upToCount:4*1024*1024),!data.isEmpty else { return false }
      value.update(data:data);return true
    }) {}
    return value.finalize().map{String(format:"%02x",$0)}.joined()
  }
  public static func convert(source:URL,destination:URL,modelRoot:URL,
    progress:(Int,Int)->Void={_,_ in}) throws -> Result {
    let root=modelRoot.resolvingSymlinksInPath().standardizedFileURL.path
    let output=destination.deletingLastPathComponent().resolvingSymlinksInPath().appendingPathComponent(destination.lastPathComponent).standardizedFileURL
    let manager=FileManager.default
    guard source.isFileURL,modelRoot.isFileURL,destination.isFileURL,output.path.hasPrefix(root+"/"),
      !manager.fileExists(atPath:output.path) else { throw H3CheckpointError.invalid("Ref2VA conversion requires a new output under the selected model library.") }
    let file=try SafeTensorFile(url:source)
    guard file.fileByteCount == sourceBytes,file.tensors.count == 1035,
      try hash(source) == sourceSHA256 else { throw H3CheckpointError.invalid("Ref2VA source differs from the pinned released checkpoint.") }
    try file.checkUnchanged(at:source)
    var tensors=[String:SafeTensorStreamWriter.Tensor](),infos=[String:H3TensorInfo](),sources=[String:String]()
    for (name,tensor) in file.tensors {
      let prepared=try preparedTensor(name:name,tensor:.init(dtype:tensor.dtype,shape:tensor.shape))
      infos[prepared.name]=prepared.info;sources[prepared.name]=name
      tensors[prepared.name]=try .init(dtype:prepared.info.dtype,shape:prepared.info.shape)
    }
    guard Set(castShapes.keys).isSubset(of:Set(file.tensors.keys)) else { throw H3CheckpointError.invalid("Incomplete Ref2VA fixed matrix set.") }
    _ = try H3CheckpointLayout(tensors:infos)
    let staging=output.deletingLastPathComponent().appendingPathComponent(".\(output.lastPathComponent).\(UUID().uuidString).partial")
    defer { try? manager.removeItem(at:staging) }
    let writer=try SafeTensorStreamWriter(url:staging,tensors:tensors,metadata:[
      "format":version,"source_repository":"Comfy-Org/MiniMax-H3",
      "source_revision":"e5eb578a89295337b8ff433a035929ce0279e0b6","source_sha256":sourceSHA256,
      "partition":"Ref2VA","qkv_layout":"contiguous_qkv",
      "preparation":"Legacy namespace; six fixed F32 matrices rounded to BF16; all INT8/scales/markers and remaining tensors unchanged."])
    for (index,native) in tensors.keys.sorted().enumerated() {
      let name=sources[native]!,tensor=file.tensors[name]!
      var offset:UInt64=0
      while offset < tensor.byteCount {
        let end=min(offset+4*1024*1024,tensor.byteCount)
        try autoreleasepool {
          try Task.checkCancellation();try file.checkUnchanged(at:source)
          try file.withTensorBytes(named:name,range:offset..<end,access:.buffered) { bytes in
            if castShapes[name] != nil {
              let values=(0..<bytes.count/4).map { bfloat16Bits(UInt32(littleEndian:bytes.loadUnaligned(fromByteOffset:$0*4,as:UInt32.self))).littleEndian }
              try values.withUnsafeBytes { try writer.append(tensor:native,bytes:$0) }
            } else { try writer.append(tensor:native,bytes:bytes) }
          }
        }
        offset=end
      }
      progress(index+1,tensors.count)
    }
    try file.checkUnchanged(at:source)
    let digest=try writer.finish()
    _ = try H3CheckpointLayout(url:staging)
    try Task.checkCancellation();try file.checkUnchanged(at:source)
    try manager.moveItem(at:staging,to:output)
    return Result(url:output,sha256:digest,payloadBytes:writer.payloadBytes)
  }
}
