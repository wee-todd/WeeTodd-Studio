import Foundation
import CryptoKit
import Darwin
import MLX
import TensorIO

/// Strict native safetensors admission. Weight storage remains source-format and stage-local.
public final class FlashVSRCheckpoint {
  public struct Identity: Sendable, Equatable {
    public let variant: FlashVSRVariant
    public let transformerSHA256: String, projectionSHA256: String, contextSHA256: String
    public init(variant: FlashVSRVariant, transformerSHA256: String, projectionSHA256: String, contextSHA256: String) {
      self.variant=variant; self.transformerSHA256=transformerSHA256; self.projectionSHA256=projectionSHA256; self.contextSHA256=contextSHA256
    }
  }
  public enum Component { case transformer, projection }
  public let config: FlashVSRConfig
  private let file: SafeTensorFile
  private let url: URL
  public let dtype: DType
  public static func releasedDigests(_ variant: FlashVSRVariant) -> (transformer: String, projection: String) {
    switch variant {
    case .turbo: return ("7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b","e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496")
    case .base: return ("58d64c1f1b3a559bafb69bb014f3ab2745e800c89435afdc154966eb96bc7459","a1fe997f19060171db8dd4d58c30debc78ddfcc357b63481798792a38a356d8d")
    }
  }
  public static func validateIdentity(_ identity: Identity, expected: FlashVSRVariant) throws {
    let pins=releasedDigests(expected)
    guard identity.variant == expected, identity.transformerSHA256 == pins.transformer, identity.projectionSHA256 == pins.projection,
      identity.contextSHA256 == TorchTensorImportProfile.positivePrompt.sourceSHA256 else {
      throw FlashVSRError.invalid("FlashVSR requires the paired released DiT/LQ identities and fixed context identity.")
    }
  }
  public static func loadFixedContext(url: URL) throws -> MLXArray {
    let archive=try TorchTensorArchive.open(url:url,expectedSHA256:TorchTensorImportProfile.positivePrompt.sourceSHA256,profile:.positivePrompt)
    guard let tensor=archive.tensors["tensor"], tensor.shape == [1,512,4096], tensor.dtype == "BF16" else { throw FlashVSRError.invalid("Invalid fixed prompt archive tensor.") }
    let result=try archive.withTensorBytes(named:"tensor") { MLXArray($0,[1,512,4096],type:UInt16.self).view(dtype:.bfloat16) }
    eval(result); return result
  }
  public static func validateContext(_ context: MLXArray?) throws {
    guard let context, context.shape == [1,512,4096], context.dtype == .bfloat16 || context.dtype == .float32 else {
      throw FlashVSRError.invalid("FlashVSR requires its fixed positive prompt tensor [1,512,4096].")
    }
  }
  public init(url: URL, variant: FlashVSRVariant, component: Component, dtype: DType = .bfloat16) throws {
    guard dtype == .float32 || dtype == .bfloat16 else { throw FlashVSRError.invalid("FlashVSR supports FP32 fixtures or BF16 execution.") }
    let file=try SafeTensorFile(url:url), pins=Self.releasedDigests(variant)
    try Self.verifyHash(url, expected:component == .transformer ? pins.transformer : pins.projection)
    try file.checkUnchanged(at:url)
    let shapes=Self.expectedShapes(component:component)
    guard Set(file.tensors.keys) == Set(shapes.keys) else { throw FlashVSRError.invalid("FlashVSR checkpoint tensor set differs from the released architecture.") }
    for (name,shape) in shapes {
      guard let desc=file.tensors[name], desc.shape == shape.map(UInt64.init), ["F32","BF16","F16"].contains(desc.dtype) else { throw FlashVSRError.invalid("Invalid FlashVSR tensor: \(name).") }
    }
    self.url=url; self.file=file; config=FlashVSRConfig(variant:variant); self.dtype=dtype
  }
  public func tensor(_ name: String) throws -> MLXArray {
    try file.checkUnchanged(at:url)
    guard let desc=file.tensors[name] else { throw FlashVSRError.invalid("Missing FlashVSR tensor \(name).") }
    let shape=desc.shape.map(Int.init)
    let value=try file.withTensorBytes(named:name) { bytes -> MLXArray in
      switch desc.dtype {
      case "F32": return MLXArray(bytes,shape,type:Float.self)
      case "F16": return MLXArray(bytes,shape,type:Float16.self)
      case "BF16": return MLXArray(bytes,shape,type:UInt16.self).view(dtype:.bfloat16)
      default: throw FlashVSRError.invalid("Unsupported FlashVSR dtype.")
      }
    }.asType(dtype)
    eval(value)
    return value
  }
  public static func verifyHash(_ url: URL, expected: String) throws {
    guard url.isFileURL else { throw FlashVSRError.invalid("FlashVSR weights must be local regular files.") }
    let stream=try FileHandle(forReadingFrom:url); defer { try? stream.close() }
    var before=stat(), pathname=stat()
    func unchanged(_ a:stat,_ b:stat) -> Bool {
      a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
      a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
      a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
    }
    guard fstat(stream.fileDescriptor,&before) == 0, before.st_mode & S_IFMT == S_IFREG,
      fstatat(AT_FDCWD,url.path,&pathname,0) == 0, unchanged(before,pathname) else { throw FlashVSRError.invalid("FlashVSR source changed before hashing.") }
    var hash=SHA256()
    while try autoreleasepool(invoking: { () throws -> Bool in
      try Task.checkCancellation()
      guard let data=try stream.read(upToCount:4*1024*1024), !data.isEmpty else { return false }
      hash.update(data:data); return true
    }) {}
    var after=stat(), finalPath=stat()
    guard fstat(stream.fileDescriptor,&after) == 0, fstatat(AT_FDCWD,url.path,&finalPath,0) == 0,
      unchanged(before,after), unchanged(before,finalPath) else { throw FlashVSRError.invalid("FlashVSR source changed during hashing.") }
    guard hash.finalize().map({ String(format:"%02x",$0) }).joined() == expected else { throw FlashVSRError.invalid("FlashVSR checkpoint content identity does not match the selected release.") }
  }
  static func expectedShapes(component: Component) -> [String:[Int]] {
    if component == .projection {
      return ["conv1.weight":[2048,768,4,3,3],"conv1.bias":[2048],"norm1.weight":[2048,1,1,1],"conv2.weight":[3072,2048,4,3,3],"conv2.bias":[3072],"norm2.weight":[3072,1,1,1],"linear_layers.0.weight":[1536,3072],"linear_layers.0.bias":[1536]]
    }
    var shapes: [String:[Int]]=["patch_embedding.weight":[1536,16,1,2,2],"patch_embedding.bias":[1536],"head.modulation":[1,2,1536]]
    func linear(_ name:String,_ input:Int,_ output:Int) { shapes[name+".weight"]=[output,input]; shapes[name+".bias"]=[output] }
    linear("text_embedding.0",4096,1536); linear("text_embedding.2",1536,1536)
    linear("time_embedding.0",256,1536); linear("time_embedding.2",1536,1536); linear("time_projection.1",1536,9216); linear("head.proj",1536,64)
    for layer in 0..<30 {
      let p="blocks.\(layer)."
      for attention in ["self_attn","cross_attn"] {
        for key in ["q","k","v","o"] { linear(p+attention+"."+key,1536,1536) }
        for key in ["norm_q","norm_k"] { shapes[p+attention+"."+key+".weight"]=[1536] }
      }
      linear(p+"ffn.0",1536,8960); linear(p+"ffn.2",8960,1536)
      shapes[p+"norm3.weight"]=[1536]; shapes[p+"norm3.bias"]=[1536]; shapes[p+"modulation"]=[1,6,1536]
    }
    return shapes
  }
}
