import CryptoKit
import Darwin
import Foundation
import MLX
import TensorIO

/// Private, ephemeral activation carry. These files contain decoded features,
/// never model parameters; their owner removes only its UUID directory.
final class FlashVSRDecoderCarryStore {
  struct State { let caches:[String:MLXArray];let upsampleStarted:Set<String> }
  private let root:URL
  private let sourceID:String,modelIdentity:String
  private var records:[String:(URL,String,UInt64)]=[:]
  var storedBytes:UInt64 { records.values.reduce(0) { $0+$1.2 } }
  init(parent:URL?=nil,sourceID:String,modelIdentity:String)throws {
    let parent=parent ?? FileManager.default.temporaryDirectory
    guard parent.isFileURL,!sourceID.isEmpty,!modelIdentity.isEmpty else { throw FlashVSRError.invalid("Wan activation carry requires a local spool and source/model identity.") }
    try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:true)
    root=parent.appendingPathComponent("WeeTodd-FlashVSR-carry-"+UUID().uuidString,isDirectory:true)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
    self.sourceID=sourceID;self.modelIdentity=modelIdentity
  }
  deinit { try? FileManager.default.removeItem(at:root) }
  func contains(_ key:String)->Bool { records[key] != nil }
  func save(key:String,caches:[String:MLXArray],upsampleStarted:Set<String>)throws {
    try Task.checkCancellation()
    guard !caches.isEmpty,caches.count<=64 else { throw FlashVSRError.invalid("Wan activation carry has an invalid feature set.") }
    var header:[String:SafeTensorStreamWriter.Tensor]=[:],bytes:UInt64=0
    for (name,value) in caches {
      guard value.ndim==5,value.dtype == .float32 || value.dtype == .bfloat16 else { throw FlashVSRError.invalid("Wan activation carry must be FP32/BF16 five-dimensional features.") }
      let tensor=try SafeTensorStreamWriter.Tensor(dtype:value.dtype == .float32 ? "F32":"BF16",shape:value.shape.map(UInt64.init))
      header[name]=tensor;bytes += tensor.byteCount
    }
    guard bytes<=1024*1024*1024 else { throw FlashVSRError.invalid("Wan tile carry exceeds the 1 GiB per-tile activation bound; reduce the selected tile size.") }
    let filename=SHA256.hash(data:Data(key.utf8)).map { String(format:"%02x",$0) }.joined()
    let target=root.appendingPathComponent(filename+".state"),temporary=root.appendingPathComponent(UUID().uuidString+".partial")
    defer { try? FileManager.default.removeItem(at:temporary) }
    let writer=try SafeTensorStreamWriter(url:temporary,tensors:header,
      metadata:["format":"flashvsr-wan-carry-v1","source":sourceID,"model":modelIdentity,"tile":key,"upsamplers":upsampleStarted.sorted().joined(separator:",")])
    for name in caches.keys.sorted() {
      let value=caches[name]!.reshaped([-1]),width=caches[name]!.dtype == .float32 ? 4:2
      let window=4*1024*1024/width
      for start in stride(from:0,to:value.size,by:window) {
        try Task.checkCancellation()
        let slice=value[start..<min(start+window,value.size)]
        if width==4 { try slice.asArray(Float.self).withUnsafeBytes { try writer.append(tensor:name,bytes:$0) } }
        else { try slice.view(dtype:.uint16).asArray(UInt16.self).withUnsafeBytes { try writer.append(tensor:name,bytes:$0) } }
      }
    }
    let hash=try writer.finish();try Task.checkCancellation()
    guard Darwin.rename(temporary.path,target.path)==0 else { throw FlashVSRError.invalid("Cannot atomically update Wan activation carry; check spool disk space.") }
    records[key]=(target,hash,bytes)
  }
  func load(key:String)throws->State {
    try Task.checkCancellation()
    guard let record=records[key] else { throw FlashVSRError.invalid("Wan activation carry is missing its selected tile.") }
    try FlashVSRCheckpoint.verifyHash(record.0,expected:record.1)
    let file=try SafeTensorFile(url:record.0)
    guard file.metadata["format"]=="flashvsr-wan-carry-v1",file.metadata["source"]==sourceID,
      file.metadata["model"]==modelIdentity,file.metadata["tile"]==key,file.tensors.count<=64 else { throw FlashVSRError.invalid("Wan activation carry identity changed.") }
    var caches:[String:MLXArray]=[:]
    for (name,tensor) in file.tensors {
      guard tensor.shape.count==5,["F32","BF16"].contains(tensor.dtype) else { throw FlashVSRError.invalid("Malformed Wan activation carry tensor.") }
      let value=try file.withTensorBytes(named:name) { bytes in
        tensor.dtype=="F32" ? MLXArray(bytes,tensor.shape.map(Int.init),type:Float.self):MLXArray(bytes,tensor.shape.map(Int.init),type:UInt16.self).view(dtype:.bfloat16)
      }
      eval(value);caches[name]=value
    }
    try file.checkUnchanged(at:record.0)
    let upsampleStarted=Set((file.metadata["upsamplers"] ?? "").split(separator:",").map(String.init))
    return State(caches:caches,upsampleStarted:upsampleStarted)
  }
}
