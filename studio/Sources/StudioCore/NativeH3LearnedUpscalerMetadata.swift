import CoreFoundation
import CryptoKit
import Darwin
import Foundation

/// Model-free editor admission; the worker independently validates execution.
public enum NativeH3LearnedUpscalerMetadata {
  public struct Inspection:Equatable {
    public let path:String
    /// Same engine pin domain: 8-byte SafeTensors prefix followed by JSON header.
    public let headerSHA256:String
    public let tensorBytes:UInt64
  }
  static var expectedShapes:[String:[Int]] {
    var shapes:[String:[Int]]=["conv_in.bias":[512],"conv_in.weight":[512,24,3,3,3],
      "conv_out.bias":[24],"conv_out.weight":[24,512,3,3,3],
      "embed.0.bias":[64],"embed.0.weight":[64,1],"embed.2.bias":[64],"embed.2.weight":[64,64],
      "norm_out.bias":[512],"norm_out.weight":[512]]
    for family in ["in_blocks","out_blocks"] {
      for index in 0..<18 {
        let entries:[String:[Int]]=index%3==1
          ? ["dwconv.bias":[512],"dwconv.weight":[512,1,5,1,1],"norm.bias":[512],"norm.weight":[512],
             "pwconv.bias":[512],"pwconv.weight":[512,512,1,1,1]]
          : ["emb_layers.1.bias":[1024],"emb_layers.1.weight":[1024,64],"in_layers.0.bias":[512],"in_layers.0.weight":[512],
             "in_layers.2.bias":[512],"in_layers.2.weight":[512,512,3,3,3],
             "out_layers.2.bias":[512],"out_layers.2.weight":[512,512,3,3,3],"out_norm.bias":[512],"out_norm.weight":[512]]
        for (key,shape) in entries {shapes["\(family).\(index).\(key)"]=shape}
      }
    };return shapes
  }
  public static func inspect(path:String,expectedHeaderSHA256:String?=nil) throws -> Inspection {
    let url=URL(fileURLWithPath:try H3JointLatentArtifact.localPath(path)).standardizedFileURL.resolvingSymlinksInPath()
    if let expectedHeaderSHA256 {guard H3JointLatentArtifact.validSHA(expectedHeaderSHA256) else {throw StudioError.invalid("Invalid H3 learned-upscaler header identity.")}}
    func invalid()->StudioError { .invalid("Select a compatible 24-channel H3 BF16 learned latent upscaler; its bounded checkpoint header or identity is invalid.") }
    let descriptor=Darwin.open(url.path,O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard descriptor>=0 else {throw invalid()}
    let file=FileHandle(fileDescriptor:descriptor,closeOnDealloc:true);defer {try? file.close()}
    var before=stat()
    guard fstat(descriptor,&before)==0,before.st_mode & S_IFMT == S_IFREG,before.st_size>8,
      let prefix=try file.read(upToCount:8),prefix.count==8 else {throw invalid()}
    let length=prefix.enumerated().reduce(UInt64(0)) {$0 | UInt64($1.element) << (8*$1.offset)}
    guard length>1,length<=1_048_576,UInt64(before.st_size)>=length+8,
      let bytes=try file.read(upToCount:Int(length)),bytes.count==Int(length),
      let header=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {throw invalid()}
    let expected=expectedShapes
    guard Set(header.keys).subtracting(["__metadata__"])==Set(expected.keys),
      header["__metadata__"]==nil || header["__metadata__"] as? [String:String] != nil else {throw invalid()}
    func unsigned(_ value:Any) throws -> UInt64 {
      guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
        n.doubleValue>=0,n.doubleValue.rounded()==n.doubleValue,n.doubleValue<9_007_199_254_740_992 else {throw invalid()};return n.uint64Value
    }
    var spans:[(UInt64,UInt64)]=[]
    for (name,shape) in expected {
      try Task.checkCancellation()
      guard let tensor=header[name] as? [String:Any],Set(tensor.keys)==["dtype","shape","data_offsets"],
        tensor["dtype"] as? String=="BF16",let actual=tensor["shape"] as? [Any],actual.count==shape.count,
        let offsets=tensor["data_offsets"] as? [Any],offsets.count==2 else {throw invalid()}
      guard try actual.map(unsigned)==shape.map({UInt64($0)}) else {throw invalid()}
      let start=try unsigned(offsets[0]),end=try unsigned(offsets[1]),count=UInt64(shape.reduce(1,*))*2
      guard end>=start,end-start==count,end<=UInt64(before.st_size)-length-8 else {throw invalid()};spans.append((start,end))
    }
    spans.sort {$0.0<$1.0};var end:UInt64=0
    for span in spans {guard span.0==end else {throw invalid()};end=span.1}
    guard end==UInt64(before.st_size)-length-8 else {throw invalid()}
    var after=stat(),current=stat()
    guard fstat(descriptor,&after)==0,lstat(url.path,&current)==0,
      before.st_dev==after.st_dev,before.st_ino==after.st_ino,before.st_size==after.st_size,
      before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,
      current.st_dev==after.st_dev,current.st_ino==after.st_ino,current.st_size==after.st_size,
      current.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,current.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
      current.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,current.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec else {throw invalid()}
    try Task.checkCancellation()
    let digest=SHA256.hash(data:prefix+bytes).map {String(format:"%02x",$0)}.joined()
    guard expectedHeaderSHA256==nil || expectedHeaderSHA256==digest else {throw invalid()}
    return Inspection(path:url.path,headerSHA256:digest,tensorBytes:end)
  }
}
