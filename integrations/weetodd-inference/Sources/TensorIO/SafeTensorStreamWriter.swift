import Foundation
import Darwin
import CryptoKit

/// Sequential, exclusive safetensors output with a predeclared header. Tensor
/// payloads are supplied in sorted-name order through at most 4 MiB windows.
/// A failed/incomplete writer removes its output; complete pages are immutable.
public final class SafeTensorStreamWriter {
  public struct Tensor: Sendable {
    public let dtype:String
    public let shape:[UInt64]
    public let byteCount:UInt64
    public init(dtype:String,shape:[UInt64]) throws {
      let widths:[String:UInt64]=["F32":4,"F16":2,"BF16":2,"U32":4,"U8":1]
      guard let width=widths[dtype],shape.count<=16 else {
        throw CheckpointError.invalid("Unsupported streaming tensor dtype or rank.")
      }
      var count:UInt64=shape.contains(0) ? 0 : 1
      for dimension in shape {
        guard dimension<=UInt64(Int.max) else { throw CheckpointError.invalid("Unaddressable tensor dimension.") }
        let product=count.multipliedReportingOverflow(by:dimension)
        guard !product.overflow else { throw CheckpointError.invalid("Streaming tensor shape overflows.") }
        count=product.partialValue
      }
      let bytes=count.multipliedReportingOverflow(by:width)
      guard !bytes.overflow else { throw CheckpointError.invalid("Streaming tensor bytes overflow.") }
      self.dtype=dtype;self.shape=shape;byteCount=bytes.partialValue
    }
  }
  public let payloadBytes:UInt64
  private let url:URL
  private let names:[String]
  private let tensors:[String:Tensor]
  private var fd:Int32 = -1
  private var index=0
  private var offset:UInt64=0
  private var completed=false
  private var created=false
  private var hash=SHA256()
  /// Runs the same header/offset admission as construction, without creating
  /// an output file or reading any payload. Converters use it before work.
  public static func validateHeader(tensors:[String:Tensor],metadata:[String:String]=[:],
    maximumHeaderBytes:Int=1024*1024) throws {
    _ = try makeHeader(tensors:tensors,metadata:metadata,maximumHeaderBytes:maximumHeaderBytes)
  }
  private static func makeHeader(tensors:[String:Tensor],metadata:[String:String],
    maximumHeaderBytes:Int) throws -> (data:Data,payloadBytes:UInt64) {
    guard !tensors.isEmpty,tensors.keys.allSatisfy({ !$0.isEmpty && $0 != "__metadata__" }),
      (1...64*1024*1024).contains(maximumHeaderBytes) else {
      throw CheckpointError.invalid("Invalid streaming safetensors header allowance.")
    }
    var cursor:UInt64=0,header:[String:Any]=["__metadata__":metadata]
    for name in tensors.keys.sorted() {
      let tensor=tensors[name]!,end=cursor.addingReportingOverflow(tensor.byteCount)
      guard !end.overflow,end.partialValue<=UInt64(Int64.max)-UInt64(maximumHeaderBytes)-8 else {
        throw CheckpointError.invalid("Streaming page byte offsets overflow.")
      }
      header[name]=["dtype":tensor.dtype,"shape":tensor.shape,"data_offsets":[cursor,end.partialValue]]
      cursor=end.partialValue
    }
    var data=try JSONSerialization.data(withJSONObject:header,options:[.sortedKeys])
    data.append(contentsOf:repeatElement(UInt8(32),count:(8-data.count%8)%8))
    guard data.count<=maximumHeaderBytes else { throw CheckpointError.invalid("Streaming header exceeds allowance.") }
    return (data,cursor)
  }
  public init(url:URL,tensors:[String:Tensor],metadata:[String:String]=[:],maximumHeaderBytes:Int=1024*1024) throws {
    guard url.isFileURL else { throw CheckpointError.invalid("Streaming output must be local.") }
    let planned=try Self.makeHeader(tensors:tensors,metadata:metadata,maximumHeaderBytes:maximumHeaderBytes)
    self.url=url;self.tensors=tensors;names=tensors.keys.sorted()
    payloadBytes=planned.payloadBytes
    let data=planned.data
    fd=Darwin.open(url.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,S_IRUSR|S_IWUSR)
    guard fd>=0 else { throw CheckpointError.invalid("Cannot create exclusive streaming tensor output.") }
    created=true
    var ready=false
    defer { if !ready { Darwin.close(fd);fd = -1;Darwin.unlink(url.path);created=false } }
    var length=UInt64(data.count).littleEndian
    try withUnsafeBytes(of:&length) { try write($0) }
    try data.withUnsafeBytes { try write($0) }
    ready=true
  }
  deinit {
    if fd>=0 { Darwin.close(fd) }
    if created && !completed { Darwin.unlink(url.path) }
  }
  private func write(_ bytes:UnsafeRawBufferPointer) throws {
    try Task.checkCancellation()
    var done=0
    while done<bytes.count {
      let n=Darwin.write(fd,bytes.baseAddress!.advanced(by:done),bytes.count-done)
      if n<0 && errno==EINTR { continue }
      guard n>0 else { throw CheckpointError.invalid("Cannot write streaming tensor output.") }
      done += n
    }
    hash.update(bufferPointer:bytes)
  }
  public func append(tensor name:String,bytes:UnsafeRawBufferPointer) throws {
    guard !completed,fd>=0,index<names.count,names[index]==name,bytes.count<=4*1024*1024,
      UInt64(bytes.count)<=tensors[name]!.byteCount-offset else {
      throw CheckpointError.invalid("Streaming tensor order/window/byte count differs from its header.")
    }
    try write(bytes);offset += UInt64(bytes.count)
    if offset==tensors[name]!.byteCount { index += 1;offset=0 }
  }
  /// Flush and validate the actual header/payload before returning its SHA-256.
  public func finish() throws -> String {
    guard !completed,fd>=0,index==names.count,offset==0 else {
      throw CheckpointError.invalid("Streaming tensor page is incomplete.")
    }
    try Task.checkCancellation()
    guard fsync(fd)==0 else { throw CheckpointError.invalid("Cannot synchronize tensor page.") }
    let closing=fd;fd = -1
    guard Darwin.close(closing)==0 else { throw CheckpointError.invalid("Cannot close tensor page.") }
    _ = try SafeTensorFile(url:url)
    completed=true
    return hash.finalize().map { String(format:"%02x",$0) }.joined()
  }
}
