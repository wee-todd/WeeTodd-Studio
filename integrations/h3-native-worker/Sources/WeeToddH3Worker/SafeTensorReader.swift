import Foundation
import Darwin
import Accelerate

enum ProbeError: Error, CustomStringConvertible {
  case invalid(String)
  var description: String { switch self { case .invalid(let reason): return reason } }
}
struct TensorRecord: Decodable {
  let dtype: String
  let shape: [Int]
  let data_offsets: [Int]
  var count: Int { shape.reduce(1, *) }
}
struct FloatTensor { let shape: [Int]; var values: [Float] }
final class SafeTensorReader {
  let path: String
  let fd: Int32
  let records: [String: TensorRecord]
  let offset: Int
  let size: Int
  let identity: [Int64]
  private static func stamp(_ s: stat) -> [Int64] {
    [Int64(s.st_dev),Int64(s.st_ino),s.st_size,Int64(s.st_mtimespec.tv_sec),Int64(s.st_mtimespec.tv_nsec),Int64(s.st_ctimespec.tv_sec),Int64(s.st_ctimespec.tv_nsec)]
  }
  init(_ path: String) throws {
    let opened = Darwin.open(path, O_RDONLY)
    guard opened >= 0 else { throw ProbeError.invalid("Cannot open tensor file") }
    var success=false
    defer { if !success { Darwin.close(opened) } }
    var st=stat(); guard fstat(opened,&st)==0 else { throw ProbeError.invalid("Cannot stat tensor file") }
    let lengthData=try Self.bytes(opened,0,8)
    let n=lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard n>=2 && n<=16*1024*1024 && n+8<=UInt64(st.st_size) else { throw ProbeError.invalid("Invalid safetensors header size") }
    let raw=try Self.bytes(opened,8,Int(n))
    guard var header=try JSONSerialization.jsonObject(with:raw) as? [String:Any] else { throw ProbeError.invalid("Invalid safetensors header") }
    header.removeValue(forKey:"__metadata__")
    var parsed: [String:TensorRecord]=[:]
    let widths=["F32":4,"F16":2,"BF16":2,"I8":1,"U8":1,"I32":4]
    for (name,value) in header {
      guard value is [String:Any] else { throw ProbeError.invalid("Invalid tensor record: \(name)") }
      let record=try JSONDecoder().decode(TensorRecord.self,from:JSONSerialization.data(withJSONObject:value))
      guard let width=widths[record.dtype], record.data_offsets.count==2, record.shape.count<=8 else { throw ProbeError.invalid("Unsupported tensor record: \(name)") }
      var count=1
      for dim in record.shape {
        let product=count.multipliedReportingOverflow(by:dim)
        guard dim>=0 && !product.overflow else { throw ProbeError.invalid("Invalid tensor shape") }
        count=product.partialValue
      }
      let bytes=count.multipliedReportingOverflow(by:width)
      let lo=record.data_offsets[0], hi=record.data_offsets[1]
      guard !bytes.overflow,lo>=0,hi>=lo,hi-lo==bytes.partialValue,hi<=Int(st.st_size)-Int(n)-8 else { throw ProbeError.invalid("Invalid tensor span: \(name)") }
      parsed[name]=record
    }
    var cursor=0
    for rec in parsed.values.sorted(by:{$0.data_offsets[0]<$1.data_offsets[0]}) {
      guard rec.data_offsets[0]==cursor else { throw ProbeError.invalid("Overlapping or unclaimed tensor bytes") }
      cursor=rec.data_offsets[1]
    }
    guard cursor==Int(st.st_size)-Int(n)-8 else { throw ProbeError.invalid("Incomplete tensor payload") }
    self.path=path; self.fd=opened; self.records=parsed; self.offset=Int(n)+8; self.size=Int(st.st_size);self.identity=Self.stamp(st)
    success=true
  }
  deinit { Darwin.close(fd) }
  func checkIdentity() throws {
    var a=stat(),b=stat()
    guard fstat(fd,&a)==0, lstat(path,&b)==0,Self.stamp(a)==identity,Self.stamp(b)==identity else { throw ProbeError.invalid("Tensor file changed during reading") }
  }
  private static func bytes(_ fd:Int32,_ offset:Int,_ count:Int) throws -> Data {
    var data=Data(count:count)
    try data.withUnsafeMutableBytes { buffer in
      var done=0
      while done<count {
        let n=pread(fd,buffer.baseAddress!.advanced(by:done),count-done,off_t(offset+done))
        if n<0 && errno==EINTR { continue }
        guard n>0 else { throw ProbeError.invalid("Truncated tensor read") };done+=n
      }
    }
    return data
  }
  func raw(_ name:String) throws -> Data {
    guard let record=records[name] else { throw ProbeError.invalid("Missing tensor: \(name)") }
    try checkIdentity()
    let d=try Self.bytes(fd,offset+record.data_offsets[0],record.data_offsets[1]-record.data_offsets[0])
    try checkIdentity();return d
  }
  func read(_ name:String) throws -> FloatTensor {
    guard let rec=records[name] else { throw ProbeError.invalid("Missing tensor: \(name)") }
    if rec.dtype=="I8" {
      guard name.hasSuffix(".weight"),rec.shape.count==2 else { throw ProbeError.invalid("Unsupported signed I8 tensor") }
      let base=String(name.dropLast(7))
      let markerName=base+".comfy_quant"
      guard let mr=records[markerName], mr.dtype=="U8",mr.shape.count==1,mr.count>0,mr.count<=4096 else { throw ProbeError.invalid("Invalid quantization marker") }
      guard let marker=try JSONSerialization.jsonObject(with:raw(markerName)) as? [String:Any], marker["format"] as? String=="int8_tensorwise",Set(marker.keys).isSubset(of:["format","convrot","convrot_groupsize"]) else { throw ProbeError.invalid("Unsupported quantization") }
      var rotated=false
      if let raw=marker["convrot"] {
        guard CFGetTypeID(raw as CFTypeRef)==CFBooleanGetTypeID(),let value=raw as? Bool else { throw ProbeError.invalid("Invalid ConvRot flag") }
        rotated=value
      }
      var group=0
      if rotated {
        group=256
        if let raw=marker["convrot_groupsize"] {
          guard CFGetTypeID(raw as CFTypeRef) != CFBooleanGetTypeID(),let value=raw as? NSNumber,let integer=Int(exactly:value.doubleValue) else { throw ProbeError.invalid("Invalid ConvRot group type") }
          group=integer
        }
      }
      guard group==0 || ([4,16,64,256,1024].contains(group) && rec.shape[1]%group==0) else { throw ProbeError.invalid("Invalid ConvRot group") }
      let scaleName=base+".weight_scale"
      guard records[scaleName]?.dtype=="F32",records[scaleName]?.shape==[rec.shape[0],1] else { throw ProbeError.invalid("Invalid row scale shape") }
      let scales=try read(scaleName).values
      guard scales.allSatisfy({$0.isFinite && $0>0}) else { throw ProbeError.invalid("Invalid row scale value") }
      var values=[Float](repeating:0,count:rec.count)
      var basis=[Float]()
      if group>0 {
        basis=[Float](repeating:1/sqrt(Float(group)),count:group*group)
        for row in 0..<group { for col in 0..<group {
          var div=1;var sign:Float=1
          while div<group { if (row/div)%4+(col/div)%4==3 { sign = -sign };div*=4 }
          basis[row*group+col] *= sign
        }}
      }
      try checkIdentity()
      let columns=rec.shape[1]
      for start in stride(from:0,to:rec.shape[0],by:1024) {
        let rows=min(1024,rec.shape[0]-start), count=rows*columns
        let data=try Self.bytes(fd,offset+rec.data_offsets[0]+start*columns,count)
        var chunk=[Float](repeating:0,count:count)
        data.withUnsafeBytes { bytes in
          let ptr=bytes.bindMemory(to:Int8.self)
          for i in 0..<count { chunk[i]=Float(ptr[i])*scales[start+i/columns] }
        }
        if group>0 {
          var decoded=[Float](repeating:0,count:count)
          chunk.withUnsafeBufferPointer { a in basis.withUnsafeBufferPointer { b in decoded.withUnsafeMutableBufferPointer { c in
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,Int32(count/group),Int32(group),Int32(group),1,a.baseAddress!,Int32(group),b.baseAddress!,Int32(group),0,c.baseAddress!,Int32(group))
          }}}
          values.replaceSubrange(start*columns..<(start*columns+count),with:decoded)
        } else { values.replaceSubrange(start*columns..<(start*columns+count),with:chunk) }
      }
      try checkIdentity();return FloatTensor(shape:rec.shape,values:values)
    }
    if boundedHostIO {
      let widths=["F32":4,"BF16":2,"F16":2,"I32":4]
      guard let width=widths[rec.dtype] else { throw ProbeError.invalid("Unsupported tensor value type: \(rec.dtype)") }
      var values=[Float](repeating:0,count:rec.count)
      try checkIdentity()
      // Fixed-size read scratch: never allocate a second full raw payload.
      for start in stride(from:0,to:rec.count,by:262144) {
        let count=min(262144,rec.count-start)
        let data=try Self.bytes(fd,offset+rec.data_offsets[0]+start*width,count*width)
        data.withUnsafeBytes { b in
          for i in 0..<count {
            switch rec.dtype {
            case "F32": values[start+i]=Float(bitPattern:b.loadUnaligned(fromByteOffset:i*4,as:UInt32.self).littleEndian)
            case "BF16": values[start+i]=Float(bitPattern:UInt32(b.loadUnaligned(fromByteOffset:i*2,as:UInt16.self).littleEndian)<<16)
            case "F16": values[start+i]=Float(Float16(bitPattern:b.loadUnaligned(fromByteOffset:i*2,as:UInt16.self).littleEndian))
            default: values[start+i]=Float(b.loadUnaligned(fromByteOffset:i*4,as:Int32.self).littleEndian)
            }
          }
        }
      }
      try checkIdentity();return FloatTensor(shape:rec.shape,values:values)
    }
    let data=try raw(name)
    let values:[Float]=try data.withUnsafeBytes { b in
      switch rec.dtype {
      case "F32": return (0..<rec.count).map { Float(bitPattern: b.loadUnaligned(fromByteOffset:$0*4,as:UInt32.self).littleEndian) }
      case "BF16": return (0..<rec.count).map { Float(bitPattern: UInt32(b.loadUnaligned(fromByteOffset:$0*2,as:UInt16.self).littleEndian)<<16) }
      case "F16": return (0..<rec.count).map { Float(Float16(bitPattern:b.loadUnaligned(fromByteOffset:$0*2,as:UInt16.self).littleEndian)) }
      case "I32": return (0..<rec.count).map { Float(b.loadUnaligned(fromByteOffset:$0*4,as:Int32.self).littleEndian) }
      default: throw ProbeError.invalid("Unsupported tensor value type: \(rec.dtype)")
      }
    }
    return FloatTensor(shape:rec.shape,values:values)
  }
}
