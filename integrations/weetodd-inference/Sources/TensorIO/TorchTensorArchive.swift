import CryptoKit
import Darwin
import Foundation

/// Read-only, bounded metadata interpreter for pinned PyTorch ZIP archives.
/// GLOBAL/REDUCE/BUILD are descriptor tokens with explicit schemas. No Python
/// objects, executable callbacks, arbitrary classes or imports are instantiated.
public final class TorchTensorArchive {
  public static let importerVersion = "flashvsr-torch-zip-v1"
  public let sourceSHA256:String
  public let profile:TorchTensorImportProfile
  public let tensors:[String:TorchTensorMetadata]
  public let fileByteCount:UInt64
  public let byteOrder:String
  public var conversionKey:String { Self.conversionKey(sourceSHA256:sourceSHA256,profile:profile) }
  public static func conversionKey(sourceSHA256:String,profile:TorchTensorImportProfile,
    importerVersion:String = TorchTensorArchive.importerVersion)->String {
    digest(Data((importerVersion+"\n"+profile.id+"\n"+sourceSHA256).utf8))
  }
  private let fd:Int32
  private let url:URL
  private let identity:Identity
  private struct Identity:Equatable {
    let device:dev_t,inode:ino_t,size:off_t
    let modifiedSeconds:Int,modifiedNanos:Int,changedSeconds:Int,changedNanos:Int
    init(_ s:stat) {
      device=s.st_dev;inode=s.st_ino;size=s.st_size
      modifiedSeconds=s.st_mtimespec.tv_sec;modifiedNanos=s.st_mtimespec.tv_nsec
      changedSeconds=s.st_ctimespec.tv_sec;changedNanos=s.st_ctimespec.tv_nsec
    }
  }
  private struct Member {
    let name:String
    let range:Range<UInt64>
  }
  private static func invalid(_ message:String)->CheckpointError { .invalid("Torch archive: "+message) }
  private static func digest(_ data:Data)->String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
  private static func add(_ a:UInt64,_ b:UInt64)throws->UInt64 {
    let r=a.addingReportingOverflow(b);guard !r.overflow else { throw invalid("integer offset overflow.") };return r.partialValue
  }
  private static func multiply(_ a:UInt64,_ b:UInt64)throws->UInt64 {
    let r=a.multipliedReportingOverflow(by:b);guard !r.overflow else { throw invalid("integer tensor overflow.") };return r.partialValue
  }
  private static func read(_ fd:Int32,_ offset:UInt64,_ count:Int)throws->Data {
    guard count>=0,count<=4*1024*1024,offset<=UInt64(Int64.max)-UInt64(count) else { throw invalid("read exceeds bounded window.") }
    try Task.checkCancellation()
    var data=Data(count:count)
    try data.withUnsafeMutableBytes { bytes in
      var n=0
      while n<count {
        let r=pread(fd,bytes.baseAddress!.advanced(by:n),count-n,off_t(offset)+off_t(n))
        if r<0 && errno==EINTR { continue };guard r>0 else { throw invalid("truncated or unreadable source.") };n += r
      }
    }
    return data
  }
  public static func open(url:URL,expectedSHA256:String,profile:TorchTensorImportProfile)throws->TorchTensorArchive {
    try Task.checkCancellation()
    guard url.isFileURL,expectedSHA256==profile.sourceSHA256,expectedSHA256.count==64,
      expectedSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw invalid("source hash must match the pinned import profile.")
    }
    let fd=Darwin.open(url.path,O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_NOFOLLOW)
    guard fd>=0 else { throw invalid("cannot open regular local source.") }
    var accepted=false
    defer { if !accepted { Darwin.close(fd) } }
    var status=stat()
    guard fstat(fd,&status)==0,status.st_mode&S_IFMT==S_IFREG,status.st_size>=22 else { throw invalid("source is not a regular ZIP file.") }
    let size=UInt64(status.st_size),identity=Identity(status)
    var hasher=SHA256(),position:UInt64=0
    while position<size {
      let data=try read(fd,position,Int(min(size-position,4*1024*1024)));hasher.update(data:data);position += UInt64(data.count)
    }
    let hash=hasher.finalize().map { String(format:"%02x",$0) }.joined()
    guard hash==expectedSHA256 else { throw invalid("source SHA-256 differs from the pinned archive.") }
    let members=try zipMembers(fd:fd,size:size,profile:profile)
    func contents(_ name:String)throws->Data {
      guard let m=members[name],m.range.count<=1024*1024 else { throw invalid("missing or oversized metadata member: "+name) }
      return try read(fd,m.range.lowerBound,Int(m.range.count))
    }
    let order=String(data:try contents("byteorder"),encoding:.utf8)
    guard order=="little" || order=="big",try contents("version")==Data("3\n".utf8) else { throw invalid("unsupported byte order or archive version.") }
    let descriptors=try Interpreter(data:contents("data.pkl"),profile:profile).parse()
    var tensors:[String:TorchTensorMetadata]=[:],usedStorages=Set<String>()
    var storageDeclarations:[String:Storage]=[:]
    for (name,view) in descriptors {
      let storage=view.storage
      if let previous=storageDeclarations[storage.key],previous != storage { throw invalid("conflicting shared storage declarations.") }
      storageDeclarations[storage.key]=storage
      guard let member=members["data/"+storage.key],UInt64(member.range.count)==(try multiply(storage.count,storage.width)) else {
        throw invalid("storage declaration differs from its ZIP payload.")
      }
      usedStorages.insert(storage.key)
      var elements:UInt64=view.shape.contains(0) ? 0:1
      for n in view.shape { elements=try multiply(elements,n) }
      let byteCount=try multiply(elements,storage.width)
      guard byteCount<=UInt64(Int64.max) else { throw invalid("logical tensor exceeds signed addressable size.") }
      var last=view.offset
      if elements>0 {
        for (n,stride) in zip(view.shape,view.strides) { last=try add(last,try multiply(n-1,stride)) }
        guard last<storage.count else { throw invalid("tensor view lies outside storage.") }
      } else { guard view.offset<=storage.count else { throw invalid("empty view lies outside storage.") } }
      var step:UInt64=1,contiguous=true
      for (n,stride) in zip(view.shape,view.strides).reversed() {
        if n>1 && stride != step { contiguous=false };step=try multiply(step,n)
      }
      tensors[name]=TorchTensorMetadata(dtype:storage.dtype,shape:view.shape,storageKey:storage.key,
        storageByteRange:member.range,storageOffset:view.offset,strides:view.strides,byteCount:byteCount,isContiguous:contiguous)
    }
    guard usedStorages.count<=profile.maximumStorages,
      Set(members.keys.filter { $0.hasPrefix("data/") }.map { String($0.dropFirst(5)) })==usedStorages else { throw invalid("unreferenced or excessive ZIP storages.") }
    var after=stat()
    guard fstat(fd,&after)==0,Identity(after)==identity else { throw invalid("source changed during inspection.") }
    let result=TorchTensorArchive(fd:fd,url:url,identity:identity,sourceSHA256:hash,profile:profile,
      tensors:tensors,size:size,byteOrder:order!)
    accepted=true
    try result.checkUnchanged()
    return result
  }
  private init(fd:Int32,url:URL,identity:Identity,sourceSHA256:String,profile:TorchTensorImportProfile,
    tensors:[String:TorchTensorMetadata],size:UInt64,byteOrder:String) {
    self.fd=fd;self.url=url;self.identity=identity;self.sourceSHA256=sourceSHA256;self.profile=profile
    self.tensors=tensors;fileByteCount=size;self.byteOrder=byteOrder
  }
  deinit { Darwin.close(fd) }
  public func checkUnchanged()throws {
    var s=stat(),p=stat()
    guard fstat(fd,&s)==0,lstat(url.path,&p)==0,Identity(s)==identity,Identity(p)==identity else {
      throw Self.invalid("source changed or pathname replaced; reopen before use.")
    }
    try Task.checkCancellation()
  }
  public func withTensorBytes<T>(named name:String,_ body:(UnsafeRawBufferPointer)throws->T)throws->T {
    guard let tensor=tensors[name] else { throw Self.invalid("missing tensor: "+name) }
    return try withTensorBytes(named:name,range:0..<tensor.byteCount,body)
  }
  /// Returns at most 4 MiB of logical, little-endian tensor bytes. The source
  /// mapping and returned buffer are scoped to body; no full tensor is copied.
  public func withTensorBytes<T>(named name:String,range:Range<UInt64>,
    _ body:(UnsafeRawBufferPointer)throws->T)throws->T {
    try checkUnchanged()
    guard let t=tensors[name],range.upperBound<=t.byteCount,range.count<=4*1024*1024 else { throw Self.invalid("tensor window exceeds bounds or 4 MiB.") }
    if range.isEmpty { return try body(UnsafeRawBufferPointer(start:nil,count:0)) }
    let width:UInt64=t.dtype=="F32" ? 4:2
    if t.isContiguous && byteOrder=="little" {
      let start=t.storageByteRange.lowerBound+t.storageOffset*width+range.lowerBound
      let page=UInt64(getpagesize()),aligned=start-start%page,prefix=start-aligned,length=Int(prefix+UInt64(range.count))
      guard let ptr=mmap(nil,length,PROT_READ,MAP_PRIVATE,fd,off_t(aligned)),ptr != MAP_FAILED else { throw Self.invalid("cannot map bounded tensor window.") }
      defer { munmap(ptr,length) }
      let result=try body(UnsafeRawBufferPointer(start:ptr.advanced(by:Int(prefix)),count:range.count))
      try checkUnchanged();return result
    }
    // Gather strided/byte-swapped views with a bounded source cache. A slice may
    // start or end inside an element, so consumers need not align their windows.
    var output=Data(count:range.count),cache=Data(),cacheStart=UInt64.max
    try output.withUnsafeMutableBytes { (destination:UnsafeMutableRawBufferPointer) in
      var logical=range.lowerBound
      while logical<range.upperBound {
        if logical%65536==0 { try Task.checkCancellation() }
        let element=logical/width,inElement=logical%width
        var index=element,storageIndex=t.storageOffset
        for axis in t.shape.indices.reversed() {
          storageIndex += (index%t.shape[axis])*t.strides[axis];index /= t.shape[axis]
        }
        let absolute=t.storageByteRange.lowerBound+storageIndex*width+(byteOrder=="big" ? width-1-inElement:inElement)
        if cacheStart==UInt64.max || absolute<cacheStart || absolute-cacheStart>=UInt64(cache.count) {
          cacheStart=absolute-absolute%65536
          cache=try Self.read(fd,cacheStart,Int(min(65536,fileByteCount-cacheStart)))
        }
        destination[Int(logical-range.lowerBound)]=cache[Int(absolute-cacheStart)];logical += 1
      }
    }
    let result=try output.withUnsafeBytes(body);try checkUnchanged();return result
  }
  /// Stream deterministic sorted tensors to a sibling temporary file, then
  /// atomically publish without replacing an existing file or symbolic link.
  /// Real model caches belong under the user's ai-models library.
  @discardableResult public func importSafetensors(to destination:URL)throws->String {
    try checkUnchanged()
    guard destination.isFileURL else { throw Self.invalid("output must be local.") }
    let temporary=destination.deletingLastPathComponent().appendingPathComponent("."+destination.lastPathComponent+"."+UUID().uuidString+".partial")
    defer { try? FileManager.default.removeItem(at:temporary) }
    var header:[String:SafeTensorStreamWriter.Tensor]=[:]
    for (name,t) in tensors { header[name]=try .init(dtype:t.dtype,shape:t.shape) }
    let writer=try SafeTensorStreamWriter(url:temporary,tensors:header,
      metadata:["format":"pt","source_sha256":sourceSHA256,"importer_version":Self.importerVersion,"import_profile":profile.id,"conversion_key":conversionKey])
    for name in tensors.keys.sorted() {
      let size=tensors[name]!.byteCount
      if size==0 { try writer.append(tensor:name,bytes:UnsafeRawBufferPointer(start:nil,count:0)) }
      var offset:UInt64=0
      while offset<size {
        let end=min(offset+4*1024*1024,size)
        try withTensorBytes(named:name,range:offset..<end) { try writer.append(tensor:name,bytes:$0) };offset=end
      }
    }
    let hash=try writer.finish();try checkUnchanged()
    guard renamex_np(temporary.path,destination.path,UInt32(RENAME_EXCL))==0 else { throw Self.invalid("cannot atomically publish exclusive output.") }
    return hash
  }
}

private extension TorchTensorArchive {
  struct Cursor {
    let data:Data
    var position=0
    var remaining:Int { data.count-position }
    mutating func bytes(_ count:Int)throws->Data {
      guard count>=0,count<=remaining else { throw invalid("truncated metadata record.") }
      defer { position += count };return data.subdata(in:position..<position+count)
    }
    mutating func byte()throws->UInt8 { guard remaining>0 else { throw invalid("truncated metadata opcode.") };defer { position += 1 };return data[position] }
    mutating func u16()throws->UInt16 { try bytes(2).withUnsafeBytes { $0.loadUnaligned(as:UInt16.self).littleEndian } }
    mutating func u32()throws->UInt32 { try bytes(4).withUnsafeBytes { $0.loadUnaligned(as:UInt32.self).littleEndian } }
    mutating func u64()throws->UInt64 { try bytes(8).withUnsafeBytes { $0.loadUnaligned(as:UInt64.self).littleEndian } }
    mutating func string(_ n:Int)throws->String {
      guard n<=65536,let value=String(data:try bytes(n),encoding:.utf8),!value.utf8.contains(0) else { throw invalid("invalid metadata string.") };return value
    }
    mutating func line()throws->String {
      guard let end=data[position...].firstIndex(of:10),end-position<=256 else { throw invalid("unterminated GLOBAL descriptor.") }
      let result=try string(end-position);position += 1;return result
    }
  }
  private static func zipMembers(fd:Int32,size:UInt64,profile:TorchTensorImportProfile)throws->[String:Member] {
    let tailSize=Int(min(size,65557)),tailStart=size-UInt64(tailSize),tail=try read(fd,tailStart,tailSize)
    var endIndex:Int?
    for i in stride(from:tail.count-22,through:0,by:-1) {
      if Array(tail[i..<i+4])==[0x50,0x4b,0x05,0x06] {
        var c=Cursor(data:tail,position:i+20)
        if i+22+Int(try c.u16())==tail.count { endIndex=i;break }
      }
    }
    guard let endIndex else { throw invalid("missing ZIP end record.") }
    let endOffset=tailStart+UInt64(endIndex)
    var end=Cursor(data:tail,position:endIndex+4)
    guard try end.u16()==0,try end.u16()==0 else { throw invalid("multi-disk ZIP is unsupported.") }
    let diskCount=try end.u16(),totalCount=try end.u16()
    guard diskCount==totalCount else { throw invalid("ZIP directory count differs across disks.") }
    var count=UInt64(totalCount),centralSize=UInt64(try end.u32()),centralStart=UInt64(try end.u32()),centralLimit=endOffset
    let zip64Sentinel=count==UInt64(UInt16.max) || centralSize==UInt64(UInt32.max) || centralStart==UInt64(UInt32.max)
    if endOffset>=20 {
      var locator=Cursor(data:try read(fd,endOffset-20,20))
      if try locator.u32()==0x07064b50 {
        guard try locator.u32()==0 else { throw invalid("multi-disk ZIP64 is unsupported.") }
        let offset=try locator.u64()
        guard try locator.u32()==1,offset<=endOffset-20, endOffset-20-offset>=56 else { throw invalid("invalid ZIP64 locator.") }
        var z=Cursor(data:try read(fd,offset,56))
        guard try z.u32()==0x06064b50,try z.u64()==44 else { throw invalid("unsupported ZIP64 end record.") }
        _ = try z.u16();_ = try z.u16()
        guard try z.u32()==0,try z.u32()==0 else { throw invalid("multi-disk ZIP64 is unsupported.") }
        let perDisk=try z.u64();count=try z.u64();centralSize=try z.u64();centralStart=try z.u64();centralLimit=offset
        guard count==perDisk,offset+56==endOffset-20,
          totalCount==UInt16.max || count==UInt64(totalCount) else { throw invalid("inconsistent ZIP64 end record.") }
      } else if zip64Sentinel { throw invalid("missing ZIP64 locator.") }
    } else if zip64Sentinel { throw invalid("missing ZIP64 locator.") }
    guard count>=5,count<=UInt64(profile.maximumStorages+4),centralSize<=1024*1024,
      centralStart<=centralLimit,centralSize==centralLimit-centralStart else { throw invalid("ZIP central directory exceeds profile bounds.") }
    var central=Cursor(data:try read(fd,centralStart,Int(centralSize)))
    var members:[String:Member]=[:],intervals:[Range<UInt64>]=[]
    for _ in 0..<count {
      try Task.checkCancellation()
      guard try central.u32()==0x02014b50 else { throw invalid("invalid ZIP central member.") }
      _ = try central.u16();_ = try central.u16()
      let flags=try central.u16(),method=try central.u16()
      guard flags & ~UInt16(0x0808)==0,method==0 else { throw invalid("encrypted or compressed ZIP members are unsupported.") }
      _ = try central.u32();let crc=try central.u32()
      var compressed=UInt64(try central.u32()),uncompressed=UInt64(try central.u32())
      let nameLength=Int(try central.u16()),extraLength=Int(try central.u16()),commentLength=Int(try central.u16())
      guard try central.u16()==0 else { throw invalid("multi-disk ZIP member.") }
      _ = try central.u16();_ = try central.u32();var localOffset=UInt64(try central.u32())
      let fullName=try central.string(nameLength),extra=try central.bytes(extraLength);_ = try central.bytes(commentLength)
      let needsUncompressed=uncompressed==UInt64(UInt32.max),needsCompressed=compressed==UInt64(UInt32.max),needsOffset=localOffset==UInt64(UInt32.max)
      var extras=Cursor(data:extra),saw64=false
      while extras.remaining>0 {
        let tag=try extras.u16(),length=Int(try extras.u16());var item=Cursor(data:try extras.bytes(length))
        if tag==1 {
          guard !saw64 else { throw invalid("duplicate ZIP64 extra field.") };saw64=true
          if needsUncompressed { uncompressed=try item.u64() };if needsCompressed { compressed=try item.u64() };if needsOffset { localOffset=try item.u64() }
          guard item.remaining==0 else { throw invalid("unexpected ZIP64 extra data.") }
        }
      }
      guard !(needsUncompressed || needsCompressed || needsOffset) || saw64,
        compressed==uncompressed,localOffset<=centralStart,centralStart-localOffset>=30 else { throw invalid("invalid ZIP member range.") }
      let prefix=profile.archiveRoot+"/"
      guard fullName.hasPrefix(prefix) else { throw invalid("unexpected archive root.") }
      let name=String(fullName.dropFirst(prefix.count)),parts=name.split(separator:"/",omittingEmptySubsequences:false)
      let allowed=name=="data.pkl" || name=="byteorder" || name=="version" || name==".data/serialization_id" ||
        (parts.count==2 && parts[0]=="data" && !parts[1].isEmpty && parts[1].utf8.allSatisfy { (48...57).contains($0) })
      guard allowed,!parts.contains(".."),!parts.contains("."),!name.contains("\\"),members[name]==nil else { throw invalid("unsafe, duplicate or unexpected ZIP member name.") }
      var local=Cursor(data:try read(fd,localOffset,30))
      guard try local.u32()==0x04034b50 else { throw invalid("invalid ZIP local header.") }
      _ = try local.u16()
      guard try local.u16()==flags,try local.u16()==method else { throw invalid("local and central ZIP methods differ.") }
      _ = try local.u32();let localCRC=try local.u32(),localCompressed=try local.u32(),localSize=try local.u32()
      let localNameLength=Int(try local.u16()),localExtraLength=Int(try local.u16())
      let payloadStart=try add(localOffset,UInt64(30+localNameLength+localExtraLength)),payloadEnd=try add(payloadStart,uncompressed)
      guard payloadEnd<=centralStart,localNameLength==nameLength,
        try read(fd,localOffset+30,localNameLength)==Data(fullName.utf8) else { throw invalid("ZIP payload is out of range or local name differs.") }
      if flags&8==0 {
        // Small installed archives use data descriptors. Also admit ordinary
        // stored ZIP members, including local ZIP64 size sentinels.
        guard localCRC==crc,(UInt64(localCompressed)==compressed || localCompressed==UInt32.max),
          (UInt64(localSize)==uncompressed || localSize==UInt32.max) else { throw invalid("local ZIP sizes differ from directory.") }
      }
      var recordEnd=payloadEnd
      if flags&8 != 0 {
        guard centralStart-payloadEnd>=16 else { throw invalid("truncated ZIP data descriptor.") }
        var descriptor=Cursor(data:try read(fd,payloadEnd,16))
        guard try descriptor.u32()==0x08074b50,try descriptor.u32()==crc,
          UInt64(try descriptor.u32())==compressed,UInt64(try descriptor.u32())==uncompressed else { throw invalid("ZIP data descriptor differs from directory.") }
        recordEnd=try add(payloadEnd,16)
      }
      intervals.append(localOffset..<recordEnd);members[name]=Member(name:name,range:payloadStart..<payloadEnd)
    }
    guard central.remaining==0 else { throw invalid("unindexed ZIP directory bytes.") }
    var previous:UInt64=0
    for (index,range) in intervals.sorted(by:{ $0.lowerBound<$1.lowerBound }).enumerated() {
      guard (index>0 || range.lowerBound==0),range.lowerBound>=previous else { throw invalid("overlapping ZIP members.") };previous=range.upperBound
    }
    guard members["data.pkl"] != nil,members["byteorder"] != nil,members["version"] != nil,members[".data/serialization_id"] != nil else { throw invalid("incomplete known-source ZIP profile.") }
    return members
  }

  struct Storage:Equatable {
    let key:String,dtype:String,location:String
    let count:UInt64
    var width:UInt64 { dtype=="F32" ? 4:2 }
  }
  struct View {
    let storage:Storage
    let offset:UInt64
    let shape:[UInt64],strides:[UInt64]
  }
  final class DictionaryValue {
    let ordered:Bool
    var values:[String:Value]=[:]
    var built=false
    init(ordered:Bool) { self.ordered=ordered }
  }
  indirect enum Value {
    case marker,global(String),string(String),integer(UInt64),tuple([Value]),boolean(Bool)
    case dictionary(DictionaryValue),storage(Storage),tensor(View)
  }
  final class Interpreter {
    var cursor:Cursor
    let profile:TorchTensorImportProfile
    var stack:[Value]=[],memo:[UInt32:Value]=[:]
    var operations=0
    init(data:Data,profile:TorchTensorImportProfile)throws { cursor=Cursor(data:data);self.profile=profile }
    func pop()throws->Value { guard let v=stack.popLast() else { throw invalid("pickle stack underflow.") };return v }
    func string(_ v:Value)throws->String { guard case .string(let s)=v else { throw invalid("expected descriptor string.") };return s }
    func number(_ v:Value)throws->UInt64 { guard case .integer(let n)=v else { throw invalid("expected nonnegative descriptor integer.") };return n }
    func tuple(_ v:Value)throws->[Value] { guard case .tuple(let t)=v else { throw invalid("expected descriptor tuple.") };return t }
    func marked()throws->[Value] {
      guard let i=stack.lastIndex(where:{ if case .marker=$0 { return true };return false }) else { throw invalid("missing pickle tuple mark.") }
      let values=Array(stack[(i+1)...]);stack.removeSubrange(i...);return values
    }
    func set(_ pairs:[Value])throws {
      guard pairs.count%2==0,case .dictionary(let d)=stack.last else { throw invalid("invalid dictionary descriptor.") }
      guard d.values.count+pairs.count/2<=2048 else { throw invalid("dictionary descriptor exceeds limit.") }
      for i in stride(from:0,to:pairs.count,by:2) {
        let key=try string(pairs[i]);guard d.values[key]==nil,key.utf8.count<=1024 else { throw invalid("duplicate or oversized dictionary key.") }
        d.values[key]=pairs[i+1]
      }
    }
    func reduce()throws {
      let args=try tuple(pop()),callable=try pop()
      guard case .global(let name)=callable else { throw invalid("REDUCE callable is not a permitted descriptor.") }
      if name=="collections.OrderedDict" {
        guard args.isEmpty else { throw invalid("OrderedDict constructor arguments are unsupported.") }
        stack.append(.dictionary(DictionaryValue(ordered:true)));return
      }
      guard name=="torch._utils._rebuild_tensor_v2",args.count==6,
        case .storage(let storage)=args[0],case .boolean(false)=args[4],
        case .dictionary(let hooks)=args[5],hooks.ordered,hooks.values.isEmpty,!hooks.built else { throw invalid("malformed tensor rebuild descriptor.") }
      let offset=try number(args[1]),shape=try tuple(args[2]).map(number),strides=try tuple(args[3]).map(number)
      guard shape.count<=16,shape.count==strides.count,shape.allSatisfy({ $0<=UInt64(Int.max) }) else { throw invalid("invalid tensor rank/strides.") }
      stack.append(.tensor(View(storage:storage,offset:offset,shape:shape,strides:strides)))
    }
    func persistent()throws {
      let fields=try tuple(pop())
      guard fields.count==5,try string(fields[0])=="storage",case .global(let kind)=fields[1],
        kind==(profile.dtype=="F32" ? "torch.FloatStorage":"torch.BFloat16Storage") else { throw invalid("unsupported persistent storage descriptor.") }
      let key=try string(fields[2]),location=try string(fields[3]),count=try number(fields[4])
      guard !key.isEmpty,key.utf8.count<=16,key.utf8.allSatisfy({ (48...57).contains($0) }),
        location=="cpu" || location=="cuda:0",count<=UInt64(Int64.max)/4 else { throw invalid("invalid storage location/key/size.") }
      stack.append(.storage(Storage(key:key,dtype:profile.dtype,location:location,count:count)))
    }
    func build()throws {
      let state=try pop()
      guard profile.allowsMetadata,case .dictionary(let d)=stack.last,d.ordered,!d.built,
        case .dictionary(let attributes)=state,!attributes.ordered,attributes.values.count==1,
        case .dictionary(let metadata)=attributes.values["_metadata"],metadata.ordered,!metadata.built,metadata.values.count<=512 else { throw invalid("unsupported BUILD state.") }
      for (_,entry) in metadata.values {
        guard case .dictionary(let version)=entry,!version.ordered,version.values.count==1,
          case .integer(let n)=version.values["version"],n<=16 else { throw invalid("malformed module-version BUILD metadata.") }
      }
      d.built=true
    }
    func parse()throws->[String:View] {
      guard try cursor.byte()==0x80,try cursor.byte()==2 else { throw invalid("only pinned pickle protocol 2 is supported.") }
      while cursor.remaining>0 {
        operations += 1
        guard operations<=100000,stack.count<=4096,memo.count<=4096 else { throw invalid("pickle metadata exceeds bounded interpreter limits.") }
        if operations%1024==0 { try Task.checkCancellation() }
        switch try cursor.byte() {
        case 0x63:
          let name=try cursor.line()+"."+cursor.line()
          let allowed=["collections.OrderedDict","torch._utils._rebuild_tensor_v2",profile.dtype=="F32" ? "torch.FloatStorage":"torch.BFloat16Storage"]
          guard allowed.contains(name) else { throw invalid("unknown GLOBAL descriptor: "+name) };stack.append(.global(name))
        case 0x58: stack.append(.string(try cursor.string(Int(cursor.u32()))))
        case 0x4a:
          let raw=try cursor.u32();guard raw<=UInt32(Int32.max) else { throw invalid("negative descriptor integer.") };stack.append(.integer(UInt64(raw)))
        case 0x4b: stack.append(.integer(UInt64(try cursor.byte())))
        case 0x4d: stack.append(.integer(UInt64(try cursor.u16())))
        case 0x28: stack.append(.marker)
        case 0x29: stack.append(.tuple([]))
        case 0x74:
          let items=try marked();guard items.count<=16 else { throw invalid("tuple rank exceeds allowance.") };stack.append(.tuple(items))
        case 0x85: stack.append(.tuple([try pop()]))
        case 0x87:
          let c=try pop(),b=try pop(),a=try pop();stack.append(.tuple([a,b,c]))
        case 0x89: stack.append(.boolean(false))
        case 0x52: try reduce()
        case 0x51: try persistent()
        case 0x7d: stack.append(.dictionary(DictionaryValue(ordered:false)))
        case 0x75: try set(marked())
        case 0x73:
          let value=try pop(),key=try pop();try set([key,value])
        case 0x62: try build()
        case 0x71,0x72:
          let opcode=cursor.data[cursor.position-1],key=try opcode==0x71 ? UInt32(cursor.byte()):cursor.u32()
          guard let value=stack.last,key<4096,memo[key]==nil else { throw invalid("invalid pickle memo write.") };memo[key]=value
        case 0x68,0x6a:
          let opcode=cursor.data[cursor.position-1],key=try opcode==0x68 ? UInt32(cursor.byte()):cursor.u32()
          guard let value=memo[key] else { throw invalid("undefined pickle memo reference.") };stack.append(value)
        case 0x2e:
          guard cursor.remaining==0,stack.count==1 else { throw invalid("unexpected trailing pickle object graph.") }
          if profile.rootTensor {
            guard case .tensor(let tensor)=stack[0] else { throw invalid("expected a single root tensor.") };return ["tensor":tensor]
          }
          guard case .dictionary(let d)=stack[0],d.ordered,!d.values.isEmpty,d.values.count<=profile.maximumTensors else { throw invalid("expected bounded tensor state dictionary.") }
          var result:[String:View]=[:]
          for (key,value) in d.values {
            guard !key.isEmpty,key != "__metadata__",case .tensor(let view)=value else { throw invalid("unexpected state dictionary object.") };result[key]=view
          }
          return result
        default: throw invalid("unknown pickle opcode.")
        }
      }
      throw invalid("pickle STOP is missing.")
    }
  }
}
