import Foundation
import Darwin
import CryptoKit
import CoreFoundation

/// Weight-free transport and completed-page inspection for native model setup.
/// The worker remains the sole converter; Studio never reads tensor payloads.
public enum NativeTransformerConversion {
  public static let devSourceName = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
  private static let identityKeys:Set<String> = ["device","inode","bytes","modifiedSeconds","modifiedNanos",
    "changedSeconds","changedNanos","headerSHA256"]
  public static func request(source:String,destination:URL,sourceIdentity:[String:Any]?=nil,
    requiresIdentity:Bool=false) throws -> [String:Any] {
    let source=try local(source)
    guard destination.isFileURL else { throw StudioError.invalid("Choose a local conversion destination.") }
    let output=try local(destination.path)
    var object:[String:Any] = ["version":1,"engine":"ltx25","task":"transformer-page-conversion",
      "source_path":source.path,"output_directory":output.path]
    if let sourceIdentity { try validateIdentity(sourceIdentity);object["source_identity"]=sourceIdentity }
    else if requiresIdentity { throw StudioError.invalid("Conversion requires the source identity captured by native preflight.") }
    return object
  }
  public static func capturedIdentity(_ response:[String:Any],source:String) throws -> [String:Any] {
    guard response["nativeRuntime"] as? String == "swift-mlx",
      let identity=response["sourceIdentity"] as? [String:Any] else {
      throw StudioError.invalid("Native conversion preflight did not return a captured source identity.")
    }
    try validateIdentity(identity)
    guard NSDictionary(dictionary:identity).isEqual(to:try currentIdentity(source)) else {
      throw StudioError.invalid("Raw transformer changed during native preflight.")
    }
    return identity
  }
  /// Inspect only stat/header bytes. Identity keys match the shared converter's
  /// Codable SourceIdentity; no full checkpoint checksum is computed here.
  public static func currentIdentity(_ source:String) throws -> [String:Any] {
    let url=try local(source)
    var before=stat()
    guard fstatat(AT_FDCWD,url.path,&before,0)==0,before.st_mode&S_IFMT==S_IFREG,before.st_size>0 else {
      throw StudioError.invalid("Raw transformer is missing or is not a regular file.")
    }
    let (_,bytes)=try header(url,maximum:16*1024*1024)
    var after=stat()
    guard fstatat(AT_FDCWD,url.path,&after,0)==0,before.st_dev==after.st_dev,before.st_ino==after.st_ino,
      before.st_size==after.st_size,before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,
      before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec else {
      throw StudioError.invalid("Raw transformer changed during header inspection.")
    }
    return ["device":UInt64(UInt32(bitPattern:after.st_dev)),"inode":UInt64(after.st_ino),"bytes":UInt64(after.st_size),
      "modifiedSeconds":after.st_mtimespec.tv_sec,"modifiedNanos":after.st_mtimespec.tv_nsec,
      "changedSeconds":after.st_ctimespec.tv_sec,"changedNanos":after.st_ctimespec.tv_nsec,
      "headerSHA256":SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()]
  }
  public static func completedDirectory(_ response:[String:Any],source:String,destination:URL,
    sourceIdentity:[String:Any]) throws -> URL {
    try validateIdentity(sourceIdentity)
    let root=try local(destination.path),sourceURL=try local(source)
    guard response["nativeRuntime"] as? String == "swift-mlx",
      let returned=response["outputDirectory"] as? String,try local(returned).path==root.path,
      let manifestPath=response["manifestPath"] as? String,
      try local(manifestPath).path==root.appendingPathComponent("paged_manifest.json").path,
      sourceURL.lastPathComponent==devSourceName,
      NSDictionary(dictionary:sourceIdentity).isEqual(to:try currentIdentity(source)) else {
      throw StudioError.invalid("Native conversion result differs from its captured source or destination.")
    }
    let manifestURL=root.appendingPathComponent("paged_manifest.json")
    let resource=try manifestURL.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
    guard resource.isRegularFile==true,resource.isSymbolicLink != true,
      let size=resource.fileSize,size>0,size<=1024*1024,
      let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:manifestURL)) as? [String:Any],
      manifest["format"] as? String=="weetodd-ltx25-transformer-paged-q8-v1",manifest["kind"] as? String=="transformer",
      integer(manifest["num_layers"])==48,integer(manifest["bits"])==8,integer(manifest["group_size"])==64,
      manifest["source"] as? String==devSourceName,let metadata=manifest["metadata"] as? [String:Any],
      metadata["model_version"] as? String=="2.5.0",metadata["weetodd_baked_loras"]==nil,
      let config=metadata["config"] as? [String:Any],config["transformer"] is [String:Any],
      let provenance=manifest["conversion_provenance"] as? [String:Any],
      provenance["implementation"] as? String=="swift-mlx-affine-q8-v1",validSHA(provenance["source_sha256"]),
      let recorded=provenance["source_identity"] as? [String:Any],NSDictionary(dictionary:recorded).isEqual(to:sourceIdentity),
      let fixed=manifest["fixed"] as? [String:Any],let layers=manifest["layers"] as? [[String:Any]],layers.count==48 else {
      throw StudioError.invalid("Conversion did not publish a complete unmerged native Dev page manifest.")
    }
    let expected=try expectedPages(sourceURL)
    guard NSDictionary(dictionary:metadata).isEqual(to:expected.metadata),
      unsigned(manifest["source_tensor_bytes"])==expected.sourceBytes else {
      throw StudioError.invalid("Converted source metadata or tensor bytes differ from the captured raw checkpoint.")
    }
    var names:Set<String>=[],total:UInt64=0
    for (index,page) in ([fixed]+layers).enumerated() {
      try Task.checkCancellation()
      guard let name=page["file"] as? String,name.hasPrefix("pages/"),!name.contains("\\"),
        name.split(separator:"/",omittingEmptySubsequences:false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
        names.insert(name).inserted,validSHA(page["sha256"]),let count=integer(page["tensor_count"]),count>0,
        let bytes=unsigned(page["tensor_bytes"]),bytes>0 else {
        throw StudioError.invalid("Converted page identity/count is malformed.")
      }
      let pageURL=root.appendingPathComponent(name)
      let info=try pageURL.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
      guard info.isRegularFile==true,info.isSymbolicLink != true,
        (try local(pageURL.path)).path==pageURL.path else {
        throw StudioError.invalid("Converted pages must be regular files inside their output directory.")
      }
      let (object,raw)=try header(pageURL,maximum:index==0 ? 1024*1024 : 512*1024)
      let tensors=object.filter { $0.key != "__metadata__" }
      guard tensors.count==count,Set(tensors.keys)==Set(expected.pages[index].keys),
        let actualSize=info.fileSize,UInt64(actualSize)==UInt64(raw.count)+bytes else {
        throw StudioError.invalid("Converted page header/count/size differs from its manifest.")
      }
      var cursor:UInt64=0
      for name in tensors.keys.sorted() {
        try Task.checkCancellation()
        guard let tensor=tensors[name] as? [String:Any],let dtype=tensor["dtype"] as? String,
          let width=["F32":UInt64(4),"F16":2,"BF16":2,"U32":4][dtype],
          let shape=tensor["shape"] as? [Any],shape.count<=16,let offsets=tensor["data_offsets"] as? [Any],offsets.count==2,
          unsigned(offsets[0])==cursor,let end=unsigned(offsets[1]),end>=cursor,
          let expectedTensor=expected.pages[index][name] as? [String:Any],expectedTensor["dtype"] as? String==dtype,
          let expectedShape=expectedTensor["shape"] as? [Any],NSArray(array:shape).isEqual(to:expectedShape) else {
          throw StudioError.invalid("Converted page tensor header is invalid.")
        }
        var elements:UInt64=1
        for value in shape {
          guard let dimension=unsigned(value) else { throw StudioError.invalid("Invalid converted tensor shape.") }
          let next=elements.multipliedReportingOverflow(by:dimension)
          guard !next.overflow else { throw StudioError.invalid("Converted tensor shape overflows.") }
          elements=next.partialValue
        }
        let length=elements.multipliedReportingOverflow(by:width)
        guard !length.overflow,length.partialValue==end-cursor else { throw StudioError.invalid("Converted tensor bytes differ from shape.") }
        cursor=end
      }
      guard cursor==bytes else { throw StudioError.invalid("Converted tensor offsets differ from the complete page size.") }
      let next=total.addingReportingOverflow(bytes)
      guard !next.overflow else { throw StudioError.invalid("Converted page sizes overflow.") };total=next.partialValue
    }
    guard NSDictionary(dictionary:sourceIdentity).isEqual(to:try currentIdentity(source)),
      unsigned(manifest["output_tensor_bytes"])==total else { throw StudioError.invalid("Converted manifest byte totals differ.") }
    return root
  }
  private static func expectedPages(_ source:URL) throws -> (pages:[[String:Any]],metadata:[String:Any],sourceBytes:UInt64) {
    let (header,_)=try self.header(source,maximum:16*1024*1024)
    var pages=[[String:Any]](repeating:[:],count:49)
    let metadata=(header["__metadata__"] as? [String:String] ?? [:]).mapValues { text -> Any in
      (try? JSONSerialization.jsonObject(with:Data(text.utf8),options:.fragmentsAllowed)) ?? text
    }
    var sourceBytes:UInt64=0
    let prefixes=["base_model.model.model.diffusion_model.","base_model.model.diffusion_model.",
      "base_model.model.transformer.","base_model.model.","model.diffusion_model.","diffusion_model.","transformer."]
    for (name,value) in header where name != "__metadata__" {
      guard let tensor=value as? [String:Any],let dtype=tensor["dtype"] as? String,
        ["BF16","F16","F32"].contains(dtype),let rawShape=tensor["shape"] as? [Any] else {
        throw StudioError.invalid("Raw source tensor header is incompatible with native conversion.")
      }
      let shape=try rawShape.map { value -> UInt64 in
        guard let dimension=unsigned(value),dimension>0 else { throw StudioError.invalid("Invalid raw tensor shape.") }
        return dimension
      }
      var bytes:UInt64=dtype=="F32" ? 4 : 2
      for dimension in shape {
        let product=bytes.multipliedReportingOverflow(by:dimension)
        guard !product.overflow else { throw StudioError.invalid("Raw tensor size overflows.") };bytes=product.partialValue
      }
      let sum=sourceBytes.addingReportingOverflow(bytes)
      guard !sum.overflow else { throw StudioError.invalid("Raw source bytes overflow.") };sourceBytes=sum.partialValue
      var key=name
      if let prefix=prefixes.first(where:key.hasPrefix) { key.removeFirst(prefix.count) }
      var index=0
      if key.hasPrefix("transformer_blocks.") {
        let parts=key.split(separator:".",maxSplits:2)
        guard parts.count==3,let block=Int(parts[1]),(0..<48).contains(block) else {
          throw StudioError.invalid("Invalid raw transformer block index.")
        };index=block+1
      }
      let entries:[String:Any]
      if index>0,name.hasSuffix(".weight"),shape.count==2,shape[1]%64==0 {
        let stem=String(name.dropLast(7))
        entries=[name:["dtype":"U32","shape":[shape[0],shape[1]/4]],
          stem+".scales":["dtype":dtype,"shape":[shape[0],shape[1]/64]],
          stem+".biases":["dtype":dtype,"shape":[shape[0],shape[1]/64]]]
      } else { entries=[name:["dtype":dtype,"shape":shape]] }
      for (name,value) in entries {
        guard pages[index].updateValue(value,forKey:name)==nil else {
          throw StudioError.invalid("Converted tensor identity would collide.")
        }
      }
    }
    guard pages.allSatisfy({ !$0.isEmpty }) else { throw StudioError.invalid("Raw transformer block set is incomplete.") }
    return (pages,metadata,sourceBytes)
  }
  private static func header(_ url:URL,maximum:Int) throws -> ([String:Any],Data) {
    let file=try FileHandle(forReadingFrom:url);defer { try? file.close() }
    guard let prefix=try file.read(upToCount:8),prefix.count==8 else { throw StudioError.invalid("Missing safetensors header.") }
    let count=prefix.withUnsafeBytes { $0.loadUnaligned(as:UInt64.self).littleEndian }
    guard count>0,count<=UInt64(maximum),let data=try file.read(upToCount:Int(count)),data.count==Int(count),
      let object=try JSONSerialization.jsonObject(with:data) as? [String:Any] else { throw StudioError.invalid("Invalid bounded safetensors header.") }
    return (object,prefix+data)
  }
  private static func validateIdentity(_ value:[String:Any]) throws {
    guard Set(value.keys)==identityKeys,unsigned(value["device"]) != nil,
      let inode=unsigned(value["inode"]),inode>0,let bytes=unsigned(value["bytes"]),bytes>0,
      integer(value["modifiedSeconds"]) != nil,integer(value["changedSeconds"]) != nil,
      let modified=integer(value["modifiedNanos"]),(0..<1_000_000_000).contains(modified),
      let changed=integer(value["changedNanos"]),(0..<1_000_000_000).contains(changed),validSHA(value["headerSHA256"]) else {
      throw StudioError.invalid("Captured native source identity is malformed.")
    }
  }
  private static func integer(_ value:Any?) -> Int? {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
      n.doubleValue>=Double(Int.min),n.doubleValue<Double(Int.max),n.doubleValue==Double(n.int64Value) else { return nil }
    return Int(n.int64Value)
  }
  private static func unsigned(_ value:Any?) -> UInt64? {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
      n.doubleValue>=0,n.doubleValue<Double(Int64.max),n.doubleValue==Double(n.uint64Value) else { return nil }
    return n.uint64Value
  }
  private static func validSHA(_ value:Any?) -> Bool {
    guard let s=value as? String,s.utf8.count==64 else { return false }
    return s.utf8.allSatisfy { (48...57).contains($0)||(97...102).contains($0) }
  }
  private static func local(_ value:String) throws -> URL {
    guard value.hasPrefix("/"),!value.utf8.contains(0),value.utf8.count<=4096 else { throw StudioError.invalid("Choose bounded absolute local model paths.") }
    var ancestor=URL(fileURLWithPath:value).standardizedFileURL
    guard ancestor.path != "/" else { throw StudioError.invalid("Choose a named model source and output directory.") }
    var missing:[String]=[]
    while true {
      if let resolved=realpath(ancestor.path,nil) {
        let path=String(cString:resolved);free(resolved)
        // Preserve POSIX spelling instead of reintroducing /tmp or /var aliases.
        let suffix=missing.reversed().joined(separator:"/")
        return URL(fileURLWithPath:suffix.isEmpty ? path : (path == "/" ? "/" : path+"/")+suffix)
      }
      let failure=errno
      var info=stat()
      guard failure==ENOENT,lstat(ancestor.path,&info) != 0,errno==ENOENT,
        ancestor.path != "/",!ancestor.lastPathComponent.isEmpty else {
        throw StudioError.invalid("Model path has an inaccessible or unresolved ancestor.")
      }
      missing.append(ancestor.lastPathComponent)
      ancestor=ancestor.deletingLastPathComponent()
    }
  }
}
