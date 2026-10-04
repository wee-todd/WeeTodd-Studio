import CryptoKit
import Foundation
import LTX25Engine

/// Resume identity hashes metadata/header bytes and immutable file stats, never
/// full model payloads. Per-tensor mutation admission remains shared TensorIO.
enum MLXMovieCheckpointIdentity {
  struct Record:Codable { let identity:MLXMovieFiles.Identity,headerSHA256:String? }
  static func capture(_ components:[String:String]) throws -> String {
    var records:[String:[Record]]=[:],total=0
    for (name,path) in components.sorted(by:{ $0.key<$1.key }) {
      let root=URL(fileURLWithPath:path).resolvingSymlinksInPath().standardizedFileURL
      var pending:[(URL,Int)]=[(root,0)],files:[URL]=[]
      while let (url,depth)=pending.popLast() {
        try Task.checkCancellation()
        let values=try url.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey,.isRegularFileKey])
        guard values.isSymbolicLink != true,depth<=4 else { throw LTXError.invalid("Movie component tree contains a symbolic link or excessive nesting.") }
        if values.isDirectory==true {
          let children=try FileManager.default.contentsOfDirectory(at:url,includingPropertiesForKeys:nil,options:.skipsHiddenFiles)
          guard children.count<=4096,pending.count+files.count+children.count<=4096 else { throw LTXError.invalid("Movie component metadata tree exceeds its file limit.") }
          pending += children.map { ($0,depth+1) }
        } else {
          guard values.isRegularFile==true else { throw LTXError.invalid("Movie component must be a regular checkpoint or metadata file.") }
          files.append(url);total+=1
          guard total<=8192 else { throw LTXError.invalid("Movie component metadata exceeds its bounded total file count.") }
        }
      }
      var component:[Record]=[]
      for file in files.sorted(by:{ $0.path<$1.path }) {
        let identity=try MLXMovieFiles.Identity(file)
        let header:Data?
        if file.pathExtension == "safetensors" {
          guard identity.bytes>=8 else { throw LTXError.invalid("Movie checkpoint lacks a complete length prefix.") }
          let input=try FileHandle(forReadingFrom:file);defer { try? input.close() }
          let length=try input.read(upToCount:8) ?? Data()
          guard length.count==8 else { throw LTXError.invalid("Movie checkpoint header is incomplete.") }
          let count=length.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element)<<($1.offset*8) }
          guard count>0,count<=4*1024*1024,count<=UInt64(identity.bytes-8) else { throw LTXError.invalid("Movie checkpoint header exceeds bounded metadata admission.") }
          let bytes=try input.read(upToCount:Int(count)) ?? Data()
          guard bytes.count==count else { throw LTXError.invalid("Movie checkpoint header was truncated.") }
          header=length+bytes
        } else if identity.bytes<=32*1024*1024 { header=try Data(contentsOf:file) }
        else { header=nil }
        guard identity == (try MLXMovieFiles.Identity(file)) else { throw LTXError.invalid("Movie model metadata changed during admission.") }
        component.append(Record(identity:identity,headerSHA256:header.map { SHA256.hash(data:$0).map { String(format:"%02x",$0) }.joined() }))
      }
      records[name]=component
    }
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    return SHA256.hash(data:try encoder.encode(records)).map { String(format:"%02x",$0) }.joined()
  }
}
