import CryptoKit
import CoreFoundation
import Darwin
import Foundation

/// Complete normalized native AV rows. This is never a saved continuation tail.
public struct H3JointLatentArtifact:Codable,Equatable {
  public var manifest:String
  public var manifestSHA256:String
  public var payloadSHA256:String
  public var task:String
  public var componentIdentity:String
  public var width:Int
  public var height:Int
  public var generatedFrames:Int
  public var payloadFilename:String { "joint-latents.f32" }
  public var payloadPath:String { URL(fileURLWithPath:manifest).deletingLastPathComponent().appendingPathComponent(payloadFilename).path }
  private static func sha(_ bytes:Data) -> String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
  static func validSHA(_ text:String) -> Bool { text.utf8.count==64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
  static func localPath(_ text:String) throws -> String {
    guard text.hasPrefix("/"),text.utf8.count<=4096,!text.utf8.contains(0),!text.contains("://") else { throw StudioError.invalid("H3 joint artifacts require bounded absolute local paths.") }
    return URL(fileURLWithPath:text).standardizedFileURL.path
  }
  private static func file(_ path:String,maximum:Int,_ body:(FileHandle,Int)throws->Data) throws -> Data {
    let fd=Darwin.open(try localPath(path),O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)
    guard fd>=0 else { throw StudioError.invalid("Relink the native H3 joint artifact.") }
    let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? file.close() }
    var before=stat(),after=stat()
    guard fstat(fd,&before)==0,before.st_mode & S_IFMT == S_IFREG,(1...maximum).contains(Int(before.st_size)) else { throw StudioError.invalid("H3 joint artifact must be a bounded regular file.") }
    try Task.checkCancellation();let bytes=try body(file,Int(before.st_size))
    guard fstat(fd,&after)==0,before.st_dev==after.st_dev,before.st_ino==after.st_ino,before.st_size==after.st_size,
      before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec else { throw StudioError.invalid("H3 joint artifact changed during validation.") }
    try Task.checkCancellation();return bytes
  }
  /// Only worker metadata carries the published paths; no synthesized top-level artifact is accepted.
  public static func adopt(metadata:[String:Any]) throws -> Self? {
    let keys=["jointLatentManifest","jointLatentManifestSHA256","jointLatentPayloadSHA256"]
    guard keys.contains(where:{metadata[$0] != nil}) else { return nil }
    guard let path=metadata[keys[0]] as? String,let hash=metadata[keys[1]] as? String,let payloadHash=metadata[keys[2]] as? String,
      validSHA(hash),validSHA(payloadHash) else { throw StudioError.invalid("H3 joint artifact publication metadata is incomplete.") }
    let canonical=try localPath(path)
    let bytes=try file(canonical,maximum:1_048_576) { file,size in
      let bytes=try file.read(upToCount:size+1) ?? Data()
      guard bytes.count==size else { throw StudioError.invalid("H3 joint manifest changed while reading.") };return bytes
    }
    guard sha(bytes)==hash,let root=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
      Set(root.keys)==["format","task","width","height","generatedFrames","componentIdentity","videoFloats","audioFloats","payloadBytes","payloadSHA256"],
      let format=root["format"] as? String,["weetodd-h3-swift-joint-latents-v1","weetodd-h3-swift-joint-latents-v2-spatial"].contains(format),
      let task=root["task"] as? String,["t2va","fl2va","ref2va"].contains(task),
      let identity=root["componentIdentity"] as? String,validSHA(identity),root["payloadSHA256"] as? String==payloadHash else { throw StudioError.invalid("H3 full-latent manifest identity or digest changed.") }
    func integer(_ key:String) throws -> Int {
      guard let n=root[key] as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
        n.doubleValue.rounded()==n.doubleValue,(0...64*1024*1024).contains(n.doubleValue) else { throw StudioError.invalid("H3 full-latent geometry is invalid.") };return n.intValue
    }
    let w=try integer("width"),h=try integer("height"),frames=try integer("generatedFrames")
    let admittedCanvas = format == "weetodd-h3-swift-joint-latents-v1" ? w*h<=1376*768
      : max(w,h)<=1920 && min(w,h)<=1088 && w*h<=1920*1088
    guard (32...4096).contains(w),(32...4096).contains(h),w%32==0,h%32==0,
      admittedCanvas,
      (60...362).contains(frames),frames%17==5 else { throw StudioError.invalid("H3 full-latent canvas or frame grid is invalid.") }
    let video=(((frames-5)/17)*5+2)*(w/32)*(h/32)*96
    let audio=2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))*32
    guard format != "weetodd-h3-swift-joint-latents-v2-spatial" || video/96+audio/32<=64_000 else {throw StudioError.invalid("Expanded H3 full-latent rows exceed the 64000-row source budget.")}
    let size=try integer("payloadBytes")
    guard try integer("videoFloats")==video,try integer("audioFloats")==audio,size==(video+audio)*4,size<=64*1024*1024 else { throw StudioError.invalid("H3 full-latent payload geometry differs.") }
    let artifact=Self(manifest:canonical,manifestSHA256:hash,payloadSHA256:payloadHash,task:task,componentIdentity:identity,width:w,height:h,generatedFrames:frames)
    let digest=try file(artifact.payloadPath,maximum:64*1024*1024) { handle,count in
      guard count==size else { throw StudioError.invalid("H3 full-latent payload length differs.") }
      var hash=SHA256(),read=0
      while let chunk=try handle.read(upToCount:1024*1024),!chunk.isEmpty {
        try Task.checkCancellation();read+=chunk.count;guard read<=count else { throw StudioError.invalid("H3 full-latent payload changed.") };hash.update(data:chunk)
      }
      guard read==count else { throw StudioError.invalid("H3 full-latent payload changed.") };return Data(hash.finalize())
    }
    guard digest.map({String(format:"%02x",$0)}).joined()==payloadHash else { throw StudioError.invalid("H3 full-latent payload digest changed.") }
    return artifact
  }
  public func verify() throws {
    let actual=try Self.adopt(metadata:["jointLatentManifest":manifest,"jointLatentManifestSHA256":manifestSHA256,"jointLatentPayloadSHA256":payloadSHA256])
    guard actual==self else { throw StudioError.invalid("H3 saved full-latent artifact changed. Relink the accepted source.") }
  }
}
