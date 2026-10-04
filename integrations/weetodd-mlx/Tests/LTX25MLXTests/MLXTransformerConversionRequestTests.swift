import Foundation
import Darwin
import XCTest
@testable import LTX25MLX

final class MLXTransformerConversionRequestTests:XCTestCase {
  private let destination=URL(fileURLWithPath:"/tmp/weetodd-native-conversion-request-output")
  private var object:[String:Any] {
    ["version":1,"engine":"ltx25","task":"transformer-page-conversion",
      "source_path":"/tmp/weetodd-native-conversion-request-source.safetensors","output_directory":destination.path]
  }
  private var identity:[String:Any] {
    ["device":1,"inode":2,"bytes":8,"modifiedSeconds":0,"modifiedNanos":123,"changedSeconds":0,
      "changedNanos":456,"headerSHA256":String(repeating:"a",count:64)]
  }
  private func decode(_ object:[String:Any],required:Bool=false,output:URL?=nil) throws -> MLXTransformerConversionRequest {
    try .init(data:JSONSerialization.data(withJSONObject:object),outputDirectory:output ?? destination,
      requiresSourceIdentity:required)
  }
  func testPreflightMayOmitIdentityButConversionRequiresExactCapturedIdentity() throws {
    let request=try decode(object)
    XCTAssertNil(request.sourceIdentity)
    let temporary=try XCTUnwrap(realpath("/tmp",nil));defer { free(temporary) }
    XCTAssertEqual(request.source.path,String(cString:temporary)+"/weetodd-native-conversion-request-source.safetensors")
    XCTAssertEqual(try MLXTransformerPageConverter.canonicalLocalURL(request.source).path,request.source.path)
    XCTAssertThrowsError(try decode(object,required:true))
    var captured=object;captured["source_identity"]=identity
    let converted=try decode(captured,required:true)
    XCTAssertEqual(converted.sourceIdentity?.inode,2)
    XCTAssertEqual(converted.sourceIdentity?.headerSHA256,String(repeating:"a",count:64))
  }
  func testUnknownMissingAndWrongTaskFieldsReject() throws {
    for (key,value) in [("version",true),("version",2),("version",1.5),("engine","h3"),
      ("task","t2v"),("unexpected",1)] as [(String,Any)] {
      var changed=object;changed[key]=value
      XCTAssertThrowsError(try decode(changed),"must reject \(key)=\(value)")
    }
    for key in object.keys {
      var changed=object;changed.removeValue(forKey:key)
      XCTAssertThrowsError(try decode(changed),"must require \(key)")
    }
  }
  func testPathsMustBeBoundedAbsoluteAndOutputArgvMustMatchCanonicalPath() throws {
    for key in ["source_path","output_directory"] {
      for path in ["relative.safetensors","file:///tmp/weights","/","/tmp/zero\0tail","/tmp/"+String(repeating:"x",count:4096)] {
        var changed=object;changed[key]=path
        XCTAssertThrowsError(try decode(changed),"must reject \(key)")
      }
    }
    XCTAssertThrowsError(try decode(object,output:destination.appendingPathComponent("foreign")))
    var equivalent=object;equivalent["output_directory"]="/tmp/unused/../"+destination.lastPathComponent
    XCTAssertNoThrow(try decode(equivalent))
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let link=root.appendingPathComponent("alias")
    try FileManager.default.createSymbolicLink(at:link,withDestinationURL:root)
    var symlink=object;symlink["output_directory"]=link.appendingPathComponent("output").path
    XCTAssertNoThrow(try decode(symlink,output:root.appendingPathComponent("output")))
    symlink["output_directory"]=link.appendingPathComponent("new/nested/output").path
    let canonical=try decode(symlink,output:root.appendingPathComponent("new/nested/output"))
    let resolved=try XCTUnwrap(realpath(root.path,nil));defer { free(resolved) }
    XCTAssertEqual(canonical.outputDirectory.path,String(cString:resolved)+"/new/nested/output")
    XCTAssertEqual(try MLXTransformerPageConverter.canonicalLocalURL(canonical.outputDirectory).path,canonical.outputDirectory.path)
    XCTAssertThrowsError(try decode(symlink,output:root.appendingPathComponent("new/nested/foreign")))
    let dangling=root.appendingPathComponent("dangling")
    try FileManager.default.createSymbolicLink(at:dangling,withDestinationURL:root.appendingPathComponent("missing"))
    symlink["output_directory"]=dangling.appendingPathComponent("output").path
    XCTAssertThrowsError(try decode(symlink,output:root.appendingPathComponent("missing/output")))
    let file=root.appendingPathComponent("file");try Data([0]).write(to:file)
    symlink["output_directory"]=file.appendingPathComponent("output").path
    XCTAssertThrowsError(try decode(symlink,output:file.appendingPathComponent("output")))
  }
  func testIdentityFieldsAreExactAndBounded() throws {
    var captured=object
    for (key,value) in [("unknown",1),("inode",0),("inode",true),("bytes",0),("bytes",-1),
      ("modifiedNanos",-1),("changedNanos",1_000_000_000),("headerSHA256",String(repeating:"A",count:64)),
      ("headerSHA256","xyz")] as [(String,Any)] {
      var changed=identity;changed[key]=value;captured["source_identity"]=changed
      XCTAssertThrowsError(try decode(captured,required:true),"must reject identity \(key)")
    }
    for key in identity.keys {
      var changed=identity;changed.removeValue(forKey:key);captured["source_identity"]=changed
      XCTAssertThrowsError(try decode(captured,required:true))
    }
    captured["source_identity"]=NSNull()
    XCTAssertThrowsError(try decode(captured,required:true))
  }
}
