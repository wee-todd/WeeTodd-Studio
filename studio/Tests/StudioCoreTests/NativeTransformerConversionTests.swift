import Foundation
import Darwin
import CryptoKit
import XCTest
@testable import StudioCore

final class NativeTransformerConversionTests:XCTestCase {
  func testInstalledNativeDevConversionAdoptsAllPublishedPagesWithoutWeightReads() throws {
    let environment=ProcessInfo.processInfo.environment
    guard let source=environment["WEETODD_TEST_LTX_RAW_DEV"],
      let converted=environment["WEETODD_TEST_LTX_CONVERTED_DEV"] else {
      throw XCTSkip("Set explicit installed raw and native-converted Dev paths for header-only adoption qualification.")
    }
    let destination=URL(fileURLWithPath:converted),identity=try NativeTransformerConversion.currentIdentity(source)
    let response:[String:Any]=["nativeRuntime":"swift-mlx","outputDirectory":converted,
      "manifestPath":destination.appendingPathComponent("paged_manifest.json").path]
    let adopted=try NativeTransformerConversion.completedDirectory(response,source:source,destination:destination,sourceIdentity:identity)
    XCTAssertEqual(adopted.resolvingSymlinksInPath(),destination.resolvingSymlinksInPath())
    let manifestData=try Data(contentsOf:destination.appendingPathComponent("paged_manifest.json"))
    let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:manifestData) as? [String:Any])
    XCTAssertEqual((manifest["layers"] as? [Any])?.count,48)
    XCTAssertEqual(manifest["source"] as? String,NativeTransformerConversion.devSourceName)
    let receipt:[String:Any]=["status":"passed","scope":"Studio native conversion adoption of actual installed page headers; no inference or new conversion",
      "source":source,"source_identity":identity,"converted_directory":adopted.path,
      "manifest_sha256":NativeHeadlessJob.hash(manifestData),"validated_pages":49,
      "nativeRuntime":"swift-mlx","weighted_inference":false,"converter_rerun":false]
    if let receiptPath=environment["WEETODD_TEST_LTX_CONVERSION_RECEIPT"] {
      XCTAssertTrue(receiptPath.hasPrefix("/"))
      try JSONSerialization.data(withJSONObject:receipt,options:[.prettyPrinted,.sortedKeys])
        .write(to:URL(fileURLWithPath:receiptPath),options:.withoutOverwriting)
    }
  }
  private func tensorFile(_ url:URL,tensors:[String:[Int]],dtype:String="F32",metadata:[String:String]=[:]) throws -> Int {
    var header:[String:Any]=[:],payload=Data()
    if !metadata.isEmpty { header["__metadata__"]=metadata }
    for name in tensors.keys.sorted() {
      let shape=tensors[name]!,start=payload.count,count=shape.reduce(1,*)*4
      payload.append(Data(repeating:0,count:count))
      header[name]=["dtype":dtype,"shape":shape,"data_offsets":[start,payload.count]]
    }
    var data=try JSONSerialization.data(withJSONObject:header,options:.sortedKeys)
    data.append(Data(repeating:32,count:(8-data.count%8)%8))
    var size=UInt64(data.count).littleEndian
    try (withUnsafeBytes(of:&size) { Data($0) }+data+payload).write(to:url)
    return payload.count
  }
  private func fixture(_ body:(URL,URL,[String:Any],[String:Any]) throws -> Void) throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let source=root.appendingPathComponent(NativeTransformerConversion.devSourceName),output=root.appendingPathComponent("output")
    var tensors=["model.diffusion_model.fixed":[1]]
    for i in 0..<48 { tensors["model.diffusion_model.transformer_blocks.\(i).bias"]=[1] }
    _ = try tensorFile(source,tensors:tensors,metadata:["model_version":"2.5.0","config":"{\"transformer\":{\"num_layers\":48}}"])
    let identity=try NativeTransformerConversion.currentIdentity(source.path)
    try FileManager.default.createDirectory(at:output.appendingPathComponent("pages"),withIntermediateDirectories:true)
    var records:[[String:Any]]=[]
    for i in 0..<49 {
      let name=i==0 ? "fixed.safetensors" : String(format:"layer-%03d.safetensors",i-1)
      let tensor=i==0 ? "model.diffusion_model.fixed" : "model.diffusion_model.transformer_blocks.\(i-1).bias"
      let url=output.appendingPathComponent("pages/"+name)
      let bytes=try tensorFile(url,tensors:[tensor:[1]])
      records.append(["file":"pages/"+name,"tensor_count":1,"tensor_bytes":bytes,
        "sha256":SHA256.hash(data:try Data(contentsOf:url)).map { String(format:"%02x",$0) }.joined()])
    }
    let manifest:[String:Any]=["format":"weetodd-ltx25-transformer-paged-q8-v1","kind":"transformer","num_layers":48,
      "bits":8,"group_size":64,"source":NativeTransformerConversion.devSourceName,
      "metadata":["model_version":"2.5.0","config":["transformer":["num_layers":48]]],
      "fixed":records[0],"layers":Array(records.dropFirst()),"source_tensor_bytes":196,"output_tensor_bytes":196,
      "conversion_provenance":["implementation":"swift-mlx-affine-q8-v1","source_identity":identity,
        "source_sha256":SHA256.hash(data:try Data(contentsOf:source)).map { String(format:"%02x",$0) }.joined()]]
    try JSONSerialization.data(withJSONObject:manifest).write(to:output.appendingPathComponent("paged_manifest.json"))
    let result:[String:Any]=["nativeRuntime":"swift-mlx","outputDirectory":output.path,"manifestPath":output.appendingPathComponent("paged_manifest.json").path]
    try body(source,output,identity,result)
  }
  func testStrictRequestAndCapturedIdentityWithoutPayloadReads() throws {
    try fixture { source,output,identity,_ in
      let request=try NativeTransformerConversion.request(source:source.path,destination:output,sourceIdentity:identity,requiresIdentity:true)
      XCTAssertEqual(Set(request.keys),["version","engine","task","source_path","output_directory","source_identity"])
      XCTAssertThrowsError(try NativeTransformerConversion.request(source:source.path,destination:output,requiresIdentity:true))
      XCTAssertThrowsError(try NativeTransformerConversion.request(source:"relative",destination:output))
      var foreign=identity;foreign["unexpected"]=true
      XCTAssertThrowsError(try NativeTransformerConversion.request(source:source.path,destination:output,sourceIdentity:foreign))
      XCTAssertEqual(try NativeTransformerConversion.capturedIdentity(["nativeRuntime":"swift-mlx","sourceIdentity":identity],source:source.path)["headerSHA256"] as? String,identity["headerSHA256"] as? String)
      foreign=identity;foreign["inode"]=0
      XCTAssertThrowsError(try NativeTransformerConversion.capturedIdentity(["nativeRuntime":"swift-mlx","sourceIdentity":foreign],source:source.path))
      let root=source.deletingLastPathComponent(),alias=root.appendingPathComponent("alias")
      try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:root)
      let nested=try NativeTransformerConversion.request(source:source.path,destination:alias.appendingPathComponent("new/nested/output"))
      let resolved=try XCTUnwrap(realpath(root.path,nil));defer { free(resolved) }
      XCTAssertEqual(nested["output_directory"] as? String,String(cString:resolved)+"/new/nested/output")
      let repeated=try NativeTransformerConversion.request(source:request["source_path"] as! String,
        destination:URL(fileURLWithPath:nested["output_directory"] as! String))
      XCTAssertEqual(repeated["output_directory"] as? String,nested["output_directory"] as? String)
      let dangling=root.appendingPathComponent("dangling")
      try FileManager.default.createSymbolicLink(at:dangling,withDestinationURL:root.appendingPathComponent("missing"))
      XCTAssertThrowsError(try NativeTransformerConversion.request(source:source.path,destination:dangling.appendingPathComponent("output")))
    }
  }
  func testCompletePagesMustMatchSourceHeaderAndExactDestination() throws {
    try fixture { source,output,identity,result in
      let resolved=try XCTUnwrap(realpath(output.path,nil));defer { free(resolved) }
      let canonicalOutput=String(cString:resolved)
      XCTAssertEqual(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity).path,canonicalOutput)
      let alias=source.deletingLastPathComponent().appendingPathComponent("alias")
      try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:source.deletingLastPathComponent())
      XCTAssertEqual(try NativeTransformerConversion.completedDirectory(result,source:source.path,
        destination:alias.appendingPathComponent("output"),sourceIdentity:identity).path,canonicalOutput)
      var changed=result;changed["outputDirectory"]=output.appendingPathComponent("foreign").path
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(changed,source:source.path,destination:output,sourceIdentity:identity))
      let page=output.appendingPathComponent("pages/layer-047.safetensors")
      _ = try tensorFile(page,tensors:["wrong-source-target":[1]])
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
    }
  }
  func testMissingOrSymlinkedPageAndTraversalCannotBeAdopted() throws {
    try fixture { source,output,identity,result in
      let page=output.appendingPathComponent("pages/layer-047.safetensors")
      try FileManager.default.removeItem(at:page)
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
      try FileManager.default.createSymbolicLink(at:page,withDestinationURL:output.appendingPathComponent("pages/layer-046.safetensors"))
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
      let url=output.appendingPathComponent("paged_manifest.json")
      var manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
      var fixed=manifest["fixed"] as! [String:Any];fixed["file"]="pages/../outside.safetensors";manifest["fixed"]=fixed
      try JSONSerialization.data(withJSONObject:manifest).write(to:url)
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
    }
  }
  func testSourceMutationAndMergedOrOversizedManifestReject() throws {
    try fixture { source,output,identity,result in
      let url=output.appendingPathComponent("paged_manifest.json")
      var manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
      var metadata=manifest["metadata"] as! [String:Any];metadata["weetodd_baked_loras"]=[];manifest["metadata"]=metadata
      try JSONSerialization.data(withJSONObject:manifest).write(to:url)
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
      try Data(repeating:32,count:1024*1024+1).write(to:url)
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
      try Data("changed".utf8).write(to:source)
      XCTAssertThrowsError(try NativeTransformerConversion.capturedIdentity(["nativeRuntime":"swift-mlx","sourceIdentity":identity],source:source.path))
    }
  }
}
