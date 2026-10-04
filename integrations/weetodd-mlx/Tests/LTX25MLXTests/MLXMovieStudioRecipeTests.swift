import XCTest
import Foundation
import CryptoKit
@testable import LTX25MLX

final class MLXMovieStudioRecipeTests:XCTestCase {
  private struct Fixture {
    let root:URL,originalOutput:URL,newOutput:URL
    let body:[String:Any],wrapper:[String:Any]
  }
  private func bytes(_ value:[String:Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes])
  }
  private func fixture() throws -> Fixture {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent("movie-wrapper-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    func media(_ name:String,_ data:Data) throws -> (path:String,sha:String) {
      let url=root.appendingPathComponent(name);try data.write(to:url,options:.withoutOverwriting)
      return (url.path,SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined())
    }
    let movie=try media("source.mov",Data("frozen-movie".utf8))
    let rgb=try media("source.rgb24",Data(repeating:71,count:32*32*9*3))
    let audio=try media("sidecar.wav",Data("frozen-audio".utf8))
    let first=try media("first.png",Data("first-endpoint".utf8))
    let last=try media("last.png",Data("last-endpoint".utf8))
    let old=root.appendingPathComponent("original-output"),new=root.appendingPathComponent("new-parent/take")
    let components=["video_checkpoint":"/model/video.safetensors","spatial_upscaler_checkpoint":"/model/upscaler.safetensors",
      "gemma_root":"/model/gemma","connector_checkpoint":"/model/fixed.safetensors","transformer_root":"/model/pages","audio_checkpoint":"/model/audio.safetensors"]
    let body:[String:Any]=["version":1,"engine":"ltx25","task":"video_upscale",
      "source":["path":movie.path,"sha256":movie.sha,"rgb_path":rgb.path,"rgb_sha256":rgb.sha,"width":32,"height":32,"frames":9,"fps":24,"start_seconds":0.5,"duration_seconds":9.0/24],
      "components":components,"mode":"refine","size_policy":"strict_32","output_directory":old.path,
      "prompt":"Keep the original motion.","seed":43,"refinement_strength":0.35,"anchors":"first_last","anchor_strength":0.7,"pixel_strength":1,
      "reference_images":[["role":"last","path":last.path,"strength":0.7,"crf":33],["role":"first","path":first.path,"strength":0.7,"crf":33]],
      "reference_image_sha256":[first.path:first.sha,last.path:last.sha],"audio_policy":"sidecar",
      "audio_source":["path":audio.path,"sha256":audio.sha,"start_seconds":1.25,"duration_seconds":NSNull()],
      "maximum_audio_drift_seconds":0.05,"chunking":false,"chunk_frame_megapixel_budget":260,"resume":false,"keep_chunks":false]
    let inputs:[[String:Any]]=[(movie,"video"),(rgb,"rgb24"),(audio,"audio"),(first,"image"),(last,"image")].map {
      ["path":$0.0.path,"sha256":$0.0.sha,"kind":$0.1]
    }
    let wrapper:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":body["prompt"]!,"config":[:] as [String:Any],
      "components":components,"conditioning":["task":"video_upscale","inputs":inputs],"movie_upscale":body]
    return Fixture(root:root,originalOutput:old,newOutput:new,body:body,wrapper:wrapper)
  }
  private func compile(_ wrapper:[String:Any],_ f:Fixture) throws -> MLXMovieStudioRecipe.Compiled {
    try MLXMovieStudioRecipe.compile(bytes(wrapper),outputDirectory:f.newOutput)
  }
  func testWrappedRecipeRebindsOnlyOutputAndPreservesExactPublicationBytes() throws {
    let f=try fixture(),original=try bytes(f.wrapper),compiled=try compile(f.wrapper,f)
    XCTAssertTrue(compiled.isStudioWrapper);XCTAssertEqual(compiled.originalRecipeBytes,original)
    XCTAssertEqual(compiled.request.source.startSeconds,0.5);XCTAssertEqual(compiled.request.audioSource?.startSeconds,1.25)
    XCTAssertEqual(compiled.request.referenceImages.map(\.role),["last","first"])
    XCTAssertEqual(compiled.request.outputDirectory,try MLXTransformerPageConverter.canonicalLocalURL(f.newOutput).path)
    var actual=try XCTUnwrap(JSONSerialization.jsonObject(with:compiled.requestBytes) as? [String:Any])
    actual["output_directory"]=f.originalOutput.path
    XCTAssertEqual(try bytes(actual),try bytes(f.body))
    XCTAssertFalse(FileManager.default.fileExists(atPath:f.newOutput.path))
  }
  func testBareV1HasNoRelocationAndKeepsOriginalBytes() throws {
    let f=try fixture(),original=try bytes(f.body)
    let compiled=try MLXMovieStudioRecipe.compile(original,outputDirectory:f.originalOutput)
    XCTAssertFalse(compiled.isStudioWrapper);XCTAssertEqual(compiled.originalRecipeBytes,original)
    XCTAssertEqual(compiled.requestBytes,original)
    XCTAssertThrowsError(try MLXMovieStudioRecipe.compile(original,outputDirectory:f.newOutput))
  }
  func testWrapperRejectsUnknownMissingAndConflictingDeclarations() throws {
    let f=try fixture()
    for key in ["format","engine","prompt","config","components","conditioning","movie_upscale"] {
      var v=f.wrapper;v.removeValue(forKey:key);XCTAssertThrowsError(try compile(v,f),key)
    }
    var v=f.wrapper;v["unknown"]=true;XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["format"]="weetodd-headless-v1";XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["engine"]="h3";XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["prompt"]="Different motion";XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["config"]=["seed":43];XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["components"]=["video_checkpoint":"/different.safetensors"];XCTAssertThrowsError(try compile(v,f))
    v=f.wrapper;v["conditioning"]=["task":"video_upscale","inputs":[],"ignored":true];XCTAssertThrowsError(try compile(v,f))
  }
  func testInputInventoryRequiresEveryExactRoleHashAndUniqueCanonicalPath() throws {
    let f=try fixture(),conditioning=f.wrapper["conditioning"] as! [String:Any],inputs=conditioning["inputs"] as! [[String:Any]]
    for index in inputs.indices {
      var v=f.wrapper,c=conditioning,list=inputs;list.remove(at:index);c["inputs"]=list;v["conditioning"]=c
      XCTAssertThrowsError(try compile(v,f),"missing input \(index)")
      for (key,bad) in [("sha256",String(repeating:"a",count:64)),("kind","video"),("path","/foreign-source")] where key != "kind" || index != 0 {
        list=inputs;list[index][key]=bad;c["inputs"]=list;v["conditioning"]=c;XCTAssertThrowsError(try compile(v,f),key)
      }
    }
    for altered in [inputs+[inputs[0]],inputs+[ ["path":"/extra","kind":"image","sha256":String(repeating:"a",count:64)] ]] {
      var v=f.wrapper,c=conditioning;c["inputs"]=altered;v["conditioning"]=c;XCTAssertThrowsError(try compile(v,f))
    }
    var v=f.wrapper,c=conditioning,list=inputs;list[0]["extra"]=false;c["inputs"]=list;v["conditioning"]=c
    XCTAssertThrowsError(try compile(v,f))
    list=inputs;list[0]["path"]="relative.mov";c["inputs"]=list;v["conditioning"]=c;XCTAssertThrowsError(try compile(v,f))
  }
  func testCanonicalInputAliasMatchesButCanonicalDuplicateStillRejects() throws {
    let f=try fixture(),alias=f.root.appendingPathComponent("alias")
    try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:f.root)
    var v=f.wrapper,c=v["conditioning"] as! [String:Any],list=c["inputs"] as! [[String:Any]]
    list[0]["path"]=alias.appendingPathComponent("source.mov").path;c["inputs"]=list;v["conditioning"]=c
    XCTAssertNoThrow(try compile(v,f))
    let outputAlias=alias.appendingPathComponent("new-parent/take")
    XCTAssertEqual(try MLXMovieStudioRecipe.compile(bytes(v),outputDirectory:outputAlias).request.outputDirectory,
      try MLXTransformerPageConverter.canonicalLocalURL(f.newOutput).path)
    list[1]=list[0];c["inputs"]=list;v["conditioning"]=c;XCTAssertThrowsError(try compile(v,f))
  }
  func testActualSourceMutationAndInvalidRGBLengthFailBeforeModelWork() throws {
    let f=try fixture(),source=f.body["source"] as! [String:Any]
    let url=URL(fileURLWithPath:source["path"] as! String)
    try Data("mutated-movie".utf8).write(to:url)
    XCTAssertThrowsError(try compile(f.wrapper,f))
    let g=try fixture();var v=g.wrapper,body=g.body,s=body["source"] as! [String:Any]
    let rgb=URL(fileURLWithPath:s["rgb_path"] as! String),short=Data(repeating:71,count:9)
    try short.write(to:rgb);let hash=SHA256.hash(data:short).map { String(format:"%02x",$0) }.joined()
    s["rgb_sha256"]=hash;body["source"]=s;v["movie_upscale"]=body
    var c=v["conditioning"] as! [String:Any],list=c["inputs"] as! [[String:Any]];list[1]["sha256"]=hash;c["inputs"]=list;v["conditioning"]=c
    XCTAssertThrowsError(try compile(v,g))
  }
  func testStrictNestedVersionAndMetadataBoundsCannotBeRepairedByWrapper() throws {
    let f=try fixture()
    for bad in [true as Any,2 as Any,"1" as Any] {
      var v=f.wrapper,body=f.body;body["version"]=bad;v["movie_upscale"]=body;XCTAssertThrowsError(try compile(v,f))
    }
    var v=f.wrapper,body=f.body;body["unsupported"]=true;v["movie_upscale"]=body;XCTAssertThrowsError(try compile(v,f))
    XCTAssertThrowsError(try MLXMovieStudioRecipe.compile(Data(repeating:32,count:1024*1024+1),outputDirectory:f.newOutput))
    XCTAssertThrowsError(try MLXMovieStudioRecipe.compile(bytes(f.wrapper),outputDirectory:URL(string:"https://example.com/output")!))
  }
}
