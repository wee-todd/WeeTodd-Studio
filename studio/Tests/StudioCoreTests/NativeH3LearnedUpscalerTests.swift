import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import StudioCore

final class NativeH3LearnedUpscalerTests:XCTestCase {
  func testExpandedCanvasAndActualTemporalRowsRejectBeforeMissingArtifactOrModel() throws {
    let source=H3JointLatentArtifact(manifest:"/missing/artifact.json",manifestSHA256:String(repeating:"a",count:64),
      payloadSHA256:String(repeating:"b",count:64),task:"t2va",componentIdentity:String(repeating:"c",count:64),width:960,height:544,generatedFrames:124)
    var selected=H3JointRefinementSettings(mode:.spatial,source:source);selected.expandedSpatialTarget=true
    let ordinary:[String:Any]=["components":["task":"t2va"],"config":["width":1920,"height":1088,"duration_seconds":Double(124)/24,"steps":20]]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(refinement:selected),to:ordinary,continuityMode:"independent")) {error in
      XCTAssertTrue(error.localizedDescription.contains("64000"),error.localizedDescription)
    }
    var small=source;small.width=640;small.height=576;small.generatedFrames=73
    selected.source=small
    let shortEdge:[String:Any]=["components":["task":"t2va"],"config":["width":1280,"height":1152,"duration_seconds":Double(73)/24,"steps":20]]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(refinement:selected),to:shortEdge,continuityMode:"independent")) {error in
      XCTAssertTrue(error.localizedDescription.contains("shortest edge"),error.localizedDescription)
    }
  }

  private func folder() throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent("H3Upscaler-\(UUID())")
    try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
    addTeardownBlock {try? FileManager.default.removeItem(at:url)};return url
  }
  private func sha(_ bytes:Data)->String {SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined()}
  /// Synthetic sparse file: no trained tensor payload is written or copied.
  private func checkpoint(_ root:URL,mutate:([String:Any])->[String:Any]={$0}) throws -> (URL,String) {
    var header:[String:Any]=[:],offset:UInt64=0
    for (name,shape) in NativeH3LearnedUpscalerMetadata.expectedShapes.sorted(by:{$0.key<$1.key}) {
      let count=UInt64(shape.reduce(1,*))*2
      header[name]=["dtype":"BF16","shape":shape,"data_offsets":[offset,offset+count]];offset+=count
    }
    let bytes=try JSONSerialization.data(withJSONObject:mutate(header),options:.sortedKeys)
    var length=UInt64(bytes.count).littleEndian
    let prefix=withUnsafeBytes(of:&length) {Data($0)}
    let file=root.appendingPathComponent("synthetic.safetensors")
    try (prefix+bytes).write(to:file)
    let handle=try FileHandle(forWritingTo:file);defer {try? handle.close()}
    try handle.truncate(atOffset:UInt64(8+bytes.count)+offset)
    return(file,sha(prefix+bytes))
  }
  private func artifact(_ root:URL,width:Int=768,height:Int=448,frames:Int=73,expanded:Bool=false) throws -> H3JointLatentArtifact {
    let video=((frames-5)/17*5+2)*(width/32)*(height/32)*96
    let audio=2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))*32
    let payload=Data(count:(video+audio)*4),path=root.appendingPathComponent("joint.json")
    try payload.write(to:root.appendingPathComponent("joint-latents.f32"))
    let fields:[String:Any]=["format":expanded ? "weetodd-h3-swift-joint-latents-v2-spatial" : "weetodd-h3-swift-joint-latents-v1",
      "task":"t2va","width":width,"height":height,"generatedFrames":frames,
      "componentIdentity":String(repeating:"c",count:64),"videoFloats":video,"audioFloats":audio,
      "payloadBytes":payload.count,"payloadSHA256":sha(payload)]
    let bytes=try JSONSerialization.data(withJSONObject:fields);try bytes.write(to:path)
    return try XCTUnwrap(H3JointLatentArtifact.adopt(metadata:["jointLatentManifest":path.path,
      "jointLatentManifestSHA256":sha(bytes),"jointLatentPayloadSHA256":sha(payload)]))
  }
  func testExactHeaderSchemaPrefixPinAndMutationWithoutPayloadRead() throws {
    let root=try folder(),(path,pin)=try checkpoint(root)
    let actual=try NativeH3LearnedUpscalerMetadata.inspect(path:path.path,expectedHeaderSHA256:pin)
    XCTAssertEqual(actual.tensorBytes,690_560_432);XCTAssertEqual(actual.headerSHA256,pin)
    XCTAssertThrowsError(try NativeH3LearnedUpscalerMetadata.inspect(path:path.path,expectedHeaderSHA256:String(repeating:"a",count:64)))
    _=try checkpoint(root) {fields in
      var fields=fields,value=fields["conv_in.weight"] as! [String:Any];value["shape"]=[512,128,3,3,3];fields["conv_in.weight"]=value;return fields
    }
    XCTAssertThrowsError(try NativeH3LearnedUpscalerMetadata.inspect(path:path.path))
    _=try checkpoint(root) {fields in
      var fields=fields,value=fields["conv_in.bias"] as! [String:Any];value["data_offsets"]=[true,1024];fields["conv_in.bias"]=value;return fields
    }
    XCTAssertThrowsError(try NativeH3LearnedUpscalerMetadata.inspect(path:path.path))
  }
  func testExplicitV2LearnedOrInterpolationKeepsBaseIdentityAndRejectsIgnoredControls() throws {
    let root=try folder(),source=try artifact(root),(model,pin)=try checkpoint(root)
    let ordinary:[String:Any]=["components":["task":"t2va"],"config":["width":1536,"height":896,"duration_seconds":Double(73)/24,"steps":20]]
    var selected=H3JointRefinementSettings(mode:.spatial,source:source)
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(refinement:selected),to:ordinary,continuityMode:"independent"))
    selected.expandedSpatialTarget=true
    let interpolated=try NativeH3JointPreparation.apply(.init(refinement:selected),to:ordinary,continuityMode:"independent")
    XCTAssertEqual((interpolated["refinement"] as? [String:Any])?["version"] as? Int,2)
    selected.learnedUpscalerPath=model.path;selected.learnedUpscalerHeaderSHA256=pin
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(refinement:selected),to:ordinary,continuityMode:"independent"))
    selected.resizeMethod=nil
    let recipe=try NativeH3JointPreparation.apply(.init(saveFullLatents:true,refinement:selected),to:ordinary,continuityMode:"independent")
    let fields=recipe["refinement"] as! [String:Any]
    XCTAssertEqual(fields["learned_upscaler_path"] as? String,model.path);XCTAssertEqual(fields["learned_upscaler_header_sha256"] as? String,pin)
    XCTAssertNil(fields["resize_method"]);XCTAssertEqual(source.componentIdentity,String(repeating:"c",count:64))
    XCTAssertEqual((recipe["joint_latents"] as? [String:Any])?["save_full"] as? Bool,true)
    selected.mode = .initialized
    var same=ordinary;same["config"]=["width":768,"height":448,"duration_seconds":Double(73)/24,"steps":20]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(refinement:selected),to:same,continuityMode:"independent"))
  }
  func testExpandedArtifactAdoptionReopenAndHeadlessSourceCollectionExcludeModel() throws {
    let root=try folder(),source=try artifact(root,width:1536,height:896,expanded:true)
    try source.verify();XCTAssertEqual(source.componentIdentity,String(repeating:"c",count:64))
    var project=StudioProject();project.settings.width=1536;project.settings.height=896
    var clip=Clip(engine:.h3)
    clip.duration=Double(73)/24;clip.generationWidth=1536;clip.generationHeight=896
    clip.versions=[RenderVersion(path:"/output/render.mp4",seed:42,prompt:"same",recipePath:"",jointLatentArtifact:source)]
    project.clips=[clip]
    let projectURL=root.appendingPathComponent("project.json");try ProjectStorage.write(project,to:projectURL)
    let reopened=try ProjectStorage.read(projectURL)
    XCTAssertEqual(reopened.clips[0].versions[0].jointLatentArtifact,source)
    try reopened.clips[0].versions[0].jointLatentArtifact?.verify()
    let bytes=try JSONSerialization.data(withJSONObject:["format":"weetodd-headless-v2","engine":"h3","config":["seed":42],
      "refinement":["version":2,"mode":"spatial","source_manifest":source.manifest,"source_manifest_sha256":source.manifestSHA256,
        "learned_upscaler_path":"/unavailable/model.safetensors","learned_upscaler_header_sha256":String(repeating:"a",count:64)]])
    let job=try NativeHeadlessJob(project:project,recipes:[clip.id.uuidString:.init(engine:"h3",bytes:bytes,signature:"same")],workers:["h3":"/usr/bin/true"],ffmpeg:"/usr/bin/true")
    let expectedSources=try [source.manifest,source.payloadPath].map {
      let pointer=try XCTUnwrap(Darwin.realpath($0,nil))
      defer { free(pointer) };return String(cString:pointer)
    }
    XCTAssertEqual(Set(job.sources.map(\.path)),Set(expectedSources))
    XCTAssertNoThrow(try job.verifySources())
    let jobURL=root.appendingPathComponent("job.json");try job.write(to:jobURL)
    let loadedJob=try NativeHeadlessJob.read(from:jobURL)
    XCTAssertEqual(loadedJob.sources,job.sources)
    XCTAssertEqual(loadedJob.recipes[clip.id.uuidString]?.bytes,bytes)
    let collected=try ProjectStorage.collectJointLatentArtifact(source,to:root.appendingPathComponent("collected"))
    try collected.verify();XCTAssertEqual(collected.componentIdentity,source.componentIdentity)
    var fields=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:source.manifest))) as! [String:Any]
    fields["format"]="weetodd-h3-swift-joint-latents-v1"
    let changed=try JSONSerialization.data(withJSONObject:fields);try changed.write(to:URL(fileURLWithPath:source.manifest))
    XCTAssertThrowsError(try H3JointLatentArtifact.adopt(metadata:["jointLatentManifest":source.manifest,"jointLatentManifestSHA256":sha(changed),"jointLatentPayloadSHA256":source.payloadSHA256]))
  }
  func testNilSettingsEncodingAndMappedModelPathPreservePin() throws {
    let root=try folder(),source=try artifact(root)
    var selected=H3JointRefinementSettings(mode:.spatial,source:source)
    let old=try JSONSerialization.jsonObject(with:JSONEncoder().encode(selected)) as! [String:Any]
    XCTAssertNil(old["expandedSpatialTarget"]);XCTAssertNil(old["learnedUpscalerPath"]);XCTAssertNil(old["learnedUpscalerHeaderSHA256"])
    selected.expandedSpatialTarget=true;selected.learnedUpscalerPath="/library/model.safetensors";selected.learnedUpscalerHeaderSHA256=String(repeating:"a",count:64);selected.resizeMethod=nil
    var selection=GenerationSelection();selection.h3Joint = .init(refinement:selected)
    selection.mapH3CreativePaths {"/mapped"+$0}
    XCTAssertEqual(selection.h3Joint?.refinement?.learnedUpscalerPath,"/mapped/library/model.safetensors")
    XCTAssertEqual(selection.h3Joint?.refinement?.learnedUpscalerHeaderSHA256,selected.learnedUpscalerHeaderSHA256)
  }
  func testCancelledHeaderAdmissionNeverNeedsModelTensorPayload() async throws {
    let root=try folder(),(file,_)=try checkpoint(root),filePath=file.path
    let task=Task<Void,Error>.detached { @Sendable [filePath] in
      withUnsafeCurrentTask {$0?.cancel()}
      _=try NativeH3LearnedUpscalerMetadata.inspect(path:filePath)
    }
    do {try await task.value;XCTFail("Cancelled model-header admission must throw")}
    catch {XCTAssertTrue(error is CancellationError,String(describing:error))}
  }
}
