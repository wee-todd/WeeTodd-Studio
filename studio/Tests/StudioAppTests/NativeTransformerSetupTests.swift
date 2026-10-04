import Foundation
import Darwin
import CryptoKit
import XCTest
import StudioCore
@testable import WeeToddStudio

@MainActor final class NativeTransformerSetupTests:XCTestCase {
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
  private func fixture(_ body:(URL,URL,[String:Any],[String:Any]) async throws -> Void) async throws {
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
    try await body(source,output,identity,result)
  }
  private func quoted(_ s:String) -> String { "'"+s.replacingOccurrences(of:"'",with:"'\\''")+"'" }
  private func setup(source:URL,output:URL,identity:[String:Any],result:[String:Any],failure:Bool=false,delay:Bool=false) throws -> (StudioStore,ModelSetupState,URL) {
    let root=source.deletingLastPathComponent(),template=root.appendingPathComponent("template")
    try FileManager.default.moveItem(at:output,to:template)
    let admission=root.appendingPathComponent("admission.json"),completion=root.appendingPathComponent("completion.json")
    try JSONSerialization.data(withJSONObject:["status":"success","result":["nativeRuntime":"swift-mlx","sourceIdentity":identity]]).write(to:admission)
    try JSONSerialization.data(withJSONObject:["status":"success","result":result]).write(to:completion)
    let log=root.appendingPathComponent("commands.txt"),worker=root.appendingPathComponent("worker")
    let script="""
    #!/bin/sh
    trap 'exit 130' INT TERM
    test "$2" = --request || exit 11
    test "$4" = --output || exit 12
    test "$5" = \(quoted(output.path)) || exit 13
    printf '%s\\n' "$1" >> \(quoted(log.path))
    case "$1" in
      preflight-transformer-conversion)
        /bin/cp "$3" \(quoted(root.appendingPathComponent("captured-preflight.json").path))
        \(failure ? "exit 17" : "/bin/cat "+quoted(admission.path))
        ;;
      convert-transformer)
        /bin/cp "$3" \(quoted(root.appendingPathComponent("captured-conversion.json").path))
        echo '{"event":"progress","fraction":0.1,"message":"conversion fixture entered"}'
        \(delay ? "/bin/sleep 1" : ":")
        /bin/mv \(quoted(template.path)) \(quoted(output.path)) || exit 14
        /bin/cat \(quoted(completion.path))
        ;;
      *) exit 15 ;;
    esac
    """
    try script.write(to:worker,atomically:true,encoding:.utf8)
    try FileManager.default.setAttributes([.posixPermissions:0o755],ofItemAtPath:worker.path)
    let store=StudioStore(dataDirectory:root.appendingPathComponent("app"),restoreSession:false)
    store.runtime=RuntimeSettings(root:"/unavailable",pythonPath:"/unavailable/python",profilesDirectory:root.appendingPathComponent("profiles").path)
    store.runtime.ltx25WorkerPath=worker.path
    let state=ModelSetupState()
    state.begin(try XCTUnwrap(NativeModelSetup.catalog().first { $0.components.contains { $0.key=="dev_transformer_path" } }))
    state.selection.components["dev_transformer_path"]=source.path
    return (store,state,log)
  }
  func testNativeDevConversionUsesCapturedIdentityAndAdoptsCompletedPagesWithoutPython() async throws {
    try await fixture { source,output,identity,result in
      let (store,state,log)=try setup(source:source,output:output,identity:identity,result:result)
      await state.convertDevTransformer(store:store,destination:output)
      XCTAssertTrue(state.error.isEmpty,state.error)
      let resolved=try XCTUnwrap(realpath(output.path,nil));defer { free(resolved) }
      XCTAssertEqual(state.selection.components["dev_transformer_path"],String(cString:resolved))
      XCTAssertTrue(state.resultPath.isEmpty,"Conversion alone must not create a profile")
      XCTAssertFalse(state.convertingTransformer);XCTAssertFalse(store.bridge.busy)
      XCTAssertEqual(try String(contentsOf:log,encoding:.utf8),"preflight-transformer-conversion\nconvert-transformer\n")
      let request=try JSONSerialization.jsonObject(with:Data(contentsOf:source.deletingLastPathComponent().appendingPathComponent("captured-conversion.json"))) as! [String:Any]
      XCTAssertEqual(Set(request.keys),["version","engine","task","source_path","output_directory","source_identity"])
      XCTAssertTrue(NSDictionary(dictionary:request["source_identity"] as! [String:Any]).isEqual(to:identity))
      XCTAssertNotNil(store.bridge.workerEventsPath)
      XCTAssertFalse(store.bridge.workerEventsTruncated)
    }
  }
  func testExistingPagesCannotBeOverwrittenAndPreflightFailureDoesNotAdopt() async throws {
    try await fixture { source,output,identity,result in
      let (store,state,log)=try setup(source:source,output:output,identity:identity,result:result,failure:true)
      try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
      await state.convertDevTransformer(store:store,destination:output)
      XCTAssertTrue(state.error.contains("never overwritten"));XCTAssertFalse(FileManager.default.fileExists(atPath:log.path))
      try FileManager.default.removeItem(at:output)
      await state.convertDevTransformer(store:store,destination:output)
      XCTAssertFalse(state.error.isEmpty)
      XCTAssertEqual(state.selection.components["dev_transformer_path"],source.path)
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
      XCTAssertEqual(try String(contentsOf:log,encoding:.utf8),"preflight-transformer-conversion\n")
    }
  }
  func testChangedSelectionDoesNotAdoptLateCompletedConversion() async throws {
    try await fixture { source,output,identity,result in
      let (store,state,log)=try setup(source:source,output:output,identity:identity,result:result,delay:true)
      let task=Task { await state.convertDevTransformer(store:store,destination:output) }
      for _ in 0..<400 {
        if (try? String(contentsOf:log,encoding:.utf8).contains("\nconvert-transformer\n"))==true { break }
        try await Task.sleep(nanoseconds:5_000_000)
      }
      XCTAssertTrue((try? String(contentsOf:log,encoding:.utf8).contains("\nconvert-transformer\n"))==true)
      state.selection.components["dev_transformer_path"]="/changed/selection"
      await task.value
      XCTAssertEqual(state.selection.components["dev_transformer_path"],"/changed/selection")
      XCTAssertTrue(FileManager.default.fileExists(atPath:output.appendingPathComponent("paged_manifest.json").path))
      XCTAssertTrue(state.resultPath.isEmpty)
    }
  }
  func testActualNativeConversionProcessCancellationKeepsRawSelection() async throws {
    try await fixture { source,output,identity,result in
      let (store,state,log)=try setup(source:source,output:output,identity:identity,result:result,delay:true)
      let task=Task { await state.convertDevTransformer(store:store,destination:output) }
      for _ in 0..<400 {
        if (try? String(contentsOf:log,encoding:.utf8).contains("\nconvert-transformer\n"))==true { break }
        try await Task.sleep(nanoseconds:5_000_000)
      }
      XCTAssertTrue(store.bridge.busy)
      state.cancelTransformerConversion(bridge:store.bridge)
      await task.value
      XCTAssertEqual(state.selection.components["dev_transformer_path"],source.path)
      XCTAssertFalse(state.convertingTransformer);XCTAssertFalse(store.bridge.busy)
      XCTAssertTrue(state.resultPath.isEmpty)
      XCTAssertNotNil(store.bridge.workerEventsPath)
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
    }
  }
  func testChangedSourceCannotReachConversionAndInstalledPagesNeedNoConversion() async throws {
    try await fixture { source,output,identity,result in
      let (store,state,log)=try setup(source:source,output:output,identity:identity,result:result)
      _ = try tensorFile(source,tensors:["changed.source":[2]])
      await state.convertDevTransformer(store:store,destination:output)
      XCTAssertFalse(state.error.isEmpty)
      XCTAssertEqual(state.selection.components["dev_transformer_path"],source.path)
      XCTAssertEqual(try String(contentsOf:log,encoding:.utf8),"preflight-transformer-conversion\n")
      let commands=try Data(contentsOf:log)
      state.selection.components["dev_transformer_path"]=source.deletingLastPathComponent().appendingPathComponent("template").path
      XCTAssertNil(state.rawDevTransformer)
      await state.convertDevTransformer(store:store,destination:output)
      XCTAssertEqual(try Data(contentsOf:log),commands)
    }
  }
  func testRawDevActionUsesResolvedSourceTaskIdentity() async throws {
    try await fixture { source,_,_,_ in
      let state=ModelSetupState()
      state.begin(try XCTUnwrap(NativeModelSetup.catalog().first { $0.components.contains { $0.key=="dev_transformer_path" } }))
      let root=source.deletingLastPathComponent(),aliases=root.appendingPathComponent("aliases")
      try FileManager.default.createDirectory(at:aliases,withIntermediateDirectories:true)
      let validAlias=aliases.appendingPathComponent("custom-name.safetensors")
      try FileManager.default.createSymbolicLink(at:validAlias,withDestinationURL:source)
      state.selection.components["dev_transformer_path"]=validAlias.path
      XCTAssertEqual(state.rawDevTransformer,validAlias.path)
      let foreign=root.appendingPathComponent("ltx-distilled.safetensors")
      try FileManager.default.copyItem(at:source,to:foreign)
      let wrongAlias=aliases.appendingPathComponent(NativeTransformerConversion.devSourceName)
      try FileManager.default.createSymbolicLink(at:wrongAlias,withDestinationURL:foreign)
      state.selection.components["dev_transformer_path"]=wrongAlias.path
      XCTAssertNil(state.rawDevTransformer,"A Dev-named alias must not relabel another checkpoint task")
    }
  }
  func testActualManifestResponseCannotNameUnpublishedLegacyManifest() async throws {
    try await fixture { source,output,identity,result in
      XCTAssertNoThrow(try NativeTransformerConversion.completedDirectory(result,source:source.path,destination:output,sourceIdentity:identity))
      var old=result;old["manifestPath"]=output.appendingPathComponent("manifest.json").path
      XCTAssertThrowsError(try NativeTransformerConversion.completedDirectory(old,source:source.path,destination:output,sourceIdentity:identity)) {
        XCTAssertTrue($0.localizedDescription.contains("differs"))
      }
      XCTAssertTrue(FileManager.default.fileExists(atPath:output.appendingPathComponent("paged_manifest.json").path))
      XCTAssertFalse(FileManager.default.fileExists(atPath:output.appendingPathComponent("manifest.json").path))
    }
  }

}
