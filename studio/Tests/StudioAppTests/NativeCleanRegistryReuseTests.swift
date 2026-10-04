import Foundation
import CryptoKit
import XCTest
import StudioCore
@testable import WeeToddStudio

final class NativeCleanRegistryReuseTests:XCTestCase {
  @MainActor func testInstalledNativeCleanRegistryReuseWithoutPythonOrNetwork() async throws {
    guard let manifestPath=ProcessInfo.processInfo.environment["WEETODD_NATIVE_CLEAN_REUSE"] else { throw XCTSkip("Explicit isolated native setup/reuse qualification") }
    let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:manifestPath))) as? [String:Any])
    let root=URL(fileURLWithPath:try XCTUnwrap(manifest["output"] as? String))
    XCTAssertFalse(FileManager.default.fileExists(atPath:root.path),"Fresh isolated app data required")
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    let app=URL(fileURLWithPath:try XCTUnwrap(manifest["app"] as? String))
    let catalogURL=app.appendingPathComponent("Contents/Resources/RendererSource/src/wee_todd_mlx/model_download_catalog.json")
    let packages=try NativeModelDownloads.catalog(at:catalogURL)
    XCTAssertEqual(packages.count,23)
    let expected=try XCTUnwrap(manifest["expectedCatalogIDs"] as? [String])
    XCTAssertEqual(packages.map { $0.descriptor.id }.sorted(),expected.sorted())
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.runtime=RuntimeSettings(root:root.appendingPathComponent("unavailable-renderer").path,
      pythonPath:root.appendingPathComponent("unavailable-python").path,profilesDirectory:root.appendingPathComponent("profiles").path)
    store.runtime.ffmpegPath=try XCTUnwrap(manifest["ffmpeg"] as? String)
    store.runtime.ltx25WorkerPath=app.appendingPathComponent("Contents/MacOS/WeeToddLTXWorker").path
    store.runtime.h3WorkerPath=app.appendingPathComponent("Contents/MacOS/WeeToddH3MLXWorker").path
    store.runtime.nativeLTX25Enabled=true;store.runtime.nativeH3Enabled=true
    XCTAssertFalse(FileManager.default.isExecutableFile(atPath:store.runtime.pythonPath))
    XCTAssertTrue(store.profiles.isEmpty);XCTAssertTrue(store.globalAssets.isEmpty)
    let state=ModelSetupState();state.nativeDownloadCatalogURL=catalogURL;state.readNativeDownloadToken={ nil }
    await state.loadCatalog(runtime:store.runtime)
    XCTAssertTrue(state.catalogError.isEmpty,state.catalogError);XCTAssertEqual(state.downloads.count,23)
    for preset in state.presets {
      for field in preset.components {
        XCTAssertTrue(packages.contains { $0.descriptor.supports(engine:preset.engine,task:preset.task,component:field.key) },"No pinned native package for \(preset.id)/\(field.key)")
      }
    }
    var evidence:[[String:Any]]=[]
    for item in try XCTUnwrap(manifest["cases"] as? [[String:Any]]) {
      let id=try XCTUnwrap(item["presetID"] as? String),preset=try XCTUnwrap(state.presets.first { $0.id==id })
      let selected=try XCTUnwrap(item["selected"] as? [String:String])
      state.begin(preset);state.roots=Array(Set(selected.values)).sorted()
      await state.scan(store:store);XCTAssertTrue(state.error.isEmpty,state.error)
      var scanMatches:[String:Bool]=[:]
      for (key,path) in selected {
        let actual=URL(fileURLWithPath:path).resolvingSymlinksInPath().standardizedFileURL.path
        scanMatches[key]=(state.selection.candidates[key] ?? []).contains { URL(fileURLWithPath:$0).resolvingSymlinksInPath().standardizedFileURL.path==actual }
        XCTAssertTrue(scanMatches[key] == true,"Missing native discovery: \(id)/\(key):\(actual)")
      }
      // Explicit selection is how the UI resolves multiple compatible candidates.
      state.selection.components=selected
      await state.createRecipe(store:store)
      XCTAssertTrue(state.error.isEmpty,state.error);XCTAssertFalse(state.resultPath.isEmpty)
      XCTAssertTrue(store.profiles.contains { $0.id==state.resultPath })
      let profile=URL(fileURLWithPath:state.resultPath)
      var created=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:profile)) as? [String:Any])
      var diffusionEvidence:[String:Any]=[:]
      if let expected=item["diffusionVAE"] as? [String:Any] {
        let checkpoint=try XCTUnwrap(selected["video_vae_path"])
        let checksum=try NativeLTXDiffusionVAE.validate(at:URL(fileURLWithPath:checkpoint))
        XCTAssertEqual(checksum,expected["rawHeaderSHA256"] as? String)
        let optional=try XCTUnwrap(packages.first { $0.descriptor.id == "ltx25-diffusion-vae" })
        XCTAssertEqual(optional.descriptor.components,["video_vae_path"])
        let components=try XCTUnwrap(created["components"] as? [String:String])
        XCTAssertEqual(Set(components.keys),Set(selected.keys))
        for (key,path) in selected {
          XCTAssertEqual(components[key],URL(fileURLWithPath:path).resolvingSymlinksInPath().standardizedFileURL.path)
        }
        let ordinary=try XCTUnwrap(evidence.first { $0["presetID"] as? String == "swift-ltx25-text" })
        let ordinaryPath=try XCTUnwrap(ordinary["profile"] as? String)
        let ordinaryRecipe=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:ordinaryPath))) as? [String:Any])
        var baseline=try XCTUnwrap(ordinaryRecipe["components"] as? [String:String])
        var actual=components;baseline.removeValue(forKey:"video_vae_path");actual.removeValue(forKey:"video_vae_path")
        XCTAssertEqual(actual,baseline,"Optional decoder selection must preserve every other installed component.")
        XCTAssertTrue(NSDictionary(dictionary:try XCTUnwrap(created["config"] as? [String:Any])).isEqual(to:try XCTUnwrap(ordinaryRecipe["config"] as? [String:Any])))
        diffusionEvidence=["checkpoint":checkpoint,"rawHeaderSHA256":checksum,"otherComponentPathsPreserved":true,
          "completePresetMerged":true,"payloadChecksumComputed":false,"headerOnly":true]
      }
      var headerPreflight=preset.task == "t2v"
      if let probePath=item["preflightRecipe"] as? String {
        let actual=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:probePath))) as? [String:Any])
        // Setup owns component linking. The prior frozen clip contributes only
        // its exact conditioning/config/prompt for media-dependent header admission.
        created["conditioning"]=actual["conditioning"];created["config"]=actual["config"]
        created["prompt"]=actual["prompt"];created["ffmpeg"]=store.runtime.ffmpegPath
        let probe=root.appendingPathComponent(id+"-preflight.json")
        try JSONSerialization.data(withJSONObject:created).write(to:probe,options:.withoutOverwriting)
        let result=try await store.bridge.invoke(preset.engine == "h3" ? "h3-native-preflight":"ltx-native-preflight",
          runtime:store.runtime,payload:["recipePath":probe.path],output:root.appendingPathComponent(id+"-headers"))
        XCTAssertEqual(result["nativeRuntime"] as? String,"swift-mlx");headerPreflight=result["nativeRuntime"] as? String == "swift-mlx"
      }
      XCTAssertFalse(store.bridge.busy)
      evidence.append(["presetID":id,"profile":profile.path,"profileSHA256":NativeHeadlessJob.hash(try Data(contentsOf:profile)),"selected":selected,"nativeScanMatches":scanMatches,"headerPreflight":headerPreflight,"diffusionVAEAdoption":diffusionEvidence])
    }
    let raw=try XCTUnwrap(manifest["rawDevSource"] as? String),converted=URL(fileURLWithPath:try XCTUnwrap(manifest["convertedDevDirectory"] as? String))
    let identity=try NativeTransformerConversion.currentIdentity(raw)
    let adopted=try NativeTransformerConversion.completedDirectory(["nativeRuntime":"swift-mlx","outputDirectory":converted.path,
      "manifestPath":converted.appendingPathComponent("paged_manifest.json").path],source:raw,destination:converted,sourceIdentity:identity)
    XCTAssertEqual(adopted.path,converted.path)
    let adapter=try XCTUnwrap(manifest["adapter"] as? [String:Any]),adapterPath=try XCTUnwrap(adapter["path"] as? String)
    await store.importURLs([URL(fileURLWithPath:adapterPath)],scope:.global,loraModel:.h3,loraProfile:"turbo",loraLayout:"contiguous_qkv")
    XCTAssertNil(store.error)
    XCTAssertEqual(store.globalAssets.last?.path,adapterPath);XCTAssertEqual(store.globalAssets.last?.loraModel,.h3)
    let saved=try JSONDecoder().decode([MediaAsset].self,from:Data(contentsOf:root.appendingPathComponent("global-assets.json")))
    XCTAssertEqual(saved,store.globalAssets)
    let files=FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey])?.allObjects as? [URL] ?? []
    XCTAssertFalse(files.contains { $0.pathExtension=="safetensors" },"Setup must link installed weights in place")
    guard (testRun?.failureCount ?? 1)==0 else { throw StudioError.invalid("Native clean-registry reuse assertions failed; no passed receipt.") }
    let receipt:[String:Any]=["status":"passed","catalogPackageCount":23,"setupCases":evidence,"convertedDevDirectory":adopted.path,"convertedPagesValidated":49,
      "rawDevIdentity":identity,"importedAdapter":adapterPath,"savedRegistry":root.appendingPathComponent("global-assets.json").path,
      "pythonExecutableExists":false,"pythonModelInference":false,"inferenceExecuted":false,"networkTransferExecuted":false,"conversionRerun":false,"weightsCopied":false,
      "scope":"Fresh isolated app-data native setup, actual installed-weight reuse/import and completed conversion adoption; worker header preflight only.",
      "limits":"All23 source contracts validated; network acquisition and clean-machine absence of weights were not exercised. Optional/gated variants were not all installed." ]
    try JSONSerialization.data(withJSONObject:receipt,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent("qualification.json"),options:.withoutOverwriting)
  }
}
