import Foundation
import XCTest
@testable import StudioCore

final class NativeH3OmniTests: XCTestCase {
  let digest = "68577f23f0c866e329aa8a578b2455e4ca9a26a3ca1d15ddd6d0e99091007e18"
  func fixture() throws -> (URL,[String:Any],StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let image=root.appendingPathComponent("ref.png");try Data([1,2,3]).write(to:image)
    let recipe:[String:Any] = ["format":"weetodd-headless-v2","engine":"h3","candidate":"omni",
      "components":["task":"ref2va","transformer":"/models/omni.safetensors","text_encoder":"/models/qwen","tokenizer":"/models/tokenizer.json","video_vae":"/models/video","audio_vae":"/models/audio"],
      "config":["width":768,"height":512,"duration_seconds":5,"steps":9,"seed":42,"sampling_method":"euler","drop_adaln":true],
      "conditioning":["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"],
      "fasth3_omni":["version":1,"contract_path":"/models/fastvideo_inference.json","contract_sha256":digest],"prompt":"Scene"]
    try save(recipe,root)
    let asset=MediaAsset(name:"Appearance",kind:.image,path:image.path)
    var clip=Clip(engine:.h3);clip.profileID=root.appendingPathComponent("omni.json").path
    clip.generationSelection = .init(task:"ref2va");clip.prompt="Keep this person";clip.duration=5
    clip.generationWidth=768;clip.generationHeight=512;clip.attachments=[Attachment(assetID:asset.id,role:.reference)]
    var project=StudioProject();project.clips=[clip];project.assets=[asset]
    return(root,recipe,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  func save(_ recipe:[String:Any],_ root:URL) throws {
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("omni.json"))
  }
  func request(_ p:StudioProject,_ r:[String:Any]) throws -> [String:Any] {
    ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(p)),"runtime":r,"clipID":p.clips[0].id.uuidString]
  }
  func testSeparateOmniCapabilityFixesEightEvaluationsAndReferenceTask() throws {
    let(root,_,_,_)=try fixture()
    let catalog=try NativeH3Preparation.catalog(directory:root.path)
    let generation=try XCTUnwrap(catalog.first?["generation"] as? [String:Any])
    XCTAssertEqual(generation["fasth3Omni"] as? Bool,true)
    XCTAssertEqual(generation["fasth3"] as? Bool,false)
    XCTAssertEqual(generation["supportedTasks"] as? [String],["ref2va"])
    let controls=try XCTUnwrap(generation["controls"] as? [String:Any])
    XCTAssertEqual(controls["evaluations"] as? Int,8)
    XCTAssertEqual(controls["stepsEditable"] as? Bool,false)
    XCTAssertEqual(generation["attentionPolicyEditable"] as? Bool,false)
  }
  func testOmniExportPreservesCanvasAndOrderedHashedReferencesAndReopen() throws {
    let(root,_,project,runtime)=try fixture()
    let saved=root.appendingPathComponent("shot.weetodd.json")
    try ProjectStorage.write(project,to:saved)
    let reopened=try ProjectStorage.read(saved)
    let result=try NativeH3Preparation.compose(request:request(reopened,runtime))
    let recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    XCTAssertEqual((recipe["config"] as? [String:Any])?["width"] as? Int,768)
    XCTAssertEqual((recipe["config"] as? [String:Any])?["steps"] as? Int,9)
    XCTAssertNil((recipe["config"] as? [String:Any])?["attention_policy"])
    XCTAssertEqual((recipe["fasth3_omni"] as? [String:Any])?["contract_sha256"] as? String,digest)
    let inputs=try XCTUnwrap((recipe["conditioning"] as? [String:Any])?["inputs"] as? [[String:Any]])
    XCTAssertEqual(inputs.map { $0["id"] as? String },project.clips[0].attachments.map { Optional($0.id.uuidString) })
    XCTAssertEqual((inputs[0]["sha256"] as? String)?.count,64)
  }
  func testOmniRejectsRetainedUnsupportedOverridesAndInvalidCanvas() throws {
    let(_,_,original,runtime)=try fixture()
    var bad:[StudioProject]=[]
    var p=original;p.clips[0].generationSelection!.steps=4;bad.append(p)
    p=original;p.clips[0].generationSelection!.h3Joint = .init();bad.append(p)
    p=original;p.clips[0].generationSelection!.h3AttentionPolicy = .dense;bad.append(p)
    p=original;p.clips[0].generationSelection!.h3Reference = .init();bad.append(p)
    p=original;p.clips[0].attachments[0].h3ReferencePlacement = .init();p.clips[0].attachments[0].h3ReferencePlacement!.frame = .index(0);bad.append(p)
    p=original;p.clips[0].attachments.append(Attachment(assetID:UUID(),role:.lora));bad.append(p)
    p=original;p.clips[0].duration=4;bad.append(p)
    p=original;p.clips[0].generationWidth=2048;p.clips[0].generationHeight=2048;bad.append(p)
    p=original;p.clips[0].profileID="auto";bad.append(p)
    for p in bad { XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(p,runtime))) }
  }
  func testSetupAndDownloadCatalogExposeSeparatePinnedSources() throws {
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id=="swift-h3-omni" })
    XCTAssertEqual(preset.task,"ref2va")
    XCTAssertTrue(preset.components.contains { $0.key=="omni_contract" })
    let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let packages=try NativeModelDownloads.catalog(at:root.appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json"))
    let package=try XCTUnwrap(packages.first { $0.descriptor.id=="h3-fasth3-omni-q8" })
    let weights=try XCTUnwrap(package.files.first { $0.target=="transformer.safetensors" })
    XCTAssertEqual(weights.repo,"ddalcu/FastH3-Omni-MLX-Serve-8bit")
    XCTAssertEqual(weights.revision,"3277ecd1f31f46037f8d51e4882d24139be8d112")
    XCTAssertEqual(weights.size,37_304_406_432)
    XCTAssertEqual(weights.sha256,"44fff6e29adde8ec485b1dddad02d9a134d75c52df2218e19bbf760199080ba6")
    XCTAssertEqual(package.files.first { $0.target=="fastvideo_inference.json" }?.sha256,digest)
    XCTAssertFalse(package.files.contains { $0.target.contains("tokenizer") || $0.target.contains("vae") })
  }
  func testStrictScalarAndComponentProfilesRejectBeforeCatalog() throws {
    let(root,original,_,_)=try fixture()
    for mutation in 0..<8 {
      var recipe=original,config=original["config"] as! [String:Any],components=original["components"] as! [String:Any]
      if mutation==0 {config["drop_adaln"]=1}
      if mutation==1 {config["seed"]="42"}
      if mutation==2 {var c=recipe["conditioning"] as! [String:Any];c["inputs"]=["wrong":"type"];recipe["conditioning"]=c}
      if mutation==3 {components["allow_fl2va_weights_for_ref2va"]=false}
      if mutation==4 {components["vision_encoder"]=""}
      if mutation==5 {components["processor"]=false}
      if mutation==6 {config["attention_policy"]="dense"}
      if mutation==7 {var w=recipe["fasth3_omni"] as! [String:Any];w["contract_sha256"]=digest.uppercased();recipe["fasth3_omni"]=w}
      recipe["components"]=components;recipe["config"]=config;try save(recipe,root)
      XCTAssertTrue(try NativeH3Preparation.catalog(directory:root.path).isEmpty,"Mutation \(mutation) must reject")
    }
  }
  func testOmniCatalogCannotRelabelUnpinnedSource() throws {
    let(root,_,_,_)=try fixture()
    let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let source=repository.appendingPathComponent("src/wee_todd_mlx/native_model_download_catalog.json")
    var entry=try XCTUnwrap((JSONSerialization.jsonObject(with:Data(contentsOf:source)) as? [[String:Any]])?.first { ($0["descriptor"] as? [String:Any])?["id"] as? String=="h3-fasth3-omni-q8" })
    var files=entry["files"] as! [[String:Any]];files[0]["repo"]="other/weights";entry["files"]=files
    let catalog=root.appendingPathComponent("tampered.json");try JSONSerialization.data(withJSONObject:[entry]).write(to:catalog)
    XCTAssertThrowsError(try NativeModelDownloads.catalog(at:catalog))
  }
  func testOmniSetupRejectsMalformedSourceFilesWithoutRemovingSupport() throws {
    let(root,_,_,_)=try fixture()
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id==NativeH3OmniProfile.presetID })
    var selected:[String:String]=[:]
    for field in preset.components {
      let file=root.appendingPathComponent(field.key)
      if field.kind=="directory" { try FileManager.default.createDirectory(at:file,withIntermediateDirectories:true) }
      else {try Data([1,2,3]).write(to:file)}
      if field.key=="tokenizer" { try Data("{}".utf8).write(to:file.appendingPathComponent("tokenizer.json")) }
      selected[field.key]=file.path
    }
    XCTAssertThrowsError(try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory))
    for field in preset.components {XCTAssertTrue(FileManager.default.fileExists(atPath:selected[field.key]!))}
  }

  func testPortableOmniExampleUsesExistingRecipeBackedSwiftRoute() throws {
    let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let file=root.appendingPathComponent("examples/headless/h3_fasth3_omni_ref2va.json")
    XCTAssertTrue(FileManager.default.fileExists(atPath:file.path))
    guard FileManager.default.fileExists(atPath:file.path) else {return}
    let recipe=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
    XCTAssertNoThrow(try NativeH3OmniProfile.ordinary(recipe))
    let api=root.appendingPathComponent("examples/h3_fasth3_omni_swift_api.json")
    let graph=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:api)) as? [String:[String:Any]])
    XCTAssertEqual(graph["1"]?["class_type"] as? String,"WeeToddSwiftVideoGenerate")
    XCTAssertEqual(graph.count,1)
  }
  func testImageExportRejectsOmniEvenWhenSavedProfileWasManuallySelected() throws {
    let(root,_,_,_)=try fixture()
    var draft=DrawThingsImageDraft(destination:.init(scope:.project,projectID:UUID()))
    draft.selectProvider(.nativeH3);draft.modelID=root.appendingPathComponent("omni.json").path
    draft.prompt="An image";draft.width=768;draft.height=512;draft.guidance=1;draft.strength=1
    XCTAssertThrowsError(try NativeH3ImagePreparation.recipe(draft:draft,requestID:UUID(),seed:42))
  }
  func testInstalledOmniSetupLinksVerifiedComponentsAndSymlinksWithoutCopies() throws {
    guard let source=ProcessInfo.processInfo.environment["WEETODD_OMNI_SETUP_RECIPE"] else {
      throw XCTSkip("Opt-in installed Omni source checksum and support reuse; no generation or network.")
    }
    let(root,_,original,runtime)=try fixture()
    let installed=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:source))) as? [String:Any])
    let components=try XCTUnwrap(installed["components"] as? [String:Any])
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id==NativeH3OmniProfile.presetID })
    var selected=components.compactMapValues { $0 as? String }
    selected.removeValue(forKey:"task")
    selected["omni_transformer"]=selected.removeValue(forKey:"transformer")
    selected["omni_contract"]=(installed["fasth3_omni"] as? [String:Any])?["contract_path"] as? String
    let linked=root.appendingPathComponent("linked-transformer.safetensors")
    try FileManager.default.createSymbolicLink(at:linked,withDestinationURL:URL(fileURLWithPath:try XCTUnwrap(selected["omni_transformer"])))
    selected["omni_transformer"]=linked.path
    let discovery=try NativeModelSetup.scan(presetID:preset.id,roots:[linked.path,try XCTUnwrap(selected["omni_contract"])])
    XCTAssertFalse(discovery.candidates["omni_transformer",default:[]].isEmpty)
    XCTAssertFalse(discovery.candidates["omni_contract",default:[]].isEmpty)
    let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
    let result=try XCTUnwrap(recipe["components"] as? [String:Any])
    for key in ["text_encoder","vision_encoder","tokenizer","video_vae","audio_vae","transformer"] {
      XCTAssertEqual(result[key] as? String,(components[key] as? String).map {URL(fileURLWithPath:$0).standardizedFileURL.resolvingSymlinksInPath().path})
    }
    XCTAssertNotEqual(result["transformer"] as? String,linked.path)
    try save(recipe,root)
    XCTAssertNoThrow(try NativeH3Preparation.compose(request:request(original,runtime)))
  }

  func testMixedReferenceOrderSurvivesExportAndCombinedCeilingRejects() throws {
    let(root,_,original,runtime)=try fixture()
    var project=original
    for (kind,name) in [(AssetKind.video,"movie"),(AssetKind.audio,"audio")] {
      let file=root.appendingPathComponent(name);try Data([1,2,3]).write(to:file)
      let asset=MediaAsset(name:name,kind:kind,path:file.path)
      project.assets.append(asset);project.clips[0].attachments.append(Attachment(assetID:asset.id,role:.reference))
    }
    let result=try NativeH3Preparation.compose(request:request(project,runtime))
    let recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    let inputs=try XCTUnwrap((recipe["conditioning"] as? [String:Any])?["inputs"] as? [[String:Any]])
    XCTAssertEqual(inputs.compactMap {$0["kind"] as? String},["image","video","audio"])
    XCTAssertEqual(inputs.compactMap {$0["id"] as? String},project.clips[0].attachments.map {$0.id.uuidString})
    XCTAssertTrue(inputs.allSatisfy {($0["sha256"] as? String)?.count==64})
    for _ in 0..<10 {project.clips[0].attachments.append(Attachment(assetID:project.assets[0].id,role:.reference))}
    XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(project,runtime)))
  }

}


extension NativeH3OmniTests {
  func testOmniExportPreservesOwnedReferenceSizingAndRejectsNonfullDensity() throws {
    let(root,_,original,runtime)=try fixture();var project=original
    project.clips[0].attachments[0].h3ReferencePlacement = .init(imagePixelBudgetPercent:400)
    let movie=root.appendingPathComponent("movie.mp4");try Data([1,2,3]).write(to:movie)
    let asset=MediaAsset(name:"Movie",kind:.video,path:movie.path);project.assets.append(asset)
    var attachment=Attachment(assetID:asset.id,role:.reference)
    attachment.h3ReferencePlacement = .init(videoSizePolicy:.nativeH3,videoTemporalDensity:.full)
    project.clips[0].attachments.append(attachment)
    let result=try NativeH3Preparation.compose(request:request(project,runtime))
    let recipe=try XCTUnwrap(result["recipe"] as? [String:Any])
    let inputs=try XCTUnwrap((recipe["conditioning"] as? [String:Any])?["inputs"] as? [[String:Any]])
    XCTAssertEqual(inputs[0]["image_pixel_budget_percent"] as? Int,400)
    XCTAssertEqual(inputs[1]["size_policy"] as? String,"native_h3")
    XCTAssertEqual(inputs[1]["temporal_density"] as? String,"full")
    for density in [H3VideoReferenceTemporalDensity.half,.quarter,.automatic] {
      var bad=project;bad.clips[0].attachments[1].h3ReferencePlacement?.videoTemporalDensity=density
      XCTAssertThrowsError(try NativeH3Preparation.compose(request:request(bad,runtime)))
    }
    for (kind,key,value) in [("image","image_pixel_budget_percent",true as Any),("image","image_pixel_budget_percent",Double.infinity),
      ("image","size_policy","native_h3"),("video","image_pixel_budget_percent",400),
      ("video","temporal_density","half"),("video","temporal_density",true),("video","size_policy",true),
      ("image","frame_index",0)] {
      var bad=recipe,c=recipe["conditioning"] as! [String:Any]
      c["inputs"]=[["id":"ref","kind":kind,"role":"reference","path":"/ref","sha256":String(repeating:"a",count:64),key:value]]
      bad["conditioning"]=c;XCTAssertThrowsError(try NativeH3OmniProfile.ordinary(bad),"\(key)")
    }
  }
}

extension NativeH3OmniTests {
  func testOmniExplicitReferenceClearPreservesSizingAndSoundtrack() {
    var image=H3ReferencePlacement(frame:.index(0),imagePixelBudgetPercent:400)
    NativeH3OmniProfile.clearUnsupportedReferenceSettings(&image)
    XCTAssertNil(image.frame);XCTAssertEqual(image.imagePixelBudgetPercent,400)
    var movie=H3ReferencePlacement(frame:.last,soundtrackPath:"/sound.wav",videoSizePolicy:.nativeH3,videoTemporalDensity:.quarter)
    NativeH3OmniProfile.clearUnsupportedReferenceSettings(&movie)
    XCTAssertNil(movie.frame);XCTAssertEqual(movie.soundtrackPath,"/sound.wav")
    XCTAssertEqual(movie.videoSizePolicy,.nativeH3);XCTAssertNil(movie.videoTemporalDensity)
    movie.videoTemporalDensity = .full
    NativeH3OmniProfile.clearUnsupportedReferenceSettings(&movie)
    XCTAssertEqual(movie.videoTemporalDensity,.full)
  }
}
