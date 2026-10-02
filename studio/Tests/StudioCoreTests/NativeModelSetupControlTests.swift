import XCTest
@testable import StudioCore

final class NativeModelSetupControlTests: XCTestCase {
  func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
  }
  func write(_ root: URL, name: String, header: [String: Any]) throws -> URL {
    let body = try JSONSerialization.data(withJSONObject: header)
    var count = UInt64(body.count).littleEndian
    var bytes = withUnsafeBytes(of: &count) { Data($0) }; bytes.append(body)
    let url = root.appendingPathComponent(name + ".safetensors")
    try bytes.write(to: url); return url
  }
  func factors(rank: Int = 32, downscale: String, version: String? = nil) -> [String: Any] {
    var metadata = ["reference_downscale_factor": downscale]
    if let version { metadata["model_version"] = version }
    var header: [String: Any] = ["__metadata__": metadata]
    for block in 0..<48 {
      for tail in ["attn1.to_k", "attn1.to_out.0", "attn1.to_q", "attn1.to_v",
        "attn2.to_k", "attn2.to_out.0", "attn2.to_q", "attn2.to_v", "ff.net.0.proj", "ff.net.2"] {
        let stem = "diffusion_model.transformer_blocks.\(block).\(tail)"
        let input = tail == "ff.net.2" ? 16384 : 4096
        let output = tail == "ff.net.0.proj" ? 16384 : 4096
        header[stem + ".lora_A.weight"] = ["dtype": "BF16", "shape": [rank,input], "data_offsets": [0,2]]
        header[stem + ".lora_B.weight"] = ["dtype": "BF16", "shape": [output,rank], "data_offsets": [0,2]]
      }
    }
    return header
  }
  func testControlDiscoveryRequiresCompleteRankShapeAndReferenceSignature() throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let motion = try write(root, name: "arbitrary-motion-name", header: factors(downscale: "2", version: "2.3.0"))
    let crossview = try write(root, name: "arbitrary-crossview-name", header: factors(downscale: "1"))
    var wrong = factors(downscale: "1")
    wrong["diffusion_model.transformer_blocks.47.ff.net.2.lora_B.weight"] =
      ["dtype": "BF16", "shape": [4095,32], "data_offsets": [0,2]]
    _ = try write(root, name: "crossview-wrong-shape", header: wrong)
    wrong = factors(downscale: "1"); wrong.removeValue(forKey: "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight")
    _ = try write(root, name: "crossview-incomplete", header: wrong)
    _ = try write(root, name: "motion-without-version", header: factors(downscale: "2"))
    _ = try write(root, name: "wrong-rank", header: factors(rank:64,downscale:"1"))
    _ = try write(root, name: "future-version", header: factors(downscale:"2",version:"3.0"))
    wrong = factors(downscale: "1"); wrong["__metadata__"] = ["reference_downscale_factor":"1","reference_spatial_scale_factor":"2"]
    _ = try write(root, name: "spatial-task", header: wrong)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-ltx25-motion-track",roots:[root.path])
      .candidates["motion_track_lora_path"], [motion.path])
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-ltx25-crossview",roots:[root.path])
      .candidates["crossview_lora_path"], [crossview.path])
  }
  func testTwoStageControlRecipesPreserveOrderedDedicatedAdapters() throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    for (id, keys) in [("union", ["union_lora_path"]), ("motion-track", ["motion_track_lora_path"]),
      ("crossview", ["crossview_lora_path"]), ("crossview-ingredients", ["crossview_lora_path","ingredients_lora_path"])] {
      let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-" + id })
      var selected: [String: String] = [:]
      for component in preset.components {
        let url = root.appendingPathComponent(id + "-" + component.key)
        if component.kind == "directory" { try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true) }
        else { try Data([0]).write(to:url) }
        selected[component.key] = url.path
      }
      let recipe = try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic)
      let components = recipe["components"] as! [String: Any]
      let adapters = components["ic_loras"] as! [[Any]]
      XCTAssertEqual(adapters.compactMap { $0[0] as? String }, keys.map { selected[$0]! })
      XCTAssertEqual((recipe["config"] as! [String:Any])["ic_lora_single_stage"] as? Bool, false)
      XCTAssertEqual((recipe["conditioning"] as! [String:Any])["task"] as? String, "control")
      let families = ["motion-track":"motion_track","crossview":"crossview_warp","crossview-ingredients":"crossview_ingredients"]
      XCTAssertEqual((recipe["conditioning"] as! [String:Any])["control_family"] as? String, families[id])
      XCTAssertNotNil(components["spatial_upscaler_path"])
      XCTAssertTrue(keys.allSatisfy { components[$0] == nil })
    }
  }
  func testIngredients25AdmissionRequiresCompleteFullResolutionRank128Signature() throws {
    let root = try directory();defer { try? FileManager.default.removeItem(at:root) }
    var valid = factors(rank:128,downscale:"1",version:"2.5")
    let admitted = try write(root,name:"new-version-complete-signature",header:valid)
    valid["diffusion_model.transformer_blocks.0.ff.net.0.proj.lora_B.weight"] = ["dtype":"BF16","shape":[4096,128],"data_offsets":[0,2]]
    _ = try write(root,name:"new-version-wrong-shape",header:valid)
    _ = try write(root,name:"new-version-wrong-rank",header:factors(rank:32,downscale:"1",version:"2.5"))
    _ = try write(root,name:"new-version-wrong-grid",header:factors(rank:128,downscale:"2",version:"2.5"))
    valid = factors(rank:128,downscale:"1",version:"2.5")
    valid["__metadata__"] = ["reference_downscale_factor":"1","model_version":"2.5","reference_temporal_scale_factor":"2"]
    _ = try write(root,name:"new-version-wrong-time",header:valid)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-ltx25-ingredients",roots:[root.path]).candidates["ingredients_lora_path"],[admitted.path])
  }
  func testIngredientsRecipeDefaultsStrengthFromBoundedMetadataRatherThanFilename() throws {
    let root = try directory();defer { try? FileManager.default.removeItem(at:root) }
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-ingredients" })
    var selected: [String:String] = [:]
    for field in preset.components where field.key != "ingredients_lora_path" {
      let file = root.appendingPathComponent(field.key)
      if field.kind == "directory" { try FileManager.default.createDirectory(at:file,withIntermediateDirectories:true) }
      else { try Data([0]).write(to:file) };selected[field.key] = file.path
    }
    for (version,strength) in [("2.3",1.2),("2.3.0",1.2),("2.5",1.0),("2.5.0",1.0)] {
      let adapter = try write(root,name:"misleading-ltx-2.3-name",header:factors(rank:128,downscale:"1",version:version))
      selected["ingredients_lora_path"] = adapter.path
      let recipe = try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic)
      let stack = (recipe["components"] as! [String:Any])["ic_loras"] as! [[Any]]
      XCTAssertEqual(stack[0][0] as? String,adapter.path)
      XCTAssertEqual(stack[0][1] as? Double,strength)
    }
  }
  func testH3DiscoveryRejectsRootlessOfficialAndPackedAffineCheckpointLayouts() throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    func header(prefix: String, dtype: String) -> [String: Any] {
      [prefix+"video_patch_proj.weight": ["shape":[5376,96],"dtype":dtype,"data_offsets":[0,2]],
       prefix+"audio_patch_proj.weight": ["shape":[5376,32],"dtype":dtype,"data_offsets":[0,2]],
       prefix+"condition_proj.weight": ["shape":[5376,5120],"dtype":"BF16","data_offsets":[0,2]]]
    }
    let supported = try write(root,name:"prefixed-direct",header:header(prefix:"model.diffusion_model.",dtype:"BF16"))
    _ = try write(root,name:"official-rootless",header:header(prefix:"",dtype:"F32"))
    _ = try write(root,name:"rootless-without-curve",header:header(prefix:"",dtype:"BF16"))
    _ = try write(root,name:"packed-affine",header:header(prefix:"model.diffusion_model.",dtype:"U32"))
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-h3-text",roots:[root.path]).candidates["transformer"],[supported.path])
    XCTAssertTrue(try NativeModelSetup.scan(presetID:"swift-h3-image",roots:[supported.path]).candidates["transformer"]!.isEmpty)
  }
  func testFunControlSetupUsesTextBaseAndRejectsPrunedOrIncompleteBranches() throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let preset = try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-h3-fun-control" })
    XCTAssertEqual(preset.task,"control"); XCTAssertFalse(preset.components.contains { $0.key == "vision_encoder" })
    var selected: [String:String] = [:]
    for field in preset.components {
      let url = root.appendingPathComponent(field.key)
      if field.kind == "directory" {
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
        if field.key == "tokenizer" { try Data("{}".utf8).write(to:url.appendingPathComponent("tokenizer.json")) }
      }
      else { try Data([0]).write(to:url) }
      selected[field.key] = url.path
    }
    let recipe = try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
    let components = recipe["components"] as! [String:Any], conditioning = recipe["conditioning"] as! [String:Any]
    XCTAssertEqual(components["task"] as? String,"t2va")
    XCTAssertEqual(components["tokenizer"] as? String,root.appendingPathComponent("tokenizer/tokenizer.json").path)
    XCTAssertEqual(components["fun_controlnet"] as? String,selected["fun_controlnet"])
    XCTAssertNil(components["loras"]); XCTAssertEqual(conditioning["task"] as? String,"control")
    XCTAssertTrue((conditioning["inputs"] as! [Any]).isEmpty)
    let shapes: [String:[Int]] = ["adaln_proj.linear.weight":[96768,2688],"adaln_proj.linear.bias":[96768],
      "norm1.weight":[5376],"norm2.weight":[5376],"attn.norm_q.weight":[128],"attn.norm_k.weight":[128],
      "attn.to_q.weight":[7168,5376],"attn.to_k.weight":[7168,5376],"attn.to_v.weight":[7168,5376],
      "attn.to_out.0.weight":[5376,7168],"ff.net.0.proj.weight":[28672,5376],"ff.net.2.weight":[5376,14336],
      "after_proj.weight":[5376,5376],"after_proj.bias":[5376]]
    var header: [String:Any] = [:]
    for (key,shape) in ["control_proj_in.weight":[5376,196],"control_proj_in.bias":[5376],
      "control_blocks.0.before_proj.weight":[5376,5376],"control_blocks.0.before_proj.bias":[5376]] {
      header[key] = ["dtype":"F32","shape":shape,"data_offsets":[0,2]]
    }
    for block in 0..<5 { for (key,shape) in shapes {
      header["control_blocks.\(block)."+key] = ["dtype":"BF16","shape":shape,"data_offsets":[0,2]]
    } }
    let admitted = try write(root,name:"arbitrary-branch",header:header)
    var wrong = header; wrong["control_blocks.0.adaln_proj.linear.weight"] = ["dtype":"F32","shape":[96768,8],"data_offsets":[0,2]]
    _ = try write(root,name:"basis-eight",header:wrong)
    wrong = header; wrong.removeValue(forKey:"control_blocks.4.attn.to_v.weight")
    _ = try write(root,name:"incomplete",header:wrong)
    wrong = header; wrong["control_blocks.0.attn.to_q.weight"] = ["dtype":"I8","shape":[7168,5376],"data_offsets":[0,2]]
    _ = try write(root,name:"unhandled-quantization",header:wrong)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:preset.id,roots:[root.path]).candidates["fun_controlnet"],[admitted.path])
  }
  func testH3DownloadedTokenizerFolderResolvesToCanonicalJSONAndRejectsMissingPayload() throws {
    let root=try directory();defer { try? FileManager.default.removeItem(at:root) }
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-h3-text" })
    var selected:[String:String]=[:]
    for field in preset.components {
      let file=root.appendingPathComponent(field.key)
      if field.kind == "directory" { try FileManager.default.createDirectory(at:file,withIntermediateDirectories:true) }
      else { try Data([0]).write(to:file) };selected[field.key]=file.path
    }
    let canonical=root.appendingPathComponent("canonical-tokenizer.json")
    try Data("{}".utf8).write(to:canonical)
    let tokenizer=root.appendingPathComponent("tokenizer/tokenizer.json")
    try FileManager.default.createSymbolicLink(at:tokenizer,withDestinationURL:canonical)
    let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic)
    XCTAssertEqual((recipe["components"] as! [String:Any])["tokenizer"] as? String,canonical.path)
    try FileManager.default.removeItem(at:tokenizer)
    XCTAssertThrowsError(try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic))
    selected["tokenizer"]=canonical.path
    let direct=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic)
    XCTAssertEqual((direct["components"] as! [String:Any])["tokenizer"] as? String,canonical.path)
  }
  func testRealBoundedRemoteHeadersMatchInstalledDiscoveryWhenRequested() throws {
    guard let source = ProcessInfo.processInfo.environment["WEETODD_NATIVE_HEADER_FIXTURES"] else {
      throw XCTSkip("Opt-in real bounded HTTP header qualification")
    }
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let checks = [("ltx-2.3-22b-ic-lora-motion-track-control-ref0.5", "swift-ltx25-motion-track", "motion_track_lora_path"),
      ("LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000", "swift-ltx25-crossview", "crossview_lora_path"),
      ("LTX-2.5-Licon-MSR-V1", "swift-ltx25-msr", "msr_lora_path"),
      ("ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0", "swift-ltx25-dfr-spatial", "dfr_detailing_lora_path"),
      ("ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0", "swift-ltx25-dfr-temporal-1", "dfr_temporal_upsampler_path"),
      ("ltx-2.3-22b-ic-lora-union-control-ref0.5", "swift-ltx25-union", "union_lora_path"),
      ("ltx-2.3-22b-ic-lora-ingredients-0.9", "swift-ltx25-ingredients", "ingredients_lora_path"),
      ("MiniMax-H3-Fun-Controlnet-Union", "swift-h3-fun-control", "fun_controlnet"),
      ("MiniMax-H3-Fun-Controlnet-Union-2.0", "swift-h3-fun-control", "fun_controlnet")]
    for (name,preset,key) in checks {
      let path = URL(fileURLWithPath:source).appendingPathComponent(name + ".safetensors.header.json")
      let header = try JSONSerialization.jsonObject(with:Data(contentsOf:path)) as! [String:Any]
      let file = try write(root,name:name,header:header)
      XCTAssertEqual(try NativeModelSetup.scan(presetID:preset,roots:[file.path]).candidates[key],[file.path])
    }
    let curvePath=URL(fileURLWithPath:source).appendingPathComponent("MiniMax-H3-FL2VA-pruned_bf16.safetensors.header.json")
    if FileManager.default.fileExists(atPath:curvePath.path) {
      let header=try JSONSerialization.jsonObject(with:Data(contentsOf:curvePath)) as! [String:Any]
      let file=try write(root,name:"MiniMax-H3-FL2VA-pruned_bf16",header:header)
      XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-h3-image",roots:[file.path]).candidates["transformer"],[file.path])
      XCTAssertTrue(try NativeModelSetup.scan(presetID:"swift-h3-reference",roots:[file.path]).candidates["transformer"]!.isEmpty)
      XCTAssertTrue(try NativeModelSetup.scan(presetID:"swift-h3-fun-control",roots:[file.path]).candidates["transformer"]!.isEmpty)
    }
    let v2Path = URL(fileURLWithPath:source).appendingPathComponent("LTX-2.5-Licon-MSR-V2.safetensors.header.json")
    if FileManager.default.fileExists(atPath:v2Path.path) {
      let header = try JSONSerialization.jsonObject(with:Data(contentsOf:v2Path)) as! [String:Any]
      let file = try write(root,name:"LTX-2.5-Licon-MSR-V2",header:header)
      XCTAssertEqual(header.keys.filter { $0.hasSuffix(".lora_A.weight") }.count,1152)
      XCTAssertTrue(try NativeModelSetup.scan(presetID:"swift-ltx25-msr",roots:[file.path]).candidates["msr_lora_path"]!.isEmpty)
    }
  }
}
