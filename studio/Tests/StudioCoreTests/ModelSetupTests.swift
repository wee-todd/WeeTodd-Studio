import XCTest

@testable import StudioCore

final class ModelSetupTests: XCTestCase {
  let presetJSON =
    #"{"id":"ltx25-t2v","name":"LTX 2.5 Text to Video","engine":"ltx25","task":"t2v","description":"Create a shot","components":[{"key":"checkpoint","label":"Transformer","kind":"file"},{"key":"text_encoder","label":"Text encoder","kind":"directory"}]}"#

  func testCatalogContractDecodesAndRequiresEveryComponent() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var selection = ModelSetupSelection()
    XCTAssertEqual(
      selection.missingComponents(for: preset).map(\.key), ["checkpoint", "text_encoder"])
    selection.components = ["checkpoint": "/models/checkpoint", "text_encoder": "  "]
    XCTAssertEqual(selection.missingComponents(for: preset).map(\.key), ["text_encoder"])
    selection.components["text_encoder"] = "/models/encoder"
    XCTAssertTrue(selection.missingComponents(for: preset).isEmpty)
  }

  func testScanAutoSelectsOnlyUniqueCandidatesAndKeepsExplicitChoice() {
    var selection = ModelSetupSelection()
    selection.applyScan(["checkpoint": ["/a", "/b"], "encoder": ["/encoder", "/encoder"]])
    XCTAssertNil(selection.components["checkpoint"])
    XCTAssertEqual(selection.components["encoder"], "/encoder")
    selection.components["checkpoint"] = "/manual"
    selection.applyScan(["checkpoint": ["/other"], "encoder": []])
    XCTAssertEqual(selection.components["checkpoint"], "/manual")
    XCTAssertEqual(selection.components["encoder"], "/encoder")
  }

  func testDownloadsOnlyAppearForCompatibleEngineAndLegacyCatalogStillDecodes() throws {
    let json =
      #"{"id":"encoder","name":"Encoder","description":"Prepare encoder","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"licenseNotice":"Review source terms"}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3"))
    XCTAssertFalse(download.supports(engine: "ltx25"))
    XCTAssertEqual(download.licenseNotice, "Review source terms")
    let legacy = json.replacingOccurrences(
      of: #","engines":["h3"],"licenseNotice":"Review source terms""#, with: "")
    let legacyDownload = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacy.utf8))
    XCTAssertTrue(legacyDownload.supports(engine: "ltx25"))
  }

  func testRecipeSelectionChecksEngineAndCurrentMediaRoles() throws {
    var preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var clip = Clip(engine: .ltx25)
    XCTAssertTrue(preset.supports(clip))
    clip.engine = .h3
    XCTAssertFalse(preset.supports(clip))
    preset.engine = "h3"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertFalse(preset.supports(clip))
    preset.task = "fflf"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(clip))
  }

  func testH3DownloadsFilterByTaskAndSharedComponentsRemainVisible() throws {
    let json =
      #"{"id":"ref","name":"Reference transformer","description":"Prepared pages","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["ref2va"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "ref2va"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(download.supports(engine: "h3", task: "fflf"))
    XCTAssertFalse(download.supports(engine: "ltx25", task: "ref2va"))
    let sharedJSON = json.replacingOccurrences(of: #","tasks":["ref2va"]"#, with: "")
    let shared = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(sharedJSON.utf8))
    XCTAssertTrue(shared.supports(engine: "h3", task: "fflf"))
  }

  func testComponentDownloadButtonsOnlySelectPackagesProvidingThatComponent() throws {
    let json =
      #"{"id":"support","name":"Support files","description":"Audio and task files","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["t2v","fflf"],"components":["checkpoint","audio_vae","tokenizer","processor"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "fflf", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "ref2va", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v", component: "transformer"))
    let legacyJSON = json.replacingOccurrences(
      of: #","components":["checkpoint","audio_vae","tokenizer","processor"]"#, with: "")
    let legacy = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacyJSON.utf8))
    XCTAssertTrue(legacy.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(legacy.supports(engine: "h3", task: "t2v", component: "audio_vae"))
  }

  func testLTX25BaseRecipeSupportsExistingImageAndAudioTaskFallbacks() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    for role in [MediaRole.first, .audioDriver] {
      var clip = Clip(engine: .ltx25)
      clip.attachments = [Attachment(assetID: UUID(), role: role)]
      XCTAssertTrue(preset.supports(clip))
    }
    var referenceClip = Clip(engine: .ltx25)
    referenceClip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(referenceClip))
  }

  func testMemoryModesUseBackendIdentifiers() {
    XCTAssertEqual(
      ModelSetupMemoryMode.allCases.map(\.rawValue), ["automatic", "lower_memory", "custom"])
  }

  func testNativeSetupCatalogAndRecipeWithoutPython() throws {
    let presets = NativeModelSetup.catalog()
    XCTAssertEqual(Set(presets.map(\.id)), Set(["swift-h3-text", "swift-h3-image",
      "swift-h3-reference", "swift-h3-fun-control", "swift-ltx25-text", "swift-ltx25-image",
      "swift-ltx25-dfr-spatial", "swift-ltx25-dfr-temporal-1", "swift-ltx25-dfr-temporal-2",
      "swift-ltx25-msr", "swift-ltx25-ingredients", "swift-ltx25-union",
      "swift-ltx25-motion-track", "swift-ltx25-crossview", "swift-ltx25-crossview-ingredients"]))
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let preset = try XCTUnwrap(presets.first { $0.id == "swift-ltx25-text" })
    var selected: [String: String] = [:]
    for component in preset.components {
      let url = root.appendingPathComponent(component.key)
      if component.kind == "directory" {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      } else {
        try Data("test".utf8).write(to: url)
      }
      selected[component.key] = url.path
    }
    let recipe = try NativeModelSetup.recipe(preset: preset, selected: selected,
      memoryMode: .lowerMemory)
    XCTAssertEqual(recipe["engine"] as? String, "ltx25")
    XCTAssertEqual((recipe["config"] as? [String: Any])?["stage2_steps"] as? Int, 3)
    let staged = try NativeModelSetup.stage(recipe, directory: root.appendingPathComponent("profiles").path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: staged))
    selected["audio_vae_path"] = root.appendingPathComponent("missing").path
    XCTAssertThrowsError(try NativeModelSetup.recipe(preset: preset, selected: selected,
      memoryMode: .automatic))
  }
  func testH3VisualSetupRequiresVisionTowerSeparatelyFromPagedText() throws {
    for id in ["swift-h3-image","swift-h3-reference"] {
      let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == id })
      XCTAssertTrue(preset.components.contains { $0.key == "vision_encoder" })
    }
    XCTAssertFalse(try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-h3-text" }).components.contains { $0.key == "vision_encoder" })
  }
  func testH3ScanAdmitsRawVisionButRequiresPagedText() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func write(_ name: String, _ shapes: [String: [Int]]) throws -> URL {
      let header = shapes.mapValues { ["dtype": "U8", "shape": $0, "data_offsets": [0, 1]] as [String: Any] }
      let bytes = try JSONSerialization.data(withJSONObject: header)
      var count = UInt64(bytes.count).littleEndian
      var data = withUnsafeBytes(of: &count) { Data($0) }
      data.append(bytes); data.append(0)
      let url = root.appendingPathComponent(name)
      try data.write(to: url)
      return url
    }
    _ = try write("text.safetensors", ["model.embed_tokens.weight": [151936, 5120],
      "model.layers.49.self_attn.q_proj.weight": [8192, 1280]])
    var visual = Dictionary(uniqueKeysWithValues: (0..<526).map { ("visual.fixture.\($0)", [1]) })
    visual["visual.patch_embed.proj.weight"] = [1152, 3, 2, 16, 16]
    visual["visual.blocks.26.attn.qkv.weight"] = [3456, 288]
    visual["visual.deepstack_merger_list.2.linear_fc2.weight"] = [5120, 1152]
    let vision = try write("vision.safetensors", visual)
    visual["visual.patch_embed.proj.weight"] = [1]
    _ = try write("incompatible-vision.safetensors", visual)
    let result = try NativeModelSetup.scan(presetID: "swift-h3-image", roots: [root.path])
    XCTAssertTrue(result.candidates["text_encoder", default: []].isEmpty)
    XCTAssertEqual(result.candidates["vision_encoder"], [vision.path])
  }
  func testReferenceSetupUsesInstalledSingleStageAdapterWithoutUnusedUpscaler() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    for family in ["msr","ingredients"] {
      let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-"+family })
      XCTAssertFalse(preset.components.contains { $0.key == "spatial_upscaler_path" })
      var selected:[String:String]=[:]
      for component in preset.components {
        let url=root.appendingPathComponent(component.key)
        if component.kind == "directory" { try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true) }
        else { try Data([1]).write(to:url) };selected[component.key]=url.path
      }
      let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
      let components=recipe["components"] as! [String:Any]
      XCTAssertEqual((recipe["config"] as! [String:Any])["ic_lora_single_stage"] as? Bool,true)
      XCTAssertEqual(components["spatial_upscaler_path"] as? String,"")
      let adapter=components["ic_loras"] as! [[Any]]
      XCTAssertEqual(adapter.count,1)
      XCTAssertEqual(adapter[0][0] as? String,selected[family == "msr" ? "msr_lora_path" : "ingredients_lora_path"])
      XCTAssertEqual((recipe["conditioning"] as! [String:Any])["task"] as? String,family == "msr" ? "ref2va" : "control")
    }
  }
  func testDFRSetupPresetsPreserveSpecializedFieldsAndRejectAudioClip() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    for rounds in 0...2 {
      let id=rounds == 0 ? "swift-ltx25-dfr-spatial" : "swift-ltx25-dfr-temporal-\(rounds)"
      let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == id })
      var selected:[String:String]=[:]
      for component in preset.components {
        let url=root.appendingPathComponent(component.key)
        if component.kind == "directory" {
          try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
        } else { try Data([1]).write(to:url) }
        selected[component.key]=url.path
      }
      let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
      let config=recipe["config"] as! [String:Any]
      let components=recipe["components"] as! [String:Any]
      XCTAssertEqual(config["dfr_enabled"] as? Bool,true)
      XCTAssertEqual(config["dfr_temporal_rounds"] as? Int,rounds)
      XCTAssertEqual(config["dfr_detailing_lora_strength"] as? Double,0.5)
      XCTAssertNil(components["dfr_detailing_lora_path"])
      XCTAssertEqual(config["dfr_detailing_lora_path"] as? String,selected["dfr_detailing_lora_path"])
      XCTAssertEqual((config["dfr_temporal_upsampler_path"] as? String)?.isEmpty,rounds == 0)
      var clip=Clip(engine:.ltx25)
      XCTAssertTrue(preset.supports(clip))
      clip.attachments=[Attachment(assetID:UUID(),role:.first)]
      XCTAssertTrue(preset.supports(clip))
      clip.attachments=[Attachment(assetID:UUID(),role:.audioDriver)]
      XCTAssertFalse(preset.supports(clip))
    }
  }
  func testDFRScanDistinguishesDetailAdapterAndTemporalUpscaler() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    func write(_ name:String,_ metadata:[String:String],_ tensors:[String:[Int]]) throws -> URL {
      let url=root.appendingPathComponent(name)
      var header:[String:Any]=["__metadata__":metadata]
      for (key,shape) in tensors { header[key]=["dtype":"BF16","shape":shape,"data_offsets":[0,2]] }
      let body=try JSONSerialization.data(withJSONObject:header)
      var count=UInt64(body.count).littleEndian
      var data=withUnsafeBytes(of:&count) { Data($0) }
      data.append(body); data.append(contentsOf:[0,0])
      try data.write(to:url)
      return url
    }
    var pairs:[String:[Int]]=[:]
    for block in 0..<48 {
      for target in ["attn1.to_k","attn1.to_out.0","attn1.to_q","attn1.to_v",
        "attn2.to_k","attn2.to_out.0","attn2.to_q","attn2.to_v","ff.proj_in","ff.proj_out"] {
        let stem="diffusion_model.transformer_blocks.\(block).\(target)"
        pairs[stem+".lora_A.weight"]=[32,1]
        pairs[stem+".lora_B.weight"]=[1,32]
      }
    }
    let adapter=try write("detail.safetensors",["model_version":"2.5",
      "reference_downscale_factor":"2","reference_spatial_scale_factor":"2"],pairs)
    _=try write("ordinary.safetensors",["model_version":"2.5"],pairs)
    let temporal=try write("temporal.safetensors",["config":
      "{\"_class_name\":\"LatentUpsampler\",\"in_channels\":128,\"mid_channels\":512,\"num_blocks_per_stage\":4,\"dims\":3,\"spatial_upsample\":false,\"temporal_upsample\":true}"],
      ["initial_conv.weight":[512,128,3,3,3],"upsampler.0.weight":[1024,512,3,3,3],
        "final_conv.weight":[128,512,3,3,3]])
    let scan=try NativeModelSetup.scan(presetID:"swift-ltx25-dfr-temporal-1",roots:[root.path])
    XCTAssertEqual(scan.candidates["dfr_detailing_lora_path"],[adapter.path])
    XCTAssertEqual(scan.candidates["dfr_temporal_upsampler_path"],[temporal.path])
  }

  func testReferenceAdapterScanUsesLearnedSlotsAndMetadataRatherThanNames() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    var pairs:[String:Any]=[:]
    for block in 0..<48 { for target in ["attn1.to_k","attn1.to_out.0","attn1.to_q","attn1.to_v",
      "attn2.to_k","attn2.to_out.0","attn2.to_q","attn2.to_v","ff.net.0.proj","ff.net.2"] {
      let stem="diffusion_model.transformer_blocks.\(block).\(target)"
      let input=target == "ff.net.2" ? 16384 : 4096,output=target == "ff.net.0.proj" ? 16384 : 4096
      pairs[stem+".lora_A.weight"]=["dtype":"BF16","shape":[128,input],"data_offsets":[0,2]]
      pairs[stem+".lora_B.weight"]=["dtype":"BF16","shape":[output,128],"data_offsets":[0,2]]
    } }
    func write(_ name:String,_ header:[String:Any]) throws -> URL {
      let bytes=try JSONSerialization.data(withJSONObject:header);var size=UInt64(bytes.count).littleEndian
      var data=withUnsafeBytes(of:&size) { Data($0) };data.append(bytes);data.append(contentsOf:[0,0])
      let url=root.appendingPathComponent(name);try data.write(to:url);return url
    }
    var ingredients=pairs;ingredients["__metadata__"]=["model_version":"2.3","reference_downscale_factor":"1"]
    let sheet=try write("arbitrary-one.safetensors",ingredients)
    var msr=pairs;msr["__metadata__"]=["reference_slot_embedding_type":"fourier_mlp","reference_token_order":"prepend",
      "reference_slot_time_offsets":"pic1_based_negative_time","reference_slot_embedding_num_frequencies":"16",
      "reference_slot_embedding_hidden_dim":"256","reference_slot_embedding_dim":"128"]
    for (name,shape) in ["frequencies":[16],"net.0.weight":[256,33],"net.0.bias":[256],"net.2.weight":[128,256],"net.2.bias":[128]] {
      msr["diffusion_model.reference_slot_embedding."+name]=["dtype":"BF16","shape":shape,"data_offsets":[0,2]]
    }
    let reference=try write("arbitrary-two.safetensors",msr)
    msr.removeValue(forKey:"diffusion_model.reference_slot_embedding.frequencies")
    _ = try write("misleading-msr.safetensors",msr)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-ltx25-ingredients",roots:[root.path]).candidates["ingredients_lora_path"],[sheet.path])
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-ltx25-msr",roots:[root.path]).candidates["msr_lora_path"],[reference.path])
  }

  func testNativeScanUsesHeadersAndManifestsWithoutReadingWeights() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func tensor(_ name: String, metadata: [String: String], names: [String]) throws -> URL {
      let file = root.appendingPathComponent(name)
      var header: [String: Any] = ["__metadata__": metadata]
      for key in names { header[key] = ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]] }
      let bytes = try JSONSerialization.data(withJSONObject: header)
      var prefix = UInt64(bytes.count).littleEndian
      var output = withUnsafeBytes(of: &prefix) { Data($0) }
      output.append(bytes)
      output.append(contentsOf: [0, 0, 0, 0])
      try output.write(to: file)
      return file
    }
    let transformer = try tensor("arbitrary.safetensors", metadata: [
      "model_version": "2.5.0", "config": "{\"transformer\":{\"num_layers\":48}}"
    ], names: ["model.diffusion_model.patchify_proj.weight"])
    _ = try tensor("misleading-ltx-transformer.safetensors", metadata: [:], names: ["wrong.weight"])
    let upscaler = try tensor("upscale.safetensors", metadata: [
      "config": "{\"_class_name\":\"LatentUpsampler\",\"in_channels\":128,\"dims\":3,\"spatial_upsample\":true,\"temporal_upsample\":false}"
    ], names: ["initial_conv.weight"])
    let scan = try NativeModelSetup.scan(presetID: "swift-ltx25-text", roots: [root.path])
    XCTAssertEqual(scan.candidates["transformer_path"], [transformer.path])
    XCTAssertEqual(scan.candidates["spatial_upscaler_path"], [upscaler.path])
    XCTAssertTrue(scan.candidates["audio_vae_path", default: []].isEmpty)
    XCTAssertTrue(scan.warnings.contains { $0.contains("audio VAE") })
  }

  func testNativeScanDistinguishesH3TaskPartitions() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for (name, partition, tasks) in [("FL2VA", "fl2va", ["t2va", "fl2va"]),
      ("Ref2VA", "ref2va", ["ref2va"])] {
      let folder = root.appendingPathComponent(name)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      let document: [String: Any] = ["_minimax_h3": ["partition": partition, "tasks": tasks]]
      try JSONSerialization.data(withJSONObject: document).write(to: folder.appendingPathComponent("model_index.json"))
    }
    let image = try NativeModelSetup.scan(presetID: "swift-h3-image", roots: [root.path])
    let reference = try NativeModelSetup.scan(presetID: "swift-h3-reference", roots: [root.path])
    XCTAssertEqual(image.candidates["checkpoint"], [root.appendingPathComponent("FL2VA").path])
    XCTAssertEqual(reference.candidates["checkpoint"], [root.appendingPathComponent("Ref2VA").path])
  }

  func testInstalledNativeScanFindsCompatibleStacksWhenRequested() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_NATIVE_SCAN_ROOT"] else {
      throw XCTSkip("Opt-in installed-model discovery")
    }
    let roots = [root] + (ProcessInfo.processInfo.environment["WEETODD_NATIVE_VISION_ROOT"].map { [$0] } ?? [])
    for presetID in ["swift-h3-image", "swift-h3-reference", "swift-h3-fun-control", "swift-ltx25-text",
      "swift-ltx25-dfr-temporal-2"] {
      let result = try NativeModelSetup.scan(presetID: presetID, roots: roots)
      let missing = result.candidates.filter { $0.value.isEmpty }.map(\.key).sorted()
      XCTAssertTrue(missing.isEmpty, "\(presetID) missing \(missing); \(result.warnings)")
    }
  }
  func testInstalledUnionScanAndRecipeWhenRequested() throws {
    guard let root=ProcessInfo.processInfo.environment["WEETODD_NATIVE_SCAN_ROOT"],
      let adapter=ProcessInfo.processInfo.environment["WEETODD_NATIVE_UNION_ADAPTER"] else { throw XCTSkip("Opt-in installed Union setup") }
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.id == "swift-ltx25-union" })
    let scan=try NativeModelSetup.scan(presetID:preset.id,roots:[root,adapter])
    let selected=try Dictionary(uniqueKeysWithValues:preset.components.map { ($0.key,try XCTUnwrap(scan.candidates[$0.key]?.first,$0.key)) })
    let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.lowerMemory)
    XCTAssertEqual((recipe["config"] as? [String:Any])?["ic_lora_single_stage"] as? Bool,false)
    let components=recipe["components"] as! [String:Any]
    XCTAssertEqual((components["ic_loras"] as? [[Any]])?.first?.first as? String,selected["union_lora_path"])
    XCTAssertNil(components["union_lora_path"])
    XCTAssertNotNil(components["spatial_upscaler_path"])
  }
}
