import Foundation

public struct NativeModelScanResult {
  public let candidates: [String: [String]]
  public let warnings: [String]
}

/// Weight-free setup for the Swift audiovisual workers. Specialized DFR presets
/// retain explicit adapter contracts and are preflighted by the shared worker.
public enum NativeModelSetup {
  public static let engines: Set<String> = ["h3", "ltx25"]

  /// Inspect bounded headers and manifests in user-selected folders. This never
  /// reads tensor payloads, copies weights, or starts the optional Python runtime.
  /// The worker still validates the selected complete stack before generation.
  public static func scan(presetID: String, roots: [String]) throws -> NativeModelScanResult {
    guard let preset = catalog().first(where: { $0.id == presetID }), !roots.isEmpty else {
      throw StudioError.invalid("Choose a native preset and at least one model folder.")
    }
    let fm = FileManager.default
    var candidates = Dictionary(uniqueKeysWithValues: preset.components.map { ($0.key, [String]()) })
    var warnings: [String] = []
    var seen: Set<String> = []
    var pending = roots.reversed().map { URL(fileURLWithPath: $0) }
    var inspected = 0
    while let next = pending.popLast() {
      try Task.checkCancellation()
      inspected += 1
      if inspected > 20_000 {
        warnings.append("Scan entry limit reached. Choose a smaller model folder.")
        break
      }
      let url = next.standardizedFileURL.resolvingSymlinksInPath()
      guard seen.insert(url.path).inserted else { continue }
      var directory: ObjCBool = false
      guard fm.fileExists(atPath: url.path, isDirectory: &directory) else {
        if roots.contains(next.path) { warnings.append("Model folder is unavailable: \(next.path)") }
        continue
      }
      let kind = directory.boolValue ? "directory" : "file"
      if kind == "directory" || url.pathExtension == "safetensors" {
        for component in preset.components where (component.accepts ?? [component.kind]).contains(kind) {
          if (try? NativeModelInspector.matches(url, key: component.key,
            engine: preset.engine, task: preset.task)) == true {
            candidates[component.key, default: []].append(url.path)
          }
        }
      }
      guard directory.boolValue else { continue }
      let children = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles])) ?? []
      for child in children.sorted(by: { $0.path > $1.path }) {
        if [".git", ".cache", "pages"].contains(child.lastPathComponent) ||
          child.lastPathComponent.hasPrefix("._") { continue }
        pending.append(child)
      }
    }
    for component in preset.components {
      let matches = Array(Set(candidates[component.key, default: []])).sorted()
      candidates[component.key] = matches
      if matches.isEmpty { warnings.append("No compatible \(component.label) found. Import it directly if installed elsewhere.") }
      else if matches.count > 1 { warnings.append("Multiple candidates for \(component.label); choose the intended component.") }
    }
    warnings.append("Discovery checks headers and manifests. Worker preflight validates the selected stack.")
    return NativeModelScanResult(candidates: candidates, warnings: warnings)
  }

  public static func catalog() -> [ModelSetupPreset] {
    func component(_ key: String, _ label: String, _ accepts: [String]) -> ModelSetupComponent {
      ModelSetupComponent(key: key, label: label, kind: accepts[0], accepts: accepts)
    }
    let h3 = [
      component("checkpoint", "H3 task manifest", ["directory"]),
      component("transformer", "H3 Comfy INT8 or compatible direct transformer", ["file"]),
      component("text_encoder", "Qwen3-VL text encoder pages", ["directory"]),
      component("processor", "Qwen3-VL processor", ["directory"]),
      component("tokenizer", "Qwen tokenizer", ["directory", "file"]),
      component("video_vae", "H3 video VAE", ["file"]),
      component("audio_vae", "H3 folded-weight audio VAE", ["file"]),
    ]
    let ltx = [
      component("transformer_path", "LTX 2.5 distilled transformer", ["directory", "file"]),
      component("text_encoder_path", "LTX 2.5 Gemma 4 pack", ["directory", "file"]),
      component("video_vae_path", "LTX 2.5 video VAE", ["file"]),
      component("audio_vae_path", "LTX 2.5 audio VAE and vocoder", ["file"]),
      component("spatial_upscaler_path", "LTX spatial latent upscaler", ["file"]),
    ]
    let ordinary = [("h3", "MiniMax H3", h3), ("ltx25", "LTX 2.5", ltx)].flatMap { engine, title, fields in
      (engine == "h3" ? [("text", "t2v", "Text to video"),
        ("image", "fflf", "Image to video"), ("reference", "ref2va", "Reference to video")]
        : [("text", "t2v", "Text to video"), ("image", "fflf", "Image to video")])
        .map { suffix, task, label in
          ModelSetupPreset(id: "swift-\(engine)-\(suffix)", name: "\(title) · \(label) · Swift",
            engine: engine, task: task,
            description: "Reuse installed components in place. Text-only setup runs Swift worker preflight now; image and reference tasks run it after clip media is attached.",
            components: fields + (engine == "h3" && task != "t2v"
              ? [component("vision_encoder","Qwen3-VL vision tower",["file","directory"])] : []))
        }
    }
    let detail = component("dfr_detailing_lora_path", "Pixel-Spatial x2 DFR adapter", ["file"])
    let temporal = component("dfr_temporal_upsampler_path", "LTX temporal x2 latent upscaler", ["file"])
    let dfr = (0...2).map { rounds in
      ModelSetupPreset(id: rounds == 0 ? "swift-ltx25-dfr-spatial" : "swift-ltx25-dfr-temporal-\(rounds)",
        name: rounds == 0 ? "LTX 2.5 · DFR spatial · Swift" : "LTX 2.5 · DFR + \(rounds) temporal round\(rounds == 1 ? "" : "s") · Swift",
        engine: "ltx25", task: "t2v",
        description: "Reuse installed DFR components in place. Swift worker preflight checks the full adapter stack before creating this profile.",
        components: ltx + [detail] + (rounds == 0 ? [] : [temporal]))
    }
    let references=[("msr","ref2va","MSR images","msr_lora_path","MSR learned-slot adapter"),
      ("ingredients","control","Ingredients sheet","ingredients_lora_path","Ingredients adapter")].map {
      family,task,label,key,adapter in
      ModelSetupPreset(id:"swift-ltx25-"+family,name:"LTX 2.5 · "+label+" · Swift",engine:"ltx25",task:task,
        description: family == "ingredients"
          ? "Experimental single-stage reference-sheet generation. LTX 2.5 adapters start at strength 1.0; legacy LTX 2.3 adapters retain 1.2. Attach a described sheet and prepare the clip for Swift worker preflight."
          : "Experimental single-stage reference generation. Link installed components, attach described images and prepare the clip for Swift worker preflight. Identity and audio quality remain under qualification.",
        components:ltx.filter { $0.key != "spatial_upscaler_path" }+[component(key,adapter,["file"])])
    }
    let union=ModelSetupPreset(id:"swift-ltx25-union",name:"LTX 2.5 · Union Control · Swift",engine:"ltx25",task:"control",
      description:"Experimental two-stage control generation. Attach one preprocessed Canny, depth or pose movie. Studio freezes a quarter-canvas RGB guide; the Union adapter is active only in stage one.",
      components:ltx+[component("union_lora_path","LTX 2.3 Union rank-64 adapter",["file"])])
    let controls = [("motion-track", "Motion Track", "motion_track_lora_path", "Motion Track rank-32 adapter"),
      ("crossview", "CrossView", "crossview_lora_path", "CrossView v2 rank-32 adapter")].map {
      family, label, key, adapter in
      ModelSetupPreset(id: "swift-ltx25-" + family, name: "LTX 2.5 · " + label + " · Swift",
        engine: "ltx25", task: "control",
        description: family == "motion-track"
          ? "Experimental two-stage control generation. Attach one movie with drawn or tracked trajectories. The Motion Track adapter is active only in stage one."
          : "Experimental two-stage view generation. Attach an already prepared depth-warp movie and its source movie in that order. The CrossView adapter is active only in stage one.",
        components: ltx + [component(key, adapter, ["file"])])
    }
    let crossviewIngredients = ModelSetupPreset(id: "swift-ltx25-crossview-ingredients",
      name: "LTX 2.5 · CrossView + Ingredients · Swift", engine: "ltx25", task: "control",
      description: "Experimental two-stage generation using a prepared depth warp, its source movie and a complete Ingredients reference sheet fitted with black padding. Both compatible adapters are active in stage one; the refinement uses a clean transformer.",
      components: ltx + [component("crossview_lora_path", "CrossView v2 rank-32 adapter", ["file"]),
        component("ingredients_lora_path", "Ingredients adapter", ["file"])])
    let h3Fun = ModelSetupPreset(id: "swift-h3-fun-control", name: "MiniMax H3 · Fun Union Control · Swift",
      engine: "h3", task: "control",
      description: "Experimental generation using one prepared Canny, depth, HED, MLSD or pose movie. Choose the original full-width five- or ten-block Fun branch and a compatible full-width H3 transformer. Turbo LoRAs cannot be combined with this control route.",
      components: h3 + [component("fun_controlnet", "H3 Fun Union full-width control branch", ["file"])])
    return ordinary + [h3Fun] + dfr + references + [union] + controls + [crossviewIngredients]
  }

  public static func recipe(preset: ModelSetupPreset, selected: [String: String],
    memoryMode: ModelSetupMemoryMode) throws -> [String: Any] {
    guard catalog().contains(where: { $0.id == preset.id }),
      Set(selected.keys) == Set(preset.components.map(\.key)) else {
      throw StudioError.invalid("Choose one component for every native setup field.")
    }
    let manager = FileManager.default
    var components: [String: Any] = [:]
    for field in preset.components {
      guard let raw = selected[field.key], !raw.isEmpty, raw.hasPrefix("/"), !raw.utf8.contains(0) else {
        throw StudioError.invalid("Select an absolute path for \(field.label).")
      }
      let path = URL(fileURLWithPath: raw).standardizedFileURL.resolvingSymlinksInPath().path
      var directory: ObjCBool = false
      guard manager.fileExists(atPath: path, isDirectory: &directory),
        (field.accepts ?? [field.kind]).contains(directory.boolValue ? "directory" : "file") else {
        throw StudioError.invalid("\(field.label) is missing or has the wrong file type.")
      }
      if preset.engine == "h3",field.key == "tokenizer",directory.boolValue {
        let tokenizer=URL(fileURLWithPath:path).appendingPathComponent("tokenizer.json").resolvingSymlinksInPath()
        let info=try? tokenizer.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
        guard info?.isRegularFile == true,let size=info?.fileSize,size>0,size<=32*1024*1024 else {
          throw StudioError.invalid("Choose a Qwen tokenizer folder containing a regular tokenizer.json under 32 MiB.")
        }
        components[field.key]=tokenizer.path
      } else { components[field.key] = path }
    }
    let lowMemory = memoryMode == .lowerMemory ||
      (memoryMode == .automatic && ProcessInfo.processInfo.physicalMemory <= 64 * 1_073_741_824)
    var config: [String: Any]
    if preset.engine == "h3" {
      components["task"] = ["t2v": "t2va", "fflf": "fl2va", "ref2va": "ref2va", "control": "t2va"][preset.task]!
      config = ["width": 384, "height": 256, "duration_seconds": 2.5,
        "steps": 20, "seed": 0, "drop_adaln": true, "resolution_mode": "custom",
        "memory_mode": lowMemory ? "low_memory_bf16" : "normal",
        "projection_backend": "mlx", "sampling_method": "euler"]
    } else {
      config = ["pipeline_mode": "distilled", "width": 768, "height": 512,
        "duration_seconds": 5.0, "frame_rate": 24.0, "seed": 0,
        "stage1_steps": 8, "stage2_steps": 3, "low_memory": true,
        "low_ram_streaming": false]
      let controlKeys: [String: [String]] = [
        "swift-ltx25-union": ["union_lora_path"],
        "swift-ltx25-motion-track": ["motion_track_lora_path"],
        "swift-ltx25-crossview": ["crossview_lora_path"],
        "swift-ltx25-crossview-ingredients": ["crossview_lora_path", "ingredients_lora_path"]]
      if let keys = controlKeys[preset.id] {
        let adapters = try keys.map { key -> [Any] in
          guard let adapter = components.removeValue(forKey: key) as? String else {
            throw StudioError.invalid("Choose every dedicated control adapter.")
          }
          return [adapter, 1.0]
        }
        components["ic_loras"] = adapters; config["ic_lora_single_stage"] = false
      }
      if ["swift-ltx25-msr","swift-ltx25-ingredients"].contains(preset.id) {
        let msr=preset.id == "swift-ltx25-msr",key=msr ? "msr_lora_path" : "ingredients_lora_path"
        guard let adapter=components[key] as? String else { throw StudioError.invalid("Choose the dedicated reference adapter.") }
        if !msr { components.removeValue(forKey:key) }
        let strength=msr ? 1.0 : NativeModelInspector.ingredientsDefaultStrength(URL(fileURLWithPath:adapter))
        components["ic_loras"]=[[adapter,strength]];components["spatial_upscaler_path"]=""
        if msr { components["msr_lora_strength"]=strength }
        config["ic_lora_single_stage"]=true
      }
      if preset.id.hasPrefix("swift-ltx25-dfr-") {
        let rounds = preset.id == "swift-ltx25-dfr-spatial" ? 0
          : preset.id == "swift-ltx25-dfr-temporal-1" ? 1 : 2
        guard let detail = components.removeValue(forKey: "dfr_detailing_lora_path") as? String else {
          throw StudioError.invalid("Choose the Pixel-Spatial x2 DFR adapter.")
        }
        let temporal = components.removeValue(forKey: "dfr_temporal_upsampler_path") as? String ?? ""
        config["width"] = 512; config["height"] = 256; config["duration_seconds"] = 2.0
        config["dfr_enabled"] = true
        config["dfr_detailing_lora_path"] = detail
        config["dfr_detailing_lora_strength"] = 0.5
        config["dfr_temporal_rounds"] = rounds
        config["dfr_temporal_upsampler_path"] = temporal
      }
    }
    var conditioning: [String: Any] = ["version": 1, "task": preset.task,
      "inputs": [], "audio_policy": "generated"]
    let controlFamilies = ["swift-ltx25-motion-track": "motion_track",
      "swift-ltx25-crossview": "crossview_warp", "swift-ltx25-crossview-ingredients": "crossview_ingredients"]
    if let family = controlFamilies[preset.id] { conditioning["control_family"] = family }
    return ["format": "weetodd-headless-v2", "engine": preset.engine,
      "candidate": preset.id, "components": components, "config": config,
      "prompt": "A continuous scene with synchronized sound.", "conditioning": conditioning]
  }

  public static func stage(_ recipe: [String: Any], directory: String) throws -> String {
    let root = URL(fileURLWithPath: directory).standardizedFileURL
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let destination = root.appendingPathComponent("swift-setup-\(UUID().uuidString).json")
    let data = try JSONSerialization.data(withJSONObject: recipe, options: [.prettyPrinted, .sortedKeys])
    guard data.count <= 1024 * 1024 else { throw StudioError.invalid("Model recipe exceeds 1 MiB.") }
    try data.write(to: destination, options: .atomic)
    return destination.path
  }
}

private enum NativeModelInspector {
  static func ingredientsDefaultStrength(_ url: URL) -> Double {
    let metadata = (try? header(url)["__metadata__"]) as? [String:String] ?? [:]
    return ["2.5","2.5.0"].contains(metadata["model_version"] ?? "") ? 1.0 : 1.2
  }
  private static func document(_ url: URL) throws -> [String: Any] {
    let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
    guard size > 0, size <= 2 * 1024 * 1024,
      let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
      throw StudioError.invalid("Invalid bounded model manifest.")
    }
    return value
  }

  private static func header(_ url: URL) throws -> [String: Any] {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
      throw StudioError.invalid("Truncated model header.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * $1.offset)) }
    guard length > 0, length <= 16 * 1024 * 1024,
      let bytes = try file.read(upToCount: Int(length)), bytes.count == Int(length),
      let value = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw StudioError.invalid("Invalid bounded SafeTensors header.")
    }
    return value
  }

  private static func embedded(_ value: Any?) -> [String: Any]? {
    if let object = value as? [String: Any] { return object }
    if let string = value as? String,
      let data = string.data(using: .utf8), data.count <= 2 * 1024 * 1024 {
      return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
    return nil
  }

  private static func completeControlAdapter(_ tensors: [String: Any], rank: Int) -> Bool {
    let keys = tensors.keys.filter { $0 != "__metadata__" }
    guard keys.count == 960 else { return false }
    var targets = Set<String>()
    for name in keys where name.hasSuffix(".lora_A.weight") {
      let partner = name.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
      guard let down = tensors[name] as? [String: Any], let up = tensors[partner] as? [String: Any],
        ["BF16", "F16", "F32"].contains(down["dtype"] as? String ?? ""),
        ["BF16", "F16", "F32"].contains(up["dtype"] as? String ?? ""),
        let a = down["shape"] as? [Int], let b = up["shape"] as? [Int],
        a.count == 2, b.count == 2, a[0] == rank, b[1] == rank else { return false }
      var stem = String(name.dropLast(".lora_A.weight".count))
      let prefixes = ["base_model.model.model.diffusion_model.", "base_model.model.diffusion_model.",
        "base_model.model.transformer.", "base_model.model.", "model.diffusion_model.",
        "diffusion_model.", "transformer."]
      if let prefix = prefixes.first(where: stem.hasPrefix) { stem.removeFirst(prefix.count) }
      stem += "."
      for (old, new) in [(".to_out.0.", ".to_out."), (".ff.net.0.proj.", ".ff.proj_in."),
        (".ff.net.2.", ".ff.proj_out.")] { stem = stem.replacingOccurrences(of: old, with: new) }
      stem.removeLast()
      let parts = stem.split(separator: ".")
      guard parts.count >= 4, parts[0] == "transformer_blocks", let block = Int(parts[1]),
        (0..<48).contains(block), String(block) == parts[1] else { return false }
      let tail = parts.dropFirst(2).joined(separator: ".")
      let attention = ["attn1", "attn2"].flatMap { family in
        ["to_k", "to_out", "to_q", "to_v"].map { family + "." + $0 }
      }
      let shape: [Int]
      if attention.contains(tail) { shape = [4096, 4096] }
      else if tail == "ff.proj_in" { shape = [16384, 4096] }
      else if tail == "ff.proj_out" { shape = [4096, 16384] }
      else { return false }
      guard a[1] == shape[1], b[0] == shape[0], targets.insert(stem).inserted else { return false }
    }
    return targets.count == 480
  }

  static func matches(_ url: URL, key: String, engine: String, task: String) throws -> Bool {
    let directory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
    if engine == "h3" {
      if key == "fun_controlnet" {
        guard !directory else { return false }
        let tensors = try header(url)
        let prefixes = ["model.diffusion_model.", "diffusion_model.", "controlnet.", ""].filter {
          tensors[$0 + "control_proj_in.weight"] != nil
        }
        guard prefixes.count == 1, let prefix = prefixes.first else { return false }
        let count = tensors[prefix + "control_blocks.9.adaln_proj.linear.weight"] != nil ? 10 : 5
        var expected: [String: [Int]] = ["control_proj_in.weight": [5376,196], "control_proj_in.bias": [5376],
          "control_blocks.0.before_proj.weight": [5376,5376], "control_blocks.0.before_proj.bias": [5376]]
        let shapes: [String: [Int]] = ["adaln_proj.linear.weight": [96768,2688], "adaln_proj.linear.bias": [96768],
          "norm1.weight": [5376], "norm2.weight": [5376], "attn.norm_q.weight": [128], "attn.norm_k.weight": [128],
          "attn.to_q.weight": [7168,5376], "attn.to_k.weight": [7168,5376], "attn.to_v.weight": [7168,5376],
          "attn.to_out.0.weight": [5376,7168], "ff.net.0.proj.weight": [28672,5376], "ff.net.2.weight": [5376,14336],
          "after_proj.weight": [5376,5376], "after_proj.bias": [5376]]
        for block in 0..<count { for (name, shape) in shapes { expected["control_blocks.\(block)." + name] = shape } }
        guard Set(tensors.keys.filter { $0 != "__metadata__" }) == Set(expected.keys.map { prefix + $0 }) else { return false }
        return expected.allSatisfy { name, shape in
          guard let info = tensors[prefix + name] as? [String: Any] else { return false }
          return info["shape"] as? [Int] == shape && ["BF16", "F16", "F32"].contains(info["dtype"] as? String ?? "")
        }
      }
      if key == "vision_encoder" {
        var file=url
        if directory {
          if let manifest=try? document(url.appendingPathComponent("paged_text_encoder_manifest.json")),
            manifest["format"] as? String == "weetodd-h3-qwen-paged-v2",
            let vision=manifest["vision"] as? [String:Any],vision["file"] as? String == "pages/vision.safetensors" {
            file=url.appendingPathComponent("pages/vision.safetensors")
          } else { file=url.appendingPathComponent("text_encoder.safetensors") }
        }
        let tensors=try header(file)
        return tensors.keys.filter { $0.hasPrefix("visual.") }.count == 529
          && (tensors["visual.patch_embed.proj.weight"] as? [String:Any])?["shape"] as? [Int] == [1152,3,2,16,16]
          && (tensors["visual.blocks.26.attn.qkv.weight"] as? [String:Any])?["shape"] as? [Int] == [3456,288]
          && (tensors["visual.deepstack_merger_list.2.linear_fc2.weight"] as? [String:Any])?["shape"] as? [Int] == [5120,1152]
      }
      if key == "checkpoint" {
        guard directory, let info = try? document(url.appendingPathComponent("model_index.json"))["_minimax_h3"] as? [String: Any],
          let partition = info["partition"] as? String, let tasks = info["tasks"] as? [String] else { return false }
        let expected = task == "ref2va" ? "ref2va" : "fl2va"
        return partition == expected && tasks.contains(["t2v", "control"].contains(task) ? "t2va" : task == "fflf" ? "fl2va" : task)
      }
      if directory {
        switch key {
        case "text_encoder":
          guard let config = try? document(url.appendingPathComponent("config.json")) else { return false }
          let text = embedded(config["text_config"]) ?? embedded(config["text_encoder"]) ?? config
          return (text["hidden_size"] as? Int ?? text["hidden"] as? Int) == 5120 &&
            FileManager.default.fileExists(atPath: url.appendingPathComponent("paged_text_encoder_manifest.json").path)
        case "processor":
          guard let config = (try? document(url.appendingPathComponent("processor_config.json"))) ??
            (try? document(url.appendingPathComponent("preprocessor_config.json"))) else { return false }
          return config["processor_class"] as? String == "Qwen3VLProcessor"
        case "tokenizer":
          guard let config = try? document(url.appendingPathComponent("tokenizer_config.json")) else { return false }
          return ["Qwen2Tokenizer", "Qwen2TokenizerFast"].contains(config["tokenizer_class"] as? String ?? "") &&
            FileManager.default.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path)
        default: return false
        }
      }
      let tensors = try header(url)
      if key == "transformer" {
        func shape(_ name:String) -> [Int]? { (tensors[name] as? [String:Any])?["shape"] as? [Int] }
        let prefixes = ["model.diffusion_model.", "diffusion_model.", ""].filter {
          shape($0 + "video_patch_proj.weight") != nil
        }
        guard prefixes.count == 1, let prefix = prefixes.first,
          shape(prefix + "video_patch_proj.weight") == [5376,96],
          shape(prefix + "audio_patch_proj.weight") == [5376,32],
          shape(prefix + "condition_proj.weight") == [5376,5120] else { return false }
        func dtype(_ name: String) -> String? { (tensors[prefix + name] as? [String: Any])?["dtype"] as? String }
        if task == "fflf" && !prefix.isEmpty { return false }
        if task == "ref2va" && prefix.isEmpty { return false }
        if task == "control" && (prefix.isEmpty
          || shape(prefix + "time_embedder.proj_out.weight") != [2688,5376]
          || shape(prefix + "blocks.0.adaln_proj.linear.weight") != [96768,2688]
          || shape(prefix + "final_layer.adaln_proj.linear.weight") != [10752,2688]) {
          return false
        }
        if !prefix.isEmpty {
          return ["BF16", "F16"].contains(dtype("video_patch_proj.weight") ?? "")
            && ["BF16", "F16"].contains(dtype("audio_patch_proj.weight") ?? "")
            && ["BF16", "F16"].contains(dtype("condition_proj.weight") ?? "")
        }
        let metadata = tensors["__metadata__"] as? [String: String] ?? [:]
        return shape("adaln_t_table") == [1001,64]
          && (tensors["adaln_t_table"] as? [String: Any])?["dtype"] as? String == "F32"
          && metadata["partition"]?.uppercased() == "FL2VA"
          && metadata["adaln_curve_grid"] == "1001" && metadata["adaln_curve_rank"] == "64"
          && metadata["adaln_curve_centered"] == "true"
          && dtype("video_patch_proj.weight") == "F32" && dtype("audio_patch_proj.weight") == "F32"
          && ["BF16", "F16"].contains(dtype("condition_proj.weight") ?? "")
      }
      guard ["video_vae", "audio_vae"].contains(key) else { return false }
      let metadata = tensors["__metadata__"] as? [String: String] ?? [:]
      return key == "video_vae"
        ? metadata["minimax_h3_video_vae"] != nil && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
        : metadata["minimax_h3_audio_vae"] != nil && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
    }
    if directory {
      guard ["transformer_path", "text_encoder_path"].contains(key),
        let manifest = try? document(url.appendingPathComponent("paged_manifest.json")),
        let fixed = manifest["fixed"] as? [String: Any], let file = fixed["file"] as? String,
        FileManager.default.fileExists(atPath: url.appendingPathComponent(file).path) else { return false }
      let metadata = manifest["metadata"] as? [String: Any] ?? [:]
      if key == "transformer_path" {
        let config = embedded(metadata["config"])
        return manifest["kind"] as? String == "transformer" &&
          (metadata["model_version"] as? String)?.hasPrefix("2.5") == true && config?["transformer"] is [String: Any]
      }
      let gemma = embedded(metadata["gemma_config"])
      return manifest["kind"] as? String == "gemma" && gemma?["model_type"] as? String == "gemma4_unified"
    }
    let tensors = try header(url)
    let metadata = tensors["__metadata__"] as? [String: String] ?? [:]
    let config = embedded(metadata["config"]) ?? [:]
    switch key {
    case "motion_track_lora_path", "crossview_lora_path":
      let motion = key == "motion_track_lora_path"
      guard metadata["reference_downscale_factor"] == (motion ? "2" : "1"),
        metadata["reference_spatial_scale_factor"] == nil,
        (metadata["reference_temporal_scale_factor"] ?? "1") == "1",
        !metadata.keys.contains(where: { $0.hasPrefix("reference_slot_") }),
        metadata["adapter_family"] == nil || metadata["adapter_family"] == (motion ? "motion_track" : "crossview"),
        metadata["model_version"].map({ ["2.3", "2.3.0", "2.5"].contains($0) }) ?? !motion else { return false }
      return completeControlAdapter(tensors, rank: 32)
    case "union_lora_path":
      let keys=tensors.keys.filter { $0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_B.weight") }
      let a=keys.filter { $0.hasSuffix(".lora_A.weight") }
      return keys.count == 960 && a.count == 480 && metadata["model_version"]?.hasPrefix("2.3") == true
        && metadata["reference_downscale_factor"] == "2" && a.allSatisfy { name in
          let partner=name.replacingOccurrences(of:".lora_A.weight",with:".lora_B.weight")
          guard let down=(tensors[name] as? [String:Any])?["shape"] as? [Int],down.count == 2,
            let up=(tensors[partner] as? [String:Any])?["shape"] as? [Int],up.count == 2 else { return false }
          return down[0] == 64 && up[1] == 64
        }
    case "msr_lora_path","ingredients_lora_path":
      let keys=tensors.keys.filter { $0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_B.weight") }
      let a=keys.filter { $0.hasSuffix(".lora_A.weight") }
      guard keys.count == 960,a.count == 480,a.allSatisfy({ name in
        let partner=name.replacingOccurrences(of:".lora_A.weight",with:".lora_B.weight")
        guard let down=(tensors[name] as? [String:Any])?["shape"] as? [Int],down.count == 2,
          let up=(tensors[partner] as? [String:Any])?["shape"] as? [Int],up.count == 2 else { return false }
        return down[0] == 128 && up[1] == 128
      }) else { return false }
      if key == "ingredients_lora_path" {
        return ["2.3","2.3.0","2.5","2.5.0"].contains(metadata["model_version"] ?? "")
          && metadata["reference_downscale_factor"] == "1"
          && metadata["reference_spatial_scale_factor"] == nil
          && (metadata["reference_temporal_scale_factor"] ?? "1") == "1"
          && (metadata["adapter_family"] ?? "ingredients_reference_sheet") == "ingredients_reference_sheet"
          && completeControlAdapter(tensors,rank:128)
      }
      guard metadata["reference_slot_embedding_type"] == "fourier_mlp",
        metadata["reference_token_order"] == "prepend",
        metadata["reference_slot_time_offsets"] == "pic1_based_negative_time",
        metadata["reference_slot_embedding_num_frequencies"] == "16",
        metadata["reference_slot_embedding_hidden_dim"] == "256",
        metadata["reference_slot_embedding_dim"] == "128" else { return false }
      let shapes:[String:[Int]]=["frequencies":[16],"net.0.weight":[256,33],"net.0.bias":[256],"net.2.weight":[128,256],"net.2.bias":[128]]
      return ["diffusion_model.reference_slot_embedding.","reference_slot_embedding."].contains { prefix in
        shapes.allSatisfy { name,shape in (tensors[prefix+name] as? [String:Any])?["shape"] as? [Int] == shape }
      }
    case "dfr_detailing_lora_path":
      guard metadata["model_version"] == "2.5",
        metadata["reference_downscale_factor"] == "2",
        metadata["reference_spatial_scale_factor"] == "2" else { return false }
      let keys = Array(tensors.keys.filter { $0 != "__metadata__" })
      let a = keys.filter { $0.hasSuffix(".lora_A.weight") }
      let b = Set(keys.filter { $0.hasSuffix(".lora_B.weight") })
      return keys.count == 960 && a.count == 480 && b.count == 480 && a.allSatisfy { name in
        let partner = name.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
        guard b.contains(partner), let info = tensors[name] as? [String: Any],
          let shape = info["shape"] as? [Int], shape.count == 2, shape[0] == 32,
          let other = tensors[partner] as? [String: Any],
          let otherShape = other["shape"] as? [Int], otherShape.count == 2,
          otherShape[1] == 32 else { return false }
        return true
      }
    case "dfr_temporal_upsampler_path":
      guard config["_class_name"] as? String == "LatentUpsampler",
        config["in_channels"] as? Int == 128, config["mid_channels"] as? Int == 512,
        config["num_blocks_per_stage"] as? Int == 4, config["dims"] as? Int == 3,
        config["spatial_upsample"] as? Bool == false,
        config["temporal_upsample"] as? Bool == true else { return false }
      return (tensors["initial_conv.weight"] as? [String: Any])?["shape"] as? [Int] == [512,128,3,3,3]
        && (tensors["upsampler.0.weight"] as? [String: Any])?["shape"] as? [Int] == [1024,512,3,3,3]
        && (tensors["final_conv.weight"] as? [String: Any])?["shape"] as? [Int] == [128,512,3,3,3]
    case "transformer_path":
      return metadata["model_version"]?.hasPrefix("2.5") == true && config["transformer"] is [String: Any] &&
        tensors.keys.contains(where: { $0.hasSuffix("patchify_proj.weight") })
    case "text_encoder_path":
      return embedded(metadata["gemma_config"])?["model_type"] as? String == "gemma4_unified" &&
        tensors.keys.contains("model.embed_tokens.weight")
    case "video_vae_path":
      return config["vae"] is [String: Any] && tensors.keys.contains(where: { $0.hasPrefix("decoder.") })
    case "audio_vae_path":
      return tensors.keys.contains(where: { $0.hasPrefix("audio_vae.decoder.") }) &&
        tensors.keys.contains(where: { $0.hasPrefix("vocoder.") })
    case "spatial_upscaler_path":
      return config["_class_name"] as? String == "LatentUpsampler" &&
        config["in_channels"] as? Int == 128 && config["dims"] as? Int == 3 &&
        config["spatial_upsample"] as? Bool == true && config["temporal_upsample"] as? Bool == false
    default: return false
    }
  }
}
