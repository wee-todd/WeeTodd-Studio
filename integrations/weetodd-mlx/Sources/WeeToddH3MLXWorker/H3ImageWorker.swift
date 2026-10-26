import Foundation
import H3MLX
import InferenceContracts

enum H3ImageWorker {
  static func execute(envelope: NativeVideoJobEnvelope, recipeData: Data, render: Bool,
    emitProgress: ([String: Any]) -> Void) throws -> [String: Any] {
    let started = Date()
    let prepared = try H3ImageRecipe.compile(data: recipeData) { path, digest, geometry in
      let loaded = try H3StillReferenceMedia.load(path: path, outputGeometry: geometry, pixelBudgetPercent: 100)
      guard loaded.sourceSHA256 == digest else {
        throw H3CheckpointError.invalid("An H3 image reference changed. Check settings again.")
      }
      return loaded.reference
    }
    let decoder = try H3VideoVAELayout(url: prepared.decoder)
    try prepared.mode.validateDecoder(imageOnly: decoder.imageOnly)
    let rows: Int, evaluations: Int
    if let text = prepared.text {
      let checked = try H3T2VARunner.preflight(text)
      rows = checked.packedRows; evaluations = checked.evaluations
    } else {
      let checked = try H3Ref2VAStillRunner.preflight(prepared.reference!)
      rows = checked.packedRows; evaluations = checked.evaluations
    }
    let root = try JSONSerialization.jsonObject(with: recipeData) as! [String: Any]
    let fields = root["still_image"] as! [String: Any]
    let components = root["components"] as! [String: Any]
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    var paths = [prepared.decoder, executable]
    let metalLibrary = executable.deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("Resources/LTXNative/mlx.metallib")
    if FileManager.default.fileExists(atPath: metalLibrary.path) { paths.append(metalLibrary) }
    for key in ["transformer", "text_encoder", "tokenizer", "video_vae"] {
      if let path = components[key] as? String { paths.append(URL(fileURLWithPath: path)) }
    }
    if prepared.reference != nil, let path = components["vision_encoder"] as? String { paths.append(URL(fileURLWithPath: path)) }
    paths += (prepared.text?.loRAAdapters ?? prepared.reference!.loRAAdapters).map(\.url)
    let fingerprint = try H3ImageIdentity.fingerprint(recipe: root, components: paths)
    if let expected = fields["expected_fingerprint"] as? String, expected != fingerprint {
      throw H3CheckpointError.invalid("H3 model files or image settings changed after preflight. Check settings again.")
    }
    let geometry = prepared.geometry
    var normalizedRecipe = root
    var savedFields = fields; savedFields.removeValue(forKey: "expected_fingerprint")
    normalizedRecipe["still_image"] = savedFields
    let seed = prepared.text?.seed ?? prepared.reference!.seed
    let configuration: [String: Any] = ["width": geometry.width, "height": geometry.height,
      "steps": evaluations, "seed": seed, "decoderMode": prepared.mode.rawValue,
      "frameSelection": prepared.selection.rawValue,
      "selectedFrame": prepared.mode.selectedIndex(prepared.selection), "sourceTask": prepared.sourceTask]
    let normalized: [String: Any] = ["engine": "h3", "prompt": root["prompt"]!,
      "modelID": components["transformer"]!, "configuration": configuration, "recipe": normalizedRecipe]
    var result: [String: Any] = ["nativeRuntime": "swift-mlx", "jobID": envelope.jobID.uuidString,
      "requestID": fields["request_id"] ?? envelope.recipeSHA256,
      "eligibility": "allowed", "issues": [], "fingerprint": fingerprint,
      "normalizedRequest": normalized, "packedRows": rows, "evaluations": evaluations,
      "framesSampled": geometry.frames, "audioDecoded": false,
      "preflightSeconds": Date().timeIntervalSince(started)]
    guard render else { return result }
    let output = URL(fileURLWithPath: envelope.outputDirectory)
    let manager = FileManager.default
    guard !manager.fileExists(atPath: output.path), manager.fileExists(atPath: output.deletingLastPathComponent().path) else {
      throw H3CheckpointError.invalid("H3 image output must be a new directory with an existing parent.")
    }
    let staging = output.deletingLastPathComponent().appendingPathComponent(".h3-image-" + UUID().uuidString)
    try manager.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? manager.removeItem(at: staging) }
    var boundary = Date(), timings: [String: Double] = [:]
    let progress: (String, Int, Int) -> Void = { stage, completed, total in
      if let name = H3WorkerStageBoundary.name(stage: stage, completed: completed, total: total) {
        let now = Date(); timings[name] = now.timeIntervalSince(boundary); boundary = now
      }
      let message: String
      if stage.hasPrefix("sampling_block_"), let step = Int(stage.dropFirst("sampling_block_".count)) {
        message = "Sampling H3 image · step \(step) of \(evaluations)"
      } else {
        switch stage {
        case "sampling": message = "Sampling H3 image · \(completed) of \(evaluations)"
        case "text": message = "Encoding the image prompt"
        case "transformer_prepare": message = "Preparing the H3 model"
        case "reference_video_encode": message = "Encoding reference images"
        case "transformer_weights_released": message = "H3 sampling complete"
        default: message = "Preparing the H3 image"
        }
      }
      var event: [String: Any] = ["event": "progress", "stage": stage, "message": message]
      if let fraction = H3WorkerProgress.fraction(stage: stage, completed: completed, total: total, evaluations: evaluations) { event["fraction"] = fraction }
      emitProgress(event)
    }
    let sampled: H3SampledLatents
    if let text = prepared.text { sampled = try H3T2VARunner.sample(text, progress: progress) }
    else { sampled = try H3Ref2VAStillRunner.sample(prepared.reference!, progress: progress) }
    try Task.checkCancellation()
    emitProgress(["event": "progress", "stage": "image_decode", "fraction": 0.85, "message": "Decoding H3 still image"])
    let memory = prepared.text?.videoDecodeMemoryMode ?? prepared.reference?.videoDecodeMemoryMode
    let precision = prepared.text?.videoDecodePrecision ?? prepared.reference!.videoDecodePrecision
    let png = try H3ImageOutput.decode(videoRows: sampled.rows.video, geometry: geometry,
      decoder: prepared.decoder, mode: prepared.mode, selection: prepared.selection,
      memoryMode: memory, precision: precision)
    timings["imageDecode"] = Date().timeIntervalSince(boundary)
    try Task.checkCancellation()
    try png.write(to: staging.appendingPathComponent("image.png"), options: .withoutOverwriting)
    result["asset"] = ["path": output.appendingPathComponent("image.png").path,
      "width": geometry.width, "height": geometry.height]
    result["receiptPath"] = output.appendingPathComponent("receipt.json").path
    result["stageSeconds"] = timings; result["totalSeconds"] = Date().timeIntervalSince(started)
    try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
      .write(to: staging.appendingPathComponent("receipt.json"), options: .withoutOverwriting)
    try manager.moveItem(at: staging, to: output)
    return result
  }
}
