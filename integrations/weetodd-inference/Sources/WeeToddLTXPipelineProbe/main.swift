import InferenceMedia
import CoreGraphics
import AdapterRuntime
import Darwin
import Foundation
import ImageIO
import LTX25Audio
import LTX25Engine
import LTX25NNC
import LTX25Text
import LTX25Video
import Metal
import UniformTypeIdentifiers

/// Developer integration proof only. This executes the released distilled
/// stage-one schedule or explicit spatial-upscaler two-stage recipe. Conditioning
/// inputs and a Studio route remain separate work. Ordered standard LoRAs are active.
@main
struct PipelineProbe {
  struct Request: Codable {
    let gemmaRoot: String
    let transformerRoot: String
    let connectorCheckpoint: String
    let videoCheckpoint: String
    let audioCheckpoint: String
    let prompt: String
    let width: Int
    let height: Int
    let frames: Int
    let fps: Double
    let seed: UInt64
    let outputDirectory: String
    let spatialUpscalerCheckpoint: String?
    let loras: [LoRAAdapter]?

    enum CodingKeys: String, CodingKey, CaseIterable {
      case gemmaRoot = "gemma_root", transformerRoot = "transformer_root"
      case connectorCheckpoint = "connector_checkpoint", videoCheckpoint = "video_checkpoint"
      case audioCheckpoint = "audio_checkpoint", prompt, width, height, frames, fps, seed
      case outputDirectory = "output_directory"
      case spatialUpscalerCheckpoint = "spatial_upscaler_checkpoint"
      case loras
    }
  }

  final class Events {
    let started = Date()
    private let device = MTLCreateSystemDefaultDevice()
    private var stream: FileHandle?
    private(set) var samples: [[String: Any]] = []
    private(set) var observedPeakMetalBytes = 0
    func emit(_ stage: String, _ detail: [String: Any] = [:]) throws {
      let metal = device?.currentAllocatedSize ?? 0
      observedPeakMetalBytes = max(observedPeakMetalBytes, metal)
      var event = detail
      event["stage"] = stage; event["metal_allocated_bytes"] = metal
      event["elapsed_seconds"] = Date().timeIntervalSince(started)
      let bytes = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) + Data([10])
      try FileHandle.standardOutput.write(contentsOf: bytes)
      try stream?.write(contentsOf: bytes)
      samples.append(event)
    }
    func record(to url: URL) throws {
      stream = try PipelineProbe.newFile(url)
      for event in samples {
        try stream?.write(contentsOf: JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) + Data([10]))
      }
    }
    func close() throws { try stream?.close(); stream = nil }
    deinit { try? stream?.close() }
  }

  static let sigmas: [Double] = [1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]
  static let scope = "native-swift-ltx25-t2av-integration-probe"

  static func main() {
    do { try run() }
    catch {
      let event: [String: Any] = ["stage": "failed", "scope": scope, "error": String(describing: error)]
      if let bytes = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) {
        try? FileHandle.standardError.write(contentsOf: bytes + Data([10]))
      }
      exit(2)
    }
  }

  static func loadRequest(at url: URL) throws -> Request {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var info = stat()
    guard fstat(file.fileDescriptor, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
      info.st_size >= 0, info.st_size <= 256 * 1024 else {
      throw BlockError.invalid("Request must be a local regular JSON file of at most 256 KiB.")
    }
    // URL resource values can be cached after a caller rewrites the same URL.
    // Read through the inspected descriptor with a hard cap, including growth
    // between metadata inspection and payload access.
    guard let data = try file.read(upToCount: 256 * 1024 + 1), data.count <= 256 * 1024 else {
      throw BlockError.invalid("Request grew beyond its 256 KiB budget.")
    }
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys).isSubset(of: Set(Request.CodingKeys.allCases.map(\.rawValue))),
      Set(Request.CodingKeys.allCases.map(\.rawValue)).subtracting(["spatial_upscaler_checkpoint","loras"]).isSubset(of: Set(object.keys)),
      object["spatial_upscaler_checkpoint"] == nil || object["spatial_upscaler_checkpoint"] is String,
      object["loras"] == nil || object["loras"] is [[String:Any]] else {
      throw BlockError.invalid("Request has missing or unsupported fields; only explicit model paths, prompt, geometry, seed, ordered loras and output_directory are accepted.")
    }
    let value = try JSONDecoder().decode(Request.self, from: data)
    guard (value.loras?.count ?? 0) <= 16 else { throw BlockError.invalid("Use at most 16 ordered LoRAs.") }
    guard value.prompt.utf8.count <= 128 * 1024,
      !value.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw BlockError.invalid("Prompt must contain text and fit within 128 KiB UTF-8.")
    }
    for path in [value.gemmaRoot, value.transformerRoot, value.connectorCheckpoint,
      value.videoCheckpoint, value.audioCheckpoint, value.outputDirectory] + [value.spatialUpscalerCheckpoint].compactMap({ $0 }) {
      guard path.hasPrefix("/"), !path.utf8.contains(0), path.utf8.count <= 4096 else {
        throw BlockError.invalid("All model/output paths must be explicit absolute local paths.")
      }
    }
    return value
  }

  static func recipeReport(seed: UInt64,twoStage: Bool) -> [String:Any] {
    let streams: [[String:Any]] = twoStage ? [
      ["purpose":"initial","seed":seed,"draw_order":"initial_video, initial_audio"],
      ["purpose":"ancestral","seed":seed &+ 10000,"draw_order":"ancestral_video, ancestral_audio per nonterminal stage-one step"],
      ["purpose":"refinement","seed":seed &+ 2,"draw_order":"refinement_video, refinement_audio"],
    ] : [["purpose":"legacy","seed":seed,
      "draw_order":"initial_video, initial_audio, then ancestral_video and ancestral_audio in step order"]]
    return ["recipe":twoStage ? DistilledTwoStageRecipe.identifier : "legacy-stage-one",
      "model_evaluations":twoStage ? 11 : 8,"noise_algorithm":GaussianNoise.algorithm,
      "noise_streams":streams,"seed_offset_arithmetic":"UInt64 wrapping addition",
      "noise_draw_order":twoStage ? "independent initial, ancestral and refinement streams; video before audio in each stream" : streams[0]["draw_order"]!,
      "qualification":twoStage
        ? "two-stage integration with neural spatial upscaling; Studio routing, reference identity, cross-backend seed parity and production perceptual quality remain unqualified"
        : "single stage-one integration proof; not the full two-stage recipe, Studio route, seed parity or production quality qualification"]
  }

  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    let preflightOnly = args.count == 2 && args[0] == "--preflight"
    guard args.count == 1 || preflightOnly else {
      throw BlockError.invalid("Usage: WeeToddLTXPipelineProbe [--preflight] REQUEST.json")
    }
    let requestURL = URL(fileURLWithPath: args.last!)
    let request = try loadRequest(at: requestURL)
    let recipe = try request.spatialUpscalerCheckpoint.map { _ in
      try DistilledTwoStageRecipe(width: request.width,height: request.height,frames: request.frames,fps: request.fps,seed: request.seed)
    }
    let geometry = try AVGeometry(width: request.width, height: request.height, frames: request.frames, fps: request.fps)
    var videoConfiguration = VideoDecodeConfiguration(); videoConfiguration.frameRate = request.fps
    let videoPlan = try VideoDecodePlan(shape: geometry.videoShape, configuration: videoConfiguration)
    let configuration = try AVBlockConfiguration(videoTokens: geometry.videoTokens,
      audioTokens: geometry.audioFrames, textTokens: 1024)
    try AVBlockRunner.validateAllocation(configuration: configuration)
    let schedule = try SamplingSchedule(sigmas: sigmas, eta: 1)
    let expectedSamples = try AudioDecoder.sampleCount(latentFrames: geometry.audioFrames)
    let audioBudget = try AudioDecoder.estimatedPeakBytes(latentFrames: geometry.audioFrames)
    guard geometry.audioFrames <= 501, audioBudget <= 2 * 1024 * 1024 * 1024 else {
      throw BlockError.invalid("Audio geometry exceeds the decoder's default memory admission.")
    }
    let output = URL(fileURLWithPath: request.outputDirectory).standardizedFileURL
    let parent = try output.deletingLastPathComponent().resourceValues(forKeys: [.isDirectoryKey])
    var outputStatus = stat()
    let absent = Darwin.lstat(output.path, &outputStatus) != 0 && errno == ENOENT
    guard parent.isDirectory == true, absent else {
      throw BlockError.invalid("Output directory must be new, with an existing parent directory.")
    }
    let gemma = URL(fileURLWithPath: request.gemmaRoot)
    let transformer = URL(fileURLWithPath: request.transformerRoot)
    let connector = URL(fileURLWithPath: request.connectorCheckpoint)
    let videoCheckpoint = URL(fileURLWithPath: request.videoCheckpoint)
    let audioCheckpoint = URL(fileURLWithPath: request.audioCheckpoint)
    let adapters = try request.loras.map(LTXAdapterCompatibility.weightStack)
    let events = Events()
    try events.emit("preflight_start")
    // Keep release identity explicit: matching tensor shapes alone do not make a
    // developer checkpoint suitable for the fixed distilled schedule.
    let manifestURL = transformer.appendingPathComponent("paged_manifest.json")
    let manifestInfo = try manifestURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard manifestInfo.isRegularFile == true, let manifestBytes = manifestInfo.fileSize,
      manifestBytes <= 1024 * 1024,
      let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any],
      manifest["source"] as? String == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw BlockError.invalid("This probe requires the released distilled transformer manifest source.")
    }
    try autoreleasepool {
      _ = try DenoiserWeights(root: transformer, configuration: configuration,adapters: adapters)
      _ = try VideoDecoder(checkpoint: videoCheckpoint)
      _ = try AudioDecoder(checkpoint: audioCheckpoint)
      let encoder = try LTX25TextEncoder(gemmaRoot: gemma, connectorURL: connector)
      _ = try TextEncodingPlan(promptTokens: encoder.tokenize(request.prompt).count)
      if let recipe,let path = request.spatialUpscalerCheckpoint {
        _ = try DistilledSamplingRunner(recipe: recipe,transformerRoot: transformer,
          upscalerCheckpoint: URL(fileURLWithPath: path),statisticsCheckpoint: videoCheckpoint,loras: request.loras ?? [])
      }
    }
    try events.emit("preflight_complete", ["video_latent_shape": geometry.videoShape,
      "video_output_shape": videoPlan.outputShape, "audio_latent_frames": geometry.audioFrames,
      "audio_samples": expectedSamples, "video_activation_admission_bytes": videoPlan.admittedActivationBytes,
      "audio_admission_bytes": audioBudget, "model_weight_headers_validated": true, "embedded_tokenizer_validated": true,
      "checkpoint_payload_hashes_verified": false, "preflight_only": preflightOnly,
      "active_lora_pairs": adapters?.activePairCount ?? 0,
      "lora_workspace_admission_bytes": adapters?.admittedWorkspaceBytes ?? 0,
      "recipe": recipe == nil ? "legacy-stage-one" : DistilledTwoStageRecipe.identifier,
      "model_evaluations": recipe == nil ? 8 : 11])
    if preflightOnly { return }

    // mkdir is atomic and fails on an existing entry, including a dangling
    // symlink. Never reuse or overwrite an earlier run's directory.
    guard mkdir(output.path, mode_t(0o700)) == 0 else {
      throw BlockError.invalid("Cannot create new output directory: \(String(cString: strerror(errno)))")
    }
    do {
      try events.record(to: output.appendingPathComponent("progress.jsonl"))
      try JSONEncoder().encode(request).write(to: output.appendingPathComponent("request.json"), options: .withoutOverwriting)
      let frameDirectory = output.appendingPathComponent("frames", isDirectory: true)
      guard mkdir(frameDirectory.path, mode_t(0o700)) == 0 else { throw BlockError.invalid("Cannot create frame directory.") }
      var tokenIDs: [Int] = []
      var stageSeconds: [String: Double] = [:]
      let latents: AVLatents = try autoreleasepool {
        let textStarted = Date()
        try events.emit("text_start")
        let conditioning = try autoreleasepool {
          let encoder = try LTX25TextEncoder(gemmaRoot: gemma, connectorURL: connector)
          return try encoder.encode(prompt: request.prompt) { event in
            try events.emit("text_progress", ["component": event.stage, "completed": event.completed, "total": event.total])
          }
        }
        tokenIDs = conditioning.tokenIDs
        // Developer-only numerical evidence: retain the two conditioning arrays
        // to compare the integrated prompt against independent reference outputs.
        for (name, values) in [("video", conditioning.video), ("audio", conditioning.audio)] {
          try values.withUnsafeBytes { bytes in
            try Data(bytes).write(to: output.appendingPathComponent("conditioning-\(name).f32"), options: .withoutOverwriting)
          }
        }
        stageSeconds["text"] = Date().timeIntervalSince(textStarted)
        try events.emit("text_weights_released", ["prompt_token_count": conditioning.tokenIDs.count,
          "context_token_count": conditioning.tokenCount])
        let samplingStarted = Date()
        try events.emit("sampling_start")
        let result: AVLatents = try autoreleasepool {
          if let recipe,let path = request.spatialUpscalerCheckpoint {
            let runner = try DistilledSamplingRunner(recipe: recipe,transformerRoot: transformer,
              upscalerCheckpoint: URL(fileURLWithPath: path),statisticsCheckpoint: videoCheckpoint,loras: request.loras ?? [])
            let result = try runner.evaluate(videoContext: conditioning.video,audioContext: conditioning.audio) { component,completed,total in
              try events.emit("two_stage_progress",["component":component,"completed":completed,"total":total])
            }
            for (stage,seconds) in runner.stageSeconds { stageSeconds[stage] = seconds }
            return result
          }
          let weights = try DenoiserWeights(root: transformer, configuration: configuration,adapters: adapters)
          let runner = try LTXSamplingRunner(configuration: configuration)
          var random = GaussianNoise(seed: request.seed)
          let videoNoise = try random.values(count: geometry.videoTokens * 128)
          let audioNoise = try random.values(count: geometry.audioFrames * 128)
          let inputs: [String: [Float]] = ["video_latent": videoNoise, "audio_latent": audioNoise,
            "video_text": conditioning.video, "audio_text": conditioning.audio,
            "video_positions": geometry.videoPositions, "audio_positions": geometry.audioPositions]
          return try runner.evaluate(inputs, schedule: schedule,
            fixedWeights: { try weights.readFixed($0, shape: $1) },
            blockWeights: { try weights.readBlock($0, name: $1, shape: $2) },
            noise: { _, _, count in try random.values(count: count) },
            stageProgress: { step, event in
              try events.emit("sampling_component", ["step": step, "component": event.stage,
                "completed_blocks": event.completedBlocks, "reported_metal_bytes": event.metalAllocatedBytes])
            }, progress: { event in
              try events.emit("sampling_step", ["completed": event.completedSteps, "total": event.totalSteps,
                "sigma": event.sigma, "next_sigma": event.nextSigma])
            })
        }
        stageSeconds["sampling"] = Date().timeIntervalSince(samplingStarted)
        return result
      }
      try events.emit("sampling_weights_released")
      var writtenFrames = 0
      let videoStarted = Date()
      try events.emit("video_decode_start")
      try autoreleasepool {
        let decoder = try VideoDecoder(checkpoint: videoCheckpoint)
        try decoder.decode(latent: geometry.unpackVideo(latents.video), shape: geometry.videoShape,
          configuration: videoConfiguration) { chunk in
          guard chunk.startFrame == writtenFrames, chunk.frameCount == 1,
            chunk.width == request.width, chunk.height == request.height, chunk.frameRate == request.fps else {
            throw BlockError.invalid("Decoded video chunk does not match requested timing or geometry.")
          }
          let frameURL = frameDirectory.appendingPathComponent(String(format: "%06d.png", writtenFrames))
          try autoreleasepool { try writePNG(chunk.rgb, width: chunk.width, height: chunk.height, to: frameURL) }
          writtenFrames += 1
          try events.emit("video_frame", ["completed": writtenFrames, "total": request.frames])
        }
      }
      guard writtenFrames == request.frames else { throw BlockError.invalid("Decoded video frame count is incomplete.") }
      stageSeconds["video_decode"] = Date().timeIntervalSince(videoStarted)
      try events.emit("video_weights_released")
      let audioStarted = Date()
      try events.emit("audio_decode_start")
      let audioFrames = try autoreleasepool {
        let decoder = try AudioDecoder(checkpoint: audioCheckpoint)
        let waveform = try decoder.decode(latent: geometry.unpackAudio(latents.audio), latentFrames: geometry.audioFrames) {
          try events.emit("audio_component", ["component": $0])
        }
        guard waveform.sampleRate == 48000, waveform.channels == 2, waveform.frameCount == expectedSamples else {
          throw BlockError.invalid("Decoded audio sample count/rate/channels differ from preflight.")
        }
        try writeWAV(waveform, to: output.appendingPathComponent("audio.wav"))
        return waveform.frameCount
      }
      stageSeconds["audio_decode"] = Date().timeIntervalSince(audioStarted)
      try events.emit("audio_weights_released")
      let videoDuration = Double(writtenFrames) / request.fps
      let audioDuration = Double(audioFrames) / 48000
      var report: [String: Any] = ["scope": scope, "status": "complete", "prompt": request.prompt,
        "seed": request.seed, "noise_algorithm": GaussianNoise.algorithm,
        "sigmas": sigmas, "eta": 1, "noise_strength": 1, "model_evaluations": recipe == nil ? 8 : 11,
        "recipe": recipe == nil ? "legacy-stage-one" : DistilledTwoStageRecipe.identifier,
        "stage2_sigmas": recipe?.second.sigmas ?? [], "stage2_eta": 0,
        "text_token_ids": tokenIDs, "text_context_tokens": 1024, "maximum_prompt_tokens": 1024,
        "conditioning_arrays": "conditioning-video.f32 [1024,4096], conditioning-audio.f32 [1024,2048]; little-endian Float32",
        "width": request.width, "height": request.height, "fps": request.fps,
        "video_frames": writtenFrames, "video_duration_seconds": videoDuration,
        "audio_samples_per_channel": audioFrames, "audio_sample_rate": 48000, "audio_channels": 2,
        "audio_duration_seconds": audioDuration, "audio_minus_video_seconds": audioDuration - videoDuration,
        "audio_timing_policy": "actual causal decoder samples retained; no padding, stretching or silent trimming",
        "frame_pattern": "frames/%06d.png", "audio_file": "audio.wav", "wav_sample_format": "IEEE_FLOAT32",
        "stage_seconds": stageSeconds, "elapsed_seconds": Date().timeIntervalSince(events.started),
        "observed_peak_metal_bytes": events.observedPeakMetalBytes,
        "memory_metric": "sampled Metal currentAllocatedSize; not a complete OS peak-memory measurement",
        "checkpoint_payload_hashes_verified": false,
        "active_lora_pairs": adapters?.activePairCount ?? 0,
        "lora_workspace_admission_bytes": adapters?.admittedWorkspaceBytes ?? 0,
        "loras": (request.loras ?? []).map { ["path":$0.path,"strength":$0.strength,"enabled":$0.enabled] as [String:Any] },
        "lora_application": "ordered Float32 B@A at base matrix load, both stages; standard transformer adapters only; task/style quality not qualified"]
      report.merge(recipeReport(seed: request.seed,twoStage: recipe != nil)) { _,new in new }
      try writeJSON(report, to: output.appendingPathComponent("report.json"))
      try events.emit("complete", ["report": output.appendingPathComponent("report.json").path,
        "video_frames": writtenFrames, "audio_samples_per_channel": audioFrames])
      try events.close()
    } catch {
      try? events.emit("failed", ["error": String(describing: error)])
      try? writeJSON(["scope": scope, "status": "failed", "error": String(describing: error),
        "partial_outputs": true, "elapsed_seconds": Date().timeIntervalSince(events.started)],
        to: output.appendingPathComponent("failure.json"))
      try? events.close()
      throw error
    }
  }

  static func writeJSON(_ object: [String: Any], to url: URL) throws {
    try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
      .write(to: url, options: .withoutOverwriting)
  }

  static func newFile(_ url: URL) throws -> FileHandle {
    let fd = Darwin.open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode_t(0o600))
    guard fd >= 0 else { throw BlockError.invalid("Cannot create new file \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
    return FileHandle(fileDescriptor: fd, closeOnDealloc: true)
  }

  static func writePNG(_ rgb:[Float],width:Int,height:Int,to url:URL) throws {
    try MediaOutput.writePNG(rgb,width:width,height:height,to:url)
  }
  static func writeWAV(_ wave:AudioWaveform,to url:URL) throws {
    try MediaOutput.writeWAV(samples:wave.samples,sampleRate:wave.sampleRate,channels:wave.channels,to:url)
  }
}
