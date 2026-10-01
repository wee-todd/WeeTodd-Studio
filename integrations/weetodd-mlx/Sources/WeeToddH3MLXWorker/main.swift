import CoreGraphics
import Darwin
import Foundation
import H3MLX
import ImageIO
import InferenceContracts
import InferenceMedia
import MLX
import UniformTypeIdentifiers

@main struct H3MLXWorker {
  private static func invalid(_ message: String) -> NSError {
    NSError(domain: "WeeToddH3MLXWorker", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message])
  }

  private static func emit(_ object: [String: Any]) throws {
    let line = try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
    try FileHandle.standardOutput.write(contentsOf: line + Data([10]))
  }

  private static func readRequest(_ path: String) throws -> Data {
    guard path.hasPrefix("/"), !path.utf8.contains(0) else {
      throw invalid("An absolute request path is required.")
    }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw invalid("Cannot open the native video request.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size <= 1024 * 1024 else {
      throw invalid("The native video request must be a regular file under 1 MiB.")
    }
    let data = try handle.read(upToCount: 1024 * 1024 + 1) ?? Data()
    guard data.count <= 1024 * 1024 else { throw invalid("The request exceeds 1 MiB.") }
    return data
  }

  private static func preview(_ rgb: Data, width: Int, height: Int,
    output: URL) throws {
    guard rgb.count == width * height * 3,
      let provider = CGDataProvider(data: rgb as CFData),
      let source = CGImage(width: width, height: height,
        bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: 0),
        provider: provider, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent) else {
      throw invalid("Cannot prepare the H3 decoded preview.")
    }
    let scale = min(1, 640 / Double(max(width, height)))
    let previewWidth = max(1, Int(Double(width) * scale))
    let previewHeight = max(1, Int(Double(height) * scale))
    guard let context = CGContext(data: nil, width: previewWidth,
      height: previewHeight, bitsPerComponent: 8,
      bytesPerRow: previewWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
      throw invalid("Cannot allocate the bounded H3 preview.")
    }
    context.interpolationQuality = .medium
    context.draw(source, in: CGRect(x: 0, y: 0,
      width: previewWidth, height: previewHeight))
    guard let image = context.makeImage() else { throw invalid("Cannot resize H3 preview.") }
    let bytes = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(bytes,
      UTType.png.identifier as CFString, 1, nil) else {
      throw invalid("Cannot encode H3 preview.")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw invalid("Cannot finish H3 preview.")
    }
    try (bytes as Data).write(to: output, options: .atomic)
  }

  private static func mux(ffmpeg: URL, directory: URL) throws {
    let log = directory.appendingPathComponent("mux.log")
    guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
      throw invalid("Cannot create the H3 mux log.")
    }
    let handle = try FileHandle(forWritingTo: log)
    defer { try? handle.close() }
    let process = Process()
    process.executableURL = ffmpeg
    process.arguments = ["-v", "error", "-nostdin", "-n",
      "-i", directory.appendingPathComponent("video.mp4").path,
      "-i", directory.appendingPathComponent("audio.wav").path,
      "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
      "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
      directory.appendingPathComponent("render.mp4").path]
    process.standardOutput = handle
    process.standardError = handle
    try Task.checkCancellation()
    try process.run()
    defer {
      if process.isRunning {
        process.terminate()
        usleep(100_000)
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    while process.isRunning {
      try Task.checkCancellation()
      usleep(10_000)
    }
    guard process.terminationStatus == 0,
      (try directory.appendingPathComponent("render.mp4")
        .resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) > 0 else {
      throw invalid("FFmpeg failed to mux the H3 audio and video.")
    }
  }

  private static func execute() throws {
    let started = Date()
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 5,
      ["preflight", "render"].contains(arguments[0]),
      arguments[1] == "--request", arguments[3] == "--output" else {
      throw invalid("Usage: WeeToddH3MLXWorker preflight|render --request ENVELOPE --output DIRECTORY")
    }
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    if executable.deletingLastPathComponent().lastPathComponent == "MacOS" {
      let library = executable.deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Resources/LTXNative/mlx.metallib")
      guard FileManager.default.fileExists(atPath: library.path) else {
        throw invalid("The packaged MLX Metal library is missing.")
      }
      GPU.metallib = library
    }
    try Task.checkCancellation()
    let envelope = try NativeVideoJobEnvelope.decode(
      readRequest(arguments[2]), expectedEngine: .h3,
      expectedOutputDirectory: arguments[4])
    let recipeData = try readRequest(envelope.recipePath)
    try envelope.validateRecipe(recipeData)
    var recipe = try JSONSerialization.jsonObject(with: recipeData) as! [String: Any]
    let selectedTask = (recipe["components"] as? [String: Any])?["task"] as? String ?? "t2va"
    let conditioningTask = (recipe["conditioning"] as? [String: Any])?["task"] as? String ?? "t2v"
    let continuation: H3Continuation.Plan?
    if let fields = recipe.removeValue(forKey: "continuation") {
      let object = fields as? [String: Any]
      let saveNumber = object?["save_context"] as? NSNumber
      guard selectedTask == "t2va", conditioningTask == "t2v",
        let object,
        Set(object.keys).isSubset(of: ["version", "context_frames",
          "source_context", "source_manifest_sha256", "save_context"]),
        object["version"] as? Int == 2,
        let context = object["context_frames"] as? Int,
        (object["source_context"] == nil || object["source_context"] is String),
        (object["source_context"] as? String).map({
          $0.hasPrefix("/") && !$0.utf8.contains(0) && !$0.contains("://")
        }) ?? true,
        (object["source_manifest_sha256"] == nil ||
          object["source_manifest_sha256"] is String),
        (object["save_context"] == nil ||
          (saveNumber != nil && CFGetTypeID(saveNumber!) == CFBooleanGetTypeID())),
        let config = recipe["config"] as? [String: Any],
        let duration = config["duration_seconds"] as? Double else {
        throw invalid("Swift H3 continuation v2 requires a text-to-AV recipe and supported fields.")
      }
      let source = (object["source_context"] as? String).map { URL(fileURLWithPath: $0) }
      continuation = try H3Continuation.Plan(contextFrames: context,
        requestedDuration: duration, sourceManifest: source,
        sourceSHA256: object["source_manifest_sha256"] as? String,
        saveContext: object["save_context"] as? Bool ?? false)
      if continuation!.sourceManifest != nil {
        var sampleConfig = config
        sampleConfig["duration_seconds"] = min(Double(continuation!.generatedFrames) / 24, 15)
        recipe["config"] = sampleConfig
      }
    } else { continuation = nil }
    let reportedTask = continuation != nil ? "continuation" :
      conditioningTask == "extension" ? "extension" :
      conditioningTask == "a2v" ? "a2v" : selectedTask
    let configuredFFmpeg = envelope.ffmpegPath ?? recipe["ffmpeg"] as? String ?? ""
    guard configuredFFmpeg.hasPrefix("/"),
      FileManager.default.isExecutableFile(atPath: configuredFFmpeg) else {
      throw invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    var sourceImages: [[String: Any]] = []
    let stillRequest: H3Ref2VAStillRequest?
    let endpointRequest: H3FL2VARequest?
    let textRequest: H3T2VARequest?
    if selectedTask == "ref2va" && conditioningTask == "extension" {
      stillRequest = try H3StudioRecipe.compileExtension(data: recipeData) {
        path, expectedSHA256 in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        let loaded = try H3VideoReferenceMedia.load(path: path,
          ffmpeg: URL(fileURLWithPath: configuredFFmpeg),
          retainCompleteAudio: true, preserveTail: true)
        try source.verify()
        guard loaded.decodedFrames < 361 else {
          throw invalid("Swift H3 external extension needs a source of at most 15 seconds.")
        }
        sourceImages.append(["path": path, "sha256": expectedSHA256,
          "kind": "extension_source", "decodedFrames": loaded.decodedFrames,
          "conditionFrames": loaded.reference.frameCount,
          "soundtrackSamples": loaded.reference.audio?.frames ?? 0])
        return (loaded.reference, loaded.lastFrame)
      }
      endpointRequest = nil
      textRequest = nil
    } else if selectedTask == "ref2va" && conditioningTask == "a2v" {
      stillRequest = try H3StudioRecipe.compileA2V(data: recipeData) {
        path, expectedSHA256, start, duration in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        if duration == 0 {
          let loaded = try H3StillReferenceMedia.load(path: path)
          guard loaded.sourceSHA256 == expectedSHA256 else {
            throw invalid("An H3 A2V opening image changed after preparation.")
          }
          sourceImages.append(["path": path, "sha256": expectedSHA256,
            "kind": "image", "anchor": 0])
          return .image(loaded.reference)
        }
        let audio = try H3AudioReferenceMedia.load(path: path,
          ffmpeg: URL(fileURLWithPath: configuredFFmpeg),
          startSeconds: start, durationSeconds: duration)
        try source.verify()
        sourceImages.append(["path": path, "sha256": expectedSHA256,
          "kind": "audio_driver", "sourceStartSeconds": start,
          "sourceDurationSeconds": duration, "preparedSamples": audio.frames,
          "sampleRate": 32_000])
        return .audio(audio)
      }
      let inputs = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      guard inputs.map({ $0["sha256"] as? String }) ==
        sourceImages.map({ $0["sha256"] as? String }) else {
        throw invalid("An H3 A2V source changed after preparation.")
      }
      endpointRequest = nil
      textRequest = nil
    } else if selectedTask == "ref2va" {
      stillRequest = try H3StudioRecipe.compileMediaReferences(data: recipeData) {
        path, kind, expectedSHA256 in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        if kind == "image" {
          let image = try H3StillReferenceMedia.load(path: path)
          guard image.sourceSHA256 == expectedSHA256 else {
            throw invalid("An H3 reference image changed after recipe preparation.")
          }
          sourceImages.append(["path": path, "sha256": image.sourceSHA256,
            "kind": "image", "width": image.sourceWidth,
            "height": image.sourceHeight])
          return .image(image.reference)
        }
        if kind == "audio" {
          let audio = try H3AudioReferenceMedia.load(path: path,
            ffmpeg: URL(fileURLWithPath: configuredFFmpeg))
          try source.verify()
          sourceImages.append(["path": path, "sha256": expectedSHA256,
            "kind": "audio", "preparedSamples": audio.frames,
            "sampleRate": 32_000])
          return .audio(audio)
        }
        let video = try H3VideoReferenceMedia.load(path: path,
          ffmpeg: URL(fileURLWithPath: configuredFFmpeg))
        try source.verify()
        sourceImages.append(["path": path, "sha256": expectedSHA256,
          "kind": "video", "preparedFrames": video.reference.frameCount,
          "decodedFrames": video.decodedFrames,
          "soundtrackSamples": video.reference.audio?.frames ?? 0,
          "width": video.reference.width, "height": video.reference.height])
        return .video(video.reference)
      }
      let inputItems = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      guard inputItems.map({ $0["sha256"] as? String }) ==
        sourceImages.map({ $0["sha256"] as? String }) else {
        throw invalid("An H3 reference image changed after recipe preparation.")
      }
      textRequest = nil
      endpointRequest = nil
    } else if selectedTask == "fl2va" {
      let inputItems = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      endpointRequest = try H3StudioRecipe.compileFL2VA(data: recipeData) {
        path, first, width, height in
        let loaded = try H3FL2VAMedia.load(path: path,
          width: width, height: height, first: first)
        sourceImages.append(["path": path, "sha256": loaded.sourceSHA256,
          "width": loaded.sourceWidth, "height": loaded.sourceHeight,
          "anchor": inputItems[sourceImages.count]["frame_index"] ?? "unknown"])
        return loaded.image
      }
      guard inputItems.map({ $0["sha256"] as? String }) ==
        sourceImages.map({ $0["sha256"] as? String }) else {
        throw invalid("An H3 endpoint image changed after recipe preparation.")
      }
      stillRequest = nil
      textRequest = nil
    } else {
      textRequest = try H3StudioRecipe.compile(
        data: JSONSerialization.data(withJSONObject: recipe))
      stillRequest = nil
      endpointRequest = nil
    }
    let admission: (geometry: H3Geometry, packedRows: Int, evaluations: Int)
    let continuationIdentity: String?
    var continuationRows: H3Continuation.Rows?
    if let continuation, let textRequest {
      let identity = try H3Continuation.fingerprint(textRequest)
      continuationIdentity = identity
      if let source = continuation.sourceManifest,
        let sourceHash = continuation.sourceSHA256 {
        continuationRows = try H3Continuation.load(manifestURL: source,
          expectedSHA256: sourceHash, contextFrames: continuation.contextFrames,
          width: textRequest.geometry.width, height: textRequest.geometry.height,
          identity: identity, loadRows: arguments[0] == "render")
      }
    } else { continuationIdentity = nil }
    if let stillRequest {
      let checked = try H3Ref2VAStillRunner.preflight(stillRequest)
      admission = (checked.geometry, checked.packedRows, checked.evaluations)
    } else if let endpointRequest {
      let checked = try H3FL2VARunner.preflight(endpointRequest)
      admission = (checked.geometry, checked.packedRows, checked.evaluations)
    } else if let textRequest {
      if let continuation, continuation.sourceManifest != nil {
        let checked = try H3ContinuationRunner.preflight(textRequest,
          contextFrames: continuation.contextFrames)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      } else {
        let checked = try H3T2VARunner.preflight(textRequest)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      }
    } else {
      throw invalid("The H3 task was not admitted.")
    }
    let preflightSeconds = Date().timeIntervalSince(started)
    if arguments[0] == "preflight" {
      try emit(["status": "success", "result": [
        "nativeRuntime": "swift-mlx", "jobID": envelope.jobID.uuidString,
        "task": reportedTask, "frames": admission.geometry.frames, "fps": 24,
        "publishedFrames": continuation?.publishedFrames ?? admission.geometry.frames,
        "overlapFrames": continuation?.overlapFrames ?? 0,
        "packedRows": admission.packedRows,
        "evaluations": admission.evaluations,
        "referenceImages": sourceImages,
        "productionQualified": false]])
      return
    }

    let inferenceLease = try NativeInferenceLease.acquire {
      try? emit(["event": "progress", "stage": "waiting", "fraction": 0,
        "message": "Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }

    let output = URL(fileURLWithPath: arguments[4]).standardizedFileURL
    let parent = output.deletingLastPathComponent()
    guard FileManager.default.fileExists(atPath: parent.path),
      !FileManager.default.fileExists(atPath: output.path) else {
      throw invalid("The H3 output parent must exist and the destination must be new.")
    }
    let staging = parent.appendingPathComponent(".h3-staging-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging,
      withIntermediateDirectories: false)
    let previewURL = output.appendingPathExtension("preview.png")
    var published = false
    var writer: RawVideoWriter?
    defer {
      writer?.cancel()
      try? FileManager.default.removeItem(at: previewURL)
      if !published { try? FileManager.default.removeItem(at: staging) }
    }
    var lastPreview = Date.distantPast
    var revision = 0
    var lastFraction = 0.0
    var stageTimings: [String: [String: Any]] = [:]
    var phaseStarted = Date()
    func recordStage(_ name: String) {
      let now = Date()
      let usage = try? H3RenderResourceUsage.capture(peakMLXBytes: Memory.peakMemory)
      stageTimings[name] = ["seconds": now.timeIntervalSince(phaseStarted),
        "activeMLXBytes": Memory.activeMemory,
        "peakMLXBytes": Memory.peakMemory,
        "currentProcessFootprintBytes": usage?.currentPhysicalBytes ?? 0]
      phaseStarted = now
    }
    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory
    let publishedFrames = continuation?.publishedFrames ?? admission.geometry.frames
    let overlapFrames = continuation?.overlapFrames ?? 0
    let onFrame: (Int, Data) throws -> Void = { index, rgb in
      try Task.checkCancellation()
      guard (overlapFrames..<(overlapFrames + publishedFrames)).contains(index) else { return }
      let outputIndex = index - overlapFrames
      if writer == nil {
        writer = try RawVideoWriter(ffmpeg: URL(fileURLWithPath: configuredFFmpeg),
          output: staging.appendingPathComponent("video.mp4"),
          width: admission.geometry.width, height: admission.geometry.height,
          frames: publishedFrames, fps: 24)
      }
      try writer!.append(rgb, frame: outputIndex)
      if outputIndex == 0 || outputIndex == publishedFrames - 1 ||
        Date().timeIntervalSince(lastPreview) >= 1 {
        try preview(rgb, width: admission.geometry.width,
          height: admission.geometry.height, output: previewURL)
        lastPreview = Date()
        revision += 1
        try emit(["event": "progress", "stage": "video_decode",
          "completed": outputIndex + 1, "total": publishedFrames,
          "fraction": min(0.97, 0.84 + 0.13 * Double(outputIndex + 1) /
            Double(publishedFrames)),
          "message": "Decoding H3 video · frame \(outputIndex + 1)/\(publishedFrames)",
          "previewPath": previewURL.path, "previewRevision": revision])
      }
    }
    let onAudio: ([Float], Int) throws -> Void = { samples, sampleRate in
      if continuation == nil {
        try MediaOutput.writeWAV(samples: samples, sampleRate: sampleRate,
          channels: 2, to: staging.appendingPathComponent("audio.wav"))
        return
      }
      guard sampleRate == 32_000, samples.count.isMultiple(of: 2) else {
        throw invalid("H3 continuation needs 32 kHz stereo audio.")
      }
      let channelSamples = samples.count / 2
      let start = Int((Double(overlapFrames) / 24 * 32_000).rounded(.toNearestOrEven))
      let count = Int((Double(publishedFrames) / 24 * 32_000).rounded(.toNearestOrEven))
      guard start <= channelSamples, start + count - channelSamples <= 800 else {
        throw invalid("H3 continuation audio crop exceeds decoded samples.")
      }
      let available = min(count, channelSamples - start)
      let padding = [Float](repeating: 0, count: count - available)
      let published = Array(samples[start..<(start + available)]) + padding +
        Array(samples[(channelSamples + start)..<(channelSamples + start + available)]) + padding
      try MediaOutput.writeWAV(samples: published, sampleRate: sampleRate,
        channels: 2, to: staging.appendingPathComponent("audio.wav"))
    }
    let onProgress: (String, Int, Int) -> Void = { stage, completed, total in
      if H3WorkerStageBoundary.tracks(task: selectedTask) {
        if let boundary = H3WorkerStageBoundary.name(stage: stage,
          completed: completed, total: total) {
          recordStage(boundary)
        }
      }
      let part = total > 0 ? Double(completed) / Double(total) : 0
      var next = lastFraction
      if let sampling = H3WorkerProgress.fraction(stage: stage,
        completed: completed, total: total,
        evaluations: admission.evaluations) { next = sampling }
      else if stage == "text" { next = 0.02 + 0.04 * part }
      else if stage == "reference_video_weights_released" { next = 0.08 }
      else if stage == "video_decode" { next = 0.84 + 0.13 * part }
      else if stage == "audio_decode" { next = 0.97 + 0.02 * part }
      lastFraction = max(lastFraction, min(0.995, next))
      let message: String
      if stage.hasPrefix("sampling_block_") {
        let step = stage.dropFirst("sampling_block_".count)
        message = "Sampling H3 · step \(step)/\(admission.evaluations) · block \(completed)/\(total)"
      } else {
        message = stage.replacingOccurrences(of: "_", with: " ")
      }
      try? emit(["event": "progress", "stage": stage,
        "completed": completed, "total": total,
        "fraction": lastFraction,
        "message": message])
    }
    let result: (videoFrames: Int, audioSamplesPerChannel: Int, audioSampleRate: Int)
    var savedContextSHA256: String?
    let onLatents: ([Float], [Float]) throws -> Void = { video, audio in
      guard let continuation, continuation.saveContext,
        let identity = continuationIdentity else { return }
      let tail = try H3Continuation.tail(video: video, audio: audio,
        geometry: admission.geometry, contextFrames: continuation.contextFrames)
      let saved = try H3Continuation.save(tail, plan: continuation,
        width: admission.geometry.width, height: admission.geometry.height,
        identity: identity, directory: staging.appendingPathComponent("continuation"))
      savedContextSHA256 = saved.sha256
    }
    if let stillRequest {
      let rendered = try H3Ref2VAStillRunner.run(stillRequest,
        onFrame: onFrame, onAudio: onAudio, progress: onProgress)
      result = (rendered.videoFrames, rendered.audioSamplesPerChannel,
        rendered.audioSampleRate)
    } else if let endpointRequest {
      let rendered = try H3FL2VARunner.run(endpointRequest,
        onFrame: onFrame, onAudio: onAudio, progress: onProgress)
      result = (rendered.videoFrames, rendered.audioSamplesPerChannel,
        rendered.audioSampleRate)
    } else if let textRequest {
      if let continuation, let rows = continuationRows {
        let rendered = try H3ContinuationRunner.run(textRequest,
          contextFrames: continuation.contextFrames, context: rows,
          onFrame: onFrame, onAudio: onAudio,
          onLatents: onLatents, progress: onProgress)
        result = (rendered.videoFrames, rendered.audioSamplesPerChannel,
          rendered.audioSampleRate)
      } else {
        let rendered = try H3T2VARunner.run(textRequest,
          onFrame: onFrame, onAudio: onAudio,
          onLatents: onLatents, progress: onProgress)
        result = (rendered.videoFrames, rendered.audioSamplesPerChannel,
          rendered.audioSampleRate)
      }
    } else {
      throw invalid("The H3 task was not admitted.")
    }
    guard let writer else { throw invalid("H3 returned no decoded video frames.") }
    try writer.finish()
    try Task.checkCancellation()
    try mux(ffmpeg: URL(fileURLWithPath: configuredFFmpeg), directory: staging)
    if H3WorkerStageBoundary.tracks(task: selectedTask) { recordStage("mux") }
    let usage = try H3RenderResourceUsage.capture(peakMLXBytes: Memory.peakMemory)
    let metadata: [String: Any] = ["status": "complete",
      "nativeRuntime": "swift-mlx", "productionQualified": false,
      "jobID": envelope.jobID.uuidString,
      "task": reportedTask, "referenceImages": sourceImages,
      "stageTimings": stageTimings,
      "frames": publishedFrames, "fps": 24,
      "sampledFrames": result.videoFrames,
      "overlapFrames": overlapFrames,
      "audioSamplesPerChannel": continuation == nil
        ? result.audioSamplesPerChannel
        : Int((Double(publishedFrames) / 24 * 32_000)
          .rounded(.toNearestOrEven)),
      "sampledAudioSamplesPerChannel": result.audioSamplesPerChannel,
      "audioSampleRate": result.audioSampleRate,
      "peakMLXBytes": usage.peakMLXBytes,
      "peakProcessFootprintBytes": usage.peakPhysicalBytes,
      "currentProcessFootprintBytes": usage.currentPhysicalBytes,
      "processMemoryScope": "Swift H3 worker; external FFmpeg excluded",
      "preflightSeconds": preflightSeconds,
      "seconds": Date().timeIntervalSince(started)]
    var completeMetadata = metadata
    if let savedContextSHA256 {
      completeMetadata["continuationManifest"] = output
        .appendingPathComponent("continuation")
        .appendingPathComponent("manifest.json").path
      completeMetadata["continuationManifestSHA256"] = savedContextSHA256
    }
    try recipeData.write(to: staging.appendingPathComponent("studio-recipe.json"),
      options: .withoutOverwriting)
    try JSONSerialization.data(withJSONObject: completeMetadata,
      options: [.prettyPrinted, .sortedKeys]).write(
        to: staging.appendingPathComponent("result.json"),
        options: .withoutOverwriting)
    try Task.checkCancellation()
    try FileManager.default.moveItem(at: staging, to: output)
    published = true
    try emit(["status": "success", "result": H3WorkerReceipt.renderResult(
      video: output.appendingPathComponent("render.mp4"),
      metadata: completeMetadata, jobID: envelope.jobID)])
  }

  static func main() async {
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let job = Task.detached { try execute() }
    let signals = [SIGINT, SIGTERM].map { number in
      let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
      source.setEventHandler { @Sendable in job.cancel() }
      source.resume()
      return source
    }
    do { try await job.value }
    catch {
      try? emit(["status": error is CancellationError ? "cancelled" : "error",
        "error": String(describing: error)])
      for source in signals { source.cancel() }
      exit(error is CancellationError ? 130 : 1)
    }
    for source in signals { source.cancel() }
  }
}
