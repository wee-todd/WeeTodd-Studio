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

  private static func importContext(_ arguments: [String], session: H3PythonContinuationImporter.ImportSession? = nil) throws {
    guard arguments.count == 11, arguments[1] == "--request",
      arguments[3] == "--manifest", arguments[5] == "--manifest-sha",
      arguments[7] == "--identity", arguments[9] == "--output" else {
      throw invalid("Usage: WeeToddH3MLXWorker import-context --request ENVELOPE --manifest LEGACY_MANIFEST --manifest-sha SHA256 --identity EXPECTED_IDENTITY_JSON --output FRESH_DIRECTORY")
    }
    let envelope = try NativeVideoJobEnvelope.decode(readRequest(arguments[2]),
      expectedEngine: .h3, expectedOutputDirectory: arguments[10])
    let recipeData = try readRequest(envelope.recipePath)
    try envelope.validateRecipe(recipeData)
    guard var recipe = try JSONSerialization.jsonObject(with: recipeData) as? [String: Any] else {
      throw invalid("Explicit Python-context import requires a native H3 recipe.")
    }
    let task = (recipe["components"] as? [String: Any])?["task"] as? String
    let conditioning = recipe["conditioning"] as? [String: Any]
    guard (task == "t2va" && conditioning?["task"] as? String == "t2v") ||
      (task == "fl2va" && conditioning?["task"] as? String == "fflf") else {
      throw invalid("Explicit Python-context import supports T2VA and FL2VA identities only.")
    }
    recipe.removeValue(forKey: "continuation")
    let effectiveRecipe = try JSONSerialization.data(withJSONObject: recipe)
    let text: H3T2VARequest?
    let frames: H3FL2VARequest?
    if task == "fl2va" {
      let inputs = conditioning?["inputs"] as? [[String: Any]] ?? []
      var source = 0
      frames = try H3StudioRecipe.compileFL2VA(data: effectiveRecipe) { path, first, width, height in
        let loaded = try H3FL2VAMedia.load(path: path, width: width, height: height, first: first)
        guard source < inputs.count, inputs[source]["sha256"] as? String == loaded.sourceSHA256 else {
          throw invalid("An import recipe endpoint source changed.")
        }
        source += 1
        return loaded.image
      }
      text = nil
    } else {
      text = try H3StudioRecipe.compile(data: effectiveRecipe)
      frames = nil
    }
    let manifestData = try readRequest(arguments[4])
    guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
      let count = manifest["context_frames"] as? NSNumber,
      CFGetTypeID(count) != CFBooleanGetTypeID(),
      count.doubleValue.rounded() == count.doubleValue,
      H3Continuation.allowedContextFrames.contains(count.intValue) else {
      throw invalid("The legacy context frame count is invalid.")
    }
    let imported: H3PythonContinuationImporter.Publication
    if let frames {
      imported = try H3PythonContinuationImporter.importToNative(
        manifestURL: URL(fileURLWithPath: arguments[4]), expectedSHA256: arguments[6],
        expectedIdentityJSON: readRequest(arguments[8]), request: frames,
        contextFrames: count.intValue, output: URL(fileURLWithPath: arguments[10]), session: session)
    } else if let text {
      imported = try H3PythonContinuationImporter.importToNative(
        manifestURL: URL(fileURLWithPath: arguments[4]), expectedSHA256: arguments[6],
        expectedIdentityJSON: readRequest(arguments[8]), request: text,
        task: "t2va", contextFrames: count.intValue,
        output: URL(fileURLWithPath: arguments[10]), session: session)
    } else { throw invalid("The context import task was not admitted.") }
    try emit(["status": "success", "operation": "import-context", "inferenceExecuted": false,
      "manifest": imported.manifestURL.path, "manifestSHA256": imported.manifestSHA256,
      "payloadSHA256": imported.payloadSHA256, "originReceiptSHA256": imported.originReceiptSHA256,
      "crossRuntimeGenerationParityQualified": false])
  }

  private static func importContextBatch(_ arguments: [String]) throws {
    guard arguments.count == 3, arguments[1] == "--requests",
      let root = try JSONSerialization.jsonObject(with: readRequest(arguments[2])) as? [String: Any],
      Set(root.keys) == ["format", "jobs"],
      root["format"] as? String == "weetodd-h3-python-context-import-batch-v1",
      let jobs = root["jobs"] as? [[String]], (1...6).contains(jobs.count),
      jobs.allSatisfy({ $0.count == 11 && $0.first == "import-context" }),
      Set(jobs.map { URL(fileURLWithPath: $0[10]).standardizedFileURL.path }).count == jobs.count,
      jobs.allSatisfy({ !FileManager.default.fileExists(atPath: $0[10]) }) else {
      throw invalid("Explicit H3 batch import requires one to six fresh, distinct import-context jobs.")
    }
    let session = H3PythonContinuationImporter.ImportSession()
    for job in jobs {
      try Task.checkCancellation()
      try session.checkUnchanged()
      try importContext(job, session: session)
      try session.checkUnchanged()
    }
    try emit(["status": "success", "operation": "import-context-batch",
      "inferenceExecuted": false, "contextsImported": jobs.count,
      "componentFilesHashed": session.filesHashed, "componentBytesHashed": session.bytesHashed,
      "hashLeaseReuses": session.hashReuses, "componentBytesReuseValidated": session.bytesReuseValidated,
      "crossRuntimeGenerationParityQualified": false])
  }

  private static func execute() throws {
    let started = Date()
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.first == "import-context-batch" {
      try importContextBatch(arguments)
      return
    }
    if arguments.first == "import-context" {
      try importContext(arguments)
      return
    }
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
    let motionRecipe = try H3MotionFidelityRecipe.prepare(data: recipeData)
    let jointRefinement = try H3JointRefinementRecipe.prepare(data: motionRecipe?.ordinaryRecipe ?? recipeData)
    let canvasAdmission = jointRefinement?.targetGeometry.canvasAdmission ?? H3CanvasAdmission.ordinary
    let baseRecipeData = jointRefinement?.ordinaryRecipe ?? motionRecipe?.ordinaryRecipe ?? recipeData
    var recipe = try JSONSerialization.jsonObject(with: baseRecipeData) as! [String: Any]
    let selectedTask = (recipe["components"] as? [String: Any])?["task"] as? String ?? "t2va"
    let conditioningTask = (recipe["conditioning"] as? [String: Any])?["task"] as? String ?? "t2v"
    let refContinuation: H3Ref2VAContinuationRecipe.Prepared?
    let flContinuation: H3FL2VAContinuationRecipe.Prepared?
    let continuation: H3Continuation.Plan?
    if selectedTask == "ref2va", recipe["continuation"] != nil {
      let prepared = try H3Ref2VAContinuationRecipe.prepare(data: recipeData)
      refContinuation = prepared; flContinuation = nil; continuation = prepared.plan
      recipe = try JSONSerialization.jsonObject(with: prepared.sampledRecipe) as! [String: Any]
    } else if selectedTask == "fl2va", recipe["continuation"] != nil {
      refContinuation = nil
      let prepared = try H3FL2VAContinuationRecipe.prepare(data: recipeData)
      flContinuation = prepared
      continuation = prepared.plan
      recipe = try JSONSerialization.jsonObject(with: prepared.sampledRecipe) as! [String: Any]
    } else if let fields = recipe.removeValue(forKey: "continuation") {
      refContinuation = nil; flContinuation = nil
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
    } else { continuation = nil; flContinuation = nil; refContinuation = nil }
    let referenceRecipeData = refContinuation?.sampledRecipe ?? baseRecipeData
    let reportedTask = motionRecipe != nil ? "motion_fidelity" : refContinuation != nil ? "ref2va_continuation" : flContinuation != nil ? "fl2va_continuation" : continuation != nil ? "continuation" :
      conditioningTask == "extension" ? "extension" :
      conditioningTask == "a2v" ? "a2v" :
      conditioningTask == "control" ? "control" : selectedTask
    let configuredFFmpeg = envelope.ffmpegPath ?? recipe["ffmpeg"] as? String ?? ""
    guard configuredFFmpeg.hasPrefix("/"),
      FileManager.default.isExecutableFile(atPath: configuredFFmpeg) else {
      throw invalid("Select an executable FFmpeg in Runtime Settings.")
    }
    var sourceImages: [[String: Any]] = []
    let stillRequest: H3Ref2VAStillRequest?
    let endpointRequest: H3FL2VARequest?
    let textRequest: H3T2VARequest?
    if selectedTask == "t2va" && conditioningTask == "control" {
      textRequest = try H3StudioRecipe.compileControl(data: baseRecipeData) {
        path, expectedSHA256, geometry in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        let video = try H3FunControlMedia.load(path: path,
          ffmpeg: URL(fileURLWithPath: configuredFFmpeg), geometry: geometry)
        try source.verify()
        sourceImages.append(["path": path, "sha256": expectedSHA256,
          "kind": "control", "preparedFrames": video.frameCount,
          "width": video.width, "height": video.height])
        return video
      }
      stillRequest = nil
      endpointRequest = nil
    } else if selectedTask == "ref2va" && conditioningTask == "extension" {
      stillRequest = try H3StudioRecipe.compileExtension(data: baseRecipeData) {
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
      let inputItems = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      stillRequest = try H3StudioRecipe.compileA2V(data: referenceRecipeData,
        driverTargetFrame: refContinuation?.plan.overlapFrames ?? 0,
        visibleDurationSeconds: refContinuation?.plan.sourceManifest == nil ? nil
          : refContinuation.map { Double($0.plan.publishedFrames) / 24 },
        canvasAdmission: canvasAdmission) {
        path, expectedSHA256, start, duration, geometry, controls in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        if duration == 0 {
          let loaded = try H3StillReferenceMedia.load(path: path,
            outputGeometry: controls == nil ? nil : geometry,
            pixelBudgetPercent: controls?.imagePixelBudgetPercent)
          guard loaded.sourceSHA256 == expectedSHA256 else {
            throw invalid("An H3 A2V opening image changed after preparation.")
          }
          sourceImages.append(["path": path, "sha256": expectedSHA256,
            "kind": "image", "anchor": inputItems[sourceImages.count]["frame_index"] ?? "unknown",
            "preparedWidth": loaded.reference.width,
            "preparedHeight": loaded.reference.height,
            "preparedRGBSHA256": loaded.preparedSHA256,
            "preparationPolicy": loaded.preparationPolicy,
            "sourceOrientation": loaded.sourceOrientation,
            "imagePixelBudgetPercent": loaded.reference.pixelBudgetPercent as Any? ?? NSNull()])
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
      stillRequest = try H3StudioRecipe.compileMediaReferences(data: referenceRecipeData, canvasAdmission: canvasAdmission) {
        path, kind, expectedSHA256, geometry, controls in
        let source = try NativeMediaSource(path: path, sha256: expectedSHA256)
        try source.verify()
        if kind == "image" {
          let image = try H3StillReferenceMedia.load(path: path,
            outputGeometry: controls == nil ? nil : geometry,
            pixelBudgetPercent: controls?.imagePixelBudgetPercent)
          guard image.sourceSHA256 == expectedSHA256 else {
            throw invalid("An H3 reference image changed after recipe preparation.")
          }
          sourceImages.append(["path": path, "sha256": image.sourceSHA256,
            "kind": "image", "width": image.sourceWidth,
            "height": image.sourceHeight,
            "preparedWidth": image.reference.width,
            "preparedHeight": image.reference.height,
            "preparedRGBSHA256": image.preparedSHA256,
            "preparationPolicy": image.preparationPolicy,
            "sourceOrientation": image.sourceOrientation,
            "imagePixelBudgetPercent": image.reference.pixelBudgetPercent as Any? ?? NSNull()])
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
          ffmpeg: URL(fileURLWithPath: configuredFFmpeg),
          outputGeometry: controls == nil ? nil : geometry, controls: controls)
        try source.verify()
        sourceImages.append(["path": path, "sha256": expectedSHA256,
          "kind": "video", "preparedFrames": video.reference.frameCount,
          "decodedFrames": video.decodedFrames,
          "soundtrackSamples": video.reference.audio?.frames ?? 0,
          "width": video.reference.width, "height": video.reference.height,
          "persistentFrames": video.reference.persistentFrameCount,
          "sourceLatentFrames": video.reference.sourceLatentFrames,
          "sizePolicy": video.reference.controls?.videoSizePolicy?.rawValue as Any? ?? NSNull(),
          "temporalDensity": video.reference.temporalDecision?.policy.rawValue as Any? ?? NSNull(),
          "resolvedTemporalDensity": video.reference.temporalDecision?.density as Any? ?? NSNull(),
          "persistentSourceFrameIndices": video.reference.temporalDecision?.indices as Any? ?? NSNull()])
        return .video(video.reference)
      }
      let inputItems = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      let expectedDigests: [String?] = inputItems.flatMap { input in
        var digests: [String?] = [input["sha256"] as? String]
        if input["soundtrack_path"] != nil { digests.append(input["soundtrack_sha256"] as? String) }
        return digests
      }
      guard expectedDigests == sourceImages.map({ $0["sha256"] as? String }) else {
        throw invalid("An H3 reference image changed after recipe preparation.")
      }
      textRequest = nil
      endpointRequest = nil
    } else if selectedTask == "fl2va" {
      let inputItems = ((recipe["conditioning"] as? [String: Any])?["inputs"]
        as? [[String: Any]]) ?? []
      endpointRequest = try H3StudioRecipe.compileFL2VA(data: flContinuation?.sampledRecipe ?? baseRecipeData, canvasAdmission: canvasAdmission) {
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
        data: JSONSerialization.data(withJSONObject: recipe), canvasAdmission: canvasAdmission)
      stillRequest = nil
      endpointRequest = nil
    }
    let motionSource: H3MotionFidelityMedia.Source?
    let motionAdmission: H3T2VARunner.Admission?
    if let motionRecipe {
      guard let textRequest, endpointRequest == nil, stillRequest == nil else {
        throw invalid("Motion Fidelity requires a plain native H3 T2VA repair recipe.")
      }
      let source = try H3MotionFidelityMedia.inspect(path: motionRecipe.sourcePath,
        sha256: motionRecipe.sourceSHA256, ffprobe: motionRecipe.ffprobe,
        startSeconds: motionRecipe.sourceIn, durationSeconds: motionRecipe.durationSeconds,
        settings: motionRecipe.settings)
      motionSource = source
      motionAdmission = try H3MotionFidelityRunner.preflight(base: textRequest,
        source: source, settings: motionRecipe.settings)
      sourceImages.append(["path": source.identity.path, "sha256": source.identity.sha256,
        "kind": "motion_source", "sourceStartSeconds": source.startSeconds,
        "sourceDurationSeconds": Double(source.frames) / 24,
        "width": source.width, "height": source.height, "sourceFrames": source.frames,
        "sourceAudioPreserved": true])
    } else { motionSource = nil; motionAdmission = nil }
    var admission: (geometry: H3Geometry, packedRows: Int, evaluations: Int)
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
    } else if let continuation, let endpointRequest {
      let identity = try H3Continuation.fingerprint(endpointRequest)
      continuationIdentity = identity
      if let source = continuation.sourceManifest, let sourceHash = continuation.sourceSHA256 {
        continuationRows = try H3Continuation.load(manifestURL: source,
          expectedSHA256: sourceHash, contextFrames: continuation.contextFrames,
          width: endpointRequest.base.geometry.width, height: endpointRequest.base.geometry.height,
          identity: identity, loadRows: arguments[0] == "render", task: "fl2va")
      }
    } else if let continuation, let stillRequest {
      let identity = try H3Continuation.fingerprint(stillRequest)
      continuationIdentity = identity
      if let source = continuation.sourceManifest, let sourceHash = continuation.sourceSHA256 {
        continuationRows = try H3Continuation.load(manifestURL: source,
          expectedSHA256: sourceHash, contextFrames: continuation.contextFrames,
          width: stillRequest.geometry.width, height: stillRequest.geometry.height,
          identity: identity, loadRows: arguments[0] == "render", task: "ref2va")
      }
    } else { continuationIdentity = nil }
    let jointTask = stillRequest != nil ? "ref2va" : endpointRequest != nil ? "fl2va" : "t2va"
    let jointIdentity: String?
    var jointBaseRequest: H3T2VARequest?, jointVision: URL?
    var fullInitialRows: H3JointLatentArtifact.Rows?
    var jointSourceManifest: H3JointLatentArtifact.Manifest?
    if jointRefinement != nil {
      let base: H3T2VARequest
      let vision: URL?
      if let textRequest { base = textRequest; vision = nil }
      else if let endpointRequest { base = endpointRequest.base; vision = endpointRequest.vision }
      else if let stillRequest {
        base = try H3T2VARequest(prompt: stillRequest.prompt,
          width: stillRequest.geometry.width, height: stillRequest.geometry.height,
          durationSeconds: min(Double(stillRequest.geometry.frames) / 24, 15),
          seed: stillRequest.seed, requestedSteps: stillRequest.requestedSteps,
          transformer: stillRequest.transformer, qwenPages: stillRequest.qwenPages,
          tokenizer: stillRequest.tokenizer, videoVAE: stillRequest.videoVAE,
          audioVAE: stillRequest.audioVAE, canvasAdmission: stillRequest.geometry.canvasAdmission)
        vision = stillRequest.qwenVision
      } else { throw invalid("Missing H3 initialized task request.") }
      jointBaseRequest = base; jointVision = vision
      let identity = try H3JointLatentArtifact.componentIdentity(base: base, task: jointTask, vision: vision)
      jointIdentity = identity
      if let source = jointRefinement!.sourceManifest, let digest = jointRefinement!.sourceSHA256 {
        let loaded = try H3JointLatentArtifact.load(manifestURL: source, expectedSHA256: digest,
          expectedTask: jointTask, expectedComponentIdentity: identity)
        try H3JointRefinementRecipe.validateSource(loaded.0, prepared: jointRefinement!)
        jointSourceManifest = loaded.0
        fullInitialRows = loaded.1
      }
    } else { jointIdentity = nil }
    if let motionAdmission {
      admission = (motionAdmission.geometry, motionAdmission.packedRows, motionAdmission.evaluations)
    } else if let stillRequest {
      let checked = try H3Ref2VAStillRunner.preflight(stillRequest,
        contextFrames: refContinuation?.plan.sourceManifest == nil ? 0 : refContinuation!.plan.contextFrames, refinement: jointRefinement?.controls)
      admission = (checked.geometry, checked.packedRows, checked.evaluations)
    } else if let endpointRequest {
      if let continuation, continuation.sourceManifest != nil {
        let checked = try H3FL2VAContinuationRunner.preflight(endpointRequest,
          contextFrames: continuation.contextFrames)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      } else {
        let checked = try H3FL2VARunner.preflight(endpointRequest, refinement: jointRefinement?.controls)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      }
    } else if let textRequest {
      if let continuation, continuation.sourceManifest != nil {
        let checked = try H3ContinuationRunner.preflight(textRequest,
          contextFrames: continuation.contextFrames)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      } else {
        let checked = try H3T2VARunner.preflight(textRequest, refinement: jointRefinement?.controls)
        admission = (checked.geometry, checked.packedRows, checked.evaluations)
      }
    } else {
      throw invalid("The H3 task was not admitted.")
    }
    let videoDecodeMemoryMode = stillRequest?.videoDecodeMemoryMode
      ?? endpointRequest?.base.videoDecodeMemoryMode ?? textRequest?.videoDecodeMemoryMode
    let videoDecodeDiagnostics = H3VideoDecodeMemoryMode.diagnostics(for: videoDecodeMemoryMode)
    let preflightSeconds = Date().timeIntervalSince(started)
    if arguments[0] == "preflight" {
      try emit(["status": "success", "result": [
        "nativeRuntime": "swift-mlx", "jobID": envelope.jobID.uuidString,
        "task": reportedTask, "frames": admission.geometry.frames, "fps": 24,
        "publishedFrames": motionSource?.frames ?? continuation?.publishedFrames ?? admission.geometry.frames,
        "overlapFrames": continuation?.overlapFrames ?? 0,
        "packedRows": admission.packedRows,
        "evaluations": admission.evaluations,
        "referenceImages": sourceImages,
        "productionQualified": false, "videoDecode": videoDecodeDiagnostics]])
      return
    }

    let output = URL(fileURLWithPath: arguments[4]).standardizedFileURL
    let parent = output.deletingLastPathComponent()
    guard FileManager.default.fileExists(atPath: parent.path),
      !FileManager.default.fileExists(atPath: output.path) else {
      throw invalid("The H3 output parent must exist and the destination must be new.")
    }

    let inferenceLease = try NativeInferenceLease.acquire {
      try? emit(["event": "progress", "stage": "waiting", "fraction": 0,
        "message": "Waiting for another local inference job"])
    }
    defer { inferenceLease.release() }

    Memory.clearCache()
    Memory.peakMemory = Memory.activeMemory

    var learnedReport: [String:Any]?
    if let source = jointSourceManifest, let initial = fullInitialRows,
      let checkpoint = jointRefinement?.learnedUpscaler,
      let digest = jointRefinement?.learnedUpscalerHeaderSHA256, let base = jointBaseRequest {
      let upscaled = try H3LearnedLatentUpscaler.upscale(rows: MLXArray(initial.video,
        [1, try source.geometry.videoRows, 96]), source: source.geometry, target: admission.geometry,
        checkpointURL: checkpoint, expectedHeaderSHA256: digest, videoVAE: base.videoVAE) {
          completed,total in
          try? emit(["event":"progress", "stage":"learned_spatial_upscale",
            "fraction":H3WorkerProgress.fraction(stage:"learned_spatial_upscale",completed:completed,total:total,evaluations:admission.evaluations) ?? 0.01,
            "completed":completed,"total":total,"message":"Upscaling H3 latents · \(completed)/\(total)"])
        }
      fullInitialRows = H3JointLatentArtifact.Rows(video: upscaled.videoRows.asArray(Float.self), audio: initial.audio)
      try fullInitialRows!.validate(geometry: admission.geometry)
      learnedReport = ["headerSHA256":upscaled.headerSHA256,"residentWeightBytes":upscaled.residentWeightBytes,
        "maximumConvolutionWorkspaceBytes":upscaled.maximumConvolutionWorkspaceBytes,
        "targetTilesPerConvolution":upscaled.targetTilesPerConvolution,"weightsReleasedBeforeTransformer":true,
        "loadSeconds":upscaled.loadSeconds,"upscaleSeconds":upscaled.upscaleSeconds]
      Stream.gpu.synchronize(); Memory.clearCache()
      try emit(["event":"progress", "stage":"learned_spatial_weights_released", "completed":1,"total":1,"fraction":0.02])
    }
    if let source = jointSourceManifest, let initial = fullInitialRows,
      let method = jointRefinement?.resizeMethod {
      let resized = try H3SpatialLatentResize.rows(MLXArray(initial.video,
        [1, try source.geometry.videoRows, 96]), source: source.geometry,
        target: admission.geometry, method: method)
      fullInitialRows = H3JointLatentArtifact.Rows(video: resized.asArray(Float.self), audio: initial.audio)
      try fullInitialRows!.validate(geometry: admission.geometry)
      Stream.gpu.synchronize(); Memory.clearCache()
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
    let publishedFrames = motionSource?.frames ?? continuation?.publishedFrames ?? admission.geometry.frames
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
      if H3WorkerStageBoundary.tracks(task: reportedTask) {
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
      else if stage == "control_video_encode" { next = 0.01 + 0.01 * part }
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
    var savedContextSHA256: String?,savedPayloadSHA256:String?
    var savedJointSHA256: String?, savedJointPayloadSHA256: String?
    let onLatents: ([Float], [Float]) throws -> Void = { video, audio in
      if let identity = jointIdentity, let base = jointBaseRequest {
        guard try H3JointLatentArtifact.componentIdentity(base: base, task: jointTask, vision: jointVision) == identity else {
          throw invalid("H3 refinement components changed during execution.")
        }
        if let source = jointRefinement?.sourceManifest, let digest = jointRefinement?.sourceSHA256 {
          try H3JointLatentArtifact.verify(manifestURL: source, expectedSHA256: digest,
            expectedTask: jointTask, expectedComponentIdentity: identity)
        }
      }
      if jointRefinement?.saveFullLatents == true, let identity = jointIdentity {
        let saved = try H3JointLatentArtifact.save(rows: .init(video: video, audio: audio),
          geometry: admission.geometry, task: jointTask, componentIdentity: identity,
          directory: staging.appendingPathComponent("joint-latents"))
        savedJointSHA256 = saved.manifestSHA256; savedJointPayloadSHA256 = saved.payloadSHA256
      }
      guard let continuation, continuation.saveContext,
        let identity = continuationIdentity else { return }
      let tail = try H3Continuation.tail(video: video, audio: audio,
        geometry: admission.geometry, contextFrames: continuation.contextFrames)
      let saved = try H3Continuation.save(tail, plan: continuation,
        width: admission.geometry.width, height: admission.geometry.height,
        identity: identity, directory: staging.appendingPathComponent("continuation"),
        task: stillRequest != nil ? "ref2va" : endpointRequest == nil ? "t2va" : "fl2va")
      savedContextSHA256 = saved.sha256;savedPayloadSHA256=saved.payloadSHA256
    }
    var actualMotionPlan: H3MotionFidelityPlan?
    if let motionRecipe, let source = motionSource, let textRequest {
      let prepared = try H3MotionFidelityRunner.prepare(base: textRequest, source: source,
        settings: motionRecipe.settings, ffmpeg: URL(fileURLWithPath: configuredFFmpeg),
        scratch: staging.appendingPathComponent("motion-audio-scratch"), progress: onProgress)
      actualMotionPlan = prepared.plan
      let checked = try H3T2VARunner.preflight(prepared.request,
        refinement: H3JointRefinement(strength: motionRecipe.settings.strength,
          startVideoSigma: motionRecipe.settings.strength, evaluations: motionRecipe.settings.evaluations))
      admission = (checked.geometry, checked.packedRows, prepared.plan.noop ? 0 : checked.evaluations)
      try JSONEncoder().encode(prepared.plan).write(to: staging.appendingPathComponent("motion-plan.json"), options: .atomic)
      let rendered = try H3MotionFidelityRunner.run(prepared, source: source,
        onFrame: onFrame, onAudio: onAudio, onLatents: onLatents, progress: onProgress)
      result = (rendered.videoFrames, rendered.audioSamplesPerChannel, rendered.audioSampleRate)
    } else if let stillRequest {
      let rendered = try H3Ref2VAStillRunner.run(stillRequest,
        contextFrames: continuationRows == nil ? 0 : refContinuation!.plan.contextFrames,
        context: continuationRows, initialRows: fullInitialRows, refinement: jointRefinement?.controls,
        onFrame: onFrame, onAudio: onAudio,
        onLatents: onLatents, progress: onProgress)
      result = (rendered.videoFrames, rendered.audioSamplesPerChannel,
        rendered.audioSampleRate)
    } else if let endpointRequest {
      if let continuation, let rows = continuationRows {
        let rendered = try H3FL2VAContinuationRunner.run(endpointRequest,
          contextFrames: continuation.contextFrames, context: rows,
          onFrame: onFrame, onAudio: onAudio, onLatents: onLatents, progress: onProgress)
        result = (rendered.videoFrames, rendered.audioSamplesPerChannel, rendered.audioSampleRate)
      } else {
        let rendered = try H3FL2VARunner.run(endpointRequest,
          initialRows: fullInitialRows, refinement: jointRefinement?.controls,
          onFrame: onFrame, onAudio: onAudio, onLatents: onLatents, progress: onProgress)
        result = (rendered.videoFrames, rendered.audioSamplesPerChannel, rendered.audioSampleRate)
      }
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
          initialRows: fullInitialRows, refinement: jointRefinement?.controls,
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
    if H3WorkerStageBoundary.tracks(task: reportedTask) { recordStage("mux") }
    let usage = try H3RenderResourceUsage.capture(peakMLXBytes: Memory.peakMemory)
    let metadata: [String: Any] = ["status": "complete",
      "nativeRuntime": "swift-mlx", "productionQualified": false,
      "jobID": envelope.jobID.uuidString,
      "task": reportedTask, "referenceImages": sourceImages,
      "stageTimings": stageTimings, "videoDecode": videoDecodeDiagnostics,
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
    if let plan = actualMotionPlan, let source = motionSource {
      completeMetadata["motionFidelity"] = ["version": 1,
        "sourcePath": source.identity.path, "sourceSHA256": source.identity.sha256,
        "sourceIn": source.startSeconds, "sourceDuration": Double(plan.sourceFrames) / 24,
        "sourceFrames": plan.sourceFrames, "expandedFrames": plan.expandedFrames,
        "paddedFrames": plan.paddedFrames, "recovery": plan.recovery,
        "noop": plan.noop, "adaptiveAnalysisPerformed": plan.adaptiveAnalysisPerformed,
        "sourceMediaInspected": true, "actualSamplingEvaluations": admission.evaluations,
        "sourceAudioPreserved": true,
        "planPath": output.appendingPathComponent("motion-plan.json").path]
    }
    if let savedJointSHA256, let savedJointPayloadSHA256 {
      completeMetadata["jointLatentManifest"] = output.appendingPathComponent("joint-latents/manifest.json").path
      completeMetadata["jointLatentManifestSHA256"] = savedJointSHA256
      completeMetadata["jointLatentPayloadSHA256"] = savedJointPayloadSHA256
    }
    if let prepared = jointRefinement, let controls = prepared.controls {
      completeMetadata["refinement"] = ["version": prepared.version, "mode": prepared.mode!,
        "strength": controls.strength, "preserveAudio": controls.preserveAudio,
        "sourceManifest": prepared.sourceManifest!.path, "sourceManifestSHA256": prepared.sourceSHA256!]
    }

    if let learnedReport { completeMetadata["learnedSpatialUpscaler"] = learnedReport }

    if let savedContextSHA256,let savedPayloadSHA256 {
      completeMetadata["continuationManifest"] = output
        .appendingPathComponent("continuation")
        .appendingPathComponent("manifest.json").path
      completeMetadata["continuationManifestSHA256"] = savedContextSHA256
      completeMetadata["continuationPayloadSHA256"] = savedPayloadSHA256
    }
    try recipeData.write(to: staging.appendingPathComponent("studio-recipe.json"),
      options: .withoutOverwriting)
    try JSONSerialization.data(withJSONObject: completeMetadata,
      options: [.prettyPrinted, .sortedKeys]).write(
        to: staging.appendingPathComponent("result.json"),
        options: .withoutOverwriting)
    try Task.checkCancellation()
    try motionSource?.identity.verify()
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
