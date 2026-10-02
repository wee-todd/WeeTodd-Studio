import AVFoundation
import CryptoKit
import Darwin
import Foundation

/// Immutable native movie jobs freeze the same headless recipes consumed by Studio and ComfyUI.
/// This host owns sequential execution and finishing; the existing Swift worker owns inference.
public struct NativeHeadlessJob: Codable {
  public static let format = "weetodd-studio-native-job-v1"
  public struct Recipe: Codable {
    public var engine: String
    public var bytes: Data
    public var signature: String
    public var report: Data
    public init(engine: String, bytes: Data, signature: String, report: Data = Data()) {
      self.engine = engine; self.bytes = bytes; self.signature = signature; self.report = report
    }
  }
  public struct Source: Codable, Equatable {
    var path: String
    var size: UInt64
    var modified: Date
    static func capture(_ path: String) throws -> Source {
      let attributes = try FileManager.default.attributesOfItem(atPath: path)
      guard attributes[.type] as? FileAttributeType == .typeRegular,
        let size = attributes[.size] as? NSNumber, size.uint64Value > 0,
        let modified = attributes[.modificationDate] as? Date else {
        throw StudioError.invalid("Relink the native job input: \(path)")
      }
      return Source(path: path, size: size.uint64Value, modified: modified)
    }
  }
  public var project: StudioProject
  public var recipes: [String: Recipe]
  public var workers: [String: String]
  public var ffmpeg: String
  public var sources: [Source]
  private struct Envelope: Codable { var format: String; var payload: Data; var payloadSHA256: String }
  public static func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  public static func fileHash(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    var digest = SHA256()
    while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
      digest.update(data: chunk)
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
  public init(project: StudioProject, recipes: [String: Recipe], workers: [String: String], ffmpeg: String) throws {
    self.project = project; self.recipes = recipes; self.workers = workers; self.ffmpeg = ffmpeg
    sources = []
    try validate()
    let owners = try sceneOwners()
    var paths = Set(project.clips.filter { recipes[$0.id.uuidString] == nil && owners[$0.id.uuidString] == nil }.map(\.sourcePath))
    // Preparation may freeze movie tails and RGB guides. Protect these and the original
    // editorial sources; component/checkpoint admission remains owned by each worker.
    for clip in project.clips {
      if !clip.extensionSource.isEmpty { paths.insert(clip.extensionSource) }
    }
    func inputs(_ value: Any, parent: String = "") {
      if let object = value as? [String: Any] {
        for (key, item) in object {
          if ["path", "sourcePath", "manifest"].contains(key), let path = item as? String,
            path.hasPrefix("/") { paths.insert(path) }
          if !["components", "loras"].contains(key) { inputs(item, parent: key) }
        }
      } else if let items = value as? [Any] { items.forEach { inputs($0, parent: parent) } }
    }
    for recipe in recipes.values { inputs(try JSONSerialization.jsonObject(with: recipe.bytes)) }
    sources = try paths.sorted().map(Source.capture)
  }
  public static func validateFinishing(_ project: StudioProject) throws {
    try project.settings.validate()
    guard project.settings.format == .proRes || (0...51).contains(project.settings.quality) else {
      throw StudioError.invalid("Native H.264 finishing quality must be from 0 to 51.")
    }
    guard !project.clips.isEmpty, project.titles.isEmpty, project.audio.isEmpty,
      project.audioMixPolicy == nil || ["legacy-v1", "studio-v1"].contains(project.audioMixPolicy!),
      project.settings.format != .pngSequence,
      project.settings.upscaling == .off, project.settings.interpolation == .off,
      ["fit", "fill"].contains(project.settings.fit) else {
      throw StudioError.invalid("Native headless jobs support cut movies with embedded source audio. Titles, added audio, PNG sequences and enhancement finishing require the Python job exporter.")
    }
    for clip in project.clips {
      let settings = clip.settings(in: project)
      try settings.validate()
      guard settings == project.settings else {
        throw StudioError.invalid("Native cut assembly requires the movie's finishing settings on every clip. Per-clip finishing overrides require the Python job exporter.")
      }
      guard clip.duration.isFinite, clip.duration > 0, clip.sourceIn.isFinite, clip.sourceIn >= 0,
        clip.volume.isFinite, (0...2).contains(clip.volume), (clip.sourcePan ?? 0) == 0,
        clip.transition == "cut", clip.motionFidelity?.enabled != true,
        settings.upscaling == .off, settings.interpolation == .off else {
        throw StudioError.invalid("Native headless jobs cannot silently omit pan, transitions, enhancements, motion fidelity or unsupported finishing.")
      }
    }
  }
  public func validate() throws {
    try Self.validateFinishing(project)
    let owners = try sceneOwners()
    let ids = Set(project.clips.map { $0.id.uuidString })
    guard Set(recipes.keys).isSubset(of: ids), ids.count == project.clips.count else {
      throw StudioError.invalid("Native job recipes must target unique existing clips.")
    }
    for clip in project.clips {
      if let recipe = recipes[clip.id.uuidString] {
        guard ["h3", "ltx25"].contains(recipe.engine), recipe.engine == clip.engine.rawValue,
          recipe.bytes.count <= 1024 * 1024, !recipe.bytes.isEmpty,
          let document = try JSONSerialization.jsonObject(with: recipe.bytes) as? [String: Any],
          document["format"] as? String == "weetodd-headless-v2",
          document["engine"] as? String == recipe.engine,
          let worker = workers[recipe.engine], FileManager.default.isExecutableFile(atPath: worker) else {
          throw StudioError.invalid("A native clip requires its frozen headless recipe and executable Swift worker.")
        }
        if let source = clip.continuity?.sourceClipID ?? clip.extensionClipID,
          recipes[source.uuidString] != nil {
          throw StudioError.invalid("Render and accept the continuity source before exporting its dependent clip.")
        }
      } else if clip.sourcePath.isEmpty && owners[clip.id.uuidString] == nil {
        throw StudioError.invalid("Every reused clip needs an accepted source take.")
      }
    }
    guard FileManager.default.isExecutableFile(atPath: ffmpeg) else {
      throw StudioError.invalid("Native headless movie finishing requires an executable FFmpeg path.")
    }
  }
  struct Scene: Codable, Equatable {
    var version: Int
    var members: [ContinuousSceneMember]
    var frameRate: Double
    var publicationMode: String
    enum CodingKeys: String, CodingKey {
      case version, members
      case frameRate = "frame_rate"
      case publicationMode = "publication_mode"
    }
  }
  func scene(_ recipe: Recipe) throws -> Scene? {
    guard !recipe.report.isEmpty,
      let report = try JSONSerialization.jsonObject(with: recipe.report) as? [String: Any],
      let raw = report["scene"] else { return nil }
    let scene = try JSONDecoder().decode(Scene.self, from: JSONSerialization.data(withJSONObject: raw))
    guard scene.version == 1, scene.frameRate.isFinite, scene.frameRate > 0,
      ["single_decode_native_latent_chain", "windowed_decode_native_latent_chain"].contains(scene.publicationMode) else {
      throw StudioError.invalid("Native job has an invalid prepared scene report.")
    }
    return scene
  }
  func sceneOwners() throws -> [String: String] {
    var owners: [String: String] = [:]
    for (id, recipe) in recipes {
      let raw = try JSONSerialization.jsonObject(with: recipe.bytes) as? [String: Any]
      let report = try scene(recipe)
      guard (raw?["scene"] == nil) == (report == nil) else {
        throw StudioError.invalid("A frozen scene recipe needs its exact prepared member report.")
      }
      guard let report else { continue }
      guard let leader = project.clips.first(where: { $0.id.uuidString == id }),
        try project.continuousSceneMembers(for: leader).map(\.id) == report.members.map(\.clipID),
        report.members.first?.clipID == leader.id else {
        throw StudioError.invalid("A native scene recipe must own its complete contiguous scene.")
      }
      for member in report.members {
        let key = member.clipID.uuidString
        guard owners[key] == nil, key == id || recipes[key] == nil else {
          throw StudioError.invalid("A scene member cannot render independently in the same job.")
        }
        owners[key] = id
      }
    }
    return owners
  }
  public func verifySources() throws {
    for source in sources where try Source.capture(source.path) != source {
      throw StudioError.invalid("A frozen native job source changed. Re-export the current edit: \(source.path)")
    }
  }
  public func write(to url: URL) throws {
    guard !FileManager.default.fileExists(atPath: url.path) else {
      throw StudioError.invalid("Choose a new job filename. Existing jobs are not overwritten.")
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let payload = try encoder.encode(self)
    let envelope = Envelope(format: Self.format, payload: payload, payloadSHA256: Self.hash(payload))
    try encoder.encode(envelope).write(to: url, options: .withoutOverwriting)
  }
  public static func read(from url: URL, workerOverrides: [String: String] = [:]) throws -> NativeHeadlessJob {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      ((attributes[.size] as? NSNumber)?.intValue ?? Int.max) <= 64 * 1024 * 1024 else {
      throw StudioError.invalid("Native job must be a regular JSON file under 64 MiB.")
    }
    let decoder = JSONDecoder(), envelope = try decoder.decode(Envelope.self, from: Data(contentsOf: url))
    guard envelope.format == Self.format, Self.hash(envelope.payload) == envelope.payloadSHA256 else {
      throw StudioError.invalid("Native job manifest changed. Re-export the current edit.")
    }
    var job = try decoder.decode(Self.self, from: envelope.payload)
    let selectedEngines = Set(job.recipes.values.map(\.engine))
    guard Set(workerOverrides.keys).isSubset(of: selectedEngines) else {
      throw StudioError.invalid("A worker override must select an engine rendered by this native job.")
    }
    job.workers.merge(workerOverrides) { _, new in new }
    try job.validate(); try job.verifySources(); return job
  }
}

/// Signals cancel the same process that owns the weighted stage, including before its first event.
public final class NativeHeadlessCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var cancelled = false
  private var process: Process?
  public init() {}
  public func cancel() { lock.lock(); cancelled = true; let child = process; lock.unlock(); if child?.isRunning == true { child?.interrupt() } }
  func attach(_ child: Process?) throws {
    lock.lock(); process = child; let stopped = cancelled; lock.unlock()
    if stopped { if child?.isRunning == true { child?.interrupt() }; throw CancellationError() }
  }
  func check() throws { lock.lock(); let stopped = cancelled; lock.unlock(); if stopped { throw CancellationError() } }
}

public enum NativeHeadlessExecutor {
  public typealias Worker = (String, URL, URL, String) throws -> [String: Any]
  private struct State: Codable {
    var jobSHA256: String
    var workerSHA256: [String: String]
    var completed: [String: String] = [:]
    var mediaSHA256: [String: String] = [:]
    var project: StudioProject
    var movieSHA256: String?
  }
  /// CLI event protocol matches the ordinary Studio/Comfy worker handoff.
  public static func worker(_ executable: String, recipe: URL, output: URL, mode: String,
    cancellation: NativeHeadlessCancellation, emit: (Data) -> Void) throws -> [String: Any] {
    try cancellation.check()
    guard !FileManager.default.fileExists(atPath: output.path) else { throw StudioError.invalid("Native worker output already exists.") }
    let bytes = try Data(contentsOf: recipe), engine = (try JSONSerialization.jsonObject(with: bytes) as? [String: Any])?["engine"] as? String
    let id = UUID().uuidString
    let envelope: [String: Any] = ["version": 1, "jobID": id, "engine": engine ?? "",
      "recipePath": recipe.path, "recipeSHA256": NativeHeadlessJob.hash(bytes), "outputDirectory": output.path]
    let request = output.deletingLastPathComponent().appendingPathComponent(".\(id).request.json")
    try JSONSerialization.data(withJSONObject: envelope).write(to: request)
    defer { try? FileManager.default.removeItem(at: request) }
    let process = Process(), pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = [mode, "--request", request.path, "--output", output.path]
    process.standardOutput = pipe
    let log = output.deletingLastPathComponent().appendingPathComponent("\(id).stderr.log")
    FileManager.default.createFile(atPath: log.path, contents: nil)
    let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
    process.standardError = handle
    let eventsURL = output.deletingLastPathComponent().appendingPathComponent("\(id).events.jsonl")
    FileManager.default.createFile(atPath: eventsURL.path, contents: nil)
    let events = try FileHandle(forWritingTo: eventsURL); defer { try? events.close() }
    try process.run()
    defer { if process.isRunning { process.interrupt() }; process.waitUntilExit(); try? cancellation.attach(nil) }
    try cancellation.attach(process)
    var pending = Data(), terminal: [String: Any]?
    func line(_ data: Data) throws {
      guard data.count <= 65536, let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw StudioError.invalid("Invalid native worker event.")
      }
      try events.write(contentsOf: data + Data([10]))
      emit(data + Data([10]))
      if value["status"] != nil {
        guard terminal == nil else { throw StudioError.invalid("Duplicate native worker completion.") }
        terminal = value
      }
    }
    while true {
      try cancellation.check()
      let chunk = pipe.fileHandleForReading.availableData
      if chunk.isEmpty { break }
      pending.append(chunk)
      while let newline = pending.firstIndex(of: 10) {
        try line(Data(pending[..<newline])); pending.removeSubrange(...newline)
      }
      guard pending.count <= 65536 else { throw StudioError.invalid("Oversized native worker event.") }
    }
    if !pending.isEmpty { try line(pending) }
    process.waitUntilExit(); try cancellation.check()
    guard process.terminationStatus == 0, terminal?["status"] as? String == "success",
      let result = terminal?["result"] as? [String: Any],
      (result["jobID"] as? String).flatMap(UUID.init(uuidString:)) == UUID(uuidString: id),
      result["nativeRuntime"] as? String == "swift-mlx" else {
      throw StudioError.invalid("Native worker failed or returned a different completion identity: \(terminal?["error"] as? String ?? "no valid receipt")")
    }
    if mode == "render" {
      guard let path = result["video"] as? String else { throw StudioError.invalid("Native worker returned no movie.") }
      let url = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
      let root = output.resolvingSymlinksInPath().standardizedFileURL.path + "/"
      guard url.path.hasPrefix(root), FileManager.default.isReadableFile(atPath: url.path) else {
        throw StudioError.invalid("Native worker movie escaped its output directory.")
      }
    }
    return result
  }
  public static func run(job: NativeHeadlessJob, output: URL, resume: Bool = false,
    preflightOnly: Bool = false, cancellation: NativeHeadlessCancellation = NativeHeadlessCancellation(),
    emit: @escaping (Data) -> Void = { FileHandle.standardOutput.write($0) },
    worker customWorker: Worker? = nil) async throws -> [String: Any] {
    try job.validate(); try job.verifySources(); try cancellation.check()
    try await preflightFinishing(project: job.project, ffmpeg: job.ffmpeg, cancellation: cancellation)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let lockURL = output.appendingPathComponent(".job.lock")
    let fd = Darwin.open(lockURL.path, O_CREAT | O_RDWR, 0o600)
    guard fd >= 0, flock(fd, LOCK_EX | LOCK_NB) == 0 else {
      if fd >= 0 { Darwin.close(fd) }; throw StudioError.invalid("Another process owns this native job output.")
    }
    defer { flock(fd, LOCK_UN); Darwin.close(fd) }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let identity = NativeHeadlessJob.hash(try encoder.encode(job))
    var hashes: [String: String] = [:]
    for engine in Set(job.recipes.values.map(\.engine)) { hashes[engine] = try NativeHeadlessJob.fileHash(URL(fileURLWithPath: job.workers[engine]!)) }
    var state = State(jobSHA256: identity, workerSHA256: hashes, project: job.project)
    let stateURL = output.appendingPathComponent("job-state.json")
    if FileManager.default.fileExists(atPath: stateURL.path) {
      guard resume else { throw StudioError.invalid("This output already contains a job. Use --resume or a new directory.") }
      state = try JSONDecoder().decode(State.self, from: Data(contentsOf: stateURL))
      guard state.jobSHA256 == identity, state.workerSHA256 == hashes else {
        throw StudioError.invalid("The native job or Swift worker changed. Use the original runtime or re-export.")
      }
      for (id, path) in state.completed {
        guard state.mediaSHA256[id] == (try? NativeHeadlessJob.fileHash(URL(fileURLWithPath: path))) else {
          throw StudioError.invalid("A completed native take changed. Preserve it and use a new output directory.")
        }
      }
    }
    let resumedGenerations = state.completed.count
    var newlyGenerated = 0
    let call: Worker = customWorker ?? { executable, recipe, destination, mode in
      try worker(executable, recipe: recipe, output: destination, mode: mode, cancellation: cancellation, emit: emit)
    }
    let recipeRoot = output.appendingPathComponent("recipes")
    try FileManager.default.createDirectory(at: recipeRoot, withIntermediateDirectories: true)
    func recipeURL(_ clip: Clip, _ recipe: NativeHeadlessJob.Recipe) throws -> URL {
      let url = recipeRoot.appendingPathComponent(clip.id.uuidString + ".json")
      if FileManager.default.fileExists(atPath: url.path) {
        guard try Data(contentsOf: url) == recipe.bytes else { throw StudioError.invalid("The frozen job recipe changed.") }
      } else { try recipe.bytes.write(to: url, options: .withoutOverwriting) }
      return url
    }
    // Admit every pending recipe and every reused source before the first weighted stage.
    let owners = try job.sceneOwners()
    for clip in job.project.clips {
      if let owner = owners[clip.id.uuidString], owner != clip.id.uuidString, state.completed[owner] == nil { continue }
      if let recipe = job.recipes[clip.id.uuidString], state.completed[clip.id.uuidString] == nil {
        _ = try call(job.workers[recipe.engine]!, recipeURL(clip, recipe), output.appendingPathComponent("preflight-\(clip.id)"), "preflight")
      } else { try await inspect(clip: state.project.clips.first { $0.id == clip.id }!) }
    }
    if preflightOnly { return ["status": "success", "native_runtime": "swift-mlx", "python_inference": false, "generations": job.recipes.count, "newlyGenerated": 0, "resumedGenerations": resumedGenerations,
      "clips": job.project.clips.count] }
    try encoder.encode(state).write(to: stateURL, options: .atomic)
    for index in job.project.clips.indices {
      let clip = job.project.clips[index], id = clip.id.uuidString
      guard let recipe = job.recipes[id], state.completed[id] == nil else { continue }
      try job.verifySources(); try cancellation.check()
      let target = output.appendingPathComponent("take-\(id)")
      var published = false
      defer { if !published { try? FileManager.default.removeItem(at: target) } }
      let frozen = try recipeURL(clip, recipe)
      let result = try call(job.workers[recipe.engine]!, frozen, target, "render")
      guard let path = result["video"] as? String else { throw StudioError.invalid("Native worker returned no movie.") }
      let report = recipe.report.isEmpty ? [:] : (try JSONSerialization.jsonObject(with: recipe.report) as? [String: Any] ?? [:])
      let descriptor = try report["generation"].map {
        try JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: $0))
      }
      let fingerprint = report["resolvedFingerprint"] as? String
      let stats = RenderStats(result: result)
      if let scene = try job.scene(recipe) {
        guard let raw = result["scene"],
          try JSONDecoder().decode(NativeHeadlessJob.Scene.self,
            from: JSONSerialization.data(withJSONObject: raw)) == scene else {
          throw StudioError.invalid("The completed scene does not match its frozen member ranges.")
        }
        let takeID = UUID()
        var versions: [UUID: RenderVersion] = [:]
        for range in scene.members {
          let member = job.project.clips.first { $0.id == range.clipID }!
          var visible = member; visible.sourcePath = path; visible.sourceIn = range.sourceIn; visible.duration = range.duration
          try await inspect(clip: visible, requireAudio: true)
          versions[member.id] = RenderVersion(path: path, seed: member.seed, prompt: member.prompt,
            recipePath: frozen.path, stats: stats, generationSettings: descriptor,
            resolvedFingerprint: fingerprint, usableSourceIn: range.sourceIn, usableDuration: range.duration,
            sceneMembers: scene.members, sceneTakeID: takeID, sceneInputFingerprint: recipe.signature,
            sceneFrameRate: scene.frameRate)
        }
        try state.project.acceptContinuousScene(versions: versions, members: scene.members)
        var asset = MediaAsset(name: "Continuous scene · " + clip.name, kind: .video, path: path, scope: .project)
        asset.duration = scene.members.reduce(0) { $0 + $1.duration }; asset.fps = scene.frameRate
        state.project.assets.append(asset)
      } else {
        guard result["scene"] == nil else { throw StudioError.invalid("Native worker returned an unexpected scene.") }
        var updated = clip; updated.sourcePath = path; updated.sourceIn = result["usable_source_in"] as? Double ?? 0
        let available = result["usable_duration"] as? Double ?? clip.duration
        guard available.isFinite, available > 0 else { throw StudioError.invalid("Native worker returned no usable interval.") }
        if result["use_complete_duration"] as? Bool == true { updated.duration = available }
        else { updated.duration = min(clip.duration, available) }
        try await inspect(clip: updated, requireAudio: true)
        let composed = try JSONSerialization.jsonObject(with: recipe.bytes) as? [String: Any]
        var version = RenderVersion(path: path, seed: clip.seed,
          prompt: composed?["prompt"] as? String ?? clip.prompt, recipePath: frozen.path,
          stats: stats, generationSettings: descriptor, resolvedFingerprint: fingerprint,
          usableSourceIn: updated.sourceIn, usableDuration: available)
        if let raw = result["continuation_artifact"] {
          version.continuationArtifact = try JSONDecoder().decode(ContinuationArtifact.self, from: JSONSerialization.data(withJSONObject: raw))
        }
        updated.versions.append(version); updated.renderedSignature = recipe.signature
        state.project.clips[index] = updated
        var asset = MediaAsset(name: clip.name + " render", kind: .video, path: path, scope: .clip, owner: clip.id)
        asset.duration = try await AVURLAsset(url: URL(fileURLWithPath: path)).load(.duration).seconds
        state.project.assets.append(asset)
      }
      state.completed[id] = path
      state.mediaSHA256[id] = try NativeHeadlessJob.fileHash(URL(fileURLWithPath: path))
      try encoder.encode(state).write(to: stateURL, options: .atomic); published = true
      newlyGenerated += 1
    }
    try job.verifySources(); try cancellation.check()
    let suffix = job.project.settings.format == .mp4 ? "mp4" : "mov"
    let movie = output.appendingPathComponent("movie.\(suffix)")
    if let expected = state.movieSHA256 {
      guard expected == (try? NativeHeadlessJob.fileHash(movie)) else { throw StudioError.invalid("The finished native movie changed.") }
    } else {
      guard !FileManager.default.fileExists(atPath: movie.path) else { throw StudioError.invalid("An unverified finished movie already exists.") }
      try await finish(project: state.project, ffmpeg: job.ffmpeg, movie: movie, cancellation: cancellation)
      state.movieSHA256 = try NativeHeadlessJob.fileHash(movie)
      try encoder.encode(state).write(to: stateURL, options: .atomic)
    }
    try ProjectStorage.write(state.project, to: output.appendingPathComponent("result.weetodd"))
    let result: [String: Any] = ["status": "success", "native_runtime": "swift-mlx", "python_inference": false,
      "video": movie.path, "project": output.appendingPathComponent("result.weetodd").path,
      "generations": job.recipes.count, "newlyGenerated": newlyGenerated,
      "resumedGenerations": resumedGenerations, "clips": job.project.clips.count]
    try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]).write(to: output.appendingPathComponent("result.json"), options: .atomic)
    return result
  }
  /// Codec/filter availability is checked without media or model inference. Output,
  /// time and cancellation are bounded even for an invalid configured executable.
  static func preflightFinishing(project: StudioProject, ffmpeg: String,
    cancellation: NativeHeadlessCancellation) async throws {
    func capabilities(_ option: String) throws -> Set<String> {
      try cancellation.check()
      let log = FileManager.default.temporaryDirectory.appendingPathComponent("NativeFFmpeg-" + UUID().uuidString)
      guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
        throw StudioError.invalid("Cannot inspect FFmpeg finishing capabilities.")
      }
      defer { try? FileManager.default.removeItem(at: log) }
      let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
      let process = Process(); process.executableURL = URL(fileURLWithPath: ffmpeg)
      process.arguments = ["-hide_banner", option]; process.standardOutput = handle; process.standardError = handle
      try process.run()
      defer {
        if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
        process.waitUntilExit(); try? cancellation.attach(nil)
      }
      try cancellation.attach(process)
      let started = Date()
      while process.isRunning {
        try cancellation.check()
        let size = (try FileManager.default.attributesOfItem(atPath: log.path)[.size] as? NSNumber)?.intValue ?? Int.max
        guard size <= 1024 * 1024, Date().timeIntervalSince(started) <= 10 else {
          throw StudioError.invalid("FFmpeg capability inspection exceeded its bounded output or time.")
        }
        // Foundation Process owns a run loop on the launching thread. Keep this
        // bounded, weight-free probe on that thread through its final wait.
        Thread.sleep(forTimeInterval: 0.02)
      }
      try cancellation.check()
      guard process.terminationStatus == 0 else { throw StudioError.invalid("FFmpeg cannot report its finishing capabilities.") }
      try handle.synchronize()
      let size = (try FileManager.default.attributesOfItem(atPath: log.path)[.size] as? NSNumber)?.intValue ?? Int.max
      guard size <= 1024 * 1024 else { throw StudioError.invalid("FFmpeg capability output is too large.") }
      let text = try String(contentsOf: log, encoding: .utf8)
      return Set(text.split(separator: "\n").compactMap { line in
        let fields = line.split(whereSeparator: { $0.isWhitespace })
        return fields.count >= 2 ? String(fields[1]) : nil
      })
    }
    let encoders = try capabilities("-encoders")
    let neededEncoders: Set<String> = project.settings.format == .proRes ? ["prores_ks", "pcm_s16le"] : ["libx264", "aac"]
    guard neededEncoders.isSubset(of: encoders) else {
      throw StudioError.invalid("FFmpeg is missing native movie encoders: " + neededEncoders.subtracting(encoders).sorted().joined(separator: ", "))
    }
    let filters = try capabilities("-filters")
    let neededFilters: Set<String> = ["trim", "setpts", "scale", project.settings.fit == "fill" ? "crop" : "pad",
      "setsar", "fps", "format", "concat", "anullsrc", "atrim", "asetpts", "aresample", "aformat", "volume", "apad"]
    guard neededFilters.isSubset(of: filters) else {
      throw StudioError.invalid("FFmpeg is missing native finishing filters: " + neededFilters.subtracting(filters).sorted().joined(separator: ", "))
    }
    let muxers = try capabilities("-muxers")
    let muxer = project.settings.format == .mp4 ? "mp4" : "mov"
    guard muxers.contains(muxer) else { throw StudioError.invalid("FFmpeg is missing the " + muxer + " movie muxer.") }
  }
  private static func inspect(clip: Clip, requireAudio: Bool = false) async throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: clip.sourcePath))
    let duration = try await asset.load(.duration).seconds
    if requireAudio {
      guard !(try await asset.loadTracks(withMediaType: .audio)).isEmpty else {
        throw StudioError.invalid("Native worker returned no synchronized audio track.")
      }
    }
    guard !(try await asset.loadTracks(withMediaType: .video)).isEmpty,
      duration.isFinite, duration + 0.05 >= clip.sourceIn + clip.duration else {
      throw StudioError.invalid("The accepted native take does not cover its frozen editorial interval.")
    }
  }
  private static func finish(project: StudioProject, ffmpeg: String, movie: URL,
    cancellation: NativeHeadlessCancellation) async throws {
    var arguments = ["-v", "error", "-nostdin"], filters: [String] = [], labels = ""
    let settings = project.settings
    for (index, clip) in project.clips.enumerated() {
      try await inspect(clip: clip)
      arguments += ["-i", clip.sourcePath]
      let start = String(format: "%.12f", clip.sourceIn), duration = String(format: "%.12f", clip.duration)
      let scale = settings.fit == "fill"
        ? "scale=\(settings.width):\(settings.height):force_original_aspect_ratio=increase,crop=\(settings.width):\(settings.height)"
        : "scale=\(settings.width):\(settings.height):force_original_aspect_ratio=decrease,pad=\(settings.width):\(settings.height):(ow-iw)/2:(oh-ih)/2"
      filters.append("[\(index):v:0]trim=start=\(start):duration=\(duration),setpts=PTS-STARTPTS,\(scale),setsar=1,fps=\(settings.fps),format=yuv420p[v\(index)]")
      let audio = try await AVURLAsset(url: URL(fileURLWithPath: clip.sourcePath)).loadTracks(withMediaType: .audio)
      if audio.isEmpty {
        filters.append("anullsrc=r=48000:cl=stereo,atrim=duration=\(duration)[a\(index)]")
      } else {
        filters.append("[\(index):a:0]atrim=start=\(start):duration=\(duration),asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,volume=\(clip.volume),apad,atrim=duration=\(duration)[a\(index)]")
      }
      labels += "[v\(index)][a\(index)]"
    }
    filters.append(labels + "concat=n=\(project.clips.count):v=1:a=1[v][a]")
    arguments += ["-filter_complex", filters.joined(separator: ";"), "-map", "[v]", "-map", "[a]"]
    if settings.format == .proRes { arguments += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le"] }
    else { arguments += ["-c:v", "libx264", "-crf", String(settings.quality), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"] }
    let partial = movie.deletingLastPathComponent().appendingPathComponent(".movie.partial." + movie.pathExtension)
    defer { try? FileManager.default.removeItem(at: partial) }
    arguments += ["-n", partial.path]
    func assemble() throws {
      let process = Process(); process.executableURL = URL(fileURLWithPath: ffmpeg); process.arguments = arguments
      let log = movie.deletingLastPathComponent().appendingPathComponent("finishing.log")
      FileManager.default.createFile(atPath: log.path, contents: nil)
      let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
      process.standardOutput = handle; process.standardError = handle
      try process.run()
      defer { if process.isRunning { process.interrupt() }; process.waitUntilExit(); try? cancellation.attach(nil) }
      try cancellation.attach(process)
      // Keep Foundation Process on its launching thread until it exits.
      while process.isRunning { try cancellation.check(); Thread.sleep(forTimeInterval: 0.02) }
      try cancellation.check()
      guard process.terminationStatus == 0 else { throw StudioError.invalid("Native movie finishing failed. Inspect finishing.log.") }
    }
    try assemble()
    try FileManager.default.moveItem(at: partial, to: movie)
  }
}
