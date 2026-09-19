import AVFoundation
import AppKit
import Combine
import Foundation
import StudioCore
import UniformTypeIdentifiers

struct RuntimeSettings: Codable {
  var root: String
  var pythonPath: String
  var profilesDirectory: String
  var ffmpegPath = ""
  var ffprobePath = ""
  var rifePath = ""
  var rifeWeights = ""
  var metalPath = ""
  var drawThingsHelperPath: String?
  var acceleration: AccelerationSettings?
  var loraFolders: [LoRAFolder]?
  var voiceModels: VoiceModelSettings?
  var generationSettings: Self {
    var value = self
    value.loraFolders = nil
    value.voiceModels = nil
    return value
  }

  static func restoring(_ data: Data?, defaults: Self) -> Self {
    var value = data.flatMap { try? JSONDecoder().decode(Self.self, from: $0) } ?? defaults
    if value.drawThingsHelperPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      ?? true {
      value.drawThingsHelperPath = defaults.drawThingsHelperPath
    }
    return value
  }

  static func defaults() -> Self {
    let root =
      ProcessInfo.processInfo.environment["WEETODD_ROOT"] ?? Bundle.main.object(
        forInfoDictionaryKey: "WeeToddRuntimeRoot") as? String ?? ""
    let support = StudioStore.supportDirectory
    var value = Self(
      root: root, pythonPath: root.isEmpty ? "" : root + "/.venv/bin/python",
      profilesDirectory: support.appendingPathComponent("Profiles").path)
    value.metalPath =
      Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/StudioMetal").path
    let helper = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/WeeToddDrawThings").path
    if FileManager.default.isExecutableFile(atPath: helper) { value.drawThingsHelperPath = helper }
    return value
  }
}
struct ModelProfile: Identifiable, Codable {
  var id: String
  var name: String
  var engine: String
  var task: String
  var generation: GenerationDescriptor?
}

struct BridgeResponseBuffer {
  private(set) var data = Data()
  private let maximum: Int
  private let retained: Int
  init(workflow: Bool) {
    // A complete workflow checkpoint plus its success envelope must survive any
    // trimming of earlier progress lines, including a trim mid-response.
    maximum = workflow ? WorkflowCheckpoint.maximumBytes + 1024 * 1024 : 2_000_000
    retained = workflow ? WorkflowCheckpoint.maximumBytes + 64 * 1024 : 1_000_000
  }
  mutating func append(_ part: Data) {
    if part.count > maximum - data.count {
      // Trim before appending so one unusually large pipe read cannot grow the
      // retained buffer beyond its bound. Preserve the same rolling-tail contract.
      if part.count >= retained {
        data = Data(part.suffix(retained))
      } else {
        data = Data(data.suffix(retained - part.count))
        data.append(part)
      }
    } else {
      data.append(part)
    }
  }
}

@MainActor final class Bridge: ObservableObject {
  @Published var busy = false
  @Published var log = ""
  @Published var message = "Ready"
  @Published var fraction: Double = 0
  @Published var startedAt: Date?
  @Published var lastOutputAt: Date?
  @Published var livePreview: BridgeProgressEvent?
  typealias Invocation = @MainActor (String, RuntimeSettings, [String: Any], URL?) async throws -> [String: Any]
  private let invocation: Invocation?
  init(invocation: Invocation? = nil) { self.invocation = invocation }
  func independent() -> Bridge { Bridge(invocation: invocation) }
  private var process: Process?
  private var cancellationRequested = false
  func cancel() {
    cancellationRequested = true
    message = "Cancelling and releasing render resources…"
    process?.interrupt()
  }
  func invoke(
    _ command: String, runtime: RuntimeSettings, payload: [String: Any], output: URL? = nil
  ) async throws -> [String: Any] {
    guard !busy else {
      throw StudioError.invalid("Another job is active. Wait or cancel it first.")
    }
    if let invocation {
      busy = true
      defer { busy = false }
      return try await invocation(command, runtime, payload, output)
    }
    guard FileManager.default.isExecutableFile(atPath: runtime.pythonPath),
      FileManager.default.fileExists(atPath: runtime.root + "/scripts/studio_bridge.py")
    else {
      throw StudioError.invalid(
        "Select the WeeTodd repository and its Python environment in Runtime Settings.")
    }
    busy = true
    cancellationRequested = false
    startedAt = Date()
    lastOutputAt = startedAt
    fraction = 0
    log = ""
    livePreview = nil
    let previewFile = command == "dt-generate-image" ? output?.appendingPathComponent("live-preview.png") : nil
    defer {
      busy = false; process = nil; livePreview = nil
      if let previewFile { try? FileManager.default.removeItem(at: previewFile) }
    }
    var env = ProcessInfo.processInfo.environment
    if command == "setup-download" {
      message = "Checking macOS Keychain — respond to its permission dialog if shown…"
      let token = try await BackgroundCredential.read { try ModelDownloadToken.read() }
      env = try ModelDownloadToken.environment(env, savedToken: token)
    }
    if command.hasPrefix("dt-"), let connection = payload["connection"] as? [String: Any],
      let reference = connection["credentialRef"] as? String {
      message = "Checking macOS Keychain — respond to its permission dialog if shown…"
      if let secret = try await BackgroundCredential.read(using: { try DrawThingsCredential.read(reference) }) {
        env["WEETODD_DT_CREDENTIAL"] = secret
      }
    }
    guard !cancellationRequested else { throw CancellationError() }
    env["PYTHONUNBUFFERED"] = "1"
    let input = StudioStore.supportDirectory.appendingPathComponent(
      "Requests/\(UUID().uuidString).json")
    try FileManager.default.createDirectory(
      at: input.deletingLastPathComponent(), withIntermediateDirectories: true)
    var body = payload
    body["runtime"] = try runtime.object()
    try JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]).write(
      to: input, options: .atomic)
    busy = true
    startedAt = Date()
    lastOutputAt = startedAt
    fraction = 0
    message = command.capitalized + "…"
    log = ""
    let task = Process()
    task.executableURL = URL(fileURLWithPath: runtime.pythonPath)
    task.arguments = [runtime.root + "/scripts/studio_bridge.py", command, "--request", input.path]
    if let output { task.arguments! += ["--output", output.path] }
    task.currentDirectoryURL = URL(fileURLWithPath: runtime.root)
    task.environment = env
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    process = task
    do { try task.run() } catch {
      busy = false
      process = nil
      throw error
    }
    defer {
      busy = false
      process = nil
      try? FileManager.default.removeItem(at: input)
    }
    let response: [String: Any] = try await withCheckedThrowingContinuation { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        var responseBuffer = BridgeResponseBuffer(workflow: command.hasPrefix("workflow-"))
        var progressStream = BridgeProgressStream()
        while true {
          let part = pipe.fileHandleForReading.availableData
          if part.isEmpty { break }
          responseBuffer.append(part)
          let text = String(decoding: part, as: UTF8.self)
          let events = progressStream.append(part)
          DispatchQueue.main.async {
            self.log = String((self.log + text).suffix(30000))
            self.lastOutputAt = Date()
            for event in events {
              self.message = event.message
              self.fraction = event.fraction ?? self.fraction
              if let previewFile, event.previewPath == previewFile.path,
                let revision = event.previewRevision, revision > (self.livePreview?.previewRevision ?? 0),
                !self.cancellationRequested {
                self.livePreview = event
              }
            }
          }
        }
        task.waitUntilExit()
        let lines = String(decoding: responseBuffer.data, as: UTF8.self).split(separator: "\n")
        let last = lines.reversed().compactMap { line -> [String: Any]? in
          guard let data = String(line).data(using: .utf8),
            let d = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            d["status"] != nil
          else { return nil }
          return d
        }.first
        if task.terminationStatus == 0, let result = last?["result"] as? [String: Any] {
          continuation.resume(returning: result)
        } else {
          continuation.resume(
            throwing: StudioError.invalid(
              last?["error"] as? String
                ?? (last?["status"] as? String == "cancelled"
                  ? "Job cancelled." : "The job failed. Open the log for details.")))
        }
      }
    }
    message = "Ready"
    fraction = 1
    return response
  }
}
extension Encodable {
  func object() throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String: Any]
  }
}

@MainActor final class StudioStore: ObservableObject {
  nonisolated static var supportDirectory: URL {
    if let override = ProcessInfo.processInfo.environment["WEETODD_STUDIO_DATA"] {
      return URL(fileURLWithPath: override)
    }
    return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("WeeTodd Studio")
  }
  @Published var project = StudioProject()
  @Published var globalAssets: [MediaAsset] = []
  @Published var loraGroups: [LoRAGroup] = []
  @Published var scannedLoRAEntries: [LoRAFolderEntry] = []
  @Published var loraFolderWarnings: [String] = []
  @Published var loraFolderScanBusy = false
  @Published var loraFolderScanKey: [LoRAFolder]?
  @Published var showLoRALibrary = false
  @Published var selectedClipID: UUID?
  @Published var selectedAssetID: UUID?
  @Published var selectedTitleID: UUID?
  @Published var selectedAudioID: UUID?
  let playbackPosition = TimelinePlaybackPosition()
  var playhead: Double {
    get { playbackPosition.seconds }
    set {
      // Idle seeks also refresh position-dependent editor actions such as keyframe placement.
      if !isPlaying { objectWillChange.send() }
      playbackPosition.seconds = newValue
    }
  }
  @Published var zoom: Double = 42
  @Published var runtime = RuntimeSettings.defaults()
  @Published var profiles: [ModelProfile] = []
  @Published var generationDescriptions: [UUID: [String: Any]] = [:]
  @Published var projectURL: URL?
  @Published var showPrompt = false
  @Published var showVoice = false
  @Published var selectedVoiceAssetID: UUID?
  lazy var audioMixBridge = bridge.independent()
  @Published var showMusic = false
  @Published var selectedMusicAssetID: UUID?
  @Published var musicModelStatus: String?
  let musicPlayer = AVPlayer()
  @Published var showMotionPrompt = false
  @Published var showWorkflows = false
  @Published var showShotList = false
  @Published var showMusicVideoWorkflow = false
  @Published var showMusicVideoProduction = false
  @Published var productionRunning = false
  @Published var productionStatus: MusicVideoProductionStatus?
  @Published var showProductionLibrary = false
  @Published var showRuntime = false
  @Published var showDrawThings = false
  @Published var drawThingsConnections: [DrawThingsConnection] = []
  let drawThingsDiscovery = DrawThingsCatalogDiscovery()
  let attachmentDigests = AttachmentDigestStore()
  var drawThingsCatalogs: [String: [String: Any]] { drawThingsDiscovery.catalogs }
  @Published var drawThingsLoRAGroups: [DrawThingsLoRAGroup] = []
  @Published var imageDraft: DrawThingsImageDraft? { didSet { persistImageWorkspace() } }
  @Published var imageEstimate: [String: Any]?
  @Published var imagePreviewPath: String? { didSet { persistImageWorkspace() } }
  @Published var referenceSheetOpen = false
  @Published var referenceImageFailure: (key: String, message: String)?
  var activeReferenceLease: ReferenceWorkspaceLease?
  var imageWorkspaceLibrary = ImageWorkspaceLibrary()
  var restoringImageWorkspace = true
  @Published var showDrawThingsConfigImport = false
  var configImportClipID: UUID?
  @Published var drawThingsClipEstimates: [UUID: [String: Any]] = [:]
  var preparedDrawThingsClip: PreparedDrawThingsClip?
  @Published var showProjectSettings = false
  @Published var showLog = false
  @Published var showActions = false
  @Published var selectedTrackID: UUID?
  @Published var error: String?
  @Published var validationErrors: [UUID: String] = [:]
  @Published var notice = "Create a clip or drop a movie onto the timeline."
  @Published var preparedPrompt = ""
  @Published var preparedRecipe: String?
  @Published var preparedReport = ""
  @Published var pendingContinuousScene: PendingContinuousSceneTake?
  @Published var showContinuousSceneReview = false
  @Published var acceptingContinuousScene = false
  @Published var connectingContinuousScene = false
  @Published var motionPromptDraft = ""
  @Published var motionRecipePrompt = ""
  @Published var motionPromptClipName = ""
  @Published var motionPromptUsingOverride = false
  @Published var motionPromptLoading = false
  @Published var motionPromptEditorError: String?
  @Published var dirty = false
  @Published var player = AVPlayer()
  @Published var isPlaying = false {
    didSet { if !isPlaying { playbackClipStates.removeAll() } }
  }
  var playbackClipStates: [UUID: ClipState] = [:]
  @Published var queue: [UUID] = []
  @Published var previewMode = "Timeline"
  @Published var preparingTimelinePlayback = false
  @Published var timelinePlaybackIssues: [UUID: String] = [:]
  @Published var timelinePlaybackWarning: String?
  var timelinePlaybackPlan: TimelinePlaybackPlan?
  var timelineBuildTask: Task<Void, Never>?
  var timelineBuildID = UUID()
  var timelineSeekID = UUID()
  var timelineSeekPending = false
  var timelineItemStatus: NSKeyValueObservation?
  var timelineClock: TimelineFallbackClock?
  var scrubWasPlaying: Bool?
  let bridge: Bridge
  let descriptionBridge: Bridge
  let dataDirectory: URL
  @Published private(set) var activeNativeRequest: UUID?
  private(set) var documentSessionID = UUID()
  @Published var preparingDrawThings = false
  var operationBusy: Bool { bridge.busy || activeNativeRequest != nil || preparingDrawThings || acceptingContinuousScene || connectingContinuousScene }
  var preparedFingerprint: String?
  var motionPromptSession: MotionPromptEditorSession?
  private var undoStates: [StudioProject] = []
  private var redoStates: [StudioProject] = []
  private var lastUndoGroup: UUID?
  private var autosaveTask: Task<Void, Never>?
  private var playbackObservation: TimelinePlayerObservation?
  private var bridgeObservation: AnyCancellable?
  private var catalogObservation: AnyCancellable?
  private var digestObservation: AnyCancellable?
  var selectedClip: Clip? { project.clips.first { $0.id == selectedClipID } }
  var allAssets: [MediaAsset] { globalAssets + project.assets }
  var selectedAsset: MediaAsset? { allAssets.first { $0.id == selectedAssetID } }
  var canUndo: Bool { !undoStates.isEmpty }
  var canRedo: Bool { !redoStates.isEmpty }

  init(dataDirectory: URL = StudioStore.supportDirectory, restoreSession: Bool = true,
       invocation: Bridge.Invocation? = nil) {
    self.dataDirectory = dataDirectory
    bridge = Bridge(invocation: invocation)
    descriptionBridge = Bridge(invocation: invocation)
    digestObservation = attachmentDigests.objectWillChange.sink { [weak self] _ in
      self?.objectWillChange.send()
    }
    catalogObservation = drawThingsDiscovery.objectWillChange.sink { [weak self] _ in
      self?.objectWillChange.send()
    }
    bridgeObservation = bridge.objectWillChange.sink { [weak self] _ in
      self?.objectWillChange.send()
    }
    playbackObservation = TimelinePlayerObservation(player: player, tick: { [weak self] time, item in
      MainActor.assumeIsolated { self?.playbackTick(time.seconds, item: item) }
    }, ended: { [weak self] item in
      MainActor.assumeIsolated { self?.playbackEnded(item: item) }
    })
    runtime = RuntimeSettings.restoring(
      try? Data(contentsOf: dataDirectory.appendingPathComponent("runtime.json")),
      defaults: runtime)
    guard restoreSession else { return }
    try? FileManager.default.createDirectory(
      at: dataDirectory.appendingPathComponent("Profiles"),
      withIntermediateDirectories: true)
    if let data = try? Data(
      contentsOf: dataDirectory.appendingPathComponent("global-assets.json")),
      let value = try? JSONDecoder().decode([MediaAsset].self, from: data)
    {
      globalAssets = value
    }
    loadLoRAGroups()
    loadDrawThingsConnections()
    loadDrawThingsLoRAGroups()
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--project"), args.count > i + 1 {
      load(URL(fileURLWithPath: args[i + 1]))
    } else { restoreAutosavedProject() }
    restoreImageWorkspaces()
    Task {
      await reloadProfiles()
      refreshPreview()
    }
  }
  /// Director can hold the first edits in a movie, before any timeline/project change.
  /// Anchor its UUID in the active restore snapshot before creating UUID-scoped drafts.
  func directorSessionURL(key: String) throws -> URL {
    try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
    try preserveRecovery()
    try ProjectStorage.write(project, to: dataDirectory.appendingPathComponent("Autosave.weetodd"))
    return dataDirectory.appendingPathComponent("Director/\(project.id.uuidString)/\(key).json")
  }
  func restoreAutosavedProject() {
    guard let recovered = try? ProjectStorage.read(dataDirectory.appendingPathComponent("Autosave.weetodd")) else { return }
    project = recovered
    selectedClipID = recovered.clips.first?.id
    dirty = true
    notice = "Recovered your autosaved project. Save it to choose a project file."
  }
  func change(undoGroup: UUID? = nil, _ body: (inout StudioProject) -> Void) {
    var updated = project
    body(&updated)
    for i in updated.clips.indices where updated.clips[i].audioDriverSelection != nil {
      let clip = updated.clips[i]
      if let old = project.clips.first(where: { $0.id == clip.id }),
        project.audioDriverRevision(for: old) != updated.audioDriverRevision(for: clip) {
        updated.clips[i].audioDriverMixKey = nil
      }
    }
    guard updated != project else { return }
    if undoGroup == nil || undoGroup != lastUndoGroup { undoStates.append(project) }
    lastUndoGroup = undoGroup
    if undoStates.count > 80 { undoStates.removeFirst() }
    redoStates.removeAll()
    project = updated
    changed()
  }
  func editClip(undoGroup: UUID? = nil, _ body: (inout Clip) -> Void) {
    guard let i = project.clips.firstIndex(where: { $0.id == selectedClipID }) else { return }
    var separated = false
    change(undoGroup: undoGroup) { project in
      let previousEngine = project.clips[i].engine
      body(&project.clips[i])
      if previousEngine != project.clips[i].engine && project.clips[i].engine != .ltx25 {
        separated = project.separateContinuousSceneMember(clipID: project.clips[i].id)
      }
    }
    if separated {
      notice = "Separated this shot from its LTX 2.5 group. It now generates independently; images, takes and edit points are preserved. Undo restores the scene."
    }
  }
  func separateContinuousScene(clipID: UUID) {
    var separated = false
    change { separated = $0.separateContinuousSceneMember(clipID: clipID) }
    if separated {
      notice = "This shot now generates independently. Images, takes and edit points are preserved. Undo restores the scene."
    }
  }
  func changed() {
    preparedDrawThingsClip = nil
    validationErrors.removeAll()
    prepareTimelinePlayback()
    dirty = true
    preparedRecipe = nil
    preparedFingerprint = nil
    preparedPrompt = ""
    preparedReport = ""
    autosaveTask?.cancel()
    autosaveTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 600_000_000)
      guard !Task.isCancelled, let self else { return }
      do {
        try self.preserveRecovery()
        try ProjectStorage.write(
          self.project, to: self.dataDirectory.appendingPathComponent("Autosave.weetodd"))
      } catch { self.notice = "Autosave needs attention: \(error.localizedDescription)" }
    }
  }
  func undo() {
    lastUndoGroup = nil
    guard let value = undoStates.popLast() else { return }
    redoStates.append(project)
    project = value
    changed()
    refreshPreview()
  }
  func redo() {
    lastUndoGroup = nil
    guard let value = redoStates.popLast() else { return }
    undoStates.append(project)
    project = value
    changed()
    refreshPreview()
  }
  func select(_ id: UUID) {
    preparedDrawThingsClip = nil
    lastUndoGroup = nil
    selectedClipID = id
    selectedTitleID = nil
    selectedAudioID = nil
    selectedTrackID = nil
    preparedRecipe = nil
    pausePlayback()
    if previewMode != "Movie" { prepareTimelinePlayback() }
    if let index = project.clips.firstIndex(where: { $0.id == id }) {
      seek(project.start(of: index))
    }
  }
  func addClip(_ engine: Engine = .ltx25) {
    var c = Clip(name: "Shot \(project.clips.count + 1)", engine: engine)
    if engine != .movie && engine != .drawThings { c.generationSelection = GenerationSelection() }
    if engine == .drawThings {
      c.drawThings = DrawThingsSelection(profileID: drawThingsConnections.first?.id ?? "",
        modelID: "", modelFamily: "")
    }
    change { $0.clips.append(c) }
    select(c.id)
  }
  func deleteClip() {
    guard let id = selectedClipID else { return }
    change {
      $0.clips.removeAll { $0.id == id }
      $0.audio.removeAll { $0.anchor?.clipID == id }
      $0.assets.removeAll { $0.scope == .clip && $0.owner == id }
    }
    selectedClipID = project.clips.first?.id
    refreshPreview()
  }
  func duplicateClip() {
    guard var c = selectedClip else { return }
    let oldID = c.id
    c.id = UUID()
    c.name += " copy"
    // Duplicate clip-store links, retaining the same underlying media files.
    var links: [MediaAsset] = []
    for old in project.assets where old.owner == oldID && old.scope == .clip {
      var linked = old
      linked.id = UUID()
      linked.owner = c.id
      for i in c.attachments.indices where c.attachments[i].assetID == old.id {
        c.attachments[i].assetID = linked.id
      }
      links.append(linked)
    }
    change { p in
      let i = p.clips.firstIndex { $0.id == oldID } ?? p.clips.count - 1
      p.clips.insert(c, at: i + 1)
      p.assets.append(contentsOf: links)
    }
    select(c.id)
  }
  func split() {
    guard let span = TimelinePlaybackPlan(project: project).span(at: playhead) else { return }
    do {
      var p = project
      let next = try p.split(span.clipID, at: playhead - span.start)
      change { $0 = p }
      select(next)
    } catch { self.error = error.localizedDescription }
  }
  func save(asNew: Bool = false) {
    var target = asNew ? nil : projectURL
    if target == nil {
      let panel = NSSavePanel()
      panel.title = "Save WeeTodd Project"
      panel.nameFieldStringValue = project.name + ".weetodd"
      panel.canCreateDirectories = true
      guard panel.runModal() == .OK else { return }
      target = panel.url
    }
    guard let target else { return }
    do {
      try ProjectStorage.write(project, to: target)
      projectURL = target
      dirty = false
      notice = "Saved \(target.lastPathComponent)"
    } catch { self.error = error.localizedDescription }
  }
  func openProject() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.json, UTType(filenameExtension: "weetodd") ?? .data]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    load(url)
  }
  /// Keep recovery separate for each opened document, including copies sharing a project UUID.
  func preserveRecovery() throws {
    guard dirty else { return }
    let folder = dataDirectory.appendingPathComponent("Recovery")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let stem = folder.appendingPathComponent(documentSessionID.uuidString)
    try ProjectStorage.write(project, to: stem.appendingPathExtension("weetodd"))
    let provenance = ["name": project.name, "originalPath": projectURL?.path ?? ""]
    try JSONEncoder().encode(provenance).write(to: stem.appendingPathExtension("json"), options: .atomic)
  }
  func replaceDocument(_ value: StudioProject, url: URL?, isDirty: Bool) throws {
    // Failure leaves the current document and its file association intact.
    try preserveRecovery()
    // Startup restores this active snapshot, independently of historical recovery files.
    // Write before replacing in-memory state so a failed write leaves the old document active.
    try ProjectStorage.write(value, to: dataDirectory.appendingPathComponent("Autosave.weetodd"))
    autosaveTask?.cancel()
    cancelMotionPromptEditor()
    referenceSheetOpen = false; imageDraft = nil; imagePreviewPath = nil
    musicPlayer.pause(); musicPlayer.replaceCurrentItem(with: nil)
    showVoice = false; selectedVoiceAssetID = nil
    showMusic = false; selectedMusicAssetID = nil; musicModelStatus = nil
    undoStates.removeAll(); redoStates.removeAll(); lastUndoGroup = nil
    invalidateTimelinePlayback()
    documentSessionID = UUID()
    project = value
    projectURL = url
    selectedClipID = value.clips.first?.id
    selectedAssetID = nil; selectedTitleID = nil; selectedAudioID = nil; selectedTrackID = nil
    playhead = 0; queue.removeAll()
    generationDescriptions.removeAll(); validationErrors.removeAll(); drawThingsClipEstimates.removeAll()
    preparedDrawThingsClip = nil; preparedRecipe = nil; preparedFingerprint = nil
    preparedPrompt = ""; preparedReport = ""
    pendingContinuousScene = nil; showContinuousSceneReview = false
    showPrompt = false; error = nil
    dirty = isDirty
    refreshPreview()
  }
  func load(_ url: URL) {
    do {
      try replaceDocument(ProjectStorage.read(url), url: url, isDirty: false)
      notice = "Opened \(url.lastPathComponent)"
    } catch { self.error = error.localizedDescription }
  }
  func newProject() {
    do {
      try replaceDocument(StudioProject(), url: nil, isDirty: false)
      notice = "New movie. Previous unsaved edits are available in Recovery."
    } catch { self.error = error.localizedDescription }
  }
  func chooseImports(scope: AssetScope = .project, addToTimeline: Bool = false) {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    guard panel.runModal() == .OK else { return }
    Task { await importURLs(panel.urls, scope: scope, addToTimeline: addToTimeline) }
  }
  func importURLs(
    _ urls: [URL], scope: AssetScope, addToTimeline: Bool = false, loraModel: LoRAModel? = nil,
    loraProfile: String? = nil, loraLayout: String? = nil, loraAdalnInputGrid: String? = nil
  ) async {
    if scope == .clip && selectedClipID == nil && !addToTimeline {
      error = "Select a clip before importing into its asset store."
      return
    }
    for url in urls {
      do {
        var inspection: [String: Any] = ["path": url.path]
        if let loraModel { inspection["loraModel"] = loraModel.rawValue }
        if loraModel == .h3 {
          if let loraProfile { inspection["loraProfile"] = loraProfile }
          if let loraLayout { inspection["loraLayout"] = loraLayout }
          if let loraAdalnInputGrid { inspection["loraAdalnInputGrid"] = loraAdalnInputGrid }
        }
        let info = try await bridge.invoke("inspect", runtime: runtime, payload: inspection)
        let kind = AssetKind(rawValue: info["kind"] as? String ?? "video") ?? .video
        var asset = MediaAsset(
          name: url.deletingPathExtension().lastPathComponent, kind: kind, path: url.path,
          scope: scope, owner: scope == .clip ? selectedClipID : nil)
        asset.duration = info["duration"] as? Double ?? 0
        asset.width = info["width"] as? Int ?? 0
        asset.height = info["height"] as? Int ?? 0
        asset.fps = info["fps"] as? Double ?? 0
        asset.text = info["text"] as? String ?? ""
        if kind == .lora {
          asset.loraModel =
            (info["loraModel"] as? String).flatMap(LoRAModel.init(rawValue:)) ?? loraModel
          if asset.loraModel == .h3 {
            asset.loraProfile = info["loraProfile"] as? String ?? (loraModel == .h3 ? loraProfile : nil)
            asset.loraLayout = info["loraLayout"] as? String ?? (loraModel == .h3 ? loraLayout : nil)
            asset.loraAdalnInputGrid = info["loraAdalnInputGrid"] as? String ?? (loraModel == .h3 ? loraAdalnInputGrid : nil)
          }
        }
        if addToTimeline && (kind == .video || kind == .image) {
          var c = Clip(name: asset.name, engine: .movie)
          c.sourcePath = asset.path
          c.duration = kind == .image ? 5 : max(0.1, asset.duration)
          asset.scope = .clip
          asset.owner = c.id
          change {
            $0.clips.append(c)
            $0.assets.append(asset)
          }
          select(c.id)
        } else if scope == .global {
          globalAssets.append(asset)
          saveGlobals()
        } else {
          change { $0.assets.append(asset) }
        }
        selectedAssetID = asset.id
        notice = "Linked \(asset.name) to \(asset.scope.rawValue) assets."
      } catch {
        self.error = error.localizedDescription
        break
      }
    }
  }
  func saveGlobals() {
    do {
      try JSONEncoder().encode(globalAssets).write(
        to: dataDirectory.appendingPathComponent("global-assets.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func useAsset(_ asset: MediaAsset, role: MediaRole, time: Double = 0) {
    if asset.kind == .text {
      editClip { $0.prompt = asset.text }
      return
    }
    guard selectedClip != nil else {
      error = "Select or create a clip first."
      return
    }
    if role == .lora {
      applyLoRAMembers([LoRAMember(asset: asset)])
      return
    }
    if let clip = selectedClip, !clip.canAssignMedia(asset, role: role) {
      error = "Choose a supported reference purpose from the asset menu. Movies may need a reference sheet or control guide; LTX audio uses Audio driver."
      return
    }
    if role == .first || role == .last, let id = selectedClipID {
      assignEndpoint(asset, to: id, role: role)
      return
    }
    if let action = selectedClip?.referenceActions(for: asset).first(where: {
      $0.role == role && $0.preparation == nil
    }) {
      do {
        var updated = selectedClip!
        try updated.attachReference(asset, action: action)
        editClip { $0 = updated }
      } catch { self.error = error.localizedDescription }
      return
    }
    editClip { c in
      if [.first, .last, .audioDriver].contains(role) {
        c.attachments.removeAll { $0.role == role }
      }
      c.attachments.append(Attachment(assetID: asset.id, role: role, time: time))
    }
  }
  func addAssetToTimeline(_ asset: MediaAsset) {
    if asset.kind == .audio {
      var a = AudioRegion(assetID: asset.id, path: asset.path)
      a.duration = max(0.1, asset.duration)
      a.trackID = selectedTrackID ?? project.audioTracks.first?.id
      change { $0.audio.append(a) }
      selectedAudioID = a.id
      selectedTitleID = nil
    } else if [.video, .image, .sequence].contains(asset.kind) {
      var c = Clip(name: asset.name, engine: .movie)
      c.sourcePath = asset.path
      c.duration = asset.kind == .image ? 5 : max(0.1, asset.duration)
      var linked = asset
      linked.id = UUID()
      linked.scope = .clip
      linked.owner = c.id
      change {
        $0.clips.append(c)
        $0.assets.append(linked)
      }
      select(c.id)
    }
  }
  func addTitle() {
    var t = TitleOverlay()
    t.start =
      selectedClipID.flatMap { id in project.clips.firstIndex { $0.id == id } }.map {
        project.start(of: $0)
      } ?? 0
    change { $0.titles.append(t) }
    selectedTitleID = t.id
    selectedAudioID = nil
  }
  func relink(_ asset: MediaAsset) {
    let panel = NSOpenPanel()
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let old = asset.path
    if let i = globalAssets.firstIndex(where: { $0.id == asset.id }) {
      globalAssets[i].path = url.path
      saveGlobals()
    }
    change { p in
      for i in p.assets.indices where p.assets[i].id == asset.id { p.assets[i].path = url.path }
      for i in p.clips.indices where p.clips[i].sourcePath == old {
        p.clips[i].sourcePath = url.path
      }
      for i in p.audio.indices where p.audio[i].path == old { p.audio[i].path = url.path }
      p.mapMusicSourcePaths { $0 == old ? url.path : $0 }
      p.mapVoicePaths { $0 == old ? url.path : $0 }
    }
    refreshPreview()
  }
  var timelineAudioLease: AudioMixLease?

  func collectMedia() {
    let panel = NSOpenPanel()
    panel.title = "Choose a folder for the portable project"
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let folder = panel.url else { return }
    do {
      let bundle = folder.appendingPathComponent(
        project.name + "-" + String(UUID().uuidString.prefix(6)))
      let media = bundle.appendingPathComponent("Media")
      try FileManager.default.createDirectory(at: media, withIntermediateDirectories: true)
      var p = project
      var copied: [String: String] = [:]
      let used = Set(p.clips.flatMap { $0.attachments.map(\.assetID) } + p.audio.map(\.assetID))
      for var asset in globalAssets where used.contains(asset.id) {
        asset.scope = .project
        p.assets.append(asset)
      }
      func collect(_ value: String) throws -> String {
        if value.isEmpty { return value }
        if let known = copied[value] { return known }
        let source = URL(fileURLWithPath: value)
        let target = media.appendingPathComponent(
          UUID().uuidString + "-" + source.lastPathComponent)
        try FileManager.default.copyItem(at: source, to: target)
        let relative = "Media/" + target.lastPathComponent
        copied[value] = relative
        return relative
      }
      for i in p.assets.indices where p.assets[i].kind != .lora {
        p.assets[i].path = try collect(p.assets[i].path)
        if let generation = p.assets[i].musicGeneration {
          if let known = copied[generation.artifacts] {
            p.assets[i].musicGeneration?.artifacts = known
          } else {
            let target = media.appendingPathComponent(UUID().uuidString + "-music")
            var collected = try ProjectStorage.collectMusicArtifacts(generation, to: target)
            collected.artifacts = "Media/" + target.lastPathComponent
            copied[generation.artifacts] = collected.artifacts
            p.assets[i].musicGeneration = collected
          }
        }
        if p.assets[i].kind == .sequence { p.assets[i].text = try collect(p.assets[i].text) }
      }
      for i in p.clips.indices {
        p.clips[i].sourcePath = try collect(p.clips[i].sourcePath)
        if p.clips[i].motionResult != nil {
          p.clips[i].motionResult!.path = try collect(p.clips[i].motionResult!.path)
          p.clips[i].motionResult!.sourcePath = try collect(p.clips[i].motionResult!.sourcePath)
          p.clips[i].motionResult!.report = try collect(p.clips[i].motionResult!.report)
          if let recipe = p.clips[i].motionResult!.recipePath {
            p.clips[i].motionResult!.recipePath = try collect(recipe)
          }
        }
        p.clips[i].extensionSource = try collect(p.clips[i].extensionSource)
        p.clips[i].depthDirectory = try collect(p.clips[i].depthDirectory)
        p.clips[i].motionDirectory = try collect(p.clips[i].motionDirectory)
        for j in p.clips[i].versions.indices {
          p.clips[i].versions[j].path = try collect(p.clips[i].versions[j].path)
          if let artifact = p.clips[i].versions[j].continuationArtifact {
            if let known = copied[artifact.manifest] {
              p.clips[i].versions[j].continuationArtifact?.manifest = known
            } else {
              let context = media.appendingPathComponent(UUID().uuidString + "-context")
              var collected = try ProjectStorage.collectContinuationArtifact(artifact, to: context)
              collected.manifest = "Media/" + context.lastPathComponent + "/manifest.json"
              copied[artifact.manifest] = collected.manifest
              p.clips[i].versions[j].continuationArtifact = collected
            }
          }
        }
      }
      for i in p.audio.indices { p.audio[i].path = try collect(p.audio[i].path) }
      try p.mapMusicSourcePaths(collect)
      try p.mapVoicePaths(collect)
      // Collected media have new identities; rebuild drivers from these portable sources.
      for i in p.clips.indices where p.clips[i].audioDriverSelection != nil { p.clips[i].audioDriverMixKey = nil }
      let target = bundle.appendingPathComponent(project.name + ".weetodd")
      try ProjectStorage.write(p, to: target)
      notice = "Collected \(copied.count) media files. Model weights remain shared."
      NSWorkspace.shared.activateFileViewerSelecting([target])
    } catch { self.error = error.localizedDescription }
  }
  func saveRuntime(reloadProfiles: Bool = true) {
    do {
      try JSONEncoder().encode(runtime).write(
        to: dataDirectory.appendingPathComponent("runtime.json"), options: .atomic)
      if reloadProfiles { Task { await self.reloadProfiles() } }
    } catch { self.error = error.localizedDescription }
  }
  func reloadProfiles() async {
    guard !runtime.root.isEmpty, !bridge.busy else { return }
    do {
      let r = try await bridge.invoke("catalog", runtime: runtime, payload: [:])
      profiles = (r["profiles"] as? [[String: Any]] ?? []).compactMap { d in
        guard let id = d["id"] as? String, let name = d["name"] as? String,
          let engine = d["engine"] as? String, let task = d["task"] as? String
        else { return nil }
        let generation = (d["generation"] as? [String: Any]).flatMap {
          try? JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: $0))
        }
        return ModelProfile(id: id, name: name, engine: engine, task: task, generation: generation)
      }
    } catch { notice = error.localizedDescription }
  }
  func importRecipes() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.json]
    panel.allowsMultipleSelection = true
    guard panel.runModal() == .OK else { return }
    do {
      let folder = URL(fileURLWithPath: runtime.profilesDirectory)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      for url in panel.urls {
        let d = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard d?["format"] as? String == "weetodd-headless-v2" else {
          throw StudioError.invalid("Import a WeeTodd headless v2 recipe, not a ComfyUI graph.")
        }
        var target = folder.appendingPathComponent(url.lastPathComponent)
        if FileManager.default.fileExists(atPath: target.path) {
          target = folder.appendingPathComponent(UUID().uuidString + "-" + url.lastPathComponent)
        }
        try FileManager.default.copyItem(at: url, to: target)
      }
      Task { await reloadProfiles() }
    } catch { self.error = error.localizedDescription }
  }
  func payload() throws -> [String: Any] {
    [
      "project": try project.object(),
      "globalAssets": try JSONSerialization.jsonObject(with: JSONEncoder().encode(globalAssets)),
      "clipID": selectedClipID?.uuidString ?? "",
    ]
  }
  func generationRequestKey(for clip: Clip) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let relevantProfiles = profiles.filter { $0.engine == clip.engine.rawValue }.sorted { $0.id < $1.id }
    let profileMetadata = relevantProfiles.map { profile in
      let attributes = try? FileManager.default.attributesOfItem(atPath: profile.id)
      return profile.id + "|" + String(describing: attributes?[.modificationDate])
        + "|" + String(describing: attributes?[.size])
    }.joined(separator: "\n")
    let referenced = Set(clip.attachments.map(\.assetID))
    let mediaMetadata = allAssets.filter { referenced.contains($0.id) }.sorted { $0.id.uuidString < $1.id.uuidString }.map { asset in
      let attributes = try? FileManager.default.attributesOfItem(atPath: asset.path)
      return asset.path + "|" + String(describing: attributes?[.modificationDate])
        + "|" + String(describing: attributes?[.size])
    }.joined(separator: "\n")
    return clip.generationFingerprint + ((try? encoder.encode(runtime.generationSettings).base64EncodedString()) ?? "")
      + project.continuityDependencyFingerprint(for: clip)
      + continuousSceneDependencyKey(for: clip)
      + GenerationSelection.assetFingerprint(for: clip, assets: allAssets)
      + ((try? encoder.encode(relevantProfiles).base64EncodedString()) ?? "")
      + ((try? encoder.encode(loraGroups).base64EncodedString()) ?? "")
      + String(clip.settings(in: project).fps) + profileMetadata + mediaMetadata
  }
  func generationDescriptionTaskKey(for clip: Clip) -> String {
    // Reopening clears descriptions even when the selected clip's render inputs are identical.
    // Keep session identity out of generationRequestKey so saved takes remain reusable.
    documentSessionID.uuidString + "|" + generationRequestKey(for: clip)
  }
  func describeGeneration() async {
    guard let clip = selectedClip, clip.engine != .movie, clip.engine != .drawThings else { return }
    let key = generationRequestKey(for: clip)
    let session = documentSessionID
    do {
      var result = try await descriptionBridge.independent().invoke("describe-generation", runtime: runtime, payload: try payload())
      guard documentSessionID == session, selectedClipID == clip.id,
        selectedClip.map({ generationRequestKey(for: $0) }) == key else { return }
      result["studioInput"] = key
      result["studioEngine"] = clip.engine.rawValue
      result["studioTask"] = clip.inferredTask
      result["studioProfile"] = clip.profileID
      generationDescriptions[clip.id] = result
      let readiness = result["readinessErrors"] as? [String] ?? []
      validationErrors[clip.id] = readiness.isEmpty ? nil : readiness.joined(separator: "\n")
    } catch {
      guard documentSessionID == session, selectedClipID == clip.id,
        selectedClip.map({ generationRequestKey(for: $0) }) == key else { return }
      validationErrors[clip.id] = error.localizedDescription
    }
  }
  func generateSelected() async {
    guard !operationBusy else { return }
    let target = selectedClipID, session = documentSessionID
    if !canGenerateSelected { await prepareSelected() }
    guard selectedClipID == target, documentSessionID == session, canGenerateSelected else { return }
    await renderPrepared()
  }
  func prepareSelected() async {
    guard !operationBusy else { return }
    let target = selectedClipID, originalSession = documentSessionID
    if selectedClip?.audioDriverSelection != nil {
      guard await prepareAudioDriver(), selectedClipID == target,
        documentSessionID == originalSession else { return }
    }
    guard let clip = selectedClip else { return }
    if clip.engine == .drawThings { await prepareDrawThingsClip(); return }
    let requestID = UUID()
    activeNativeRequest = requestID
    defer { if activeNativeRequest == requestID { activeNativeRequest = nil } }
    let session = documentSessionID
    let projectID = project.id
    let key = generationRequestKey(for: clip)
    let settings = runtime
    func isCurrent() -> Bool {
      documentSessionID == session && project.id == projectID && selectedClipID == clip.id
        && selectedClip.map { generationRequestKey(for: $0) == key } == true
    }
    preparedRecipe = nil; preparedFingerprint = nil
    do {
      let body = try payload()
      var description = try await descriptionBridge.invoke("describe-generation", runtime: settings, payload: body)
      guard isCurrent() else {
        notice = "Preflight stopped because its project or clip changed. Prepare the current clip again."
        return
      }
      description["studioInput"] = key
      description["studioEngine"] = clip.engine.rawValue
      description["studioTask"] = clip.inferredTask
      description["studioProfile"] = clip.profileID
      generationDescriptions[clip.id] = description
      let snapshot = signature(for: clip)
      let destination = dataDirectory.appendingPathComponent("Jobs/\(requestID.uuidString)/prepared")
      let r = try await bridge.invoke("prepare", runtime: settings, payload: body, output: destination)
      guard isCurrent(), signature(for: clip) == snapshot else {
        notice = "Clip changed during preflight. Prepare it again."
        return
      }
      preparedRecipe = r["recipePath"] as? String
      preparedPrompt = r["prompt"] as? String ?? ""
      preparedReport = String(decoding: try JSONSerialization.data(withJSONObject: r["report"] ?? [:],
        options: [.prettyPrinted, .sortedKeys]), as: UTF8.self)
      if let i = project.clips.firstIndex(where: { $0.id == clip.id }) {
        project.clips[i].validatedSignature = snapshot
      }
      validationErrors[clip.id] = nil
      preparedFingerprint = snapshot
      notice = "Preflight passed. Review the exact prompt, then render."
    } catch {
      guard isCurrent() else { return }
      self.error = error.localizedDescription
      validationErrors[clip.id] = error.localizedDescription
    }
  }
  func renderPrepared() async {
    if selectedClip?.engine == .drawThings { await renderDrawThingsClip(); return }
    guard !operationBusy, let path = preparedRecipe, let c = selectedClip else { return }
    let session = documentSessionID
    let projectID = project.id
    let submittedSignature = signature(for: c)
    let settings = runtime
    let requestID = UUID()
    activeNativeRequest = requestID
    defer { if activeNativeRequest == requestID { activeNativeRequest = nil } }
    do {
      guard signature(for: c) == preparedFingerprint else {
        throw StudioError.invalid("Clip changed. Prepare it again before rendering.")
      }
      let destination = URL(fileURLWithPath: path).deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("render")
      let prompt = preparedPrompt
      let prepared = (try? JSONSerialization.jsonObject(with: Data(preparedReport.utf8))) as? [String: Any]
      let resolved = prepared ?? generationDescriptions[c.id]
      let generationSettings = resolved?["generation"].flatMap {
        try? JSONDecoder().decode(GenerationDescriptor.self,
          from: JSONSerialization.data(withJSONObject: $0))
      }
      let resolvedFingerprint = prepared?["resolvedFingerprint"] as? String
      let sceneClips = try project.continuousSceneMembers(for: c)
      let sceneKey = sceneClips.isEmpty ? "" : continuousSceneDependencyKey(for: c)
      let sceneReport = try prepared?["scene"].map { try ContinuousSceneRenderReport.decode($0) }
      if !sceneClips.isEmpty && sceneReport == nil {
        throw StudioError.invalid("Prepare the complete continuous scene before rendering.")
      }
      // Retried scenes share immutable sampling checkpoints, never candidate movies.
      let renderDestination = sceneClips.isEmpty ? destination
        : destination.appendingPathComponent(requestID.uuidString)
      let r = try await bridge.invoke(
        "render", runtime: settings, payload: ["recipePath": path], output: renderDestination)
      guard let video = r["video"] as? String else {
        throw StudioError.invalid("Renderer did not return a movie.")
      }
      let info = try await bridge.invoke("inspect", runtime: settings, payload: ["path": video])
      if !sceneClips.isEmpty {
        try receiveContinuousScene(result: r, media: info, prepared: sceneReport!,
          clips: sceneClips, requestKey: sceneKey, projectID: projectID, session: session,
          recipePath: path, generation: generationSettings, resolvedFingerprint: resolvedFingerprint)
        return
      }
      if r["scene"] != nil {
        throw StudioError.invalid("The renderer returned an unexpected scene. Movie saved at \(video)")
      }
      var renderedDuration = info["duration"] as? Double ?? c.duration
      var renderedStart = 0.0
      if let start = r["usable_source_in"] as? Double, let duration = r["usable_duration"] as? Double {
        renderedStart = start
        renderedDuration = duration
      } else if !c.extensionSource.isEmpty {
        let source = try await bridge.invoke(
          "inspect", runtime: settings, payload: ["path": c.extensionSource])
        let sourceDuration = source["duration"] as? Double ?? 0
        renderedDuration -= sourceDuration
        if c.extensionDirection == "after" { renderedStart = sourceDuration }
      }
      let continuationArtifact = try (r["continuation_artifact"] as? [String: Any]).map {
        try JSONDecoder().decode(ContinuationArtifact.self, from: JSONSerialization.data(withJSONObject: $0))
      }
      guard renderedDuration.isFinite, renderedDuration > 0, renderedStart.isFinite, renderedStart >= 0 else {
        throw StudioError.invalid("The extension returned no new frames.")
      }
      let metadata = r["metadata"] as? [String: Any] ?? [:]
      let generationConfig = metadata["generation"] as? [String: Any] ?? [:]
      let preserveEditorialDuration = (prepared?["preserveEditorialDuration"] as? Bool)
        ?? ([Engine.ltx23, .ltx25].contains(c.engine) && c.extensionSource.isEmpty
          && c.extensionDirection.isEmpty && c.continuityMode != "motion"
          && generationConfig["duration_mode"] as? String != "automatic")
      let measuredFPS = info["fps"] as? Double ?? prepared?["nativeFPS"] as? Double ?? 24
      let tolerance = 0.5 / (measuredFPS.isFinite && measuredFPS > 0 ? measuredFPS : 24)
      let mediaDuration = info["duration"] as? Double ?? renderedStart + renderedDuration
      let tooShort = preserveEditorialDuration && (
        renderedDuration + tolerance < c.duration || renderedStart + c.duration > mediaDuration + tolerance)
      guard documentSessionID == session, project.id == projectID,
        let current = project.clips.first(where: { $0.id == c.id }) else {
        throw StudioError.invalid("The destination project or clip changed. The completed video is saved at \(video)")
      }
      let stillCurrent = current == c && signature(for: current) == submittedSignature
      var finishedClip = c
      finishedClip.duration = preserveEditorialDuration ? c.duration : min(c.duration, renderedDuration)
      let finishedSignature = signature(for: finishedClip)
      change { p in
        guard let i = p.clips.firstIndex(where: { $0.id == c.id }) else { return }
        p.clips[i].versions.append(
          RenderVersion(path: video, seed: c.seed, prompt: prompt, recipePath: path,
                        stats: RenderStats(result: r), generationSettings: generationSettings,
                        resolvedFingerprint: resolvedFingerprint, usableSourceIn: renderedStart,
                        usableDuration: renderedDuration, continuationArtifact: continuationArtifact))
        if stillCurrent && !tooShort {
          p.clips[i].sourcePath = video
          p.clips[i].sourceIn = renderedStart
          p.clips[i].duration = finishedClip.duration
          p.clips[i].renderedSignature = finishedSignature
        }
        var asset = MediaAsset(
          name: c.name + " render", kind: .video, path: video, scope: .clip, owner: c.id)
        asset.duration = preserveEditorialDuration ? mediaDuration : min(c.duration, renderedDuration)
        p.assets.append(asset)
      }
      if tooShort {
        throw StudioError.invalid("Generated take is shorter than the planned shot. Timeline timing was preserved; the take is saved in clip versions and at \(video).")
      }
      if stillCurrent && selectedClipID == c.id {
        if finishedClip.duration != c.duration, let accepted = selectedClip {
          // Exact frame durations can replace the editor's rounded seconds. Refresh
          // the resolved input key before recording the completed take's signature.
          let acceptedKey = generationRequestKey(for: accepted)
          await describeGeneration()
          guard documentSessionID == session, project.id == projectID else { return }
          if let index = project.clips.firstIndex(where: { $0.id == c.id }),
            project.clips[index] == accepted,
            generationRequestKey(for: project.clips[index]) == acceptedKey,
            generationDescriptions[c.id]?["studioInput"] as? String == acceptedKey {
            project.clips[index].renderedSignature = signature(for: project.clips[index])
            changed()
          }
        }
        guard selectedClipID == c.id else { return }
        showPrompt = false
        refreshPreview()
      }
      notice = stillCurrent ? "Render complete. Added to clip versions and Clip Assets."
        : "Render saved as a version. The clip changed during generation; prepare its new settings."
    } catch { self.error = error.localizedDescription }
  }
  func extend(_ direction: String) {
    guard let old = selectedClip, !old.sourcePath.isEmpty else {
      error = "Render or import a movie before extending it."
      return
    }
    var c = Clip(
      name: old.name + " · extension", engine: old.engine == .movie ? .ltx25 : old.engine)
    c.prompt = old.prompt
    c.generationSelection = GenerationSelection(task: "extension")
    c.extensionDirection = direction
    c.extensionSource = old.sourcePath
    c.extensionClipID = old.id
    c.generationWidth = old.generationWidth
    c.generationHeight = old.generationHeight
    change { p in
      let i = p.clips.firstIndex(where: { $0.id == old.id })!
      p.clips.insert(c, at: direction == "before" ? i : i + 1)
    }
    select(c.id)
    showPrompt = true
  }
  func exportMovie() {
    let panel = NSSavePanel()
    let f = project.settings.format
    panel.nameFieldStringValue =
      project.name + (f == .pngSequence ? "-frames" : f == .mp4 ? ".mp4" : ".mov")
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await export(to: url) }
  }
  func export(to url: URL) async {
    do {
      _ = try await bridge.invoke("export", runtime: runtime, payload: try payload(), output: url)
      notice = "Exported \(url.lastPathComponent)"
      NSWorkspace.shared.activateFileViewerSelecting([url])
    } catch { self.error = error.localizedDescription }
  }
}
