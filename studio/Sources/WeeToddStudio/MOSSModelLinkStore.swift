import Foundation
import StudioCore
import ImageLayout

@MainActor extension StudioStore {
  func loadMOSSModelLinks() {
    let file = dataDirectory.appendingPathComponent("moss-model-links.json")
    guard FileManager.default.fileExists(atPath: file.path) else { return }
    do { mossModelLinks = try MOSSModelLinks.read(from: file) }
    catch {
      let backup = dataDirectory.appendingPathComponent("moss-model-links-unreadable-\(UUID().uuidString).json")
      do { try FileManager.default.moveItem(at: file, to: backup) }
      catch { mossModelLinksWritable = false }
      notice = "MOSS folder preferences could not be read. Existing preferences were preserved; choose model folders again."
    }
  }
  func saveMOSSModelLinks(_ value: MOSSModelLinks) {
    guard value != mossModelLinks else { return }
    mossModelLinks = value
    guard mossModelLinksWritable else { return }
    do { try value.write(to: dataDirectory.appendingPathComponent("moss-model-links.json")); Task { await refreshGenerationLibrary(force: true) } }
    catch { notice = "MOSS folder preferences could not be saved: \(error.localizedDescription)" }
  }
  func rememberMOSSComponents(_ settings: MOSSImageSettings) {
    var links = mossModelLinks; links.remember(MOSSModelQuery(settings)); saveMOSSModelLinks(links)
  }
  func selectMOSSAnalyzer(_ directory: String) {
    guard directory.hasPrefix("/"), !directory.utf8.contains(0) else { return }
    mossAnalyzerSelectionAttempt = UUID()
    var links = mossModelLinks; links.analyzerDirectory = directory; saveMOSSModelLinks(links)
    mossAnalyzerFamily = nil
  }
  @discardableResult
  func completeMOSSAnalyzerSelection(_ directory: String, attempt: UUID, session: UUID, draftKey: String?) -> Bool {
    guard !Task.isCancelled, documentSessionID == session, imageDraft?.storageKey == draftKey,
      mossAnalyzerSelectionAttempt == attempt, directory.hasPrefix("/"), !directory.utf8.contains(0) else { return false }
    selectMOSSAnalyzer(directory); return true
  }
  func selectMOSSPrecision(_ bits: Int?) {
    guard bits == nil || [4, 8].contains(bits!), var draft = imageDraft, draft.executionProvider == .nativeMOSS,
      var settings = draft.mossImage, settings.quantizationBits != bits else { return }
    rememberMOSSComponents(settings)
    settings.quantizationBits = bits; settings.transformerDirectory = ""
    mossModelLinks.applying(to: MOSSModelQuery(settings)).apply(to: &settings)
    draft.mossImage = settings; imageDraft = draft; imageEstimate = nil
  }
  func applyMOSSComponentLinks(_ found: MOSSImageSettings) {
    guard var draft = imageDraft, draft.executionProvider == .nativeMOSS, var settings = draft.mossImage,
      settings.variant == found.variant else { return }
    rememberMOSSComponents(found)
    settings.libraryDirectory = found.libraryDirectory; settings.sharedDirectory = found.sharedDirectory
    if settings.quantizationBits == found.quantizationBits { settings.transformerDirectory = found.transformerDirectory }
    else { settings.transformerDirectory = mossModelLinks.transformerDirectories[MOSSModelQuery(settings).profileKey] ?? "" }
    draft.mossImage = settings; imageDraft = draft; imageEstimate = nil
  }
  func resolveMOSSModelLinks(libraries: [URL]? = nil, legacyAnalyzer: String? = nil) async {
    guard let draft = imageDraft, draft.executionProvider == .nativeMOSS, !operationBusy else { return }
    let query = MOSSModelQuery(draft.mossImage ?? .init()), session = documentSessionID, key = draft.storageKey
    let attempt = UUID(); mossModelDiscoveryAttempt = attempt; mossModelDiscoveryBusy = true
    defer { if mossModelDiscoveryAttempt == attempt { mossModelDiscoveryBusy = false } }
    var remembered = mossModelLinks
    let legacyAnalyzer = legacyAnalyzer ?? (dataDirectory.standardizedFileURL == StudioStore.supportDirectory.standardizedFileURL
      ? UserDefaults.standard.string(forKey: "mossLayoutAnalyzerDirectory") : nil)
    if remembered.analyzerDirectory.isEmpty, let legacyAnalyzer, legacyAnalyzer.hasPrefix("/"), !legacyAnalyzer.utf8.contains(0) {
      remembered.analyzerDirectory = legacyAnalyzer
    }
    let preferredLibrary = nativeUpscalerModelLibrary
    let rememberedSnapshot = remembered
    let found = await Task.detached(priority: .utility) {
      var roots = libraries ?? MOSSModelDiscovery.mountedLibraries()
      if preferredLibrary.hasPrefix("/") { roots.insert(URL(fileURLWithPath: preferredLibrary), at: 0) }
      let links = MOSSModelDiscovery.discover(query: query, remembered: rememberedSnapshot, libraries: roots)
      let selected = links.applying(to: query)
      let model = URL(fileURLWithPath: links.analyzerDirectory)
      let family = !links.analyzerDirectory.isEmpty && MOSSLayoutAnalysisPreparation.compatibleModel(model)
        ? MOSSLayoutAnalysisPreparation.visionModel(model) : nil
      let recommended = MOSSLayoutAnalysisPreparation.discoverModel(libraryDirectory: selected.libraryDirectory,
        sharedDirectory: selected.sharedDirectory)
      let gemma = recommended.flatMap { MOSSLayoutAnalysisPreparation.visionModel(URL(fileURLWithPath: $0)) == .gemma4_12b ? $0 : nil }
      let tokenizer = URL(fileURLWithPath: selected.sharedDirectory).appendingPathComponent("tokenizer")
      let ready = !selected.sharedDirectory.isEmpty && ["tokenizer.json", "tokenizer_config.json"].allSatisfy {
        FileManager.default.fileExists(atPath: tokenizer.appendingPathComponent($0).path)
      }
      return (links: links, family: family, gemma: gemma, tokenizerReady: ready)
    }.value
    guard !Task.isCancelled, mossModelDiscoveryAttempt == attempt, documentSessionID == session,
      var current = imageDraft, current.executionProvider == .nativeMOSS, current.storageKey == key,
      MOSSModelQuery(current.mossImage ?? .init()) == query else { return }
    var settings = current.mossImage ?? .init(); found.links.applying(to: query).apply(to: &settings)
    if settings != current.mossImage { current.mossImage = settings; imageDraft = current; imageEstimate = nil }
    var links = mossModelLinks; links.remember(MOSSModelQuery(settings))
    for (profile, path) in found.links.transformerDirectories where links.transformerDirectories[profile] == nil {
      links.transformerDirectories[profile] = path
    }
    if links.analyzerDirectory.isEmpty { links.analyzerDirectory = found.links.analyzerDirectory }
    saveMOSSModelLinks(links)
    if mossModelLinks.analyzerDirectory == found.links.analyzerDirectory { mossAnalyzerFamily = found.family }
    mossRecommendedGemmaDirectory = found.gemma; mossSharedTokenizerAvailable = found.tokenizerReady
  }
}
