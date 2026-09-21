import Foundation

public struct CharacterDocumentPortableManifest: Codable, Equatable {
  public var format = "weetodd-character-portable-v1"
  public var originalRoot: String
  public init(originalRoot: String) { self.originalRoot = originalRoot }
}

public extension CharacterSheetDocumentStore {
  func export(_ document: CharacterSheetDocument, to destination: URL) throws {
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw StudioError.invalid("Choose an empty destination for the character export.")
    }
    let staging = destination.deletingLastPathComponent()
      .appendingPathComponent(".\(destination.lastPathComponent)-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: staging) }
    let assets = staging.appendingPathComponent("assets")
    try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
    var exported = document
    // Machine-local recovery folders are not completed media and cannot be resumed after export.
    for index in exported.pipeline.stages.indices { exported.pipeline.stages[index].recoveryDirectory = nil }
    var copied: [String: String] = [:]
    var index = 0
    try exported.transformPortablePaths { path in
      let source = URL(fileURLWithPath: path).standardizedFileURL
      guard source.isFile else { throw StudioError.invalid("Relink missing character media before exporting: \(path)") }
      if let existing = copied[source.path] { return existing }
      index += 1
      let extensionPart = source.pathExtension.isEmpty ? "" : "." + source.pathExtension
      let relative = "assets/asset-\(index)\(extensionPart)"
      try FileManager.default.copyItem(at: source, to: staging.appendingPathComponent(relative))
      copied[source.path] = relative
      return relative
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(exported).write(to: staging.appendingPathComponent("character.json"), options: .atomic)
    let inherited = directory(id: document.id).appendingPathComponent("portability.json")
    let manifest = (try? JSONDecoder().decode(CharacterDocumentPortableManifest.self,
      from: Data(contentsOf: inherited))) ?? CharacterDocumentPortableManifest(originalRoot: root.path)
    try encoder.encode(manifest).write(to: staging.appendingPathComponent("portability.json"), options: .atomic)
    try FileManager.default.moveItem(at: staging, to: destination)
  }

  func importDocument(from source: URL) throws -> CharacterSheetDocument {
    let packageRoot = source.resolvingSymlinksInPath().standardizedFileURL
    guard packageRoot.isDirectory else { throw StudioError.invalid("Choose a Character Director export folder.") }
    var document = try load(from: packageRoot.appendingPathComponent("character.json"))
    var resolved: [String: URL] = [:]
    try document.transformPortablePaths { path in
      guard !path.isEmpty, !path.hasPrefix("/"), path.split(separator: "/").first == "assets" else {
        throw StudioError.invalid("The character export contains an unsafe media path.")
      }
      let candidate = packageRoot.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
      guard candidate.isContained(in: packageRoot), candidate.isFile else {
        throw StudioError.invalid("The character export contains missing or unsafe media.")
      }
      resolved[path] = candidate
      return path
    }
    let newID = UUID()
    let target = directory(id: newID)
    let targetAssets = target.appendingPathComponent("assets")
    try FileManager.default.createDirectory(at: targetAssets, withIntermediateDirectories: true)
    do {
      var importedPaths: [String: String] = [:]
      for (relative, sourceURL) in resolved.sorted(by: { $0.key < $1.key }) {
        let destination = target.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        importedPaths[relative] = destination.path
      }
      document.id = newID
      document.draft.destination = .init(scope: .global, projectID: newID)
      document.proposals = document.proposals.map {
        var proposal = $0; proposal.documentID = newID; return proposal
      }
      for index in document.pipeline.stages.indices {
        document.pipeline.stages[index].draft?.destination = .init(scope: .global, projectID: newID)
      }
      try document.transformPortablePaths { relative in
        guard let path = importedPaths[relative] else { throw StudioError.invalid("The character export is incomplete.") }
        return path
      }
      try save(document)
      let sourceManifest = packageRoot.appendingPathComponent("portability.json")
      let manifest = (try? JSONDecoder().decode(CharacterDocumentPortableManifest.self,
        from: Data(contentsOf: sourceManifest))) ?? CharacterDocumentPortableManifest(originalRoot: packageRoot.path)
      let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(manifest).write(to: target.appendingPathComponent("portability.json"), options: .atomic)
      return document
    } catch {
      try? FileManager.default.removeItem(at: target)
      throw error
    }
  }
}

private extension CharacterSheetDocument {
  mutating func transformPortablePaths(_ transform: (String) throws -> String) throws {
    for key in sources.keys.sorted() where !(sources[key] ?? "").isEmpty { sources[key] = try transform(sources[key]!) }
    try draft.transformPortablePaths(transform)
    for index in proposals.indices where !proposals[index].sourcePath.isEmpty {
      proposals[index].sourcePath = try transform(proposals[index].sourcePath)
    }
    for index in candidates.indices {
      if !candidates[index].path.isEmpty { candidates[index].path = try transform(candidates[index].path) }
      if !candidates[index].thumbnail.isEmpty { candidates[index].thumbnail = try transform(candidates[index].thumbnail) }
      try candidates[index].generation?.transformPortablePaths(transform)
    }
    if let path = initialSheetPath, !path.isEmpty { initialSheetPath = try transform(path) }
    if var headReference {
      headReference.sourcePath = try transformIfPresent(headReference.sourcePath, transform)
      headReference.headCropPath = try transformIfPresent(headReference.headCropPath, transform)
      headReference.maskPath = try transformIfPresent(headReference.maskPath, transform)
      headReference.rgbaCutoutPath = try transformIfPresent(headReference.rgbaCutoutPath, transform)
      headReference.whiteMattePath = try transformIfPresent(headReference.whiteMattePath, transform)
      self.headReference = headReference
    }
    for index in pipeline.stages.indices {
      if let path = pipeline.stages[index].outputPath, !path.isEmpty {
        pipeline.stages[index].outputPath = try transform(path)
      }
      try pipeline.stages[index].draft?.transformPortablePaths(transform)
    }
  }
}

private extension DrawThingsImageDraft {
  mutating func transformPortablePaths(_ transform: (String) throws -> String) throws {
    if var canvas, !canvas.path.isEmpty { canvas.path = try transform(canvas.path); self.canvas = canvas }
    for index in moodboard.indices where !moodboard[index].path.isEmpty {
      moodboard[index].path = try transform(moodboard[index].path)
    }
    if var rippleReference {
      rippleReference.sourcePath = try transformIfPresent(rippleReference.sourcePath, transform)
      rippleReference.originalPath = try transformIfPresent(rippleReference.originalPath, transform)
      self.rippleReference = rippleReference
    }
  }
}

private extension ImageGeneration {
  mutating func transformPortablePaths(_ transform: (String) throws -> String) throws {
    if var rippleReference {
      rippleReference.sourcePath = try transformIfPresent(rippleReference.sourcePath, transform)
      rippleReference.originalPath = try transformIfPresent(rippleReference.originalPath, transform)
      self.rippleReference = rippleReference
    }
  }
}

private func transformIfPresent(_ path: String, _ transform: (String) throws -> String) throws -> String {
  path.isEmpty ? path : try transform(path)
}

private extension URL {
  var isFile: Bool {
    var directory: ObjCBool = false
    return FileManager.default.fileExists(atPath: path, isDirectory: &directory) && !directory.boolValue
  }
  var isDirectory: Bool {
    var directory: ObjCBool = false
    return FileManager.default.fileExists(atPath: path, isDirectory: &directory) && directory.boolValue
  }
  func isContained(in root: URL) -> Bool {
    let rootComponents = root.pathComponents
    let components = pathComponents
    return components.count > rootComponents.count && Array(components.prefix(rootComponents.count)) == rootComponents
  }
}
