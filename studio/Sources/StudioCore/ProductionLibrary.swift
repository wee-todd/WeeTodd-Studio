import Foundation
import SQLite3

/// Immutable, portable metadata snapshot. Image files remain external references.
public struct ProductionLibraryPackage: Codable, Equatable, Identifiable {
  public var format = "weetodd-production-library-v1"
  public var rootID: UUID
  public var version: Int
  public var subjects: [PlanningSubject]
  public var assets: [MediaAsset]
  public var id: UUID { rootID }
  public var root: PlanningSubject? { subjects.first { $0.id == rootID } }
  public var missingMediaPaths: [String] {
    Array(Set(assets.filter { !FileManager.default.fileExists(atPath: $0.path) }.map(\.path))).sorted()
  }
}

/// Small app-local catalog. All snapshots are immutable; each publication is one atomic transaction.
public final class ProductionLibrary {
  private var database: OpaquePointer?
  private let lock = NSRecursiveLock()
  private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
  public init(url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard sqlite3_open_v2(url.path, &database, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
      if let database { sqlite3_close(database) }; database = nil
      throw StudioError.invalid("Could not open the production library.")
    }
    sqlite3_busy_timeout(database, 3000)
    do {
      try execute("PRAGMA foreign_keys=ON")
      let schema = try rows("PRAGMA user_version").first?.first ?? "0"
      guard schema == "0" || schema == "1" else { throw StudioError.invalid("This library was created by a newer Studio version.") }
      try execute("CREATE TABLE IF NOT EXISTS packages (root TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, fingerprint TEXT NOT NULL, search_text TEXT NOT NULL, PRIMARY KEY(root, version))")
      try execute("PRAGMA user_version=1")
    } catch { sqlite3_close(database); database = nil; throw error }
  }
  deinit { sqlite3_close(database) }

  public func publish(rootID: UUID, planning: ProjectPlanning, assets: [MediaAsset]) throws -> ProductionLibraryPackage {
    lock.lock(); defer { lock.unlock() }
    var objects = try planning.resolvedObjects([rootID], includeSets: planning.subjects.first(where: { $0.id == rootID })?.kind == .environment)
    for i in objects.indices { objects[i].libraryOrigin = nil }
    let references = Set(objects.flatMap(\.referenceAssetIDs))
    let media = assets.filter { references.contains($0.id) }.sorted { $0.id.uuidString < $1.id.uuidString }
    guard Set(media.map(\.id)) == references else { throw StudioError.invalid("Resolve missing reference assets before publishing to the library.") }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    var value = ProductionLibraryPackage(rootID: rootID, version: 0, subjects: objects, assets: media)
    let fingerprint = planningDigest(value)
    try execute("BEGIN IMMEDIATE")
    do {
      let latest = try rows("SELECT version, fingerprint FROM packages WHERE root=? ORDER BY version DESC LIMIT 1", [rootID.uuidString]).first
      if latest?.last == fingerprint, let version = latest?.first.flatMap(Int.init) {
        let previous = try package(rootID: rootID, version: version)
        try execute("COMMIT"); return previous
      }
      value.version = (latest?.first.flatMap(Int.init) ?? 0) + 1
      let data = try encoder.encode(value)
      guard data.count <= 2_000_000, objects.count <= 2000 else { throw StudioError.invalid("A library package is limited to 2,000 objects and 2 MB of metadata.") }
      let text = String(decoding: data, as: UTF8.self)
      let search = objects.flatMap { [$0.name, $0.kind.rawValue] + ($0.tags ?? []) }.joined(separator: " ").lowercased()
      try execute("INSERT INTO packages VALUES (?, ?, ?, ?, ?)", [rootID.uuidString, String(value.version), text, fingerprint, search])
      try execute("COMMIT"); return value
    } catch { try? execute("ROLLBACK"); throw error }
  }
  public func search(_ query: String) throws -> [ProductionLibraryPackage] {
    lock.lock(); defer { lock.unlock() }
    let tokens = query.lowercased().split(whereSeparator: { $0.isWhitespace }).map(String.init)
    // Literal substring filtering avoids query-language surprises and remains adequate for a small metadata catalog.
    let predicates = tokens.map { _ in " AND instr(p.search_text, ?) > 0" }.joined()
    let data = try rows("SELECT p.payload FROM packages p WHERE p.version=(SELECT MAX(q.version) FROM packages q WHERE q.root=p.root)" + predicates + " ORDER BY p.search_text LIMIT 100", tokens)
    return try data.map { row in
      try JSONDecoder().decode(ProductionLibraryPackage.self, from: Data(row[0].utf8))
    }
  }
  /// Read bounded object metadata without loading media or entire package payloads into Swift.
  public func candidates(matching text: String, limit: Int = 64) throws -> [LibraryObjectCandidate] {
    lock.lock(); defer { lock.unlock() }
    let query = String(text.lowercased().prefix(64_000))
    guard !query.isEmpty else { return [] }
    let words = Array(Set(query.split(whereSeparator: { !$0.isLetter && !$0.isNumber }).filter { $0.count >= 3 }.map(String.init))).sorted().prefix(512)
    let wordJSON = String(decoding: try JSONEncoder().encode(Array(words)), as: UTF8.self)
    let values = try rows("""
      SELECT o.value, p.root, p.version FROM packages p, json_each(p.payload,'$.subjects') o
      WHERE p.version=(SELECT MAX(q.version) FROM packages q WHERE q.root=p.root)
      AND (length(json_extract(o.value,'$.name')) > 0 AND instr(?,lower(json_extract(o.value,'$.name'))) > 0
        OR EXISTS (SELECT 1 FROM json_each(o.value,'$.tags') t WHERE length(t.value)>2 AND instr(?,lower(t.value))>0)
        OR EXISTS (SELECT 1 FROM json_each(?) w WHERE instr(lower(coalesce(json_extract(o.value,'$.aliases'),'')),w.value)>0))
      ORDER BY CASE WHEN json_extract(o.value,'$.id')=p.root THEN 0 ELSE 1 END, p.root, p.version DESC LIMIT 256
      """, [query, query, wordJSON])
    var chosen: [String: LibraryObjectCandidate] = [:]
    for row in values {
      guard let root = UUID(uuidString: row[1]), let version = Int(row[2]) else { continue }
      let subject = try JSONDecoder().decode(PlanningSubject.self, from: Data(row[0].utf8))
      guard !subject.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !subject.details.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
      if chosen[subject.id.uuidString] == nil {
        chosen[subject.id.uuidString] = LibraryObjectCandidate(subject: subject, packageID: root, version: version, scope: "global")
      }
    }
    return Array(chosen.values.sorted { $0.name == $1.name ? $0.id < $1.id : $0.name < $1.name }.prefix(max(0, min(64, limit))))
  }
  public func package(rootID: UUID, version: Int) throws -> ProductionLibraryPackage {
    lock.lock(); defer { lock.unlock() }
    guard let text = try rows("SELECT payload FROM packages WHERE root=? AND version=?", [rootID.uuidString, String(version)]).first?.first else {
      throw StudioError.invalid("This library version is unavailable.")
    }
    return try JSONDecoder().decode(ProductionLibraryPackage.self, from: Data(text.utf8))
  }
  private func execute(_ sql: String, _ values: [String] = []) throws { _ = try rows(sql, values) }
  private func rows(_ sql: String, _ values: [String] = []) throws -> [[String]] {
    var statement: OpaquePointer?
    guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK else { throw failure() }
    defer { sqlite3_finalize(statement) }
    for (i, value) in values.enumerated() {
      guard sqlite3_bind_text(statement, Int32(i + 1), value, -1, transient) == SQLITE_OK else { throw failure() }
    }
    var output: [[String]] = []
    while true {
      let status = sqlite3_step(statement)
      if status == SQLITE_DONE { return output }
      guard status == SQLITE_ROW else { throw failure() }
      output.append((0..<sqlite3_column_count(statement)).map {
        guard let text = sqlite3_column_text(statement, $0) else { return "" }
        return String(cString: text)
      })
    }
  }
  private func failure() -> StudioError { .invalid("Production library: " + String(cString: sqlite3_errmsg(database))) }
}

public extension StudioProject {
  /// Pinned definitions are project-owned snapshots. Conflicting existing definitions are never overwritten.
  mutating func importLibraryPackage(_ package: ProductionLibraryPackage) throws {
    guard package.format == "weetodd-production-library-v1", package.version > 0,
          package.subjects.count <= 2000, (try JSONEncoder().encode(package)).count <= 2_000_000, package.root != nil else { throw StudioError.invalid("Invalid production library package.") }
    var source = ProjectPlanning(); source.subjects = package.subjects
    _ = try source.resolvedObjects(package.subjects.map(\.id))
    var next = self; var plan = next.planning ?? ProjectPlanning()
    for object in package.subjects {
      if let old = plan.subjects.first(where: { $0.id == object.id }) {
        guard old.revision == object.revision, old.referenceAssetIDs == object.referenceAssetIDs else {
          throw StudioError.invalid("“\(old.name)” already has a different movie definition. Keep this movie’s version or import the library into a new movie.")
        }
      } else {
        var pinned = object
        pinned.libraryOrigin = LibraryOrigin(packageID: package.rootID, version: package.version, definitionRevision: object.revision)
        plan.subjects.append(pinned)
      }
    }
    for media in package.assets {
      if let old = next.assets.first(where: { $0.id == media.id }) {
        guard old.path == media.path, old.kind == media.kind else { throw StudioError.invalid("A reference asset ID conflicts with this movie.") }
      } else {
        var linked = media; linked.scope = .project; linked.owner = nil
        next.assets.append(linked)
      }
    }
    let ids = Set(next.assets.map(\.id))
    guard package.subjects.flatMap(\.referenceAssetIDs).allSatisfy({ ids.contains($0) }) else { throw StudioError.invalid("The package has missing reference metadata.") }
    next.planning = plan; self = next
  }
}
