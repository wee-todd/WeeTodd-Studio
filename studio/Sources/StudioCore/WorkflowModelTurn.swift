import Foundation
import SQLite3

public struct WorkflowModelTurn: Decodable, Identifiable {
  public var id: String
  public var label: String
  public var kind: String
  public var status: String
  public var system: String?
  public var prompt: String?
  public var response: String?
  public var images: [String]?
  public var imageBindings: [String: String]?
  public var startedAt: Double?
  public var seconds: Double?
  public var requestKey: String?
  public var error: String?
  public var validation: JSONValue?
  public var model: JSONValue?
  public var runtime: JSONValue?
  public var settings: JSONValue?
  public var reason: String?
  public var operation: String?
  public var truncated: Bool?

  public static func legacy(_ value: [String: JSONValue], path: String) -> Self? {
    guard case .string(let response) = value["text"], value["turnID"] == nil else { return nil }
    var data = value
    data["id"] = .string("legacy:" + path)
    data["label"] = .string(path)
    data["kind"] = .string("legacy")
    data["status"] = .string("Saved response")
    data["response"] = .string(response)
    data["requestKey"] = value["key"]
    guard let encoded = try? JSONEncoder().encode(data) else { return nil }
    return try? JSONDecoder().decode(Self.self, from: encoded)
  }

  public var details: String {
    var fields: [String: JSONValue] = [:]
    fields["model"] = model; fields["runtime"] = runtime; fields["settings"] = settings
    fields["validation"] = validation
    if let reason { fields["reason"] = .string(reason) }
    if let operation { fields["operation"] = .string(operation) }
    if let requestKey { fields["requestFingerprint"] = .string(requestKey) }
    if let truncated { fields["responseTruncatedByModel"] = .boolean(truncated) }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return (try? encoder.encode(fields)).flatMap { String(data: $0, encoding: .utf8) } ?? "Unavailable"
  }
}

/// Opens only the fixed local transcript database, read-only. Messages load one turn at a time.
public enum WorkflowTurnArchive {
  public struct Row: Identifiable {
    public var id: String
    public var ordinal: Int
    public var label: String
    public var status: String
    public var kind: String
    public var seconds: Double?
    public var revision: Int
  }
  public struct Page {
    public var rows: [Row]
    public var total: Int
  }
  private static func open(_ directory: String) throws -> OpaquePointer? {
    let url = URL(fileURLWithPath: directory).appendingPathComponent("model-turns.sqlite")
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    let info = try url.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey])
    guard info.isSymbolicLink != true, (info.fileSize ?? Int.max) <= 16 * 1024 * 1024 else {
      throw StudioError.invalid("Model-turn database exceeds its inspection limit or is a symlink.")
    }
    var db: OpaquePointer?
    guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK, let db else {
      if let db { sqlite3_close(db) }
      throw StudioError.invalid("Could not open model-turn history.")
    }
    sqlite3_busy_timeout(db, 200)
    var statement: OpaquePointer?
    guard sqlite3_prepare_v2(db, "PRAGMA application_id", -1, &statement, nil) == SQLITE_OK else {
      sqlite3_close(db); throw StudioError.invalid("Invalid model-turn database.")
    }
    let valid = sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_int(statement, 0) == 0x57545431
    sqlite3_finalize(statement)
    guard valid else { sqlite3_close(db); throw StudioError.invalid("Unrecognized model-turn database.") }
    return db
  }
  private static func statement(_ db: OpaquePointer, sql: String, value: String) throws -> OpaquePointer {
    var statement: OpaquePointer?
    guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
      throw StudioError.invalid("Could not query model-turn history.")
    }
    let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    guard sqlite3_bind_text(statement, 1, value, -1, transient) == SQLITE_OK else {
      sqlite3_finalize(statement); throw StudioError.invalid("Could not select model turn.")
    }
    return statement
  }
  private static func string(_ s: OpaquePointer, _ index: Int32) -> String {
    sqlite3_column_text(s, index).map { String(cString: $0) } ?? ""
  }
  public static func list(directory: String, stepID: String, limit: Int = 100) throws -> Page {
    guard let db = try open(directory) else { return Page(rows: [], total: 0) }
    defer { sqlite3_close(db) }
    let s = try statement(db, sql: "SELECT rowid,id,label,status,kind,seconds,revision FROM turns WHERE step_id=? ORDER BY rowid LIMIT ?", value: stepID)
    defer { sqlite3_finalize(s) }
    sqlite3_bind_int(s, 2, Int32(min(max(1, limit), 10000)))
    var rows: [Row] = []
    var code = sqlite3_step(s)
    while code == SQLITE_ROW {
      rows.append(Row(id: string(s, 1), ordinal: Int(sqlite3_column_int64(s, 0)), label: string(s, 2),
                      status: string(s, 3), kind: string(s, 4),
                      seconds: sqlite3_column_type(s, 5) == SQLITE_NULL ? nil : sqlite3_column_double(s, 5),
                      revision: Int(sqlite3_column_int(s, 6))))
      code = sqlite3_step(s)
    }
    guard code == SQLITE_DONE else { throw StudioError.invalid("Model-turn history is busy; retry shortly.") }
    let count = try statement(db, sql: "SELECT count(*) FROM turns WHERE step_id=?", value: stepID)
    defer { sqlite3_finalize(count) }
    guard sqlite3_step(count) == SQLITE_ROW else { throw StudioError.invalid("Could not count model turns.") }
    return Page(rows: rows, total: Int(sqlite3_column_int(count, 0)))
  }
  public static func read(directory: String, id: String) throws -> WorkflowModelTurn? {
    guard let db = try open(directory) else { return nil }
    defer { sqlite3_close(db) }
    let s = try statement(db, sql: "SELECT detail FROM turns WHERE id=?", value: id)
    defer { sqlite3_finalize(s) }
    let code = sqlite3_step(s)
    if code == SQLITE_DONE { return nil }
    guard code == SQLITE_ROW else { throw StudioError.invalid("Model-turn history is busy; retry shortly.") }
    return try JSONDecoder().decode(WorkflowModelTurn.self, from: Data(string(s, 0).utf8))
  }
}
