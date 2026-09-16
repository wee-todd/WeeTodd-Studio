import XCTest
import SQLite3
@testable import StudioCore

final class WorkflowModelTurnTests: XCTestCase {
  func testLegacyTurnsExposeFullMessagesAndMarkMissingFields() throws {
    let data = Data(#"{"status":"completed","totalSeconds":2,"definition":{"steps":[]},"steps":{"review":{"status":"completed","calls":[{"system":"Full instructions","prompt":"Full input","text":"Full response","seconds":1}],"items":{"dog":{"calls":[{"text":"Retained response only"}]}}}}}"#.utf8)
    let snapshot = try JSONDecoder().decode(WorkflowHistorySnapshot.self, from: data)
    let turns = try XCTUnwrap(snapshot.steps["review"]?.legacyTurns)
    XCTAssertEqual(turns.count, 2)
    XCTAssertEqual(turns[0].system, "Full instructions")
    XCTAssertEqual(turns[0].prompt, "Full input")
    XCTAssertEqual(turns[0].response, "Full response")
    XCTAssertNil(turns[1].system)
    XCTAssertEqual(turns[1].response, "Retained response only")
    XCTAssertNotEqual(turns[0].id, turns[1].id)
  }

  func testArchivedTurnsAreNotDuplicatedFromTheResumeCache() throws {
    let data = Data(#"{"status":"completed","totalSeconds":2,"definition":{"steps":[]},"steps":{"review":{"status":"completed","calls":[{"turnID":"archived","system":"instructions","text":"response"}]}}}"#.utf8)
    let snapshot = try JSONDecoder().decode(WorkflowHistorySnapshot.self, from: data)
    XCTAssertEqual(snapshot.steps["review"]?.savedResponses, 1)
    XCTAssertTrue(try XCTUnwrap(snapshot.steps["review"]?.legacyTurns).isEmpty)
  }
  func testArchiveListsAndLoadsFullMessagesReadOnly() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("model-turns.sqlite")
    var db: OpaquePointer?
    XCTAssertEqual(sqlite3_open(file.path, &db), SQLITE_OK)
    defer { sqlite3_close(db) }
    let sql = """
    PRAGMA application_id=1465144369;
    CREATE TABLE turns(id TEXT, step_id TEXT, status TEXT, kind TEXT, label TEXT, seconds REAL, revision INTEGER, detail TEXT);
    INSERT INTO turns VALUES('one','review','returned','model','Read dog',1.25,2,
      '{"id":"one","label":"Read dog","kind":"model","status":"returned","system":"All instructions","prompt":"All input","response":"All output"}');
    """
    XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK)
    let page = try WorkflowTurnArchive.list(directory: directory.path, stepID: "review")
    XCTAssertEqual(page.total, 1)
    XCTAssertEqual(page.rows.first?.revision, 2)
    let turn = try XCTUnwrap(WorkflowTurnArchive.read(directory: directory.path, id: "one"))
    XCTAssertEqual(turn.system, "All instructions")
    XCTAssertEqual(turn.response, "All output")
    XCTAssertEqual(try WorkflowTurnArchive.list(directory: directory.path, stepID: "other").total, 0)
  }

}
