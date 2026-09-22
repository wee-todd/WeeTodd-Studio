import Foundation
import XCTest
@testable import DrawThingsTransport

final class TextSessionProtocolTests: XCTestCase {
  private func record(_ type: String, _ item: String = "", extra: [String: Any] = [:]) -> [String: Any] {
    ["type": type, "version": 1, "sessionID": "session", "itemID": item]
      .merging(extra) { _, new in new }
  }

  func testCancelBypassesQueuedWorkAndQueueIsBounded() throws {
    let inbox = TextSessionInbox()
    for number in 0..<32 { try inbox.append(record("generate", "\(number)")) }
    XCTAssertThrowsError(try inbox.append(record("generate", "overflow")))
    try inbox.append(record("cancel", "0"))
    XCTAssertTrue(inbox.isCancelled("0"))
    XCTAssertFalse(inbox.isCancelled("1"))
  }

  func testEOFAndFatalInputCancelExecutingWork() throws {
    let eof = TextSessionInbox()
    eof.endOfInput()
    XCTAssertTrue(eof.isCancelled("active"))
    let invalid = TextSessionInbox()
    invalid.fail("text_session_record_invalid")
    XCTAssertTrue(invalid.isCancelled("active"))
    XCTAssertThrowsError(try invalid.next())
  }

  func testSessionBindsModelAndRespectsDependencies() throws {
    let pipe = Pipe()
    let messages = [
      record("hello"), record("open", extra: ["modelPath": "/qwen_3.5_4b_i8x.ckpt"]),
      record("generate", "first", extra: ["value": ["requestID": "first", "modelPath": "/qwen_3.5_4b_i8x.ckpt"], "dependencies": []]),
      record("generate", "second", extra: ["value": ["requestID": "second", "modelPath": "/qwen_3.5_4b_i8x.ckpt"], "dependencies": ["first"]]),
      record("close"),
    ]
    for message in messages {
      var data = try JSONSerialization.data(withJSONObject: message); data.append(10)
      try pipe.fileHandleForWriting.write(contentsOf: data)
    }
    var events = [[String: Any]](), unloads = 0
    try TextSessionServer.run(input: pipe.fileHandleForReading, emit: { events.append($0) },
      generate: { value, progress, _ in
        progress(["stage": "decoding"])
        return ["text": value["requestID"]!]
      }, unload: { unloads += 1 })
    try pipe.fileHandleForWriting.close()
    XCTAssertEqual(events.filter { $0["type"] as? String == "result" }.map { $0["itemID"] as? String }, ["first", "second"])
    XCTAssertEqual(events.last?["type"] as? String, "closed")
    XCTAssertGreaterThanOrEqual(unloads, 1)
  }

  func testRunningItemReceivesCancellationWithoutWaitingForGeneration() throws {
    let pipe = Pipe()
    func write(_ value: [String: Any]) throws {
      var data = try JSONSerialization.data(withJSONObject: value); data.append(10)
      try pipe.fileHandleForWriting.write(contentsOf: data)
    }
    try write(record("hello"))
    try write(record("open", extra: ["modelPath": "/qwen_3.5_4b_i8x.ckpt"]))
    try write(record("generate", "first", extra: ["value": ["requestID": "first", "modelPath": "/qwen_3.5_4b_i8x.ckpt"], "dependencies": []]))
    var terminal = [String]()
    try TextSessionServer.run(input: pipe.fileHandleForReading, emit: { event in
      if ["result", "error", "cancelled"].contains(event["type"] as? String ?? "") {
        terminal.append(event["type"] as! String)
      }
    }, generate: { _, _, cancelled in
      try write(self.record("cancel", "first"))
      try write(self.record("close"))
      let deadline = Date().addingTimeInterval(1)
      while !cancelled() && Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
      XCTAssertTrue(cancelled())
      throw LocalTextError("text_cancelled")
    }, unload: {})
    try pipe.fileHandleForWriting.close()
    XCTAssertEqual(terminal, ["cancelled"])
  }
}
