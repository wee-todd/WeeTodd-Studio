import Darwin
import Foundation

/// A bounded input mailbox. Cancellation bypasses the serial work queue so it
/// can be observed while the executor is inside a model operation.
final class TextSessionInbox {
  private let condition = NSCondition()
  private var queue = [[String: Any]]()
  private var cancelled = Set<String>()
  private var sessionID: String?
  private var eof = false
  private var failure: String?
  private var finished = false

  func append(_ value: [String: Any]) throws {
    condition.lock(); defer { condition.unlock() }
    guard let version = value["version"] as? Int, version == 1,
      let session = value["sessionID"] as? String, !session.isEmpty, session.utf8.count <= 256,
      let item = value["itemID"] as? String, item.utf8.count <= 256,
      let type = value["type"] as? String,
      ["hello", "open", "generate", "cancel", "unload", "close"].contains(type),
      sessionID == nil || sessionID == session else {
      throw LocalTextError("text_session_record_invalid")
    }
    sessionID = session
    if type == "cancel" {
      guard !item.isEmpty, cancelled.count < 1024 else {
        throw LocalTextError("text_session_cancel_invalid")
      }
      cancelled.insert(item)
    } else {
      guard queue.count < 32 else { throw LocalTextError("text_session_queue_full") }
      queue.append(value)
    }
    condition.broadcast()
  }

  func next() throws -> [String: Any]? {
    condition.lock(); defer { condition.unlock() }
    while queue.isEmpty && !eof && failure == nil { condition.wait() }
    if let failure { throw LocalTextError(failure) }
    if eof { return nil }
    return queue.removeFirst()
  }

  func isCancelled(_ item: String) -> Bool {
    condition.lock(); defer { condition.unlock() }
    return eof || failure != nil || cancelled.contains(item)
  }

  func endOfInput() {
    condition.lock(); defer { condition.unlock() }
    eof = true; condition.broadcast()
  }

  func fail(_ code: String) {
    condition.lock(); defer { condition.unlock() }
    failure = code; condition.broadcast()
  }

  func finish() {
    condition.lock(); defer { condition.unlock() }
    finished = true; condition.broadcast()
  }

  var hasFinished: Bool {
    condition.lock(); defer { condition.unlock() }
    return finished
  }

  var identity: String {
    condition.lock(); defer { condition.unlock() }
    return sessionID ?? "invalid-session"
  }
}

public enum TextSessionServer {
  /// Private JSONL v1 transport; the inference owner is supplied by the CLI.
  /// Only the executor touches it. The reader/watchdog may only signal cancel.
  public static func run(
    input: FileHandle,
    emit: @escaping ([String: Any]) throws -> Void,
    generate: ([String: Any], ([String: Any]) -> Void, () -> Bool) throws -> [String: Any],
    unload: () -> Void
  ) throws {
    let inbox = TextSessionInbox()
    do {
      try serve(inbox: inbox, input: input, emit: emit, generate: generate, unload: unload)
    } catch {
      try? emit(["type": "error", "version": 1, "sessionID": inbox.identity, "itemID": "",
        "code": (error as? LocalTextError)?.code ?? "text_session_failed"])
      throw error
    }
  }

  private static func serve(
    inbox: TextSessionInbox, input: FileHandle, emit: @escaping ([String: Any]) throws -> Void,
    generate: ([String: Any], ([String: Any]) -> Void, () -> Bool) throws -> [String: Any],
    unload: () -> Void
  ) throws {
    let parent = getppid()
    let started = ProcessInfo.processInfo.systemUptime
    func forceExitIfStuck() {
      DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
        if !inbox.hasFinished { _exit(1) }
      }
    }
    let watchdog = DispatchSource.makeTimerSource(queue: .global())
    watchdog.schedule(deadline: .now() + 1, repeating: 1)
    watchdog.setEventHandler {
      if getppid() != parent || ProcessInfo.processInfo.systemUptime - started >= 3600 {
        inbox.endOfInput(); forceExitIfStuck()
      }
    }
    watchdog.resume()
    defer { unload(); inbox.finish(); watchdog.cancel() }
    DispatchQueue.global().async {
      var buffer = Data()
      var bytes = [UInt8](repeating: 0, count: 65536)
      do {
        while !inbox.hasFinished {
          // Foundation's bounded read may wait to fill the requested count on
          // a pipe. A live JSONL session must accept the currently available
          // bytes without waiting for EOF or a 64 KiB record.
          let count = Darwin.read(input.fileDescriptor, &bytes, bytes.count)
          if count < 0 && errno == EINTR { continue }
          guard count >= 0 else { throw LocalTextError("text_session_input_failed") }
          if count == 0 {
            inbox.endOfInput(); forceExitIfStuck(); return
          }
          buffer.append(contentsOf: bytes[..<count])
          while let end = buffer.firstIndex(of: 10) {
            let length = buffer.distance(from: buffer.startIndex, to: end)
            guard length <= 1024 * 1024,
              let record = try JSONSerialization.jsonObject(with: buffer[..<end]) as? [String: Any] else {
              throw LocalTextError("text_session_record_invalid")
            }
            buffer.removeSubrange(...end)
            try inbox.append(record)
          }
          guard buffer.count <= 1024 * 1024 else {
            throw LocalTextError("text_session_record_too_large")
          }
        }
      } catch {
        inbox.fail((error as? LocalTextError)?.code ?? "text_session_record_invalid")
        forceExitIfStuck()
      }
    }

    guard let hello = try inbox.next(), hello["type"] as? String == "hello",
      let session = hello["sessionID"] as? String, hello["itemID"] as? String == "" else {
      throw LocalTextError("text_session_hello_required")
    }
    func event(_ type: String, item: String = "", value: [String: Any] = [:]) throws {
      var record: [String: Any] = ["type": type, "version": 1, "sessionID": session, "itemID": item]
      record.merge(value) { _, new in new }
      try emit(record)
    }
    try event("hello", value: ["capabilities": ["maxActiveItems": 1, "maxQueuedItems": 32,
      "residentModel": true, "concurrentInference": false]])
    var model: String?
    var submitted = Set<String>(), completed = Set<String>()
    while let record = try inbox.next() {
      let type = record["type"] as! String, item = record["itemID"] as! String
      switch type {
      case "open":
        guard item.isEmpty, model == nil, let path = record["modelPath"] as? String,
          path.hasPrefix("/"), path.utf8.count <= 4096 else {
          throw LocalTextError("text_session_model_invalid")
        }
        model = path
        try event("opened", value: ["value": ["resident": false]])
      case "generate":
        guard !item.isEmpty, !submitted.contains(item), submitted.count < 1024,
          let request = record["value"] as? [String: Any], request["requestID"] as? String == item,
          let model, request["modelPath"] as? String == model,
          let dependencies = record["dependencies"] as? [String], dependencies.count <= 32 else {
          throw LocalTextError("text_session_item_invalid")
        }
        submitted.insert(item)
        guard dependencies.allSatisfy({ completed.contains($0) }) else {
          try event("error", item: item, value: ["code": "text_session_dependency_failed"])
          continue
        }
        if inbox.isCancelled(item) {
          try event("cancelled", item: item, value: ["code": "text_cancelled"])
          continue
        }
        try event("progress", item: item, value: ["value": ["stage": "queued"]])
        do {
          let value = try generate(request, { progress in
            try? event("progress", item: item, value: ["value": progress])
          }, { inbox.isCancelled(item) })
          if inbox.isCancelled(item) { throw LocalTextError("text_cancelled") }
          try event("result", item: item, value: ["value": value])
          completed.insert(item)
        } catch {
          unload()
          let code = (error as? LocalTextError)?.code ?? "text_session_failed"
          try event(code == "text_cancelled" ? "cancelled" : "error", item: item, value: ["code": code])
        }
      case "unload":
        guard item.isEmpty else { throw LocalTextError("text_session_record_invalid") }
        unload()
        try event("unloaded", value: ["value": ["resident": false]])
      case "close":
        guard item.isEmpty else { throw LocalTextError("text_session_record_invalid") }
        unload()
        try event("closed", value: ["value": ["resident": false]])
        return
      default: throw LocalTextError("text_session_record_invalid")
      }
    }
  }
}
