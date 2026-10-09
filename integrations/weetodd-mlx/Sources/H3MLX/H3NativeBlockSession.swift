import CoreFoundation
import Darwin
import Foundation

/// One owned, serialized native worker. No MLX work runs on the reader thread.
/// The caller owns workspace removal and may remove it only after childReaped is true.
final class H3NativeBlockSession: @unchecked Sendable {
  struct Ready: Sendable { let rows: Int; let loadSeconds: Double; let metalAllocatedBytes: UInt64 }
  struct Report: Sendable {
    let predictionID: String; let outputURL: URL; let rows: Int; let byteCount: UInt64
    let seconds: Double; let computeSeconds: Double
    let weightLoadSeconds: Double; let weightPreparationSeconds: Double; let metalAllocatedBytes: UInt64
  }
  struct LineBuffer {
    private var bytes = Data()
    mutating func append(_ data: Data) throws {
      bytes.append(data)
      var start = bytes.startIndex
      for index in bytes.indices where bytes[index] == 10 {
        guard bytes.distance(from: start, to: index) <= 65_536 else { throw H3NativeBlockSession.invalid("Worker line exceeds 64 KiB.") }
        start = bytes.index(after: index)
      }
      guard bytes.distance(from: start, to: bytes.endIndex) <= 65_536 else { throw H3NativeBlockSession.invalid("Worker line exceeds 64 KiB.") }
      // Reads are bounded and the consumer removes each complete line before another read.
      guard bytes.count <= 69_632 else { throw H3NativeBlockSession.invalid("Worker protocol buffer exceeds its bound.") }
    }
    mutating func next() throws -> Data? {
      guard let end = bytes.firstIndex(of: 10) else { return nil }
      let line = Data(bytes[..<end]); bytes.removeSubrange(...end)
      guard !line.isEmpty, String(data: line, encoding: .utf8) != nil else { throw H3NativeBlockSession.invalid("Worker line is not nonempty UTF-8.") }
      return line
    }
    func finish() throws { guard bytes.isEmpty else { throw H3NativeBlockSession.invalid("Worker ended with an incomplete line.") } }
  }
  /// The utility queue exclusively reads stderr; its bounded tail is protected by this lock.
  private final class ErrorTail: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes = Data()
    func append(_ data: Data) {
      lock.lock(); defer { lock.unlock() }
      bytes.append(data); if bytes.count > 65_536 { bytes.removeFirst(bytes.count - 65_536) }
    }
    var text: String { lock.lock(); defer { lock.unlock() }; return String(decoding: bytes, as: UTF8.self) }
  }
  private let operation = NSRecursiveLock()
  private let process = Process()
  private let inputPipe = Pipe(), outputPipe = Pipe(), errorPipe = Pipe()
  private let errors = ErrorTail(), errorDrain = DispatchGroup()
  private var lines = LineBuffer()
  private var memoryMonitor: H3NativeMemoryMonitor?
  private var retiredMemoryReport: H3NativeMemoryMonitor.Report?
  /// Parent lifetime HWM includes stages before child launch. Combined current values are
  /// non-atomic estimates; independent HWM envelopes are never a measured total peak.
  var memoryReport: H3NativeMemoryMonitor.Report? {
    operation.lock(); defer { operation.unlock() }
    return memoryMonitor?.report ?? retiredMemoryReport
  }
  private let workspace: URL
  private let rows: Int
  private let predictionTimeout: TimeInterval, shutdownGrace: TimeInterval
  private var nextID = 0
  private var launched = false, reaped = false, closed = false
  private(set) var ready: Ready
  var processIdentifier: Int32? { operation.lock(); defer { operation.unlock() }; return launched ? process.processIdentifier : nil }
  var childReaped: Bool { operation.lock(); defer { operation.unlock() }; return reaped }
  var isClosed: Bool { operation.lock(); defer { operation.unlock() }; return closed }

  init(initialURL: URL, checkpointURL: URL, adapterURL: URL, workspaceURL: URL,
    rows: Int, workerURL: URL, readyTimeout: TimeInterval = 120,
    predictionTimeout: TimeInterval = 3600, shutdownGrace: TimeInterval = 2) throws {
    guard (1...40_000).contains(rows), [readyTimeout, predictionTimeout, shutdownGrace].allSatisfy({ $0.isFinite && $0 > 0 }),
      readyTimeout <= 3600, predictionTimeout <= 86_400, shutdownGrace <= 10 else { throw Self.invalid("Invalid native worker session limits.") }
    self.rows = rows; workspace = workspaceURL.standardizedFileURL
    self.predictionTimeout = predictionTimeout; self.shutdownGrace = shutdownGrace
    ready = Ready(rows: rows, loadSeconds: 0, metalAllocatedBytes: 0)
    for url in [initialURL, checkpointURL, adapterURL, workerURL] { try Self.readableRegular(url) }
    guard FileManager.default.isExecutableFile(atPath: workerURL.path) else { throw Self.invalid("Native worker is not executable.") }
    guard [initialURL, checkpointURL, adapterURL, workerURL, workspaceURL].allSatisfy({ $0.isFileURL && $0.path.utf8.count <= 4096 }) else { throw Self.invalid("Invalid native worker paths.") }
    try Task.checkCancellation()
    if FileManager.default.fileExists(atPath: workspace.path) {
      let values = try workspace.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
      guard values.isDirectory == true, values.isSymbolicLink != true,
        try FileManager.default.contentsOfDirectory(atPath: workspace.path).isEmpty else { throw Self.invalid("Native worker workspace must be empty.") }
    } else { try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true) }
    process.executableURL = workerURL
    process.arguments = ["serve", initialURL.path, checkpointURL.path, adapterURL.path, workspace.path, "1"]
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys where key.hasPrefix("WEETODD_NNC_") { environment.removeValue(forKey: key) }
    process.environment = environment
    process.standardInput = inputPipe; process.standardOutput = outputPipe; process.standardError = errorPipe
    do {
      try process.run(); launched = true
      let monitor = try H3NativeMemoryMonitor(childPID: process.processIdentifier)
      memoryMonitor = monitor
      try monitor.start() // Before the ready wait and the child's weighted load.
      try inputPipe.fileHandleForReading.close(); try outputPipe.fileHandleForWriting.close(); try errorPipe.fileHandleForWriting.close()
      guard fcntl(inputPipe.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1) == 0,
        fcntl(inputPipe.fileHandleForWriting.fileDescriptor, F_SETFL, O_NONBLOCK) == 0 else { throw Self.invalid("Cannot configure native worker command pipe.") }
      let handle = errorPipe.fileHandleForReading, tail = errors, group = errorDrain
      group.enter()
      DispatchQueue.global(qos: .utility).async {
        defer { try? handle.close(); group.leave() }
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
          let count = buffer.withUnsafeMutableBytes { Darwin.read(handle.fileDescriptor, $0.baseAddress, $0.count) }
          if count > 0 { tail.append(Data(buffer.prefix(count))) }
          else if count == -1 && errno == EINTR { continue }
          else { break }
        }
      }
      ready = try Self.validateReady(readLine(until: Self.deadline(readyTimeout)), rows: rows)
    } catch { stopAndReap(); throw error }
  }

  static func validateReady(_ data: Data, rows: Int) throws -> Ready {
    guard (1...40_000).contains(rows) else { throw invalid("Invalid native worker row count.") }
    let value = try object(data)
    guard string(value, "event") == "ready", try integer(value, "protocol") == 1,
      try integer(value, "rows") == UInt64(rows), string(value, "precision") == "fp16",
      try integer(value, "start") == 0, try integer(value, "count") == 50,
      string(value, "residency") == "block", try integer(value, "resident_blocks") == 0,
      string(value, "projections") == "input-scaled", try integer(value, "modulation_spans") == 0,
      try boolean(value, "weight_prefetch"), try integer(value, "prefetch_slot_capacity") == 1,
      string(value, "buffer_io") == "bounded", string(value, "qkv_schedule") == "serial" else { throw invalid("Native worker ready contract differs from the admitted backend.") }
    return Ready(rows: rows, loadSeconds: try number(value, "load_seconds"), metalAllocatedBytes: try integer(value, "metal_allocated_bytes"))
  }
  static func validateProgress(_ data: Data, previous: Int) throws -> (completed: Int, total: Int) {
    let value = try object(data), completed = try integer(value, "completed")
    guard string(value, "event") == "progress", previous >= 0, previous < 50,
      completed == UInt64(previous + 1), try integer(value, "total") == 50,
      try integer(value, "resident_blocks") == 1 else { throw invalid("Native worker progress is out of sequence.") }
    for key in ["weight_load_seconds", "weight_preparation_seconds", "compute_seconds"] { _ = try number(value, key) }
    for key in ["metal_allocated_bytes", "model_scratch_bytes"] { _ = try integer(value, key) }
    guard try integer(value, "cpu_prefetch_blocks") <= 1 else { throw invalid("Native worker exceeded its CPU prefetch slot.") }
    return (Int(completed), 50)
  }
  static func validatePrediction(_ data: Data, id: String, output: URL, rows: Int, completed: Int) throws -> Report {
    let value = try object(data)
    guard (1...40_000).contains(rows), completed == 50, string(value, "event") == "prediction",
      string(value, "id") == id, string(value, "output") == output.path,
      string(value, "dtype") == "F32", try integer(value, "resident_blocks") == 1 else { throw invalid("Native worker prediction contract mismatch.") }
    var info = stat()
    guard fstatat(AT_FDCWD, output.path, &info, AT_SYMLINK_NOFOLLOW) == 0, (info.st_mode & S_IFMT) == S_IFREG,
      info.st_size == Int64(rows) * 5376 * 4 else { throw invalid("Native worker output size or file type is invalid.") }
    return Report(predictionID: id, outputURL: output, rows: rows, byteCount: UInt64(info.st_size),
      seconds: try number(value, "seconds"), computeSeconds: try number(value, "compute_seconds"),
      weightLoadSeconds: try number(value, "weight_load_seconds"), weightPreparationSeconds: try number(value, "weight_preparation_seconds"),
      metalAllocatedBytes: try integer(value, "metal_allocated_bytes"))
  }
  func predict(input: URL, output: URL, progress: (Int, Int) throws -> Void = { _, _ in }) throws -> Report {
    operation.lock(); defer { operation.unlock() }
    guard !closed, launched, process.isRunning else { throw Self.invalid("Native worker session is closed.") }
    do {
      try Task.checkCancellation(); try Self.readableRegular(input)
      guard input.path.utf8.count <= 4096, output.isFileURL, output.path.utf8.count <= 4096,
        output.standardizedFileURL.deletingLastPathComponent().path == workspace.path,
        !FileManager.default.fileExists(atPath: output.path) else { throw Self.invalid("Native worker output must be a fresh workspace file.") }
      nextID += 1; let id = String(nextID), end = Self.deadline(predictionTimeout)
      try writeCommand(["op": "predict", "id": id, "input": input.path, "output": output.path], until: end)
      var completed = 0
      while true {
        let line = try readLine(until: end), value = try Self.object(line)
        switch Self.string(value, "event") {
        case "progress":
          let parsed = try Self.validateProgress(line, previous: completed)
          completed = parsed.completed; try progress(parsed.completed, parsed.total)
        case "prediction":
          let report = try Self.validatePrediction(line, id: id, output: output, rows: rows, completed: completed)
          try Task.checkCancellation(); return report
        default: throw Self.invalid("Unexpected native worker event during prediction.")
        }
      }
    } catch { stopAndReap(); throw error }
  }
  func close() throws {
    operation.lock(); defer { operation.unlock() }
    guard !closed else { return }
    var handshakeError: Error?
    var acknowledged = false
    if launched, process.isRunning {
      do {
        let id = "close-" + String(nextID), end = Self.deadline(shutdownGrace)
        try writeCommand(["op": "close", "id": id], until: end)
        let value = try Self.object(readLine(until: end))
        guard Self.string(value, "event") == "closed", Self.string(value, "id") == id else { throw Self.invalid("Native worker close acknowledgement mismatch.") }
        acknowledged = true
      } catch { handshakeError = error }
    } else if launched { handshakeError = Self.invalid("Native worker exited before close acknowledgement.") }
    let naturalExit = stopAndReap(allowNaturalExit: acknowledged)
    if let handshakeError { throw handshakeError }
    guard naturalExit else { throw Self.invalid("Native worker required termination after close acknowledgement.") }
  }
  deinit { stopAndReap() }

  private func readLine(until end: UInt64) throws -> Data {
    while true {
      try Task.checkCancellation()
      if let line = try lines.next() { return line }
      try wait(fd: outputPipe.fileHandleForReading.fileDescriptor, events: Int16(POLLIN), until: end)
      var buffer = [UInt8](repeating: 0, count: 4096)
      let count = buffer.withUnsafeMutableBytes { Darwin.read(outputPipe.fileHandleForReading.fileDescriptor, $0.baseAddress, $0.count) }
      if count > 0 { try lines.append(Data(buffer.prefix(count))) }
      else if count == -1 && errno == EINTR { continue }
      else { try lines.finish(); throw Self.invalid("Native worker ended before its response. " + errors.text) }
    }
  }
  private func writeCommand(_ value: [String: String], until end: UInt64) throws {
    var data = try JSONSerialization.data(withJSONObject: value, options: .sortedKeys)
    guard data.count <= 65_536 else { throw Self.invalid("Native worker command exceeds its bound.") }
    data.append(10); var sent = 0
    while sent < data.count {
      try Task.checkCancellation()
      try wait(fd: inputPipe.fileHandleForWriting.fileDescriptor, events: Int16(POLLOUT), until: end)
      let count = data.withUnsafeBytes { Darwin.write(inputPipe.fileHandleForWriting.fileDescriptor, $0.baseAddress!.advanced(by: sent), data.count - sent) }
      if count > 0 { sent += count }
      else if count == -1 && [EINTR, EAGAIN].contains(errno) { continue }
      else { throw Self.invalid("Native worker command pipe closed.") }
    }
  }
  private func wait(fd: Int32, events: Int16, until end: UInt64) throws {
    while true {
      try Task.checkCancellation()
      let now = DispatchTime.now().uptimeNanoseconds
      guard now < end else { throw Self.invalid("Native worker response timed out.") }
      var descriptor = pollfd(fd: fd, events: events, revents: 0)
      let result = poll(&descriptor, 1, Int32(min(100, max(1, (end - now) / 1_000_000))))
      if result > 0 { return }
      if result == -1 && errno != EINTR { throw Self.invalid("Native worker pipe polling failed.") }
    }
  }
  @discardableResult
  private func stopAndReap(allowNaturalExit: Bool = false) -> Bool {
    guard !closed else { return reaped }
    // Stop and join before intentional terminate/wait/reap; no PID queries after it.
    if let monitor = memoryMonitor {
      monitor.stop()
      retiredMemoryReport = monitor.report
      memoryMonitor = nil
    }
    var naturalExit = true
    try? inputPipe.fileHandleForWriting.close()
    if launched {
      if allowNaturalExit {
        let end = Self.deadline(shutdownGrace)
        while process.isRunning, DispatchTime.now().uptimeNanoseconds < end { usleep(10_000) }
      }
      if process.isRunning {
        naturalExit = false
        process.terminate()
        let end = Self.deadline(shutdownGrace)
        while process.isRunning, DispatchTime.now().uptimeNanoseconds < end { usleep(10_000) }
        if process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit(); reaped = true
      if allowNaturalExit, process.terminationReason != .exit || process.terminationStatus != 0 { naturalExit = false }
    }
    try? inputPipe.fileHandleForReading.close()
    try? outputPipe.fileHandleForWriting.close()
    try? errorPipe.fileHandleForWriting.close()
    try? outputPipe.fileHandleForReading.close()
    if launched { errorDrain.wait() }
    try? errorPipe.fileHandleForReading.close()
    closed = true
    return naturalExit
  }
  private static func deadline(_ seconds: TimeInterval) -> UInt64 { DispatchTime.now().uptimeNanoseconds + UInt64(seconds * 1_000_000_000) }
  private static func invalid(_ message: String) -> H3CheckpointError { .invalid(message) }
  private static func object(_ data: Data) throws -> [String: Any] {
    guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw invalid("Native worker event is not a JSON object.") }; return value
  }
  private static func string(_ value: [String: Any], _ key: String) -> String? { value[key] as? String }
  private static func number(_ value: [String: Any], _ key: String) throws -> Double {
    guard let number = value[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.doubleValue.isFinite, number.doubleValue >= 0 else { throw invalid("Invalid native worker numeric field: " + key) }; return number.doubleValue
  }
  private static func integer(_ value: [String: Any], _ key: String) throws -> UInt64 {
    let result = try number(value, key)
    guard result.rounded() == result, result < 18_446_744_073_709_551_616 else { throw invalid("Invalid native worker integer field: " + key) }; return UInt64(result)
  }
  private static func boolean(_ value: [String: Any], _ key: String) throws -> Bool {
    guard let result = value[key] as? NSNumber, CFGetTypeID(result) == CFBooleanGetTypeID() else { throw invalid("Invalid native worker boolean field: " + key) }; return result.boolValue
  }
  private static func readableRegular(_ url: URL) throws {
    var info = stat()
    guard url.isFileURL, fstatat(AT_FDCWD, url.path, &info, 0) == 0,
      (info.st_mode & S_IFMT) == S_IFREG, FileManager.default.isReadableFile(atPath: url.path) else { throw invalid("Native worker file is not regular and readable: " + url.path) }
  }
}
