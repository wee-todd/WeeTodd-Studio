import CoreFoundation
import Darwin
import Foundation

/// Header-only native capability probe. It does not start a weighted session.
/// This backend is explicit and experimental; no Python or machine-wide search is used.
enum H3NativeBlockWorker {
  static func resolve(executableURL: URL, environment: [String: String]) throws -> URL {
    try Task.checkCancellation()
    let candidate: URL
    if let override = environment["WEETODD_H3_NATIVE_WORKER"] {
      guard !override.isEmpty, override.hasPrefix("/"), !override.utf8.contains(0),
        override.utf8.count <= 4096 else { throw invalid("Native worker override must be an absolute file path.") }
      candidate = URL(fileURLWithPath: override).standardizedFileURL
    } else {
      guard executableURL.isFileURL, executableURL.path.hasPrefix("/"),
        executableURL.path.utf8.count <= 4096 else { throw invalid("Native MLX executable path is invalid.") }
      candidate = executableURL.standardizedFileURL.deletingLastPathComponent().appendingPathComponent("WeeToddH3Worker")
    }
    try validateExecutable(candidate)
    return candidate
  }
  static func preflight(workerURL: URL) throws { try preflight(workerURL: workerURL, timeout: 10) }

  /// A shorter deadline is available to focused process tests; callers cannot extend ten seconds.
  static func preflight(workerURL: URL, timeout: TimeInterval) throws {
    guard timeout.isFinite, timeout > 0, timeout <= 10 else { throw invalid("Native worker capability deadline must be within ten seconds.") }
    try Task.checkCancellation(); try validateExecutable(workerURL)
    let process = Process(), stdout = Pipe(), stderr = Pipe()
    process.executableURL = workerURL; process.arguments = ["--capabilities"]
    process.standardInput = FileHandle.nullDevice; process.standardOutput = stdout; process.standardError = stderr
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys where key.hasPrefix("WEETODD_NNC_") { environment.removeValue(forKey: key) }
    process.environment = environment
    var launched = false
    defer {
      // Reap before returning even when parsing, pipe setup, cancellation or deadline fails.
      if launched { stopAndReap(process) }
      for handle in [stdout.fileHandleForReading, stdout.fileHandleForWriting,
        stderr.fileHandleForReading, stderr.fileHandleForWriting] { try? handle.close() }
    }
    try process.run(); launched = true
    try stdout.fileHandleForWriting.close(); try stderr.fileHandleForWriting.close()
    let outputFD = stdout.fileHandleForReading.fileDescriptor, errorFD = stderr.fileHandleForReading.fileDescriptor
    for fd in [outputFD, errorFD] {
      let flags = fcntl(fd, F_GETFL)
      guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 else { throw invalid("Cannot configure native worker capability pipe.") }
    }
    let deadline = DispatchTime.now().uptimeNanoseconds + UInt64(timeout * 1_000_000_000)
    var output = Data(), errors = Data(), outputEnded = false, errorsEnded = false
    while !(outputEnded && errorsEnded && !process.isRunning) {
      try Task.checkCancellation()
      let now = DispatchTime.now().uptimeNanoseconds
      guard now < deadline else { throw invalid("Native worker capability response timed out.") }
      var descriptors = [pollfd(fd: outputEnded ? -1 : outputFD, events: Int16(POLLIN), revents: 0),
        pollfd(fd: errorsEnded ? -1 : errorFD, events: Int16(POLLIN), revents: 0)]
      let result = descriptors.withUnsafeMutableBufferPointer {
        poll($0.baseAddress, nfds_t($0.count), Int32(min(50, max(1, (deadline - now) / 1_000_000))))
      }
      if result < 0 {
        if errno == EINTR { continue }
        throw invalid("Native worker capability pipe polling failed.")
      }
      if descriptors[0].revents != 0 { try readAvailable(outputFD, into: &output, ended: &outputEnded) }
      if descriptors[1].revents != 0 { try readAvailable(errorFD, into: &errors, ended: &errorsEnded) }
    }
    process.waitUntilExit()
    try Task.checkCancellation()
    guard process.terminationReason == .exit, process.terminationStatus == 0 else {
      throw invalid("Native worker capability command failed. " + String(decoding: errors.prefix(4096), as: UTF8.self))
    }
    try validateCapabilities(output)
  }
  static func validateCapabilities(_ data: Data) throws {
    guard !data.isEmpty, data.count <= 65_536, String(data: data, encoding: .utf8) != nil,
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      integer(object, "protocol") == 1, object["backend"] as? String == "nnc_experimental",
      integer(object, "blocks") == 50, integer(object, "max_rows") == 40_000,
      let watchdog = object["parent_watchdog"] as? NSNumber,
      CFGetTypeID(watchdog) == CFBooleanGetTypeID(), watchdog.boolValue else {
      throw invalid("Native worker capabilities differ from the admitted experimental NNC contract.")
    }
  }
  private static func integer(_ object: [String: Any], _ key: String) -> Int? {
    guard let value = object[key] as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
      value.doubleValue.isFinite, value.doubleValue >= 0, value.doubleValue <= 40_000,
      value.doubleValue.rounded() == value.doubleValue else { return nil }
    return value.intValue
  }
  private static func validateExecutable(_ url: URL) throws {
    var info = stat()
    guard url.isFileURL, url.path.hasPrefix("/"), url.path.utf8.count <= 4096,
      !url.path.utf8.contains(0), Darwin.fstatat(AT_FDCWD, url.path, &info, 0) == 0,
      (info.st_mode & S_IFMT) == S_IFREG, FileManager.default.isReadableFile(atPath: url.path),
      FileManager.default.isExecutableFile(atPath: url.path) else {
      throw invalid("Native worker must be a readable executable regular file.")
    }
  }
  private static func readAvailable(_ fd: Int32, into data: inout Data, ended: inout Bool) throws {
    var buffer = [UInt8](repeating: 0, count: 4096)
    // One bounded read per poll cycle keeps cancellation and the absolute deadline responsive,
    // including a producer which writes continuously to either stream.
    let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
    if count > 0 {
      guard data.count <= 65_536 - count else { throw invalid("Native worker capability output exceeds 64 KiB.") }
      data.append(contentsOf: buffer.prefix(count))
    } else if count == 0 { ended = true }
    else if ![EINTR, EAGAIN].contains(errno) { throw invalid("Native worker capability pipe read failed.") }
  }
  private static func stopAndReap(_ process: Process) {
    if process.isRunning {
      _ = kill(process.processIdentifier, SIGTERM)
      let end = DispatchTime.now().uptimeNanoseconds + 200_000_000
      while process.isRunning && DispatchTime.now().uptimeNanoseconds < end { usleep(10_000) }
      if process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
    }
    process.waitUntilExit()
  }
  private static func invalid(_ text: String) -> H3CheckpointError { .invalid(text) }
}
