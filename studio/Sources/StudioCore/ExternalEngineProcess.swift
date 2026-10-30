import Darwin
import Foundation

/// A bounded subprocess channel. A supervisor may establish its own process group;
/// cancellation then signals that entire group and waits for the direct child.
public enum ExternalEngineProcess {
  public struct Result: Sendable { public let exitCode: Int32; public let output: Data }
  public static func run(executable: URL, arguments: [String], directory: URL, log: URL,
    cancellation: NativeHeadlessCancellation = NativeHeadlessCancellation(), ownerPID: Int32? = nil,
    maximumOutput: Int = 8 * 1024 * 1024, captureOutput: Bool = true, onLine: (String) throws -> Void = { _ in }) throws -> Result {
    guard executable.isFileURL, directory.isFileURL, maximumOutput > 0,
      FileManager.default.isExecutableFile(atPath: executable.path) else {
      throw StudioError.invalid("Connect an executable local media engine.")
    }
    let fd = Darwin.open(log.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    guard fd >= 0 else { throw StudioError.invalid("The engine log must be a new writable file.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? handle.close() }
    let process = Process(), pipe = Pipe()
    process.executableURL = executable; process.arguments = arguments; process.currentDirectoryURL = directory
    process.standardInput = FileHandle.nullDevice; process.standardOutput = pipe; process.standardError = pipe
    let reader = pipe.fileHandleForReading
    let flags = fcntl(reader.fileDescriptor, F_GETFL)
    guard flags >= 0, fcntl(reader.fileDescriptor, F_SETFL, flags | O_NONBLOCK) == 0 else {
      throw StudioError.invalid("Cannot prepare the engine output channel.")
    }
    defer { try? reader.close(); try? pipe.fileHandleForWriting.close() }
    try cancellation.check(); try Task.checkCancellation(); try process.run()
    try pipe.fileHandleForWriting.close()
    func stop() {
      guard process.isRunning else { return }
      // The hidden CLI supervisor calls setpgid before launching weighted work.
      if kill(-process.processIdentifier, SIGINT) != 0 { process.interrupt() }
      let deadline = Date().addingTimeInterval(0.75)
      while process.isRunning && Date() < deadline { usleep(10000) }
      if process.isRunning {
        if kill(-process.processIdentifier, SIGKILL) != 0 { kill(process.processIdentifier, SIGKILL) }
      }
      process.waitUntilExit()
    }
    defer { stop(); try? cancellation.attach(nil) }
    try cancellation.attach(process)
    var pending = Data(), captured = Data(), logBytes = 0, eof = false
    var buffer = [UInt8](repeating: 0, count: 16384)
    while !eof {
      if let ownerPID, getppid() != ownerPID { cancellation.cancel() }
      try cancellation.check(); try Task.checkCancellation()
      let count = Darwin.read(reader.fileDescriptor, &buffer, buffer.count)
      if count > 0 {
        let bytes = Data(buffer.prefix(count))
        if logBytes < 8 * 1024 * 1024 {
          let writable = bytes.prefix(8 * 1024 * 1024 - logBytes)
          try handle.write(contentsOf: writable); logBytes += writable.count
        }
        if captureOutput {
          guard captured.count <= maximumOutput - count else { throw StudioError.invalid("Engine output exceeds its bounded channel.") }
          captured.append(bytes)
        } else {
          captured.append(bytes)
          if captured.count > maximumOutput { captured.removeFirst(captured.count - maximumOutput) }
        }
        pending.append(bytes)
        while let newline = pending.firstIndex(of: 10) {
          guard newline <= 65536 else { throw StudioError.invalid("Engine output line exceeds 64 KiB.") }
          let line = String(decoding: pending.prefix(upTo: newline), as: UTF8.self)
          pending.removeSubrange(...newline); try onLine(line)
        }
        guard pending.count <= 65536 else { throw StudioError.invalid("Engine output line exceeds 64 KiB.") }
      } else if count == 0 { eof = true }
      else if errno == EAGAIN || errno == EINTR { usleep(10000) }
      else { throw StudioError.invalid("Cannot read the engine output.") }
    }
    // EOF only closes the output channel; a running child can still own Metal state.
    while process.isRunning {
      if let ownerPID, getppid() != ownerPID { cancellation.cancel() }
      try cancellation.check(); try Task.checkCancellation()
      usleep(10000)
    }
    process.waitUntilExit(); try cancellation.check(); try Task.checkCancellation()
    if !pending.isEmpty { try onLine(String(decoding: pending, as: UTF8.self)) }
    return Result(exitCode: process.terminationStatus, output: captured)
  }
}
