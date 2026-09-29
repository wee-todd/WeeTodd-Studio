import Darwin
import Foundation

/// One app-owned process per invocation. Readers drain stdout and stderr concurrently;
/// cancellation escalates to SIGKILL and never returns while the direct worker is alive.
/// Workers must not spawn inference descendants. The engine's parent-death watchdog is
/// a separate requirement; Swift task cancellation cannot observe an app crash.
public enum WorkerProcess {
  public enum Error: Swift.Error, Equatable {
    case outputLineTooLong
    case unterminatedOutputLine
    case invalidConfiguration
    case pipeReadFailed(Int32)
  }
  public struct Result: Sendable {
    public let exitCode: Int32
    public let stderrTail: Data
    public let stderrWasTruncated: Bool
  }

  public static func run(executable: URL, arguments: [String],
    terminationGraceMilliseconds: Int = 2000,
    onLine: @escaping @Sendable (Data) throws -> Void) async throws -> Result {
    guard executable.isFileURL, (0...10000).contains(terminationGraceMilliseconds) else {
      throw Error.invalidConfiguration
    }
    let invocation = Invocation(executable: executable, arguments: arguments,
      grace: terminationGraceMilliseconds, onLine: onLine)
    return try await withTaskCancellationHandler {
      let result = try await Task.detached { try invocation.execute() }.value
      try Task.checkCancellation()
      return result
    } onCancel: {
      invocation.requestStop()
    }
  }

  /// Foundation Process and pipe handles are owned by execute(). Shared cancellation,
  /// errors and log state use the lock; only one stdout reader invokes the receiver.
  private final class Invocation: @unchecked Sendable {
    private let process = Process()
    private let output = Pipe()
    private let errors = Pipe()
    private let lock = NSLock()
    private let grace: Int
    private let onLine: @Sendable (Data) throws -> Void
    private var started = false
    private var exited = false
    private var stopping = false
    private var firstError: (any Swift.Error)?
    private var stderrTail = Data()
    private var truncated = false

    init(executable: URL, arguments: [String], grace: Int,
      onLine: @escaping @Sendable (Data) throws -> Void) {
      process.executableURL = executable; process.arguments = arguments
      process.standardInput = FileHandle.nullDevice
      process.standardOutput = output; process.standardError = errors
      self.grace = grace; self.onLine = onLine
    }

    func requestStop() {
      let scheduleKill = lock.withLock { () -> Bool in
        guard !stopping else { return false }
        stopping = true
        if started && !exited && process.isRunning { process.terminate() }
        return started && !exited
      }
      if scheduleKill {
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(grace)) { [self] in
          lock.withLock {
            if started && !exited && process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
          }
        }
      }
    }

    func execute() throws -> Result {
      defer {
        try? output.fileHandleForReading.close(); try? output.fileHandleForWriting.close()
        try? errors.fileHandleForReading.close(); try? errors.fileHandleForWriting.close()
      }
      for handle in [output.fileHandleForReading, errors.fileHandleForReading] {
        let flags = fcntl(handle.fileDescriptor, F_GETFL)
        guard flags >= 0, fcntl(handle.fileDescriptor, F_SETFL, flags | O_NONBLOCK) == 0 else {
          throw Error.pipeReadFailed(errno)
        }
      }
      try lock.withLock {
        if stopping { throw CancellationError() }
        try process.run()
        started = true
      }
      // Parent copies must close after launch, otherwise EOF never reaches readers.
      try? output.fileHandleForWriting.close(); try? errors.fileHandleForWriting.close()
      let readers = DispatchGroup()
      readers.enter()
      DispatchQueue.global().async { [self] in
        defer { readers.leave() }
        do { try readLines() } catch { recordFailure(error) }
      }
      readers.enter()
      DispatchQueue.global().async { [self] in
        defer { readers.leave() }
        do { try drain(errors.fileHandleForReading.fileDescriptor) { data in
          lock.withLock {
            stderrTail.append(data)
            if stderrTail.count > 65536 {
              truncated = true; stderrTail.removeFirst(stderrTail.count - 65536)
            }
          }
        } } catch { recordFailure(error) }
      }
      process.waitUntilExit()
      lock.withLock { exited = true }
      readers.wait()
      return try lock.withLock {
        if let firstError { throw firstError }
        return Result(exitCode: process.terminationStatus, stderrTail: stderrTail,
          stderrWasTruncated: truncated)
      }
    }

    private func recordFailure(_ error: any Swift.Error) {
      lock.withLock { if firstError == nil { firstError = error } }
      requestStop()
    }

    private func readLines() throws {
      var pending = Data()
      try drain(output.fileHandleForReading.fileDescriptor) { data in
        pending.append(data)
        while let newline = pending.firstIndex(of: 10) {
          let length = pending.distance(from: pending.startIndex, to: newline)
          guard length <= 65536 else { throw Error.outputLineTooLong }
          try onLine(Data(pending[..<newline]))
          pending.removeSubrange(pending.startIndex...newline)
        }
        guard pending.count <= 65536 else { throw Error.outputLineTooLong }
      }
      // On cancellation the final partial line is expected and never published.
      if !pending.isEmpty && !lock.withLock({ stopping }) { throw Error.unterminatedOutputLine }
    }

    private func drain(_ descriptor: Int32, consume: (Data) throws -> Void) throws {
      var bytes = [UInt8](repeating: 0, count: 4096)
      while true {
        let count = Darwin.read(descriptor, &bytes, bytes.count)
        if count > 0 { try consume(Data(bytes.prefix(count))); continue }
        if count == 0 { return }
        let code = errno
        if code == EINTR { continue }
        guard code == EAGAIN || code == EWOULDBLOCK else { throw Error.pipeReadFailed(code) }
        if lock.withLock({ exited }) { return }
        var descriptorState = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
        let ready = poll(&descriptorState, 1, 100)
        if ready < 0 && errno != EINTR { throw Error.pipeReadFailed(errno) }
      }
    }
  }
}
