import Darwin
import Foundation

/// The same cross-process lease used by Swift and Python generation workers.
final class ExternalInferenceLease {
  private let fd: Int32
  init(at url: URL, cancellation: NativeHeadlessCancellation, ownerPID: Int32) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    fd = Darwin.open(url.path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0o600)
    guard fd >= 0 else { throw StudioError.invalid("Cannot open the shared inference lease.") }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
      Darwin.close(fd); throw StudioError.invalid("The inference lease must be a regular file.")
    }
    do {
      while flock(fd, LOCK_EX | LOCK_NB) != 0 {
        guard errno == EWOULDBLOCK else { throw StudioError.invalid("Cannot acquire the shared inference lease.") }
        try cancellation.check()
        guard getppid() == ownerPID else { throw CancellationError() }
        usleep(100000)
      }
    } catch { Darwin.close(fd); throw error }
  }
  deinit { flock(fd, LOCK_UN); Darwin.close(fd) }
  static var defaultURL: URL {
    let configured = ProcessInfo.processInfo.environment["WEETODD_STUDIO_DATA"]
    let root = configured.map { URL(fileURLWithPath: $0) } ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/WeeTodd Studio")
    return root.appendingPathComponent("Runtime/native-inference.lock")
  }
}
public struct FlashVSREngineCommand: Codable {
  public let executable: FlashVSRFile
  public let arguments: [String]
  public let directory: String, log: String, lease: String
  public let ownerPID: Int32
  public var modelCheck: FlashVSRModelCheck? = nil
}
public struct FlashVSRModelCheck: Codable {
  public let components: FlashVSRComponents
  public let settings: FlashVSRSettings
}
public enum FlashVSREngineSupervisor {
  public static func run(commandURL: URL) throws -> Int32 {
    let size = try FileManager.default.attributesOfItem(atPath: commandURL.path)[.size] as? NSNumber
    guard let size, size.intValue > 0, size.intValue <= 1024 * 1024 else { throw StudioError.invalid("Engine supervisor command exceeds 1 MiB.") }
    let command = try JSONDecoder().decode(FlashVSREngineCommand.self, from: Data(contentsOf: commandURL))
    guard command.ownerPID > 1, getppid() == command.ownerPID,
      [command.directory, command.log, command.lease, command.executable.path].allSatisfy({ $0.hasPrefix("/") && !$0.utf8.contains(0) }),
      command.arguments.count <= 32 else { throw StudioError.invalid("Invalid local engine supervisor command.") }
    guard setpgid(0, 0) == 0 || getpgrp() == getpid() else { throw StudioError.invalid("Cannot isolate the external engine process group.") }
    let cancellation = NativeHeadlessCancellation()
    signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
    let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
      let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
      source.setEventHandler { cancellation.cancel() }; source.resume(); return source
    }
    let lease = try ExternalInferenceLease(at: URL(fileURLWithPath: command.lease), cancellation: cancellation, ownerPID: command.ownerPID)
    defer { withExtendedLifetime(signals) {}; withExtendedLifetime(lease) {} }
    try command.executable.verify()
    if let check = command.modelCheck {
      // Resolve again after waiting for the shared lease: variant shadows or
      // symlink changes must not redirect the weighted process to unrecorded files.
      try FlashVSRModelManifest.verifyLookup(check.components, settings: check.settings)
    }
    try cancellation.check()
    return try ExternalEngineProcess.run(executable: URL(fileURLWithPath: command.executable.path), arguments: command.arguments,
      directory: URL(fileURLWithPath: command.directory), log: URL(fileURLWithPath: command.log), cancellation: cancellation,
      ownerPID: command.ownerPID, maximumOutput: 65536, captureOutput: false, onLine: { line in FileHandle.standardOutput.write(Data((line + "\n").utf8)) }).exitCode
  }
}
