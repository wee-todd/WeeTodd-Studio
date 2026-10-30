import Darwin
import Foundation
import XCTest
@testable import StudioCore
final class FlashVSRSupervisorTests: XCTestCase {
  func testParentDeathStopsWeightedChildAndReleasesLease() throws {
    // Exercise the CLI from this exact build, including native SwiftPM builds.
    let cli = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("WeeToddCLI")
    guard FileManager.default.isExecutableFile(atPath: cli.path) else { throw XCTSkip("Build WeeToddCLI for the supervisor integration test") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let engine = root.appendingPathComponent("engine"), enginePID = root.appendingPathComponent("engine.pid")
    try "#!/bin/sh\necho $$ > '\(enginePID.path)'\nexec 1>&- 2>&-\ntrap '' INT\nexec /bin/sleep 30\n".write(to: engine, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: engine.path)
    let commandURL = root.appendingPathComponent("command.json"), ready = root.appendingPathComponent("ready")
    let parent = Process(); parent.executableURL = URL(fileURLWithPath: "/bin/sh")
    parent.arguments = ["-c", "while [ ! -f '\(ready.path)' ]; do /bin/sleep 0.01; done; '\(cli.path)' --flashvsr-engine-host '\(commandURL.path)' & wait"]
    parent.standardInput = FileHandle.nullDevice; parent.standardOutput = FileHandle.nullDevice; parent.standardError = FileHandle.nullDevice
    try parent.run(); defer { if parent.isRunning { parent.terminate(); parent.waitUntilExit() } }
    let lease = root.appendingPathComponent("lease")
    let command = FlashVSREngineCommand(executable: try FlashVSRFile.capture(engine), arguments: [], directory: root.path,
      log: root.appendingPathComponent("engine.log").path, lease: lease.path, ownerPID: parent.processIdentifier)
    try JSONEncoder().encode(command).write(to: commandURL); try Data().write(to: ready)
    let launched = Date().addingTimeInterval(5)
    while !FileManager.default.fileExists(atPath: enginePID.path) && Date() < launched { usleep(10000) }
    let pid = try XCTUnwrap(Int32(String(contentsOf: enginePID, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
    defer { if kill(pid, 0) == 0 { kill(pid, SIGKILL) } }
    kill(parent.processIdentifier, SIGKILL); parent.waitUntilExit()
    let ended = Date().addingTimeInterval(5)
    while kill(pid, 0) == 0 && Date() < ended { usleep(10000) }
    XCTAssertEqual(kill(pid, 0), -1, "A weighted external engine survived owner death")
    let fd = Darwin.open(lease.path, O_RDWR); defer { if fd >= 0 { close(fd) } }
    XCTAssertGreaterThanOrEqual(fd, 0)
    let released = Date().addingTimeInterval(5)
    var locked = flock(fd, LOCK_EX | LOCK_NB)
    while locked != 0 && Date() < released { usleep(10000); locked = flock(fd, LOCK_EX | LOCK_NB) }
    XCTAssertEqual(locked, 0, "The dead-owner supervisor did not release the inference lease")
  }
}
