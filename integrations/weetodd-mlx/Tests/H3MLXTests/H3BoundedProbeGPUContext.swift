// Explicit opt-in, fresh-process developer probe harness; no model runtime.
// Bind and verify the frozen Metal library before any device/allocator access.
import Foundation
import CryptoKit
import Darwin
import XCTest
import MLX
import InferenceContracts
@testable import H3MLX
import MLXRandom

enum H3BoundedProbeError: Error { case invalid(String), workspace }

// No MLX/Memory/Device access during declaration or CPU-only planner tests.
private final class H3BoundedProbeMetallib {
  let url: URL, sha256: String
  private let descriptor: Int32, identity: stat

  private static func same(_ a: stat, _ b: stat) -> Bool {
    a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size
      && a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec
      && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec
      && a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec
      && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
  }
  init(environment: [String: String]) throws {
    guard let path = environment["WEETODD_H3_BOUNDED_PROBE_METALLIB"], path.hasPrefix("/"),
      !path.utf8.contains(0), let expected = environment["WEETODD_H3_BOUNDED_PROBE_METALLIB_SHA256"],
      expected.count == 64, expected.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw H3BoundedProbeError.invalid("Declare an absolute metallib and its frozen lower-case SHA256")
    }
    let selected = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    let fd = Darwin.open(selected.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard fd >= 0 else { throw H3BoundedProbeError.invalid("Cannot open declared metallib") }
    var keep = false
    defer { if !keep { Darwin.close(fd) } }
    var before = stat(), named = stat()
    guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
      before.st_size > 0, before.st_size <= 512 * 1024 * 1024,
      fstatat(AT_FDCWD, selected.path, &named, 0) == 0, Self.same(before, named) else {
      throw H3BoundedProbeError.invalid("Declared metallib is not a bounded stable regular file")
    }
    let input = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    var digest = SHA256()
    while let bytes = try input.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty {
      try Task.checkCancellation(); digest.update(data: bytes)
    }
    let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
    guard actual == expected else { throw H3BoundedProbeError.invalid("Declared metallib SHA256 mismatch") }
    var after = stat(), finalName = stat()
    guard fstat(fd, &after) == 0, fstatat(AT_FDCWD, selected.path, &finalName, 0) == 0,
      Self.same(before, after), Self.same(before, finalName) else {
      throw H3BoundedProbeError.invalid("Declared metallib changed while hashing")
    }
    descriptor = fd; identity = before; url = selected; sha256 = actual
    keep = true
  }
  func verifyUnchanged() throws {
    var current = stat(), named = stat()
    guard fstat(descriptor, &current) == 0, fstatat(AT_FDCWD, url.path, &named, 0) == 0,
      Self.same(identity, current), Self.same(identity, named) else {
      throw H3BoundedProbeError.invalid("Declared metallib changed during the gate")
    }
  }
  deinit { Darwin.close(descriptor) }
}

private final class H3FreshGPUProbeProcess: @unchecked Sendable {
  static let shared = H3FreshGPUProbeProcess()
  private let lock = NSLock()
  private var used = false
  func claim() throws {
    lock.lock(); defer { lock.unlock() }
    guard !used else { throw H3BoundedProbeError.invalid("Run every GPU gate in a fresh XCTest process") }
    used = true
  }
}

enum H3BoundedProbeGPUContext {
  static let activeLimit = 48 * 1024 * 1024 * 1024
  static let maximumCache = 256 * 1024 * 1024

  static func checkBudget(_ box: ErrorBox) throws {
    try box.check()
    guard Memory.activeMemory <= activeLimit, Memory.peakMemory <= activeLimit else {
      throw H3BoundedProbeError.invalid("Fixed48GiB active-memory gate failed")
    }
  }

  static func run(_ body: (ErrorBox) throws -> Void) throws {
    try H3FreshGPUProbeProcess.shared.claim()
    let environment = ProcessInfo.processInfo.environment
    let library = try H3BoundedProbeMetallib(environment: environment)
    guard let declaredLease = environment["WEETODD_H3_BOUNDED_PROBE_LEASE_PATH"],
      declaredLease.hasPrefix("/"), !declaredLease.utf8.contains(0),
      URL(fileURLWithPath: declaredLease).standardizedFileURL.resolvingSymlinksInPath()
        == NativeInferenceLease.defaultURL.standardizedFileURL.resolvingSymlinksInPath() else {
      throw H3BoundedProbeError.invalid("Declare the real native inference lease; no temporary substitute")
    }
    // This is the same production lease, never a test/temp substitute.
    guard let lease = try NativeInferenceLease.tryAcquire(at: NativeInferenceLease.defaultURL) else {
      throw H3BoundedProbeError.invalid("Real native inference lease is busy; root must queue this gate")
    }
    defer { lease.release() }
    try library.verifyUnchanged()
    try MLX.withError { box in
      // FIRST GPU configuration call: nothing above touches Memory/Device/arrays.
      GPU.metallib = library.url
      try box.check()
      guard GPU.metallib?.standardizedFileURL.resolvingSymlinksInPath() == library.url else {
        throw H3BoundedProbeError.invalid("Metallib override did not retain the declared path")
      }
      try box.check()
      guard Device.defaultDevice().deviceType == .gpu else {
        throw H3BoundedProbeError.invalid("Declared Metal gate requires the GPU device")
      }
      try box.check()
      let oldLimit = Memory.memoryLimit
      try box.check()
      let oldCache = Memory.cacheLimit
      try box.check()
      defer {
        Memory.cacheLimit = oldCache
        Memory.memoryLimit = oldLimit
        if Memory.cacheLimit != oldCache || Memory.memoryLimit != oldLimit {
          box.firstError = H3BoundedProbeError.invalid("Memory/cache limits did not restore")
        }
        print("H3_BOUNDED_PROBE_LIMITS_RESTORED activeLimit=\(Memory.memoryLimit) cacheLimit=\(Memory.cacheLimit)")
      }
      Memory.memoryLimit = activeLimit
      Memory.cacheLimit = min(oldCache, maximumCache)
      try box.check()
      guard Memory.memoryLimit == activeLimit, Memory.cacheLimit <= maximumCache else {
        throw H3BoundedProbeError.invalid("Fixed48GiB/256MiB memory controls were not applied")
      }
      Stream.gpu.synchronize(); try box.check()
      Memory.clearCache(); try box.check()
      guard Memory.activeMemory == 0 else {
        throw H3BoundedProbeError.invalid("Fresh gate already owns MLX arrays before its body")
      }
      Memory.peakMemory = 0
      print("H3_BOUNDED_PROBE_CONTEXT pid=\(getpid()) lease=\(NativeInferenceLease.defaultURL.path) metallib=\(library.url.path) sha256=\(library.sha256) activeLimit=\(Memory.memoryLimit) cacheLimit=\(Memory.cacheLimit)")
      var failure: Error?
      do {
        try autoreleasepool { try body(box); try box.check() }
      } catch { failure = error }
      // Body scope has unwound before synchronization and owned-array checks.
      // A sticky MLX error must not skip a later cleanup operation.
      func cleanupIssue(_ issue: Error) {
        if failure == nil { failure = issue }
        else { print("H3_BOUNDED_PROBE_SECONDARY_CLEANUP_ERROR \(issue)") }
      }
      Stream.gpu.synchronize()
      do { try box.check() } catch { cleanupIssue(error) }
      Memory.clearCache()
      do { try box.check() } catch { cleanupIssue(error) }
      let active = Memory.activeMemory, peak = Memory.peakMemory
      print("H3_BOUNDED_PROBE_CLEANUP activeBytes=\(active) peakBytes=\(peak)")
      if active != 0 || peak > activeLimit {
        cleanupIssue(H3BoundedProbeError.invalid("Gate retained arrays or exceeded its fixed48GiB cap"))
      }
      do { try library.verifyUnchanged() } catch { cleanupIssue(error) }
      if let failure { throw failure }
    }
  }
}
