import Darwin
import Foundation

/// Scalar-only, session-owned observations. Start after launching the owned child;
/// stop and join before terminating/reaping it. Never keep this monitor across PID reuse.
final class H3NativeMemoryMonitor: @unchecked Sendable {
  struct Physical: Sendable, Equatable {
    let currentBytes: UInt64
    let lifetimeHighWaterBytes: UInt64
    let processStartAbsoluteTime: UInt64
  }
  enum Sample: Sendable, Equatable { case available(Physical), unavailable(errorCode: Int32) }
  enum Status: String, Sendable, Equatable { case notSampled = "not_sampled", available, unavailable }
  enum Lifecycle: String, Sendable, Equatable { case notStarted = "not_started", sampling, stopped }
  struct ProcessSummary: Sendable, Equatable {
    let pid: Int32
    var attempts = 0, availableSamples = 0, failedSamples = 0
    var zeroValueSamples = 0, currentAboveLifetimeSamples = 0, identityMismatchSamples = 0
    var sampledCurrentPeakBytes: UInt64?
    var reportedLifetimeHighWaterBytes: UInt64?
    var lastRawPhysical: Physical?
    var lastAnomalousPhysical: Physical?
    var lastErrorCode: Int32?
    var admittedProcessStartAbsoluteTime: UInt64?
    var status: Status { availableSamples > 0 ? .available : (attempts == 0 ? .notSampled : .unavailable) }
    /// Envelope of separately reported HWM and sampled current bytes, since these fields
    /// are not atomic and current can exceed the HWM in an individual response.
    var independentHighWaterEnvelopeBytes: UInt64? {
      guard let current = sampledCurrentPeakBytes, let lifetime = reportedLifetimeHighWaterBytes else { return nil }
      return max(current, lifetime)
    }
  }
  struct Report: Sendable, Equatable {
    let lifecycle: Lifecycle
    let intervalSeconds: Double
    let pairAttempts: Int
    let combinedSamples: Int
    let parent: ProcessSummary
    let child: ProcessSummary
    /// Sum of two sequential same-tick queries; a non-atomic estimate with no rigorous lower/upper guarantee, not a measured
    /// total physical high-water mark. Parent and child queries are not atomic.
    let sampledNonAtomicCombinedCurrentPeakEstimateBytes: UInt64?
    /// Sum of independent per-PID HWM envelopes, an upper bound for the observed
    /// interval, not a simultaneous peak. Includes pre-monitor process lifetime.
    let sumIndependentHighWaterUpperBoundBytes: UInt64?
    let overflowSamples: Int
    let lastSampleMonotonicNanoseconds: UInt64?
    let clockRegressionSamples: Int
  }
  typealias Sampler = @Sendable (Int32) -> Sample
  typealias Clock = @Sendable () -> UInt64
  private let parentPID: Int32, childPID: Int32
  private let sampler: Sampler, clock: Clock, automaticSampling: Bool
  private let operation = NSLock()
  private let queue = DispatchQueue(label: "WeeTodd.H3.NativeMemoryMonitor", qos: .utility)
  private var timer: DispatchSourceTimer?
  /// Timer handlers own this scalar state, never the monitor. This prevents a
  /// weak-self handler's last release from running monitor deinit on its own queue.
  private final class State: @unchecked Sendable {
    let parentPID: Int32, childPID: Int32
    let sampler: Sampler, clock: Clock
    var lifecycle = Lifecycle.notStarted
    var parent: ProcessSummary, child: ProcessSummary
    var pairAttempts = 0, combinedSamples = 0, overflowSamples = 0, clockRegressions = 0
    var combinedPeak: UInt64?, lastTime: UInt64?
    init(parentPID: Int32, childPID: Int32, sampler: @escaping Sampler, clock: @escaping Clock) {
      self.parentPID = parentPID; self.childPID = childPID; self.sampler = sampler; self.clock = clock
      parent = ProcessSummary(pid: parentPID); child = ProcessSummary(pid: childPID)
    }
    func sample() {
      guard lifecycle == .sampling else { return }
      let now = clock()
      if let previous = lastTime, now < previous { clockRegressions += 1 }
      lastTime = now; pairAttempts += 1
      let a = H3NativeMemoryMonitor.record(sampler(parentPID), into: &parent)
      let b = H3NativeMemoryMonitor.record(sampler(childPID), into: &child)
      if let a, let b {
        let sum = a.addingReportingOverflow(b)
        if sum.overflow { overflowSamples += 1 }
        else { combinedSamples += 1; combinedPeak = max(combinedPeak ?? 0, sum.partialValue) }
      }
    }
    func report() -> Report {
      var upper: UInt64?
      if let a = parent.independentHighWaterEnvelopeBytes, let b = child.independentHighWaterEnvelopeBytes {
        let sum = a.addingReportingOverflow(b); if !sum.overflow { upper = sum.partialValue }
      }
      return Report(lifecycle: lifecycle, intervalSeconds: 0.1, pairAttempts: pairAttempts,
        combinedSamples: combinedSamples, parent: parent, child: child,
        sampledNonAtomicCombinedCurrentPeakEstimateBytes: combinedPeak, sumIndependentHighWaterUpperBoundBytes: upper,
        overflowSamples: overflowSamples, lastSampleMonotonicNanoseconds: lastTime,
        clockRegressionSamples: clockRegressions)
    }
  }
  private let state: State

  init(parentPID: Int32 = getpid(), childPID: Int32,
    sampler: @escaping Sampler = H3NativeMemoryMonitor.systemSample,
    clock: @escaping Clock = { DispatchTime.now().uptimeNanoseconds },
    automaticSampling: Bool = true) throws {
    guard parentPID > 0, childPID > 0, parentPID != childPID else {
      throw H3CheckpointError.invalid("Native memory monitor requires distinct positive parent and owned child PIDs.")
    }
    self.parentPID = parentPID; self.childPID = childPID
    self.sampler = sampler; self.clock = clock; self.automaticSampling = automaticSampling
    state = State(parentPID: parentPID, childPID: childPID, sampler: sampler, clock: clock)
  }
  func start() throws {
    operation.lock(); defer { operation.unlock() }
    let admitted = queue.sync { () -> Bool in
      guard state.lifecycle == .notStarted else { return false }
      state.lifecycle = .sampling; state.sample(); return true
    }
    guard admitted else { throw H3CheckpointError.invalid("Native memory monitor can start only once.") }
    if automaticSampling {
      let source = DispatchSource.makeTimerSource(queue: queue)
      source.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100), leeway: .milliseconds(10))
      let scalarState = state
      source.setEventHandler { scalarState.sample() }
      timer = source; source.resume()
    }
  }
  /// Cancelling the source and synchronizing its serial queue joins any active query.
  /// Call before reap/termination; subsequent snapshots never query this PID again.
  func stop() {
    operation.lock(); defer { operation.unlock() }
    timer?.cancel(); timer = nil
    queue.sync { state.sample(); state.lifecycle = .stopped }
  }
  var report: Report { queue.sync { state.report() } }
  /// Internal deterministic-test seam. Runtime callers rely on the 100 ms timer.
  func sampleNowForTesting() { queue.sync { state.sample() } }
  deinit { stop() }

  private static func record(_ sample: Sample, into summary: inout ProcessSummary) -> UInt64? {
    summary.attempts += 1
    switch sample {
    case .unavailable(let error):
      summary.failedSamples += 1; summary.lastErrorCode = error; return nil
    case .available(let physical):
      summary.lastRawPhysical = physical
      if physical.currentBytes > physical.lifetimeHighWaterBytes {
        summary.currentAboveLifetimeSamples += 1; summary.lastAnomalousPhysical = physical
      }
      if physical.currentBytes == 0 || physical.lifetimeHighWaterBytes == 0 || physical.processStartAbsoluteTime == 0 {
        summary.zeroValueSamples += 1; summary.lastAnomalousPhysical = physical; return nil
      }
      if let expected = summary.admittedProcessStartAbsoluteTime, expected != physical.processStartAbsoluteTime {
        summary.identityMismatchSamples += 1; summary.lastAnomalousPhysical = physical; return nil
      }
      summary.admittedProcessStartAbsoluteTime = physical.processStartAbsoluteTime
      summary.availableSamples += 1
      summary.sampledCurrentPeakBytes = max(summary.sampledCurrentPeakBytes ?? 0, physical.currentBytes)
      summary.reportedLifetimeHighWaterBytes = max(summary.reportedLifetimeHighWaterBytes ?? 0, physical.lifetimeHighWaterBytes)
      return physical.currentBytes
    }
  }
  static func systemSample(_ pid: Int32) -> Sample {
    var info = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      proc_pid_rusage(pid, RUSAGE_INFO_V4,
        UnsafeMutableRawPointer(pointer).assumingMemoryBound(to: rusage_info_t?.self))
    }
    guard result == 0 else { return .unavailable(errorCode: errno) }
    return .available(Physical(currentBytes: info.ri_phys_footprint,
      lifetimeHighWaterBytes: info.ri_lifetime_max_phys_footprint,
      processStartAbsoluteTime: info.ri_proc_start_abstime))
  }
}
