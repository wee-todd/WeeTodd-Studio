import Darwin
import Foundation
import XCTest
@testable import H3MLX

final class H3NativeMemoryMonitorTests: XCTestCase {
  private final class Samples: @unchecked Sendable {
    let lock = NSLock()
    var values: [Int32: [H3NativeMemoryMonitor.Sample]]
    var calls = 0, ticks: UInt64 = 0
    init(_ values: [Int32: [H3NativeMemoryMonitor.Sample]]) { self.values = values }
    func sample(_ pid: Int32) -> H3NativeMemoryMonitor.Sample {
      lock.lock(); defer { lock.unlock() }; calls += 1
      guard var list = values[pid], !list.isEmpty else { return .unavailable(errorCode: ESRCH) }
      let result = list.removeFirst(); values[pid] = list; return result
    }
    func clock() -> UInt64 { lock.lock(); defer { lock.unlock() }; ticks += 100_000_000; return ticks }
    var count: Int { lock.lock(); defer { lock.unlock() }; return calls }
  }
  private static func physical(_ current: UInt64, _ lifetime: UInt64, start: UInt64 = 1) -> H3NativeMemoryMonitor.Sample {
    .available(.init(currentBytes: current, lifetimeHighWaterBytes: lifetime, processStartAbsoluteTime: start))
  }
  private func monitor(_ samples: Samples) throws -> H3NativeMemoryMonitor {
    try H3NativeMemoryMonitor(parentPID: 10, childPID: 20,
      sampler: { samples.sample($0) }, clock: { samples.clock() }, automaticSampling: false)
  }
  func testNotSampledAndInvalidPIDAdmissionBeforeSampler() throws {
    let samples = Samples([:]), value = try monitor(samples)
    XCTAssertEqual(value.report.lifecycle, .notStarted)
    XCTAssertEqual(value.report.parent.status, .notSampled)
    XCTAssertEqual(value.report.child.status, .notSampled)
    XCTAssertNil(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes)
    XCTAssertNil(value.report.sumIndependentHighWaterUpperBoundBytes)
    let invalid: [(Int32, Int32)] = [(0, 1), (1, 0), (-1, 2), (1, 1)]
    for (parent, child) in invalid {
      XCTAssertThrowsError(try H3NativeMemoryMonitor(parentPID: parent, childPID: child,
        sampler: { samples.sample($0) }, automaticSampling: false))
    }
    XCTAssertEqual(samples.count, 0)
  }
  func testNonAtomicPairEstimatesDifferFromSumIndependentHighWaterMarks() throws {
    let samples = Samples([10: [Self.physical(100, 1000), Self.physical(80, 1000)],
      20: [Self.physical(50, 200), Self.physical(90, 200)]])
    let value = try monitor(samples); try value.start(); value.sampleNowForTesting()
    let report = value.report
    XCTAssertEqual(report.intervalSeconds, 0.1)
    XCTAssertEqual(report.parent.sampledCurrentPeakBytes, 100)
    XCTAssertEqual(report.child.sampledCurrentPeakBytes, 90)
    XCTAssertEqual(report.sampledNonAtomicCombinedCurrentPeakEstimateBytes, 170)
    XCTAssertEqual(report.sumIndependentHighWaterUpperBoundBytes, 1200)
    XCTAssertEqual(report.combinedSamples, 2)
    XCTAssertEqual(report.lastSampleMonotonicNanoseconds, 200_000_000)
  }
  func testCurrentAboveLifetimeIsRetainedAsNonAtomicAnomaly() throws {
    let samples = Samples([10: [Self.physical(150, 100)], 20: [Self.physical(80, 200)]])
    let value = try monitor(samples); try value.start()
    XCTAssertEqual(value.report.parent.status, .available)
    XCTAssertEqual(value.report.parent.currentAboveLifetimeSamples, 1)
    XCTAssertEqual(value.report.parent.lastAnomalousPhysical?.currentBytes, 150)
    XCTAssertEqual(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes, 230)
    XCTAssertEqual(value.report.sumIndependentHighWaterUpperBoundBytes, 350)
  }
  func testUnavailableAndZeroResponsesNeverAppearAsMemoryWins() throws {
    let samples = Samples([10: [.unavailable(errorCode: EPERM), Self.physical(40, 100)],
      20: [Self.physical(0, 0), .unavailable(errorCode: ESRCH)]])
    let value = try monitor(samples); try value.start()
    XCTAssertEqual(value.report.parent.status, .unavailable)
    XCTAssertEqual(value.report.parent.lastErrorCode, EPERM)
    XCTAssertEqual(value.report.child.status, .unavailable)
    XCTAssertEqual(value.report.child.zeroValueSamples, 1)
    value.sampleNowForTesting()
    XCTAssertEqual(value.report.parent.status, .available)
    XCTAssertEqual(value.report.child.failedSamples, 1)
    XCTAssertNil(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes)
    XCTAssertNil(value.report.sumIndependentHighWaterUpperBoundBytes)
  }
  func testPIDReuseIsRecordedAndExcludedFromCombinedPeak() throws {
    let samples = Samples([10: [Self.physical(10, 10), Self.physical(10, 10)],
      20: [Self.physical(20, 20, start: 100), Self.physical(999, 999, start: 200)]])
    let value = try monitor(samples); try value.start(); value.sampleNowForTesting()
    XCTAssertEqual(value.report.child.identityMismatchSamples, 1)
    XCTAssertEqual(value.report.child.availableSamples, 1)
    XCTAssertEqual(value.report.child.lastRawPhysical?.currentBytes, 999)
    XCTAssertEqual(value.report.child.lastAnomalousPhysical?.processStartAbsoluteTime, 200)
    XCTAssertEqual(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes, 30)
  }
  private final class ClockValues: @unchecked Sendable {
    let lock = NSLock()
    var values: [UInt64]
    init(_ values: [UInt64]) { self.values = values }
    func next() -> UInt64 {
      lock.lock(); defer { lock.unlock() }
      return values.isEmpty ? 90 : values.removeFirst()
    }
  }
  func testOverflowAndRegressingClockAreExplicit() throws {
    let samples = Samples([10: [Self.physical(UInt64.max, UInt64.max), Self.physical(UInt64.max, UInt64.max)],
      20: [Self.physical(1, 1), Self.physical(1, 1)]])
    let value = try H3NativeMemoryMonitor(parentPID: 10, childPID: 20,
      sampler: { samples.sample($0) }, clock: { 0 }, automaticSampling: false)
    try value.start(); value.sampleNowForTesting()
    XCTAssertEqual(value.report.overflowSamples, 2)
    XCTAssertNil(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes)
    XCTAssertNil(value.report.sumIndependentHighWaterUpperBoundBytes)
    let clock = ClockValues([100, 90])
    let regression = try H3NativeMemoryMonitor(parentPID: 10, childPID: 20,
      sampler: { _ in Self.physical(1, 2) }, clock: { clock.next() }, automaticSampling: false)
    try regression.start(); regression.sampleNowForTesting()
    XCTAssertEqual(regression.report.clockRegressionSamples, 1)
  }
  func testStopFinalSampleThenNeverQueriesAgainAndCannotRestart() throws {
    let samples = Samples([10: [Self.physical(1, 2), Self.physical(2, 3)],
      20: [Self.physical(3, 4), Self.physical(4, 5)]])
    let value = try monitor(samples); try value.start(); value.stop()
    XCTAssertEqual(value.report.lifecycle, .stopped)
    XCTAssertEqual(value.report.pairAttempts, 2)
    XCTAssertEqual(value.report.sampledNonAtomicCombinedCurrentPeakEstimateBytes, 6)
    let count = samples.count
    value.sampleNowForTesting(); value.stop()
    XCTAssertEqual(samples.count, count)
    XCTAssertThrowsError(try value.start())
  }
  private final class BlockingSample: @unchecked Sendable {
    let lock = NSLock(), entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
    var parentCalls = 0
    func sample(_ pid: Int32) -> H3NativeMemoryMonitor.Sample {
      if pid == 10 {
        lock.lock(); parentCalls += 1; let block = parentCalls == 2; lock.unlock()
        if block { entered.signal(); _ = release.wait(timeout: .now() + 3) }
      }
      return .available(.init(currentBytes: 10, lifetimeHighWaterBytes: 20, processStartAbsoluteTime: 1))
    }
  }
  func testStopJoinsAnInFlightTimerQueryBeforeReturning() throws {
    let samples = BlockingSample()
    let value = try H3NativeMemoryMonitor(parentPID: 10, childPID: 20, sampler: { samples.sample($0) })
    try value.start()
    XCTAssertEqual(samples.entered.wait(timeout: .now() + 2), .success)
    let stopped = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { value.stop(); stopped.signal() }
    XCTAssertEqual(stopped.wait(timeout: .now() + 0.05), .timedOut)
    samples.release.signal()
    XCTAssertEqual(stopped.wait(timeout: .now() + 2), .success)
    XCTAssertEqual(value.report.lifecycle, .stopped)
  }
  func testPublicDarwinSelfSnapshotAndUnavailablePIDDoNotUseMLX() {
    switch H3NativeMemoryMonitor.systemSample(getpid()) {
    case .available(let sample):
      XCTAssertGreaterThan(sample.currentBytes, 0)
      XCTAssertGreaterThan(sample.lifetimeHighWaterBytes, 0)
      XCTAssertGreaterThan(sample.processStartAbsoluteTime, 0)
    case .unavailable(let error): XCTFail("Self proc_pid_rusage unavailable: \(error)")
    }
    if case .unavailable = H3NativeMemoryMonitor.systemSample(-1) {} else { XCTFail("Invalid PID admitted") }
  }
}
