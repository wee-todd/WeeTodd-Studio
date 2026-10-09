import XCTest
@testable import H3MLX

final class H3PreparationObservationTests: XCTestCase {
  func testDefaultPathDoesNotReadClockAndReturnsBodyValueOnce() {
    var clockCalls = 0
    var bodyCalls = 0
    let value = H3PreparationObservation.measure(.conditionProjection, observer: nil,
      clock: { clockCalls += 1; return 0 }) {
      bodyCalls += 1
      return Float(3.25) * Float(7.5)
    }
    XCTAssertEqual(value.bitPattern, (Float(3.25) * Float(7.5)).bitPattern)
    XCTAssertEqual(bodyCalls, 1)
    XCTAssertEqual(clockCalls, 0)
  }

  func testSequentialCallbacksFollowExistingBodyCompletionsAndPreserveValues() {
    var events: [String] = []
    var measurements: [H3PreparationObservation.Measurement] = []
    var now: UInt64 = 0
    let observer: H3PreparationObservation.Observer = {
      measurements.append($0)
      events.append($0.stage.rawValue)
    }
    let stages: [H3PreparationObservation.Stage] = [.conditionProjection,
      .tokenRefinement, .timeAndSmallMetadata, .adalnTables, .terminalCleanup]
    var result: [Int] = []
    for (index, stage) in stages.enumerated() {
      result.append(H3PreparationObservation.measure(stage, observer: observer,
        clock: { defer { now += 500_000_000 }; return now }) {
        events.append("body_\(index)")
        return index * index
      })
    }
    XCTAssertEqual(result, [0, 1, 4, 9, 16])
    XCTAssertEqual(events, stages.enumerated().flatMap { ["body_\($0.offset)", $0.element.rawValue] })
    XCTAssertEqual(measurements.map(\.stage), stages)
    XCTAssertTrue(measurements.allSatisfy { $0.succeeded && $0.seconds == 0.5 })
  }

  func testOriginalErrorPropagatesAndFailureIsNotReportedAsCompletion() {
    enum Stop: Error { case expected }
    var measurements: [H3PreparationObservation.Measurement] = []
    var ticks: UInt64 = 0
    XCTAssertThrowsError(try H3PreparationObservation.measure(.tokenRefinement,
      observer: { measurements.append($0) },
      clock: { defer { ticks += 250_000_000 }; return ticks }) {
      throw Stop.expected
    }) { XCTAssertTrue($0 is Stop) }
    XCTAssertEqual(measurements, [.init(stage: .tokenRefinement,
      seconds: 0.25, succeeded: false)])
  }

  func testCancellationRemainsTheOriginalErrorAndCleanupCanStillBeObserved() {
    var records: [H3PreparationObservation.Measurement] = []
    let observer: H3PreparationObservation.Observer = { records.append($0) }
    XCTAssertThrowsError(try H3PreparationObservation.measure(.adalnTables,
      observer: observer, clock: { 5 }) { throw CancellationError() }) {
      XCTAssertTrue($0 is CancellationError)
    }
    H3PreparationObservation.measure(.terminalCleanup, observer: observer, clock: { 5 }) {}
    XCTAssertEqual(records.map(\.stage), [.adalnTables, .terminalCleanup])
    XCTAssertEqual(records.map(\.succeeded), [false, true])
  }
}
