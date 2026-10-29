import XCTest

@testable import Drama3Contracts

final class StreamingTextTests: XCTestCase {
  func testLookaheadAndBoundedInput() async throws {
    let input = Drama3TextInput(capacity: 2)
    try await input.append("Hello.")
    try await input.append("World.")
    do {
      try await input.append("Too much.")
      XCTFail("Must apply backpressure")
    } catch {}
    let first = try await input.next()
    XCTAssertEqual(first?.text, "Hello.")
    XCTAssertEqual(first?.lookahead, "World.")
    await input.finish()
    let last = try await input.next()
    XCTAssertEqual(last?.text, "World.")
    XCTAssertNil(last?.lookahead)
    let end = try await input.next()
    XCTAssertNil(end)
  }
  func testCancelWakesWaitingConsumer() async throws {
    let input = Drama3TextInput()
    let waiting = Task { try await input.next() }
    await input.cancel()
    do {
      _ = try await waiting.value
      XCTFail("Must cancel")
    } catch is CancellationError {} catch { XCTFail("Unexpected \(error)") }
  }
  func testChunkBoundaryKeepsAtMostThreeSpeakers() throws {
    let chunks = try Drama3Text.chunks(
      (0..<8).map { "<|speaker:\($0)|> line \($0)." }.joined(separator: "\n"), limit: 10000)
    XCTAssertGreaterThan(chunks.count, 2)
    for chunk in chunks {
      XCTAssertLessThanOrEqual(
        Set(
          chunk.components(separatedBy: "<|speaker:").dropFirst().map {
            $0.prefix(while: { $0 != "|" })
          }
        ).count, 3)
    }
  }
}
