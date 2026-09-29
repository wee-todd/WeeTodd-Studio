import XCTest
@testable import LTX25NNC

final class AVStackTests: XCTestCase {
  private func configuration() throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
  }
  private func weights(_ index: Int, _ name: String, _ shape: [Int]) -> [Float] {
    let seed = name.utf8.reduce(index * 7) { $0 + Int($1) }
    let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return (0..<shape.reduce(1, *)).map { Float(($0 * 17 + seed) % 31 - 15) / 128 + norm }
  }
  private func inputs(_ config: AVBlockConfiguration) throws -> [String: [Float]] {
    try AVBlockRunner.expectedInputShapes(configuration: config).mapValues { shape in
      (0..<shape.reduce(1, *)).map { Float($0 % 19 - 9) / 16 }
    }
  }

  func testResidentChainMatchesSerialBlocksAndReleasesByDefault() throws {
    let config = try configuration(), initial = try inputs(config)
    let serial = try AVBlockRunner(configuration: config)
    var reference = initial
    for index in 0..<3 {
      try serial.load { self.weights(index, $0, $1) }
      let output = try serial.evaluate(reference)
      reference["video"] = output.video; reference["audio"] = output.audio
    }
    let stack = try AVStackRunner(configuration: config, blockCount: 3)
    var progress: [Int] = []
    let output = try stack.evaluate(initial, weights: weights) { event in
      progress.append(event.completedBlocks)
      XCTAssertEqual(event.residentBlocks, 1)
      XCTAssertEqual(stack.residentBlocks, 1)
    }
    XCTAssertEqual(progress, [1, 2, 3])
    XCTAssertEqual(output.video, reference["video"])
    XCTAssertEqual(output.audio, reference["audio"])
    XCTAssertEqual(stack.residentBlocks, 0)
    XCTAssertEqual(stack.lastTransferCounts.activationUploads, initial.count)
    XCTAssertEqual(stack.lastTransferCounts.activationDownloads, 2)
  }

  func testPrefetchedStackMatchesSerialAndDrainsOnObserverFailure() throws {
    enum Stop: Error { case now }
    let c = try configuration(), input = try inputs(c)
    let runner = try AVStackRunner(configuration: c, blockCount: 3)
    let reference = try runner.evaluate(input, weights: weights)
    var sawPrefetch = false
    let actual = try runner.evaluate(input, maximumPreparationBytes: 1024 * 1024, weights: weights) {
      sawPrefetch = sawPrefetch || $0.metrics.preparedWeightBytes > 0
    }
    XCTAssertTrue(sawPrefetch)
    XCTAssertEqual(reference.video, actual.video)
    XCTAssertEqual(reference.audio, actual.audio)
    XCTAssertThrowsError(try runner.evaluate(input, retainWeights: true,
      maximumPreparationBytes: 1024 * 1024, weights: weights) { _ in throw Stop.now })
    XCTAssertEqual(runner.residentBlocks, 0)
    let retry = try runner.evaluate(input, weights: weights)
    XCTAssertEqual(reference.video, retry.video)
    XCTAssertEqual(reference.audio, retry.audio)
  }

  func testFailureDropsWarmSlotAndCanRetryWithoutStaleActivations() throws {
    enum Failure: Error { case stop }
    let config = try configuration(), initial = try inputs(config)
    let stack = try AVStackRunner(configuration: config, blockCount: 2)
    let first = try stack.evaluate(initial, retainWeights: true, weights: weights)
    XCTAssertEqual(stack.residentBlocks, 1)
    XCTAssertThrowsError(try stack.evaluate(initial, retainWeights: true, weights: weights) { _ in throw Failure.stop })
    XCTAssertEqual(stack.residentBlocks, 0)
    let second = try stack.evaluate(initial, weights: weights)
    XCTAssertEqual(first.video, second.video)
    XCTAssertEqual(first.audio, second.audio)
    XCTAssertEqual(stack.residentBlocks, 0)
  }

  func testInvalidInputsNeverCallProviderAndReentrantOperationsAreRejected() throws {
    let config = try configuration(), initial = try inputs(config)
    let stack = try AVStackRunner(configuration: config, blockCount: 2)
    XCTAssertThrowsError(try stack.evaluate([:], weights: { _, _, _ in XCTFail("Must preflight first"); return [] }))
    _ = try stack.evaluate(initial, retainWeights: true, weights: weights) { _ in
      XCTAssertThrowsError(try stack.release())
      XCTAssertThrowsError(try stack.evaluate(initial, weights: self.weights))
    }
    try stack.release()
    XCTAssertEqual(stack.residentBlocks, 0)
    XCTAssertThrowsError(try AVStackRunner(configuration: config, blockCount: 0))
    XCTAssertThrowsError(try AVStackRunner(configuration: config, blockCount: 49))
  }

  func testNonfiniteGPUOutputStopsBeforeLoadingNextBlock() throws {
    let config = try configuration(), initial = try inputs(config)
    let stack = try AVStackRunner(configuration: config, blockCount: 3)
    var loaded: Set<Int> = []
    XCTAssertThrowsError(try stack.evaluate(initial, retainWeights: true, weights: { index, _, shape in
      loaded.insert(index)
      return [Float](repeating: 1e20, count: shape.reduce(1, *))
    }) { _ in XCTFail("Must not report successful progress for nonfinite output") })
    XCTAssertEqual(loaded, [0])
    XCTAssertEqual(stack.residentBlocks, 0)
  }

  @MainActor
  func testCancellationAfterProgressReleasesSlotAndStopsBeforeNextLoad() async throws {
    let config = try configuration(), initial = try inputs(config)
    let result = try await Task {
      let stack = try AVStackRunner(configuration: config, blockCount: 3)
      var loaded: Set<Int> = []
      do {
        _ = try stack.evaluate(initial, retainWeights: true, weights: { index, _, shape in
          loaded.insert(index)
          return [Float](repeating: 0, count: shape.reduce(1, *))
        }) { _ in withUnsafeCurrentTask { $0?.cancel() } }
        return false
      } catch is CancellationError { return loaded == [0] && stack.residentBlocks == 0 }
    }.value
    XCTAssertTrue(result)
  }
}
