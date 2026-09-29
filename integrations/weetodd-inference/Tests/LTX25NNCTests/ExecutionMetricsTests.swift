import XCTest
@testable import LTX25NNC

final class ExecutionMetricsTests: XCTestCase {
  func testStackSeparatesPreparationComputeAndReuseWithoutChangingResults() throws {
    let c = try AVBlockConfiguration(videoDimension: 32, audioDimension: 16, heads: 2,
      videoHeadDimension: 16, audioHeadDimension: 8, videoTokens: 5, audioTokens: 3, textTokens: 4)
    let shapes = try AVBlockRunner.expectedWeightShapes(configuration: c)
    let bytes = shapes.values.reduce(UInt64(0)) { $0 + UInt64($1.reduce(4, *)) }
    let inputs = try AVBlockRunner.expectedInputShapes(configuration: c).mapValues {
      [Float](repeating: 0.1, count: $0.reduce(1, *))
    }
    let stack = try AVStackRunner(configuration: c, blockCount: 2)
    var snapshots: [ExecutionMetrics] = []
    let first = try stack.evaluate(inputs, retainWeights: true, weights: { _, _, shape in
      [Float](repeating: 0.01, count: shape.reduce(1, *))
    }) { snapshots.append($0.metrics) }
    XCTAssertEqual(snapshots.count, 2)
    XCTAssertEqual(snapshots.last?.decodedWeightBytes, bytes * 2)
    XCTAssertGreaterThan(snapshots.last!.preparationSeconds, 0)
    XCTAssertGreaterThan(snapshots.last!.installationSeconds, 0)
    XCTAssertGreaterThan(snapshots.last!.computeSeconds, 0)
    XCTAssertGreaterThan(snapshots.last!.healthCheckSeconds, 0)
    XCTAssertEqual(stack.graphBuildCount, 1)
    XCTAssertGreaterThan(snapshots.last!.graphBuildSeconds, 0)
    snapshots.removeAll()
    let second = try stack.evaluate(inputs, weights: { _, _, shape in
      [Float](repeating: 0.01, count: shape.reduce(1, *))
    }) { snapshots.append($0.metrics) }
    XCTAssertEqual(snapshots.last?.graphBuildSeconds, 0)
    XCTAssertEqual(stack.graphBuildCount, 1)
    XCTAssertEqual(first.video, second.video)
    XCTAssertEqual(first.audio, second.audio)
    XCTAssertEqual(stack.residentBlocks, 0)
  }
}
