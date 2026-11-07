import XCTest
import MLX
@testable import RelayVSR

final class RelayMathTests: XCTestCase {
  func testGeometryPadsInputWithoutChangingDeliveredSize() throws {
    let geometry = try RelayGeometry(width: 33, height: 17)
    XCTAssertEqual(geometry.paddedWidth, 48)
    XCTAssertEqual(geometry.paddedHeight, 32)
    XCTAssertEqual(geometry.outputWidth, 66)
    XCTAssertEqual(geometry.outputHeight, 34)
    XCTAssertThrowsError(try RelayGeometry(width: 0, height: 17))
  }

  func testFlowScheduleMatchesFourStepWorkflow() throws {
    let sigmas = try RelaySchedule.sigmas(steps: 4)
    XCTAssertEqual(sigmas, [1, 24.0 / 25.0, 8.0 / 9.0, 8.0 / 11.0, 0])
    XCTAssertEqual(try RelaySchedule.sigmas(steps: 1), [1, 0])
    XCTAssertThrowsError(try RelaySchedule.sigmas(steps: 0))
  }

  func testMemoryPreflightAccountsForSolverHistory() throws {
    let geometry = try RelayGeometry(width: 1920, height: 1080)
    let budget: UInt64 = 32 * 1024 * 1024 * 1024
    XCTAssertThrowsError(try RelayMemoryPlan.validate(geometry: geometry, options: .init(),
      budget: budget))
    XCTAssertNoThrow(try RelayMemoryPlan.validate(geometry: geometry, options: .init(steps: 1),
      budget: budget))
    XCTAssertNoThrow(try RelayMemoryPlan.validate(geometry: geometry, options: .init(frameLimit: 1),
      budget: budget))
  }

  func testEndpointWindowsShareOnlyBoundaryFrame() throws {
    func windows(_ count: Int) -> [[Int]] {
      var buffer = RelayWindowBuffer<Int>(interval: 15)
      var emitted = (0..<count).compactMap { buffer.append($0) }
      if let last = buffer.finish() { emitted.append(last) }
      return emitted
    }
    XCTAssertEqual(windows(1), [[0]])
    XCTAssertEqual(windows(16), [Array(0...15)])
    XCTAssertEqual(windows(32), [Array(0...15), Array(15...30), [30, 31]])
  }

  func testWindowBoundaryUsesTheCheckpointSpecificRule() {
    let sliding = RelayWindow(height: 4, width: 4, slidesAtEdge: true)
    let truncated = RelayWindow(height: 4, width: 4, slidesAtEdge: false)
    XCTAssertEqual(sliding.range(at: 0, extent: 9, size: 4), 0..<4)
    XCTAssertEqual(truncated.range(at: 0, extent: 9, size: 4), 0..<2)
    XCTAssertEqual(sliding.range(at: 8, extent: 9, size: 4), 5..<9)
    XCTAssertEqual(truncated.range(at: 8, extent: 9, size: 4), 6..<9)
  }

  func testMetalBicubicUsesHalfPixelCoordinatesAndReplicatesEdges() {
    let pixels = MLXArray([Float(0), 0, 0, 1, 1, 1], [1, 2, 3])
    let output = RelayBicubic.upsample(pixels)
    XCTAssertEqual(output.shape, [2, 4, 3])
    let samples = output[0, 0..<4, 0].asArray(Float.self)
    XCTAssertEqual(samples, [-0.10546875, 0.2265625, 0.7734375, 1.10546875])
  }

  func testRotatedInt8ProjectionRetainsSignedWeights() {
    var bytes = Array(repeating: Int8(0), count: 32 * 256)
    bytes[3] = 16
    bytes[256 + 3] = -16
    let scales = full([32, 1], values: MLXArray(Float(0.0625)))
    let projection = RelayLinear(weight: MLXArray(bytes, [32, 256]), bias: nil,
      scale: scales, rotation: RelayLinear.rotation())
    var vector = Array(repeating: Float(0), count: 256)
    vector[0] = 1
    let result = projection(MLXArray(vector, [1, 256]).asType(.bfloat16)).asArray(Float.self)
    XCTAssertEqual(result[0], -0.0625, accuracy: 0.001)
    XCTAssertEqual(result[1], 0.0625, accuracy: 0.001)
    XCTAssertEqual(result[2], 0, accuracy: 0.001)
  }

  func testTiledAttentionMatchesIndependentPerPlaneWindows() {
    let height = 9
    let width = 17
    let tokens = height * width
    let windows = [RelayWindow(height: 4, width: 4, slidesAtEdge: true),
      RelayWindow(height: 2, width: 2, slidesAtEdge: false)]
    let plan = RelayAttentionPlan(height: height, width: width, windows: windows)
    let q = MLXArray.zeros([tokens, 2, 64])
    let k = MLXArray.zeros([tokens, 1, 64])
    let values = (0..<2).map { plane in
      let grid = MLXArray((0..<tokens).map { Float(plane * tokens + $0) }, [tokens, 1, 1])
      return RelayKV(key: k, value: broadcast(grid, to: [tokens, 1, 64]))
    }
    let output = plan(q, values)[0..<tokens, 1, 0].asArray(Float.self)
    for token in output.indices {
      var expected: [Float] = []
      for (plane, window) in windows.enumerated() {
        for y in window.range(at: token / width, extent: height, size: window.height) {
          for x in window.range(at: token % width, extent: width, size: window.width) {
            expected.append(Float(plane * tokens + y * width + x))
          }
        }
      }
      XCTAssertEqual(output[token], expected.reduce(0, +) / Float(expected.count), accuracy: 0.001)
    }
  }
}
