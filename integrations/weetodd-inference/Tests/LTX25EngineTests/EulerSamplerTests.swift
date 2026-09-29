import Metal
import XCTest
@testable import LTX25Engine

final class EulerSamplerTests: XCTestCase {
  func evaluate(sample: [Float], prediction: [Float], noise: [Float]? = nil,
                sigma: Double, next: Double, eta: Double = 1,
                clean: [Float]? = nil, mask: [Float]? = nil, channels: Int = 1) throws -> [Float] {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
          let commands = queue.makeCommandBuffer() else { throw XCTSkip("Metal device is unavailable") }
    func buffer(_ values: [Float]) -> MTLBuffer {
      values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }
    let source = buffer(sample), denoised = buffer(prediction)
    let output = device.makeBuffer(length: sample.count * 4, options: .storageModeShared)!
    let sampler = try MetalEulerSampler(device: device)
    try sampler.encode(commandBuffer: commands, sample: source, denoised: denoised,
      noise: noise.map(buffer), clean: clean.map(buffer), mask: mask.map(buffer),
      output: output, elementCount: sample.count, channels: channels,
      step: EulerStep(sigma: sigma, nextSigma: next, eta: eta))
    commands.commit(); commands.waitUntilCompleted()
    XCTAssertEqual(commands.status, .completed, String(describing: commands.error))
    return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: sample.count))
  }

  func testDeterministicStepDoesNotReapplyFractionalGuideMask() throws {
    let output = try evaluate(sample: [2, 4, 6, 8], prediction: [0, 0, 0, 0],
      sigma: 1, next: 0.5, eta: 0, clean: [1, 1, 3, 3], mask: [0.5, 0], channels: 2)
    XCTAssertEqual(output, [1.25, 2.25, 4.5, 5.5])
  }

  func testAncestralStepMatchesIndependentRectifiedFlowFixture() throws {
    // sigma=1,next=.5,eta=1 => down=.25, alpha ratio=2/3, noise=sqrt(2/9).
    let output = try evaluate(sample: [3, -3], prediction: [1, 1], noise: [1, -1],
      sigma: 1, next: 0.5)
    XCTAssertEqual(output[0], 1.47140452, accuracy: 1e-6)
    XCTAssertEqual(output[1], -0.47140452, accuracy: 1e-6)
  }

  func testAncestralNoiseCannotMovePinnedGuide() throws {
    let output = try evaluate(sample: [3, 3], prediction: [0, 0], noise: [1, 1],
      sigma: 1, next: 0.5, clean: [0.75, 0.75], mask: [0, 1])
    XCTAssertEqual(output[0], 0.75)
    XCTAssertEqual(output[1], 0.97140452, accuracy: 1e-6)
  }

  func testTerminalStepDoesNotReadNoiseOrSource() throws {
    let output = try evaluate(sample: [.nan, .nan], prediction: [0.125, -0.25],
      sigma: 0.421875, next: 0, clean: [7, 7], mask: [1, 0])
    XCTAssertEqual(output, [0.125, 7])
  }

  func testRejectsMissingNoiseAndIncompatibleMaskBeforeDispatch() throws {
    XCTAssertThrowsError(try evaluate(sample: [1], prediction: [0], sigma: 1, next: 0.5))
    XCTAssertThrowsError(try evaluate(sample: [1, 2], prediction: [0], sigma: 1, next: 0.5, eta: 0))
    XCTAssertThrowsError(try evaluate(sample: [1, 2], prediction: [0, 0], sigma: 1, next: 0.5, eta: 0,
      clean: [1, 1], mask: [1, 1], channels: 3))
    XCTAssertThrowsError(try evaluate(sample: [1], prediction: [0], sigma: 1, next: 0.5, eta: 0, mask: [1]))
  }

  func testRejectsInvalidScheduleCoefficients() {
    for (sigma, next, eta) in [(0.0, 0.0, 1.0), (0.5, 0.6, 1), (1, 0.5, 2),
                                (Double.nan, 0.5, 1), (1, -0.1, 1), (1, 0.5, -1)] {
      XCTAssertThrowsError(try EulerStep(sigma: sigma, nextSigma: next, eta: eta))
    }
  }
}
