import XCTest
import InferenceTestSupport
@testable import LTX25Audio

final class AudioTests: XCTestCase {
  func testTransposeConvolutionOrientationAndLength() throws {
    let x = AudioTensor([1, 2], height: 2, width: 1, channels: 1)
    let y = try AudioMath.transposeConv(x, weight: [1, 2, 3], bias: [0], outputChannels: 1, kernel: 3, stride: 2, padding: 1)
    XCTAssertEqual(y.values, [2, 5, 4])
  }
  func testCausalConvolutionDoesNotSeeFuture() throws {
    let x = AudioTensor([1, 2, 4], height: 3, width: 1, channels: 1)
    let y = try AudioMath.conv(x, weight: [1, 1, 1], bias: [0], outputChannels: 1, kernelHeight: 3, kernelWidth: 1, padTop: 2, padLeft: 0)
    XCTAssertEqual(y.values, [1, 3, 7])
  }
  func testReplicateTransposeFilterEdges() {
    let x = AudioTensor([1, 2], height: 2, width: 1, channels: 1)
    let y = AudioMath.upsampleFilter(x, filter: [0.25, 0.5, 0.25, 0], ratio: 2, inputPad: 1, cropLeft: 3)
    XCTAssertEqual(y.values, [1, 1.5, 2, 2])
  }
  func testResamplingPreservesDCAndExactLength() {
    let x = AudioTensor([Float](repeating: 0.3, count: 10), height: 5, width: 1, channels: 2)
    let y = AudioMath.resample48k(x)
    XCTAssertEqual(y.height, 15)
    for v in y.values { XCTAssertEqual(v, 0.3, accuracy: 0.0003) }
  }
  func testHannDecoderResamplerUsesReplicatedEndpointImpulses() {
    // Independent scalar sinc equation sampled at t/3-n, with n=-7...T+6
    // and endpoints extended before filtering. These intentionally differ from
    // generic comfy.audio.resample's zero-padded boundary impulses.
    let x = AudioTensor([1, 0, 0, 0, 0, 0, 0, 1], height: 4, width: 1, channels: 2)
    let expectedLeft: [Float] = [0.995019302, 0.690420391, 0.310259128, 0.005019302,
      -0.125408018, -0.095778582, -0.004322319, 0.053031782, 0.044564467,
      0.003217925, -0.023323899, -0.019557552]
    let expectedRight: [Float] = [0.003217925, 0.044564467, 0.053031782, -0.004322319,
      -0.095778582, -0.125408018, 0.005019302, 0.310259128, 0.690420391,
      0.995019302, 1.126087537, 1.096458101]
    let actual = AudioMath.resample48k(x)
    XCTAssertEqual(actual.height, 12)
    for t in 0..<12 {
      XCTAssertEqual(actual.values[t * 2], expectedLeft[t], accuracy: 1e-6)
      XCTAssertEqual(actual.values[t * 2 + 1], expectedRight[t], accuracy: 1e-6)
    }
  }
  func testCheckpointAndBudgetFailBeforeWeightAllocation() throws {
    try withTensorFile(tensors: [("unrelated", [1], "F32")]) { url in
      XCTAssertThrowsError(try AudioDecoder(checkpoint: url))
      XCTAssertThrowsError(try AudioDecoder(checkpoint: url, maximumLatentFrames: Int.max))
      XCTAssertThrowsError(try AudioDecoder(checkpoint: url, maximumResidentBytes: 1))
    }
  }
  func testDurationContract() throws {
    XCTAssertEqual(try AudioDecoder.sampleCount(latentFrames: 1), 480)
    XCTAssertEqual(try AudioDecoder.sampleCount(latentFrames: 26), 48480)
    XCTAssertThrowsError(try AudioDecoder.sampleCount(latentFrames: 0))
    XCTAssertThrowsError(try AudioDecoder.sampleCount(latentFrames: Int.max))
    XCTAssertLessThan(try AudioDecoder.estimatedPeakBytes(latentFrames: 501), 2 * 1024 * 1024 * 1024)
    XCTAssertGreaterThan(try AudioDecoder.estimatedPeakBytes(latentFrames: 1501), 2 * 1024 * 1024 * 1024)
  }
}

extension AudioTests {
  func testGPUConvolutionAndTransposeMatchScalarReference() throws {
    let matrix = try AudioMatrixEngine()
    let x = AudioTensor((0..<30).map { sin(Float($0) * 0.3) }, height: 5, width: 3, channels: 2)
    let w = (0..<54).map { cos(Float($0) * 0.2) }
    let cpu = try AudioMath.conv(x, weight: w, bias: [0.1, 0.2, 0.3], outputChannels: 3, kernelHeight: 3, kernelWidth: 3, padTop: 2, padLeft: 1)
    let gpu = try AudioMath.conv(x, weight: w, bias: [0.1, 0.2, 0.3], outputChannels: 3, kernelHeight: 3, kernelWidth: 3, padTop: 2, padLeft: 1, matrix: matrix)
    for (a, b) in zip(cpu.values, gpu.values) { XCTAssertEqual(a, b, accuracy: 2e-5) }
    let z = AudioTensor([1, 2, 3, 4, 5, 6], height: 3, width: 1, channels: 2)
    let tw = (0..<18).map { Float($0) * 0.1 }
    let tcpu = try AudioMath.transposeConv(z, weight: tw, bias: [0.1, 0.2, 0.3], outputChannels: 3, kernel: 3, stride: 2, padding: 1)
    let tgpu = try AudioMath.transposeConv(z, weight: tw, bias: [0.1, 0.2, 0.3], outputChannels: 3, kernel: 3, stride: 2, padding: 1, matrix: matrix)
    for (a, b) in zip(tcpu.values, tgpu.values) { XCTAssertEqual(a, b, accuracy: 2e-5) }
  }
  func testRealCheckpointAgainstReleasedMLXOracle() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_AUDIO_CHECKPOINT"],
      let oracle = ProcessInfo.processInfo.environment["WEETODD_AUDIO_ORACLE"] else {
      throw XCTSkip("Set checkpoint and independently exported oracle for real audio qualification")
    }
    struct Oracle: Decodable { let frames: Int; let latent: [Float]; let waveform: [Float] }
    let reference = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: URL(fileURLWithPath: oracle)))
    let decoder = try AudioDecoder(checkpoint: URL(fileURLWithPath: checkpoint))
    let start = Date()
    var visitedStages: [String] = []
    let output = try decoder.decode(latent: reference.latent, latentFrames: reference.frames) { stage in
      print("audio stage: \(stage)")
      visitedStages.append(stage)
      // The identical valid request would allocate and decode again without the
      // single-evaluation guard, while the outer stage still retains its tensors.
      XCTAssertThrowsError(try decoder.decode(latent: reference.latent, latentFrames: reference.frames)) { error in
        if case AudioError.invalid(let message) = error {
          XCTAssertEqual(message, "Audio decoder is already evaluating")
        } else {
          XCTFail("Unexpected recursive decode error: \(error)")
        }
      }
    }
    XCTAssertEqual(visitedStages, ["audio_vae", "vocoder", "bandwidth_extension"])
    XCTAssertEqual(output.sampleRate, 48000)
    XCTAssertEqual(output.samples.count, reference.waveform.count)
    XCTAssertThrowsError(try decoder.decode(latent: reference.latent, latentFrames: reference.frames, maximumSamples: output.frameCount + 1))
    XCTAssertThrowsError(try decoder.decode(latent: [Float.nan], latentFrames: reference.frames))
    var squared: Double = 0, maximum: Float = 0
    for (a, b) in zip(output.samples, reference.waveform) { let d = abs(a - b); maximum = max(maximum, d); squared += Double(d * d) }
    let rmse = sqrt(squared / Double(output.samples.count))
    print("audio qualification seconds=\(Date().timeIntervalSince(start)) maximum=\(maximum) rmse=\(rmse)")
    XCTAssertLessThan(maximum, 0.003)
    XCTAssertLessThan(rmse, 0.0005)
  }
}
