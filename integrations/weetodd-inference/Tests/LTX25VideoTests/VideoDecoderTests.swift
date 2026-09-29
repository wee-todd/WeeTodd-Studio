import Foundation
import XCTest
import InferenceTestSupport
import TensorIO
@testable import LTX25Video

final class VideoDecoderTests: XCTestCase {
  func testFusedLayerResidencyAdmitsUsefulGeometryWithinExistingBudget() throws {
    var c = VideoDecodeConfiguration()
    let plan = try VideoDecodePlan(shape: [1,128,5,9,16],configuration: c)
    XCTAssertEqual(plan.outputShape,[33,288,512,3])
    XCTAssertLessThanOrEqual(plan.admittedActivationBytes,512*1024*1024)
    c.fuseNormalization = false
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,5,9,16],configuration: c))
    c = VideoDecodeConfiguration(); c.windowsPerCommand = 0
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,2,2,2],configuration: c))
  }
  func testAdmissionValidatesTimingShapeAndPeakBeforeAllocating() throws {
    let config = VideoDecodeConfiguration()
    let plan = try VideoDecodePlan(shape: [1,128,3,4,5], configuration: config)
    XCTAssertEqual(plan.outputShape, [17,128,160,3])
    XCTAssertEqual(plan.duration, 17.0 / 24.0)
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,Int.max,4,5], configuration: config))
    XCTAssertThrowsError(try VideoDecodePlan(shape: [2,128,3,4,5], configuration: config))
    var invalid = config; invalid.frameRate = .nan
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,3,4,5], configuration: invalid))
    invalid = config; invalid.frameRate = Double.leastNonzeroMagnitude
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,3,4,5], configuration: invalid))
    invalid = config; invalid.maximumActivationBytes = 100
    XCTAssertThrowsError(try VideoDecodePlan(shape: [1,128,3,4,5], configuration: invalid))
  }
  func testMissingDecoderFailsAtHeaderPreflight() throws {
    try withTensorFile(tensors: [("wrong", [1], "F32")]) { url in
      XCTAssertThrowsError(try VideoDecoder(checkpoint: url))
    }
  }
  func testRealCheckpointDecodeMatchesMLX() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let checkpoint = environment["WEETODD_VIDEO_VAE"],
      let oraclePath = environment["WEETODD_VIDEO_ORACLE"] else {
      throw XCTSkip("Set WEETODD_VIDEO_VAE and WEETODD_VIDEO_ORACLE for installed-weight qualification")
    }
    struct Oracle: Decodable { let shape: [Int]; let latent: [Float]; let output: [Float] }
    let oracle: Oracle
    if oraclePath.hasSuffix(".safetensors") {
      let file = try SafeTensorFile(url: URL(fileURLWithPath: oraclePath))
      let shape = try XCTUnwrap(file.tensors["latent"]?.shape).map(Int.init)
      oracle = Oracle(shape: shape,latent: try file.readFloat32(named: "latent"),
        output: try file.readFloat32(named: "output",maximumBytes: 256*1024*1024))
    } else {
      oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
    }
    let decoder = try VideoDecoder(checkpoint: URL(fileURLWithPath: checkpoint))
    var rejected = VideoDecodeConfiguration(); rejected.maximumActivationBytes = 100
    XCTAssertThrowsError(try decoder.decode(latent: oracle.latent, shape: oracle.shape,
      configuration: rejected, receive: { _ in XCTFail("Rejected request emitted output") }))
    XCTAssertEqual(decoder.residentWeightBytes, 0)
    var cancelledWithResidentWeights = false
    XCTAssertThrowsError(try decoder.decode(latent: oracle.latent, shape: oracle.shape,
      checkCancelled: {
        if decoder.residentWeightBytes > 0 {
          cancelledWithResidentWeights = true
          throw CancellationError()
        }
      }, receive: { _ in XCTFail("Cancelled weighted stage emitted output") }))
    XCTAssertTrue(cancelledWithResidentWeights)
    XCTAssertEqual(decoder.residentWeightBytes, 0)
    var pixels: [Float] = [], peakWeights = 0
    let started = Date()
    try decoder.decode(latent: oracle.latent, shape: oracle.shape,
      checkCancelled: { peakWeights = max(peakWeights,decoder.residentWeightBytes) }) { chunk in
      XCTAssertEqual(chunk.startFrame, pixels.count / (chunk.width * chunk.height * 3))
      pixels.append(contentsOf: chunk.rgb)
    }
    XCTAssertEqual(pixels.count, oracle.output.count)
    var square: Double = 0, peak: Float = 0
    for (actual, expected) in zip(pixels,oracle.output) {
      let delta = abs(actual-expected); peak = max(peak,delta); square += Double(delta*delta)
    }
    let rmse = sqrt(square / Double(pixels.count))
    print("VIDEO_PARITY count=\(pixels.count) peak=\(peak) rmse=\(rmse) seconds=\(Date().timeIntervalSince(started)) peakWeightBytes=\(peakWeights)")
    XCTAssertLessThan(peak, 0.01)
    XCTAssertLessThan(rmse, 0.001)
    XCTAssertEqual(decoder.residentWeightBytes, 0)
    XCTAssertThrowsError(try decoder.decode(latent: oracle.latent, shape: oracle.shape,
      checkCancelled: { throw CancellationError() }, receive: { _ in XCTFail("Cancelled decode emitted output") }))
    XCTAssertEqual(decoder.residentWeightBytes, 0)
    // Consumer failure requires a full decode before the first frame callback.
    // Exercise it on the tiny fixture without duplicating large qualification renders.
    if oracle.output.count < 1_000_000 {
      XCTAssertThrowsError(try decoder.decode(latent: oracle.latent, shape: oracle.shape,
        receive: { _ in throw CancellationError() }))
    }
    XCTAssertEqual(decoder.residentWeightBytes, 0)
  }
}
