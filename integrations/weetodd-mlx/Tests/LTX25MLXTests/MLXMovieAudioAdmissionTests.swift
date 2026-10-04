import XCTest
import MLX
@testable import LTX25MLX

final class MLXMovieAudioAdmissionTests: XCTestCase {
  func testExactLongestClockAndWorkspaceRejectBeforeFFT() throws {
    let max=try MLXAudioMelPlan(samples:960639)
    XCTAssertEqual(max.melFrames,6004);XCTAssertEqual(max.latentFrames,1501)
    XCTAssertLessThan(max.ownedBufferBytes,128*1024*1024)
    XCTAssertThrowsError(try MLXAudioMelPlan(samples:960640))
    XCTAssertThrowsError(try MLXAudioMelPlan(samples:960639,maximumOwnedBufferBytes:max.ownedBufferBytes-1))
    XCTAssertEqual(try MLXAudioEncoder.latentFrames(melFrames:6004,maximumMelFrames:6004),1501)
    XCTAssertThrowsError(try MLXAudioEncoder.latentFrames(melFrames:6005,maximumMelFrames:6004))
    // Existing default remains conservative and existing short-clip API exact.
    XCTAssertThrowsError(try MLXAudioEncoder.latentFrames(melFrames:2012))
    XCTAssertEqual(try MLXAudioEncoder.latentFrames(melFrames:2001),501)
  }
  func testLowWorkspaceAndCancellationDoNotReachFFT() async throws {
    let planar=[Float](repeating:0,count:16000)
    XCTAssertThrowsError(try MLXAudioMel.encode(planar:planar,sampleRate:16000,maximumOwnedBufferBytes:1))
    let task=Task<Void,Error>.detached { @Sendable in
      withUnsafeCurrentTask { $0?.cancel() }
      _ = try MLXAudioMel.encode(planar:[Float](repeating:0,count:16000),sampleRate:16000)
    }
    do { try await task.value;XCTFail("Canceled audio reached FFT") }
    catch { XCTAssertTrue(error is CancellationError) }
  }
  func testExtendedEncoderWorkspaceRejectsBeforeAnyWeightOrActivationLoad() throws {
    let plan=try MLXAudioEncodePlan(melFrames:6004,maximumMelFrames:6004)
    XCTAssertEqual(plan.latentFrames,1501)
    XCTAssertLessThan(plan.ownedBufferBytes,2*1024*1024*1024)
    XCTAssertThrowsError(try MLXAudioEncodePlan(melFrames:6004,maximumMelFrames:6004,maximumOwnedBufferBytes:plan.ownedBufferBytes-1))
    XCTAssertThrowsError(try MLXAudioEncodePlan(melFrames:6005,maximumMelFrames:6004))
    XCTAssertThrowsError(try MLXAudioEncodePlan(melFrames:2012))
  }

}
