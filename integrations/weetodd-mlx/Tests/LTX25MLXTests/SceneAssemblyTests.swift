import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class SceneAssemblyTests: XCTestCase {
  func testPublicationExcludesCausalFrameAndTrimsAudioToEditorialDuration() throws {
    let plan = try LTX25ScenePlan(durations: [2, 2], fps: 24)
    XCTAssertEqual(plan.totalFrames, 97)
    XCTAssertEqual(MLXSceneMediaPublisher.deliveredFrames(plan: plan), 96)
    XCTAssertEqual(MLXSceneMediaPublisher.deliveredAudioFrames(plan: plan), 192000)
  }
  func testContinuationKeepsInteriorVideoAndExactAudioTail() throws {
    let video = MLXArray((0..<7).flatMap { index in
      [Float](repeating: Float(index), count: 128)
    }, [7, 128])
    let audio = MLXArray((0..<52).flatMap { index in
      [Float](repeating: Float(index), count: 128)
    }, [52, 128])
    let visual = try MLXSceneSampler.interiorVideoTail(video,
      latentFrames: 7, pixels: 1, overlapFrames: 4)
    let sound = try MLXSceneSampler.audioTail(audio, count: 27)
    XCTAssertEqual(visual.shape, [3, 128])
    XCTAssertEqual([visual[0,0].item(Float.self),visual[2,0].item(Float.self)], [3,5])
    XCTAssertEqual(sound.shape, [27, 128])
    XCTAssertEqual(sound[0,0].item(Float.self), 25)
    XCTAssertEqual(sound[26,0].item(Float.self), 51)
  }
  func testCausalVideoBlendAndJointAudioTrimMatchPlan() throws {
    let plan = try LTX25ScenePlan(durations: [2, 3], fps: 24)
    let firstVideo = MLXArray((0..<7).flatMap { index in
      [Float](repeating: Float(index), count: 128)
    }, [7, 128])
    let secondVideo = MLXArray((100..<113).flatMap { index in
      [Float](repeating: Float(index), count: 128)
    }, [13, 128])
    let firstAudio = MLXArray.ones([52, 128])
    let secondAudio = MLXArray.ones([102, 128]) * 2
    let output = try MLXSceneLatentAssembly.assemble(video: [firstVideo, secondVideo],
      audio: [firstAudio, secondAudio], plan: plan, latentHeight: 1, latentWidth: 1)
    XCTAssertEqual(output.video.shape, [16, 128])
    XCTAssertEqual(output.audio.shape, [127, 128])
    XCTAssertEqual(output.video[4, 0].item(Float.self), 4)
    XCTAssertEqual(output.video[5, 0].item(Float.self), 53.5)
    XCTAssertEqual(output.video[6, 0].item(Float.self), 103)
    XCTAssertEqual(output.audio[51, 0].item(Float.self), 1)
    XCTAssertEqual(output.audio[52, 0].item(Float.self), 2)
  }
}
