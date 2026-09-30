import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class SceneAssemblyTests: XCTestCase {
  func testDecodeWindowPlanCoversTimelineWithExactPixelOverlap() throws {
    let geometry = try AVGeometry(width: 384, height: 256, frames: 97, fps: 24)
    let plan = try MLXSceneDecodeWindowPlan(geometry: geometry,
      maximumActivationBytes: Int.max, maximumWindowFrames: 57)
    XCTAssertEqual(plan.latentRanges, [0..<8, 4..<12, 8..<13])
    XCTAssertEqual(plan.overlapFrames, 25)
    XCTAssertLessThan(plan.admittedActivationBytes,
      try MLXSceneMediaPublisher.requiredVideoActivationBytes(geometry: geometry))
  }

  func testDecodeWindowPlanRejectsUnalignedOrUnadmittedWindow() throws {
    let geometry = try AVGeometry(width: 384, height: 256, frames: 97, fps: 24)
    XCTAssertThrowsError(try MLXSceneDecodeWindowPlan(geometry: geometry,
      maximumActivationBytes: Int.max, maximumWindowFrames: 48))
    XCTAssertThrowsError(try MLXSceneDecodeWindowPlan(geometry: geometry,
      maximumActivationBytes: Int.max, maximumWindowFrames: 49))
    XCTAssertThrowsError(try MLXSceneDecodeWindowPlan(geometry: geometry,
      maximumActivationBytes: 1))
  }
  func testRGBJoinerBlends25FramesAndOmitsFinalCausalFrame() throws {
    var joiner = try MLXSceneRGBJoiner(windowCount: 2, frameBytes: 3)
    var output: [Data] = []
    for index in 0..<49 {
      try joiner.receive(window: 0, frame: index, count: 49,
        rgb: Data(repeating: 10, count: 3)) { frame, bytes in
          XCTAssertEqual(frame, output.count); output.append(bytes)
        }
    }
    for index in 0..<73 {
      try joiner.receive(window: 1, frame: index, count: 73,
        rgb: Data(repeating: 20, count: 3)) { frame, bytes in
          XCTAssertEqual(frame, output.count); output.append(bytes)
        }
    }
    try joiner.finish(expectedFrames: 96)
    XCTAssertEqual(output.count, 96)
    XCTAssertEqual(output[23][0], 10)
    XCTAssertEqual(output[24][0], 10)
    XCTAssertEqual(output[36][0], 15)
    XCTAssertEqual(output[48][0], 20)
    XCTAssertEqual(output[95][0], 20)
  }
  func testRGBJoinerCarriesExactOverlapAcrossThreeWindows() throws {
    var joiner = try MLXSceneRGBJoiner(windowCount: 3, frameBytes: 3)
    var output: [Data] = []
    for (window, count) in [57, 57, 33].enumerated() {
      for frame in 0..<count {
        try joiner.receive(window: window, frame: frame, count: count,
          rgb: Data(repeating: UInt8((window + 1) * 10), count: 3)) {
            index, bytes in
            XCTAssertEqual(index, output.count); output.append(bytes)
          }
      }
      try joiner.finishWindow(window: window)
    }
    try joiner.finish(expectedFrames: 96)
    XCTAssertEqual(output.count, 96)
    XCTAssertEqual(output[31][0], 10)
    XCTAssertEqual(output[32][0], 10)
    XCTAssertEqual(output[56][0], 20)
    XCTAssertEqual(output[64][0], 20)
    XCTAssertEqual(output[88][0], 30)
  }
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
