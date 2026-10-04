import Foundation
import XCTest
@testable import StudioCore

final class NativeH3MotionFidelityAcceptanceTests:XCTestCase {
  private func fixture(noop:Bool=false)->([String:Any],[String:Any],[String:Any]) {
    let frames=60,seconds=2.5,padded=noop ? 73 : 124
    let source:[String:Any]=["path":"/original/movie.mov","sha256":String(repeating:"a",count:64),"sourceStartSeconds":0.5,
      "sourceDurationSeconds":seconds,"sourceFrames":frames,"width":64,"height":32]
    let motion:[String:Any]=["version":1,"sourcePath":"/original/movie.mov","sourceSHA256":String(repeating:"a",count:64),"sourceIn":0.5,
      "sourceDuration":seconds,"sourceFrames":frames,"expandedFrames":noop ? frames : 120,"paddedFrames":padded,
      "recovery":(0..<frames).map {noop ? $0 : $0*2},"noop":noop,"sourceMediaInspected":true,"sourceAudioPreserved":true,
      "actualSamplingEvaluations":noop ? 0 : 8]
    let metadata:[String:Any]=["nativeRuntime":"swift-mlx","task":"motion_fidelity","frames":frames,"fps":24,
      "sampledFrames":noop ? 0 : padded,"audioSampleRate":32000,"audioSamplesPerChannel":80000,"motionFidelity":motion]
    return(["task":"motion_fidelity","motion_source":source],["nativeRuntime":"swift-mlx","use_complete_duration":true,
      "usable_source_in":0.0,"usable_duration":seconds,"metadata":metadata],
      ["videoDuration":seconds,"duration":seconds+0.02,"fps":24,"width":64,"height":32,"frames":frames])
  }
  func testExpandedAndNoopAcceptExactSourceDurationInsteadOfSamplingDuration() throws {
    for noop in [false,true] {
      let(prepared,result,media)=fixture(noop:noop)
      XCTAssertEqual(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:media),2.5)
    }
    XCTAssertNil(try NativeH3MotionFidelityAcceptance.duration(prepared:nil,result:[:],media:[:]))
  }
  func testWrongSourceGeometryCountAndSamplingStateRejectBeforeAdoption() throws {
    let(prepared,result,media)=fixture()
    var wrong=result;var metadata=wrong["metadata"] as! [String:Any],motion=metadata["motionFidelity"] as! [String:Any]
    motion["sourceSHA256"]=String(repeating:"b",count:64);metadata["motionFidelity"]=motion;wrong["metadata"]=metadata
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:wrong,media:media))
    var changed=media;changed["videoDuration"]=Double(124)/24
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:changed))
    changed=media;changed["frames"]=59
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:changed))
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:nil,result:result,media:media))
    let(noopPrepared,noopResult,noopMedia)=fixture(noop:true)
    wrong=noopResult;metadata=wrong["metadata"] as! [String:Any];motion=metadata["motionFidelity"] as! [String:Any]
    motion["actualSamplingEvaluations"]=1;metadata["motionFidelity"]=motion;wrong["metadata"]=metadata
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:noopPrepared,result:wrong,media:noopMedia))
  }
  func testFiniteStrictNumericsAndOptionalMeasuredFrameCount() throws {
    let(prepared,result,media)=fixture();var changed=media
    changed.removeValue(forKey:"frames")
    XCTAssertEqual(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:changed),2.5)
    changed["fps"]=true
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:changed))
    changed=media;changed["videoDuration"]=Double.nan
    XCTAssertThrowsError(try NativeH3MotionFidelityAcceptance.duration(prepared:prepared,result:result,media:changed))
  }
}
