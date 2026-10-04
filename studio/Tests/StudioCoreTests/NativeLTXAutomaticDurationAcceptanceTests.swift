import XCTest
@testable import StudioCore

final class NativeLTXAutomaticDurationAcceptanceTests: XCTestCase {
  private func fixture(frames: Int) -> ([String:Any],[String:Any]) {
    let policy: [String:Any] = ["minimum_seconds":1.0,"maximum_seconds":20.0,
      "head_checkpoint_path":"/models/head.safetensors","head_header_sha256":String(repeating:"a",count:64)]
    var actual = policy; actual["predicted_duration_seconds"] = 5.0; actual["resolved_frames"] = frames; actual["fps"] = 24.0
    return (["durationMode":"automatic","automaticDuration":policy,"nativeFPS":24.0],
      ["nativeRuntime":"swift-mlx","use_complete_duration":true,"usable_duration":Double(frames)/24,"usable_source_in":0.0,
       "metadata":["automatic_duration":actual,"frames":frames,"fps":24.0,"video_seconds":Double(frames)/24]])
  }
  func testAdoptsCompleteResolvedVideoDurationBothShorterAndLongerWithoutAudioTailCrop() throws {
    for frames in [65,241] {
      let (prepared,result) = fixture(frames:frames), video = Double(frames)/24
      XCTAssertEqual(try NativeLTXAutomaticDurationAcceptance.duration(prepared:prepared,result:result,
        measuredVideoDuration:video,measuredContainerDuration:video+0.2,measuredFPS:24),video)
    }
  }
  func testManualIntentNeverAdoptsAutomaticResultFlag() throws {
    let (_,result) = fixture(frames:113)
    XCTAssertNil(try NativeLTXAutomaticDurationAcceptance.duration(prepared:["durationMode":"manual"],result:result,
      measuredVideoDuration:113.0/24,measuredContainerDuration:5,measuredFPS:24))
  }
  func testChangedFrozenPolicyAndWrongGeometryFailBeforeTimelineAcceptance() throws {
    let (prepared,result) = fixture(frames:113)
    for (key,value) in [("head_header_sha256",String(repeating:"b",count:64) as Any),
      ("head_checkpoint_path","/other/head.safetensors" as Any), ("minimum_seconds",2 as Any), ("resolved_frames",114 as Any)] {
      var altered = result, metadata = altered["metadata"] as! [String:Any], actual = metadata["automatic_duration"] as! [String:Any]
      actual[key] = value; metadata["automatic_duration"] = actual; altered["metadata"] = metadata
      XCTAssertThrowsError(try NativeLTXAutomaticDurationAcceptance.duration(prepared:prepared,result:altered,
        measuredVideoDuration:113.0/24,measuredContainerDuration:5,measuredFPS:24))
    }
    XCTAssertThrowsError(try NativeLTXAutomaticDurationAcceptance.duration(prepared:prepared,result:result,
      measuredVideoDuration:5,measuredContainerDuration:5,measuredFPS:24))
  }
}
