import XCTest
@testable import StudioCore

final class NativeLTXMovieAcceptanceTests:XCTestCase {
  func testMovieAdoptsExactLongSourceRatherThanFiveSecondDefaultOrAACDuration() throws {
    let duration=Double(721)/30,sha=String(repeating:"a",count:64),rgb=String(repeating:"b",count:64)
    let prepared:[String:Any]=["task":"video_upscale","sourceFrames":721,"fps":30,"width":768,"height":512,
      "sourceMovieSHA256":sha,"sourceRGBSHA256":rgb]
    let metadata:[String:Any]=["task":"video_upscale","pythonModelInference":false,"frames":721,"fps":30,"width":768,"height":512,
      "source_movie_sha256":sha,"source_rgb_sha256":rgb]
    var result:[String:Any]=["nativeRuntime":"swift-mlx","use_complete_duration":true,"usable_source_in":0,"usable_duration":duration,"metadata":metadata]
    var media:[String:Any]=["fps":30,"width":768,"height":512,"videoDuration":duration,"duration":duration+0.01]
    XCTAssertEqual(try NativeLTXMovieAcceptance.duration(prepared:prepared,result:result,media:media),duration)
    result["usable_duration"]=5
    XCTAssertThrowsError(try NativeLTXMovieAcceptance.duration(prepared:prepared,result:result,media:media))
    result["usable_duration"]=duration;media["videoDuration"]=duration-1.0/30
    XCTAssertThrowsError(try NativeLTXMovieAcceptance.duration(prepared:prepared,result:result,media:media))
  }
  func testOnlyAnExactPreparedMovieContractMayChangeEditorialDuration() throws {
    XCTAssertNil(try NativeLTXMovieAcceptance.duration(prepared:["task":"t2v"],result:[:],media:[:]))
    XCTAssertThrowsError(try NativeLTXMovieAcceptance.duration(prepared:nil,result:["metadata":["task":"video_upscale"]],media:[:]))
    XCTAssertThrowsError(try NativeLTXMovieAcceptance.duration(prepared:["task":"video_upscale","sourceFrames":true],result:[:],media:[:]))
  }
}
