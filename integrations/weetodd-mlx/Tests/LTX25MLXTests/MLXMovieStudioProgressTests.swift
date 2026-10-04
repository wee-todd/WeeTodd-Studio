import XCTest
@testable import LTX25MLX

final class MLXMovieStudioProgressTests:XCTestCase {
  func testChunkRefinementProgressAndResumeRemainMonotonicUntilPublication() {
    var progress=MLXMovieStudioProgress(chunks:3)
    let events:[(String,Int,Int)]=[("movie_chunk_reused",1,3),("movie_chunk_started",1,3),
      ("movie_source_video_encode",21,42),("movie_source_video_encode",42,42),
      ("movie_sampling",0,3),("movie_refine:transformer",24,48),("movie_refine:transformer",48,48),
      ("movie_sampling",1,3),("movie_sampling",3,3),("movie_video_layers",42,42),
      ("movie_chunk_completed",2,3),("movie_chunk_started",2,3),("movie_sampling",3,3),
      ("movie_chunk_completed",3,3),("ready_to_publish",1,1)]
    let reports=events.map { progress.event(stage:$0.0,completed:$0.1,total:$0.2) }
    let fractions=reports.map { $0["fraction"] as! Double }
    XCTAssertEqual(fractions,fractions.sorted());XCTAssertLessThan(fractions.last!,1)
    XCTAssertGreaterThan(fractions[5],fractions[4]);XCTAssertGreaterThan(fractions[6],fractions[5])
    XCTAssertTrue((reports[5]["message"] as! String).contains("chunk 2/3"))
    XCTAssertTrue((reports[5]["message"] as! String).contains("1/3"))
    XCTAssertTrue((reports[8]["message"] as! String).contains("3/3 updates"))
  }
  func testLatentOnlyChunksAdvanceWithoutPretendingToSample() {
    var progress=MLXMovieStudioProgress(chunks:2)
    _=progress.event(stage:"movie_chunk_started",completed:0,total:2)
    let source=progress.event(stage:"movie_source_video_encode",completed:42,total:42)
    let upscale=progress.event(stage:"movie_upscaler_weights_released",completed:1,total:1)
    let decode=progress.event(stage:"movie_video_layers",completed:42,total:42)
    let complete=progress.event(stage:"movie_chunk_completed",completed:1,total:2)
    XCTAssertGreaterThan(upscale["fraction"] as! Double,source["fraction"] as! Double)
    XCTAssertGreaterThan(decode["fraction"] as! Double,upscale["fraction"] as! Double)
    XCTAssertGreaterThan(complete["fraction"] as! Double,decode["fraction"] as! Double)
    XCTAssertFalse((upscale["message"] as! String).contains("update"))
  }
}
