import XCTest
@testable import StudioCore

final class NativeLTXMovieClipCopyTests:XCTestCase {
  func testCopyRetainsSelectedSourceIntervalAndLeavesAcceptedSourceUntouched() throws {
    var source=Clip(name:"Accepted source",engine:.movie)
    source.sourcePath="/source.mp4";source.sourceIn=2;source.duration=3;source.seed=42
    let before=source
    var media=MediaAsset(name:source.name,kind:.video,path:source.sourcePath)
    media.duration=8;media.fps=30;media.width=768;media.height=448
    let made=try NativeLTXMovieClipCopy.create(source:source,media:media)
    XCTAssertEqual(source,before);XCTAssertNotEqual(made.clip.id,source.id)
    XCTAssertEqual(made.clip.duration,3);XCTAssertEqual(made.clip.seed,42)
    XCTAssertEqual(made.clip.engine,.ltx25);XCTAssertEqual(made.clip.inferredTask,"video_upscale")
    XCTAssertEqual(made.clip.attachments.first?.sourceStartSeconds,2)
    XCTAssertEqual(made.clip.attachments.first?.sourceDurationSeconds,3)
    XCTAssertEqual(made.asset.owner,made.clip.id);XCTAssertEqual(made.asset.path,source.sourcePath)
    XCTAssertFalse(made.clip.generationSelection!.ltx25MovieUpscale!.experimentalEnabled)
    let restored=try JSONDecoder().decode(Clip.self,from:JSONEncoder().encode(made.clip))
    XCTAssertEqual(restored,made.clip)
    media.kind = .image
    XCTAssertThrowsError(try NativeLTXMovieClipCopy.create(source:source,media:media))
  }
  func testOldAttachmentsKeepTheirSerializedFieldsAndInvalidIntervalsReject() throws {
    let old=Attachment(assetID:UUID(),role:.reference)
    let fields=try JSONSerialization.jsonObject(with:JSONEncoder().encode(old)) as! [String:Any]
    XCTAssertNil(fields["sourceStartSeconds"]);XCTAssertNil(fields["sourceDurationSeconds"])
    XCTAssertEqual(try JSONDecoder().decode(Attachment.self,from:JSONEncoder().encode(old)),old)
    var source=Clip(engine:.movie);source.sourcePath="/source.mp4";source.duration=10
    var media=MediaAsset(name:"Source",kind:.video,path:source.sourcePath);media.duration=2;media.fps=24
    XCTAssertThrowsError(try NativeLTXMovieClipCopy.create(source:source,media:media))
  }
}
