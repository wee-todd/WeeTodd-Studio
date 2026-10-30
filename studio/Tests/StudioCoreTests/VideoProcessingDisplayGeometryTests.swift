import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingDisplayGeometryTests:XCTestCase {
  func testRotatedSourceCanonicalizesDisplayedPixelsWithoutClockChange() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let fixture=try VideoProcessingFixture(root:root),runtime=try fixture.plan().runtime
    let original=root.appendingPathComponent("upright.mp4"),rotated=root.appendingPathComponent("rotated.mp4")
    try VideoProcessingFixture.ffmpeg(["-f","lavfi","-i","testsrc2=size=96x64:rate=24:duration=1","-an","-c:v","libx264",original.path])
    try VideoProcessingFixture.ffmpeg(["-display_rotation","90","-i",original.path,"-c","copy",rotated.path])
    let cancel=NativeHeadlessCancellation(),source=try FlashVSRFile.capture(rotated)
    let media=try FlashVSRMediaInfo.inspect(rotated,runtime:runtime,directory:root,cancellation:cancel)
    XCTAssertEqual(media.width,64);XCTAssertEqual(media.height,96)
    let (_,actual,_)=try VideoProcessingSequenceExecutor.canonical(source,media:media,output:root.appendingPathComponent("canonical.mkv"),runtime:runtime,directory:root,cancellation:cancel)
    XCTAssertEqual(actual.width,64);XCTAssertEqual(actual.height,96)
    XCTAssertEqual(actual.frames,24);XCTAssertEqual(actual.fps.value,24)
  }
  func testQuarterTurnMetadataAndUnsupportedRotationAdmission() throws {
    func read(_ value:Any) throws -> FlashVSRMediaInfo {
      let summary=try JSONSerialization.data(withJSONObject:["streams":[["codec_type":"video","width":96,"height":64,"avg_frame_rate":"24/1","pix_fmt":"yuv420p","side_data_list":[["side_data_type":"Display Matrix","rotation":value]]]]])
      return try FlashVSRMediaInfo.parse(summary:summary,timestamps:Data("0\n".utf8))
    }
    for turn in [-270,-90,90,270] {let media=try read(turn);XCTAssertEqual(media.width,64);XCTAssertEqual(media.height,96)}
    for turn in [-360,-180,0,180,360] {let media=try read(turn);XCTAssertEqual(media.width,96);XCTAssertEqual(media.height,64)}
    for turn:Any in [45,"invalid"] {XCTAssertThrowsError(try read(turn))}
  }
}
