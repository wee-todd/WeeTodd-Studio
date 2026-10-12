import XCTest
@testable import StudioCore

final class AudioPresentationTests: XCTestCase {
  func testCaptionsFollowTrimAndRegionPositionAndVoiceWinsOverlap() throws {
    var p = StudioProject(); p.audioPresentation = AudioPresentation()
    p.audioPresentation?.voiceCaptions = true; p.audioPresentation?.lyrics = true
    let v = AudioTrack(name: "Voice", role: .voice), m = AudioTrack(name: "Music", role: .music)
    p.audioTracks = [v,m]
    var a = AudioRegion(assetID: UUID(), path: "v"); a.start = 10; a.sourceIn = 2; a.duration = 4; a.trackID = v.id
    a.captions = [AudioCaption(start: 3, end: 4, text: "Hello")]
    var b = AudioRegion(assetID: UUID(), path: "m"); b.start = 10; b.duration = 5; b.trackID = m.id
    b.captions = [AudioCaption(start: 1, end: 2, text: "Sing")]; p.audio = [a,b]
    XCTAssertEqual(p.captionText(at: 11.5), "Hello")
    p.audioPresentation?.showBothCaptions = true
    XCTAssertEqual(p.captionText(at: 11.5), "Hello\nSing")
    p.audioPresentation?.voiceCaptions = false
    XCTAssertEqual(p.captionText(at: 11.5), "Sing")
    XCTAssertEqual(p.captionText(at: 12), "")
    p.audio[1].instrumental = true
    XCTAssertEqual(p.captionText(at: 11.5), "")
  }
  func testCaptionEditsDoNotChangeAudioPlaybackPlan() {
    var p = StudioProject(); var a = AudioRegion(assetID: UUID(), path: "a")
    a.duration = 5; p.audio = [a]
    let before = TimelinePlaybackPlan(project:p)
    p.audio[0].captions = [AudioCaption(start:0,end:1,text:"Edited")]
    XCTAssertEqual(before,TimelinePlaybackPlan(project:p))
  }
  func testTitleIsDrawnAboveEmptyVideoVisualizer() throws {
    var p = StudioProject(); p.audioPresentation = AudioPresentation()
    var title = TitleOverlay(); title.text = "VISIBLE TITLE"; title.position = "center"; p.titles = [title]
    let withTitle = try XCTUnwrap(AudioPresentationDrawing.image(size:CGSize(width:640,height:360),project:p,features:nil,time:1))
    p.titles = []
    let withoutTitle = try XCTUnwrap(AudioPresentationDrawing.image(size:CGSize(width:640,height:360),project:p,features:nil,time:1))
    XCTAssertNotEqual(withTitle.dataProvider?.data as Data?, withoutTitle.dataProvider?.data as Data?)
  }
  func testPresentationValidationAndRoundTrip() throws {
    var p = StudioProject(); p.audioPresentation = AudioPresentation()
    p.audioPresentation?.sensitivity = .nan
    XCTAssertThrowsError(try p.audioPresentation?.validate())
    p.audioPresentation?.sensitivity = 1
    let copy = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(p))
    XCTAssertEqual(copy.audioPresentation, p.audioPresentation)
  }
}
