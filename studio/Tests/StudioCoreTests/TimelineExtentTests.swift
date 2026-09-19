import Foundation
import XCTest
@testable import StudioCore

final class TimelineExtentTests: XCTestCase {
  func testFullSongExtendsEditingCanvasWithoutExtendingVideoExport() {
    var project = StudioProject(); var clip = Clip(); clip.duration = 12.333
    project.clips = [clip]
    var song = AudioRegion(assetID: UUID(), path: "/tmp/song.wav"); song.start = 0; song.duration = 445.3987
    project.audio = [song]
    XCTAssertEqual(project.timelineContentDuration, 445.3987)
    XCTAssertEqual(project.duration, 12.333)
    XCTAssertEqual(TimelinePlaybackPlan(project: project).duration, 12.333)
  }
  func testOffsetTitlesAndAudioDetermineTheVisibleExtent() {
    var project = StudioProject()
    var title = TitleOverlay(); title.start = 100; title.duration = 4
    var song = AudioRegion(assetID: UUID(), path: "/tmp/song.wav"); song.start = 30; song.duration = 60
    project.titles = [title]; project.audio = [song]
    XCTAssertEqual(project.timelineContentDuration, 104)
    project.titles[0].duration = .infinity
    XCTAssertEqual(project.timelineContentDuration, 90)
  }
}
